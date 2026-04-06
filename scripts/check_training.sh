#!/usr/bin/env bash
#
# Check, monitor, and clean up a detached training run launched by run_on_aws.sh.
#
# Usage:
#   ./scripts/check_training.sh                    # List all runs
#   ./scripts/check_training.sh <run-name>         # Check if training is still running
#   ./scripts/check_training.sh <run-name> logs    # Tail the training log
#   ./scripts/check_training.sh <run-name> finish  # Sync results to S3 and terminate instance

set -euo pipefail

RUN_INFO_DIR="$HOME/.aws-training-runs"

# ---- List runs ----
if [[ $# -eq 0 ]]; then
    echo "Saved training runs:"
    echo ""
    if [[ -d "$RUN_INFO_DIR" ]] && ls "${RUN_INFO_DIR}"/*.env >/dev/null 2>&1; then
        for f in "${RUN_INFO_DIR}"/*.env; do
            source "$f"
            RUN=$(basename "$f" .env)
            # Check instance state
            STATE=$(aws ec2 describe-instances \
                --instance-ids "${INSTANCE_ID}" \
                --region "${AWS_REGION}" \
                --query "Reservations[0].Instances[0].State.Name" \
                --output text 2>/dev/null || echo "unknown")
            printf "  %-40s  %-12s  %s\n" "$RUN" "$STATE" "$INSTANCE_ID"
        done
    else
        echo "  No runs found in ${RUN_INFO_DIR}"
    fi
    echo ""
    echo "Usage: $0 <run-name> [logs|finish]"
    exit 0
fi

# ---- Load run info ----
RUN_NAME="$1"
ACTION="${2:-status}"
RUN_INFO_FILE="${RUN_INFO_DIR}/${RUN_NAME}.env"

if [[ ! -f "$RUN_INFO_FILE" ]]; then
    echo "Error: Run '${RUN_NAME}' not found."
    echo "Available runs:"
    ls "${RUN_INFO_DIR}"/*.env 2>/dev/null | xargs -I{} basename {} .env | sed 's/^/  /'
    exit 1
fi

source "$RUN_INFO_FILE"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -i ${KEY_FILE}"

# ---- Check instance is still running ----
INSTANCE_STATE=$(aws ec2 describe-instances \
    --instance-ids "${INSTANCE_ID}" \
    --region "${AWS_REGION}" \
    --query "Reservations[0].Instances[0].State.Name" \
    --output text 2>/dev/null || echo "unknown")

if [[ "$INSTANCE_STATE" != "running" ]]; then
    echo "Instance ${INSTANCE_ID} is '${INSTANCE_STATE}' — not running."
    echo "Cleaning up run info."
    rm -f "$RUN_INFO_FILE"
    exit 1
fi

# ---- Actions ----
case "$ACTION" in
    status)
        echo "Run:       ${RUN_NAME}"
        echo "Instance:  ${INSTANCE_ID} (${PUBLIC_IP})"
        echo "Pipeline:  ${PIPELINE}"
        echo "State:     ${INSTANCE_STATE}"
        echo ""

        # Check if docker container is still running
        CONTAINER_RUNNING=$(ssh ${SSH_OPTS} ubuntu@"${PUBLIC_IP}" \
            "docker ps --filter name=training-${PIPELINE} --format '{{.Status}}'" 2>/dev/null || echo "")

        if [[ -n "$CONTAINER_RUNNING" ]]; then
            echo "Training:  RUNNING (${CONTAINER_RUNNING})"
        else
            echo "Training:  FINISHED (or not started)"
        fi

        echo ""
        echo "Last 5 lines of log:"
        ssh ${SSH_OPTS} ubuntu@"${PUBLIC_IP}" "tail -5 /home/ubuntu/training.log 2>/dev/null" || echo "  (no log yet)"
        echo ""
        echo "Commands:"
        echo "  $0 ${RUN_NAME} logs     # Tail full log"
        echo "  $0 ${RUN_NAME} finish   # Sync results & terminate"
        ;;

    logs)
        echo "Tailing training log (Ctrl+C to stop)..."
        echo ""
        ssh ${SSH_OPTS} ubuntu@"${PUBLIC_IP}" "tail -f /home/ubuntu/training.log"
        ;;

    finish)
        # Check if still training
        CONTAINER_RUNNING=$(ssh ${SSH_OPTS} ubuntu@"${PUBLIC_IP}" \
            "docker ps --filter name=training-${PIPELINE} --format '{{.ID}}'" 2>/dev/null || echo "")

        if [[ -n "$CONTAINER_RUNNING" ]]; then
            echo "Warning: Training is still running!"
            read -rp "Stop training and proceed? [y/N] " confirm
            if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
                echo "Aborted."
                exit 0
            fi
            ssh ${SSH_OPTS} ubuntu@"${PUBLIC_IP}" "docker stop training-${PIPELINE}" 2>/dev/null || true
        fi

        # Sync results to S3
        if [[ -n "${S3_BUCKET}" ]]; then
            echo "Syncing checkpoints to s3://${S3_BUCKET}/${RUN_NAME}/..."
            ssh ${SSH_OPTS} ubuntu@"${PUBLIC_IP}" \
                "aws s3 sync /home/ubuntu/checkpoints s3://${S3_BUCKET}/${RUN_NAME}/"
            echo "Results saved to s3://${S3_BUCKET}/${RUN_NAME}/"
        else
            echo "No S3_BUCKET configured — skipping sync."
        fi

        # Download log
        echo "Downloading training log..."
        mkdir -p "logs"
        scp ${SSH_OPTS} ubuntu@"${PUBLIC_IP}":/home/ubuntu/training.log "logs/${RUN_NAME}.log" 2>/dev/null || true
        echo "Log saved to logs/${RUN_NAME}.log"

        # Terminate
        echo ""
        read -rp "Terminate instance ${INSTANCE_ID}? [y/N] " confirm
        if [[ "$confirm" =~ ^[Yy]$ ]]; then
            aws ec2 terminate-instances --instance-ids "${INSTANCE_ID}" --region "${AWS_REGION}"
            echo "Instance ${INSTANCE_ID} terminated."
            rm -f "$RUN_INFO_FILE"
        else
            echo "Instance left running."
            echo "  aws ec2 terminate-instances --instance-ids ${INSTANCE_ID} --region ${AWS_REGION}"
        fi
        ;;

    *)
        echo "Unknown action: ${ACTION}"
        echo "Usage: $0 <run-name> [status|logs|finish]"
        exit 1
        ;;
esac
