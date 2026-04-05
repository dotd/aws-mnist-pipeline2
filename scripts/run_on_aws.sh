#!/usr/bin/env bash
#
# Automates the full AWS training workflow:
#   1. Build & push Docker image to ECR
#   2. Create security group + key pair (if needed)
#   3. Launch EC2 GPU instance
#   4. Wait for instance, SSH in, pull image, run training
#   5. Sync results to S3
#   6. Terminate instance
#
# Usage:
#   ./scripts/run_on_aws.sh                          # Train MNIST (default)
#   ./scripts/run_on_aws.sh unet                     # Train U-Net
#   ./scripts/run_on_aws.sh mnist --epochs 5         # Pass extra args
#   ./scripts/run_on_aws.sh unet --epochs 25 --wandb # U-Net with wandb
#   ./scripts/run_on_aws.sh unet --epochs 25 --wandb --wandb-project my-project # U-Net with wandb custom project

set -euo pipefail

# =============================================================================
# Configuration — edit these to match your setup
# =============================================================================
AWS_ACCOUNT_ID="148761683501"
AWS_REGION="us-east-1"
ECR_REPO_NAME="dotan-fr-my-cv-model"
IMAGE_TAG="latest"
INSTANCE_TYPE="g4dn.xlarge"
KEY_NAME="training-key"
KEY_FILE="$HOME/.ssh/${KEY_NAME}.pem"
SECURITY_GROUP_NAME="training-sg"
S3_BUCKET=""  # Set to your bucket name, e.g. "my-training-results". Leave empty to skip S3 sync.
VOLUME_SIZE_GB=50

# Deep Learning AMI (Ubuntu) — update if needed for your region
# This finds the latest Deep Learning AMI automatically
AMI_ID=""

# =============================================================================
# Derived values
# =============================================================================
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${ECR_URI}/${ECR_REPO_NAME}:${IMAGE_TAG}"

# Parse pipeline and extra args
PIPELINE="${1:-mnist}"
shift 2>/dev/null || true
EXTRA_ARGS="$*"

if [[ "$PIPELINE" == "mnist" ]]; then
    DOCKER_CMD="python run.py mnist ${EXTRA_ARGS}"
elif [[ "$PIPELINE" == "unet" ]]; then
    DOCKER_CMD="python run.py unet ${EXTRA_ARGS}"
else
    echo "Error: unknown pipeline '${PIPELINE}'. Use 'mnist' or 'unet'."
    exit 1
fi

RUN_NAME="${PIPELINE}-$(date +%Y%m%d-%H%M%S)"

echo "============================================================"
echo "  AWS Training Pipeline"
echo "  Pipeline:  ${PIPELINE}"
echo "  Instance:  ${INSTANCE_TYPE}"
echo "  Run name:  ${RUN_NAME}"
echo "============================================================"

# =============================================================================
# Step 1: Build and push Docker image to ECR
# =============================================================================
echo ""
echo "[Step 1/7] Building and pushing Docker image..."

# Authenticate with ECR
aws ecr get-login-password --region "${AWS_REGION}" | \
  docker login --username AWS --password-stdin "${ECR_URI}"

# Create ECR repo if it doesn't exist
aws ecr describe-repositories --repository-names "${ECR_REPO_NAME}" --region "${AWS_REGION}" 2>/dev/null || \
  aws ecr create-repository --repository-name "${ECR_REPO_NAME}" --region "${AWS_REGION}"

# Build for linux/amd64
docker buildx build --platform linux/amd64 -t "${ECR_REPO_NAME}:${IMAGE_TAG}" .

# Tag and push
docker tag "${ECR_REPO_NAME}:${IMAGE_TAG}" "${IMAGE_URI}"
docker push "${IMAGE_URI}"

echo "Image pushed: ${IMAGE_URI}"

# =============================================================================
# Step 2: Find the latest Deep Learning AMI
# =============================================================================
echo ""
echo "[Step 2/7] Finding latest Deep Learning AMI..."

if [[ -z "$AMI_ID" ]]; then
    AMI_ID=$(aws ec2 describe-images \
        --region "${AWS_REGION}" \
        --owners amazon \
        --filters "Name=name,Values=Deep Learning AMI (Ubuntu 20.04)*" "Name=state,Values=available" \
        --query "Images | sort_by(@, &CreationDate) | [-1].ImageId" \
        --output text)
fi

echo "Using AMI: ${AMI_ID}"

# =============================================================================
# Step 3: Create key pair (if needed)
# =============================================================================
echo ""
echo "[Step 3/7] Setting up key pair..."

if ! aws ec2 describe-key-pairs --key-names "${KEY_NAME}" --region "${AWS_REGION}" 2>/dev/null; then
    echo "Creating key pair: ${KEY_NAME}"
    aws ec2 create-key-pair \
        --key-name "${KEY_NAME}" \
        --region "${AWS_REGION}" \
        --query "KeyMaterial" \
        --output text > "${KEY_FILE}"
    chmod 400 "${KEY_FILE}"
    echo "Key saved to: ${KEY_FILE}"
else
    echo "Key pair '${KEY_NAME}' already exists."
fi

# =============================================================================
# Step 4: Create security group (if needed)
# =============================================================================
echo ""
echo "[Step 4/7] Setting up security group..."

SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=${SECURITY_GROUP_NAME}" \
    --region "${AWS_REGION}" \
    --query "SecurityGroups[0].GroupId" \
    --output text 2>/dev/null)

if [[ "$SG_ID" == "None" ]] || [[ -z "$SG_ID" ]]; then
    echo "Creating security group: ${SECURITY_GROUP_NAME}"
    SG_ID=$(aws ec2 create-security-group \
        --group-name "${SECURITY_GROUP_NAME}" \
        --description "SSH access for training instances" \
        --region "${AWS_REGION}" \
        --query "GroupId" \
        --output text)

    MY_IP=$(curl -s https://checkip.amazonaws.com)
    aws ec2 authorize-security-group-ingress \
        --group-id "${SG_ID}" \
        --protocol tcp \
        --port 22 \
        --cidr "${MY_IP}/32" \
        --region "${AWS_REGION}"
    echo "Allowed SSH from ${MY_IP}"
else
    echo "Security group '${SECURITY_GROUP_NAME}' already exists (${SG_ID})."
fi

# =============================================================================
# Step 5: Launch EC2 instance
# =============================================================================
echo ""
echo "[Step 5/7] Launching EC2 instance (${INSTANCE_TYPE})..."

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "${AMI_ID}" \
    --instance-type "${INSTANCE_TYPE}" \
    --key-name "${KEY_NAME}" \
    --security-group-ids "${SG_ID}" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${VOLUME_SIZE_GB},\"VolumeType\":\"gp3\"}}]" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${RUN_NAME}}]" \
    --region "${AWS_REGION}" \
    --query "Instances[0].InstanceId" \
    --output text)

echo "Instance launched: ${INSTANCE_ID}"
echo "Waiting for instance to be running..."

aws ec2 wait instance-running --instance-ids "${INSTANCE_ID}" --region "${AWS_REGION}"

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "${INSTANCE_ID}" \
    --region "${AWS_REGION}" \
    --query "Reservations[0].Instances[0].PublicIpAddress" \
    --output text)

echo "Instance running at: ${PUBLIC_IP}"
echo "Waiting for SSH to become available..."

for i in $(seq 1 30); do
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i "${KEY_FILE}" ubuntu@"${PUBLIC_IP}" "echo ready" 2>/dev/null; then
        break
    fi
    echo "  Attempt ${i}/30 — waiting..."
    sleep 10
done

# =============================================================================
# Step 6: Run training on the instance
# =============================================================================
echo ""
echo "[Step 6/7] Running training on EC2..."

ssh -o StrictHostKeyChecking=no -i "${KEY_FILE}" ubuntu@"${PUBLIC_IP}" << REMOTE_SCRIPT
set -euo pipefail

echo "--- Authenticating with ECR ---"
aws ecr get-login-password --region ${AWS_REGION} | \
  docker login --username AWS --password-stdin ${ECR_URI}

echo "--- Pulling image ---"
docker pull ${IMAGE_URI}

echo "--- Starting training: ${DOCKER_CMD} ---"
mkdir -p /home/ubuntu/data /home/ubuntu/checkpoints

docker run --rm --gpus all \
  -v /home/ubuntu/data:/workspace/data \
  -v /home/ubuntu/checkpoints:/workspace/checkpoints \
  ${IMAGE_URI} \
  ${DOCKER_CMD}

echo "--- Training complete ---"
REMOTE_SCRIPT

# =============================================================================
# Step 7: Sync results to S3 and terminate
# =============================================================================
echo ""
echo "[Step 7/7] Collecting results and cleaning up..."

if [[ -n "$S3_BUCKET" ]]; then
    echo "Syncing checkpoints to s3://${S3_BUCKET}/${RUN_NAME}/..."
    ssh -o StrictHostKeyChecking=no -i "${KEY_FILE}" ubuntu@"${PUBLIC_IP}" \
        "aws s3 sync /home/ubuntu/checkpoints s3://${S3_BUCKET}/${RUN_NAME}/"
    echo "Results saved to s3://${S3_BUCKET}/${RUN_NAME}/"
else
    echo "No S3_BUCKET configured — skipping sync."
    echo "You can SSH in to retrieve results: ssh -i ${KEY_FILE} ubuntu@${PUBLIC_IP}"
fi

echo ""
read -rp "Terminate instance ${INSTANCE_ID}? [y/N] " confirm
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    aws ec2 terminate-instances --instance-ids "${INSTANCE_ID}" --region "${AWS_REGION}"
    echo "Instance ${INSTANCE_ID} terminated."
else
    echo "Instance left running. Don't forget to terminate it!"
    echo "  aws ec2 terminate-instances --instance-ids ${INSTANCE_ID} --region ${AWS_REGION}"
fi

echo ""
echo "============================================================"
echo "  Done! Run: ${RUN_NAME}"
echo "============================================================"
