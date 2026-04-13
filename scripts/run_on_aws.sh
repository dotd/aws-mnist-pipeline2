#!/usr/bin/env bash
#
# Automates the full AWS training workflow:
#   1. Build base image (only if needed) and push to ECR
#   2. Find the latest Deep Learning AMI
#   3. Create key pair (if needed)
#   4. Create security group (update SSH IP)
#   5. Create IAM role + instance profile for ECR access (if needed)
#   6. Launch EC2 instance
#   7. Sync code to EC2
#   8. Launch training (detached) — safe to close laptop
#
# After training, use check_training.sh to monitor, sync results, and terminate.
#
# Usage:
#   ./scripts/run_on_aws.sh src.mnist.train_mnist --epochs 5
#   ./scripts/run_on_aws.sh src.UNET_PascalVOC_simple.train --epochs 25 --wandb
#   CONFIG_FILE=configs/custom.yaml ./scripts/run_on_aws.sh src.mnist.train_mnist --epochs 10
#
# Config: reads from configs/aws.yaml by default. Override with CONFIG_FILE env var.

set -euo pipefail

# =============================================================================
# Configuration — loaded from YAML config file
# =============================================================================
CONFIG_FILE="${CONFIG_FILE:-configs/aws.yaml}"

# Helper: extract a value from the YAML config using Python
yaml_get() {
    python3 -c "
import yaml, sys
with open('${CONFIG_FILE}') as f:
    cfg = yaml.safe_load(f)
keys = '$1'.split('.')
val = cfg
for k in keys:
    val = val[k]
print(val)
"
}

AWS_ACCOUNT_ID=$(yaml_get "aws.account_id")
AWS_REGION=$(yaml_get "aws.region")
ECR_REPO_NAME=$(yaml_get "ecr.repo_name")
IMAGE_TAG=$(yaml_get "ecr.image_tag")
INSTANCE_TYPE=$(yaml_get "ec2.instance_type")
KEY_NAME=$(yaml_get "ec2.key_name")
KEY_FILE=$(eval echo "$(yaml_get 'ec2.key_file')")  # eval to expand ~
SECURITY_GROUP_NAME=$(yaml_get "ec2.security_group_name")
VOLUME_SIZE_GB=$(yaml_get "ec2.volume_size_gb")
IAM_ROLE_NAME=$(yaml_get "iam.role_name")
IAM_INSTANCE_PROFILE_NAME=$(yaml_get "iam.instance_profile_name")
S3_BUCKET=$(yaml_get "s3.bucket")
WANDB_API_KEY=$(yaml_get "wandb.api_key")

# Deep Learning AMI (Ubuntu) — found automatically
AMI_ID=""

# =============================================================================
# Derived values
# =============================================================================
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${ECR_URI}/${ECR_REPO_NAME}:${IMAGE_TAG}"

# Parse pipeline and extra args
MODULE="${1:?Error: module is required (e.g. src.mnist.train_mnist)}"
shift
EXTRA_ARGS="$*"

# Build AWS flags for S3 sync and self-termination
AWS_TRAIN_FLAGS=""
if [[ -n "$S3_BUCKET" ]]; then
    S3_PREFIX="${MODULE}-$(date +%Y%m%d-%H%M%S)"
    AWS_TRAIN_FLAGS="--s3-bucket ${S3_BUCKET} --s3-prefix ${S3_PREFIX}"
fi
AWS_TRAIN_FLAGS="${AWS_TRAIN_FLAGS} --self-terminate"

DOCKER_CMD="python -m ${MODULE} ${EXTRA_ARGS} ${AWS_TRAIN_FLAGS}"

# Check wandb key if --wandb is requested
WANDB_ENV_FLAG=""
if echo "${EXTRA_ARGS}" | grep -q -- "--wandb"; then
    if [[ -z "$WANDB_API_KEY" ]]; then
        echo "Error: --wandb requested but WANDB_API_KEY is not set."
        echo "Set it via: export WANDB_API_KEY=your_key"
        exit 1
    fi
    WANDB_ENV_FLAG="-e WANDB_API_KEY=${WANDB_API_KEY}"
fi

RUN_NAME="${MODULE}-$(date +%Y%m%d-%H%M%S)"

echo "============================================================"
echo "  AWS Training Pipeline"
echo "  Module:    ${MODULE}"
echo "  Instance:  ${INSTANCE_TYPE}"
echo "  Run name:  ${RUN_NAME}"
echo "============================================================"

# =============================================================================
# Step 1/8: Build base image (only if needed) and push to ECR
# =============================================================================
echo ""
echo "[Step 1/8] Checking if base image rebuild is needed..."

# Authenticate with ECR
aws ecr get-login-password --region "${AWS_REGION}" | \
  docker login --username AWS --password-stdin "${ECR_URI}"

# Create ECR repo if it doesn't exist
aws ecr describe-repositories --repository-names "${ECR_REPO_NAME}" --region "${AWS_REGION}" 2>/dev/null || \
  aws ecr create-repository --repository-name "${ECR_REPO_NAME}" --region "${AWS_REGION}"

# Only rebuild if requirements.txt or Dockerfile changed, or image doesn't exist
NEEDS_BUILD=false
if ! docker image inspect "${ECR_REPO_NAME}:${IMAGE_TAG}" >/dev/null 2>&1; then
    echo "Image not found locally — building."
    NEEDS_BUILD=true
elif [[ "Dockerfile" -nt "$(docker image inspect "${ECR_REPO_NAME}:${IMAGE_TAG}" --format '{{.Created}}')" ]] || \
     [[ "requirements.txt" -nt "$(docker image inspect "${ECR_REPO_NAME}:${IMAGE_TAG}" --format '{{.Created}}')" ]]; then
    echo "Dockerfile or requirements.txt changed — rebuilding."
    NEEDS_BUILD=true
else
    echo "Base image is up to date — skipping rebuild."
fi

if [[ "$NEEDS_BUILD" == "true" ]]; then
    docker buildx build --platform linux/amd64 -t "${ECR_REPO_NAME}:${IMAGE_TAG}" .
    docker tag "${ECR_REPO_NAME}:${IMAGE_TAG}" "${IMAGE_URI}"
    docker push "${IMAGE_URI}"
    echo "Image pushed: ${IMAGE_URI}"
fi

# =============================================================================
# Step 2/8: Find the latest Deep Learning AMI
# =============================================================================
echo ""
echo "[Step 2/8] Finding latest Deep Learning AMI..."

if [[ -z "$AMI_ID" ]]; then
    AMI_ID=$(aws ec2 describe-images \
        --region "${AWS_REGION}" \
        --owners amazon \
        --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
                  "Name=state,Values=available" \
                  "Name=architecture,Values=x86_64" \
        --query "Images | sort_by(@, &CreationDate) | [-1].ImageId" \
        --output text)
fi

if [[ -z "$AMI_ID" ]] || [[ "$AMI_ID" == "None" ]]; then
    echo "Error: Could not find a Deep Learning AMI. Set AMI_ID manually in the script."
    exit 1
fi

echo "Using AMI: ${AMI_ID}"

# =============================================================================
# Step 3/8: Create key pair (if needed)
# =============================================================================
echo ""
echo "[Step 3/8] Setting up key pair..."

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
# Step 4/8: Create security group (if needed)
# =============================================================================
echo ""
echo "[Step 4/8] Setting up security group..."

SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=${SECURITY_GROUP_NAME}" \
    --region "${AWS_REGION}" \
    --query "SecurityGroups[0].GroupId" \
    --output text 2>/dev/null)

MY_IP=$(curl -s https://checkip.amazonaws.com)

if [[ "$SG_ID" == "None" ]] || [[ -z "$SG_ID" ]]; then
    echo "Creating security group: ${SECURITY_GROUP_NAME}"
    SG_ID=$(aws ec2 create-security-group \
        --group-name "${SECURITY_GROUP_NAME}" \
        --description "SSH access for training instances" \
        --region "${AWS_REGION}" \
        --query "GroupId" \
        --output text)
else
    echo "Security group '${SECURITY_GROUP_NAME}' already exists (${SG_ID})."
    # Revoke all existing SSH rules so we can set the current IP
    EXISTING_CIDRS=$(aws ec2 describe-security-groups \
        --group-ids "${SG_ID}" \
        --region "${AWS_REGION}" \
        --query "SecurityGroups[0].IpPermissions[?FromPort==\`22\`].IpRanges[].CidrIp" \
        --output text)
    for cidr in ${EXISTING_CIDRS}; do
        aws ec2 revoke-security-group-ingress \
            --group-id "${SG_ID}" \
            --protocol tcp \
            --port 22 \
            --cidr "${cidr}" \
            --region "${AWS_REGION}" 2>/dev/null || true
    done
fi

# Always ensure current IP is allowed
aws ec2 authorize-security-group-ingress \
    --group-id "${SG_ID}" \
    --protocol tcp \
    --port 22 \
    --cidr "${MY_IP}/32" \
    --region "${AWS_REGION}" 2>/dev/null || true
echo "SSH allowed from ${MY_IP}"

# =============================================================================
# Step 5/8: Create IAM role + instance profile for ECR access (if needed)
# =============================================================================
echo ""
echo "[Step 5/8] Setting up IAM instance profile..."

if ! aws iam get-role --role-name "${IAM_ROLE_NAME}" 2>/dev/null; then
    echo "Creating IAM role: ${IAM_ROLE_NAME}"
    aws iam create-role \
        --role-name "${IAM_ROLE_NAME}" \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }'

    aws iam attach-role-policy \
        --role-name "${IAM_ROLE_NAME}" \
        --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly

    aws iam attach-role-policy \
        --role-name "${IAM_ROLE_NAME}" \
        --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

    # Allow the instance to terminate itself
    aws iam put-role-policy \
        --role-name "${IAM_ROLE_NAME}" \
        --policy-name "self-terminate" \
        --policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Action": "ec2:TerminateInstances",
                "Resource": "*"
            }]
        }'
else
    echo "IAM role '${IAM_ROLE_NAME}' already exists."
fi

if ! aws iam get-instance-profile --instance-profile-name "${IAM_INSTANCE_PROFILE_NAME}" 2>/dev/null; then
    echo "Creating instance profile: ${IAM_INSTANCE_PROFILE_NAME}"
    aws iam create-instance-profile \
        --instance-profile-name "${IAM_INSTANCE_PROFILE_NAME}"

    aws iam add-role-to-instance-profile \
        --instance-profile-name "${IAM_INSTANCE_PROFILE_NAME}" \
        --role-name "${IAM_ROLE_NAME}"

    echo "Waiting for instance profile to propagate..."
    sleep 10
else
    echo "Instance profile '${IAM_INSTANCE_PROFILE_NAME}' already exists."
fi

# =============================================================================
# Step 6/8: Launch EC2 instance
# =============================================================================
echo ""
echo "[Step 6/8] Launching EC2 instance (${INSTANCE_TYPE})..."

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "${AMI_ID}" \
    --instance-type "${INSTANCE_TYPE}" \
    --key-name "${KEY_NAME}" \
    --security-group-ids "${SG_ID}" \
    --iam-instance-profile "Name=${IAM_INSTANCE_PROFILE_NAME}" \
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
# Step 7/8: Sync code to EC2
# =============================================================================
echo ""
echo "[Step 7/8] Syncing code to EC2..."

REMOTE_CODE_DIR="/home/ubuntu/code"
SSH_OPTS="-o StrictHostKeyChecking=no -i ${KEY_FILE}"

rsync -avz --filter=':- .gitignore' --exclude '.git' \
    -e "ssh ${SSH_OPTS}" \
    ./ ubuntu@"${PUBLIC_IP}":"${REMOTE_CODE_DIR}/"

echo "Code synced to ${REMOTE_CODE_DIR}"

# =============================================================================
# Step 8/8: Launch training (detached) on the instance
# =============================================================================
echo ""
echo "[Step 8/8] Launching training on EC2 (detached)..."

ssh ${SSH_OPTS} ubuntu@"${PUBLIC_IP}" << REMOTE_SCRIPT
set -euo pipefail

echo "--- Authenticating with ECR ---"
aws ecr get-login-password --region ${AWS_REGION} | \
  docker login --username AWS --password-stdin ${ECR_URI}

echo "--- Pulling base image ---"
docker pull ${IMAGE_URI}

echo "--- Starting training (detached): ${DOCKER_CMD} ---"
mkdir -p /home/ubuntu/data /home/ubuntu/checkpoints

nohup docker run --rm --gpus all \
  --name training-${MODULE} \
  ${WANDB_ENV_FLAG} \
  -v ${REMOTE_CODE_DIR}:/workspace \
  -v /home/ubuntu/data:/workspace/data \
  -v /home/ubuntu/checkpoints:/workspace/checkpoints \
  ${IMAGE_URI} \
  ${DOCKER_CMD} \
  > /home/ubuntu/training.log 2>&1 &

echo "--- Training launched in background (PID: \$!) ---"
REMOTE_SCRIPT

# Save run info for check_training.sh
RUN_INFO_DIR="$HOME/.aws-training-runs"
mkdir -p "${RUN_INFO_DIR}"
RUN_INFO_FILE="${RUN_INFO_DIR}/${RUN_NAME}.env"
cat > "${RUN_INFO_FILE}" << EOF
INSTANCE_ID=${INSTANCE_ID}
PUBLIC_IP=${PUBLIC_IP}
KEY_FILE=${KEY_FILE}
RUN_NAME=${RUN_NAME}
PIPELINE=${MODULE}
AWS_REGION=${AWS_REGION}
S3_BUCKET=${S3_BUCKET}
EOF

echo ""
echo "============================================================"
echo "  Training launched! Safe to close your laptop."
echo ""
echo "  Run name:    ${RUN_NAME}"
echo "  Instance:    ${INSTANCE_ID} (${PUBLIC_IP})"
echo "  Log file:    /home/ubuntu/training.log"
echo ""
echo "  Check status:  ./scripts/check_training.sh ${RUN_NAME}"
echo "  Tail logs:     ./scripts/check_training.sh ${RUN_NAME} logs"
echo "  Sync & stop:   ./scripts/check_training.sh ${RUN_NAME} finish"
echo "  SSH in:        ssh -i ${KEY_FILE} ubuntu@${PUBLIC_IP}"
echo "============================================================"
