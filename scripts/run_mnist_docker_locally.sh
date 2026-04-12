#!/usr/bin/env bash
#
# Build and run MNIST training in Docker locally.
# Code is mounted so no rebuild is needed for code changes.
# Data, logs, and checkpoints are mapped to local directories.
#
# Usage:
#   ./scripts/run_mnist_docker_locally.sh
#   ./scripts/run_mnist_docker_locally.sh --epochs 5 --batch-size 128
#   ./scripts/run_mnist_docker_locally.sh --epochs 10 --wandb --wandb-project my-project

# -e: exit on error, -u: error on undefined vars, -o pipefail: catch errors in pipes
set -euo pipefail

echo "Working directory: $(pwd)"

IMAGE_NAME="mnist-local"
DOCKERFILE="Dockerfile"

# ---- Build image (skips if up to date) ----
echo "Building Docker image: ${IMAGE_NAME}..."
docker buildx build --platform linux/amd64 \
    -f "${DOCKERFILE}" \
    -t "${IMAGE_NAME}:latest" \
    .

# ---- Create local directories if needed ----
mkdir -p data logs checkpoints

# ---- Load WANDB_API_KEY from api_keys/wandb.txt or env var ----
WANDB_ENV_FLAG=""
if [[ -z "${WANDB_API_KEY:-}" ]] && [[ -f "api_keys/wandb.txt" ]]; then
    WANDB_API_KEY=$(cat api_keys/wandb.txt | tr -d '[:space:]')
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_ENV_FLAG="-e WANDB_API_KEY=${WANDB_API_KEY}"
fi

# ---- Run training ----
echo "Running MNIST training in Docker..."
docker run --rm \
    ${WANDB_ENV_FLAG} \
    -v "$(pwd)":/workspace \
    -v "$(pwd)/data":/workspace/data \
    -v "$(pwd)/logs":/workspace/logs \
    -v "$(pwd)/checkpoints":/workspace/checkpoints \
    "${IMAGE_NAME}:latest" \
    python -m src.mnist.train_mnist --epochs 3 --batch-size 64 --lr 0.0005 --wandb --wandb-project my-project
