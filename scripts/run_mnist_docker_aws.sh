#!/usr/bin/env bash
#
# Run MNIST training on AWS using bash.
# Wraps scripts/run_on_aws.sh with the MNIST module.
#
# Usage:
#   ./scripts/run_mnist_docker_aws.sh
#   ./scripts/run_mnist_docker_aws.sh --epochs 5 --batch-size 128
#   ./scripts/run_mnist_docker_aws.sh --epochs 10 --wandb --wandb-project my-project

# -e: exit on error, -u: error on undefined vars, -o pipefail: catch errors in pipes
set -euo pipefail

./scripts/run_on_aws.sh src.mnist.train_mnist \
    --epochs 1 \
    --batch-size 128 \
    --lr 0.001 \
    --data-dir ./data/mnist \
    --checkpoints-dir ./checkpoints \
    --wandb \
    --wandb-project mnist-training \
    --wandb-run-name "mnnist2" \
    --s3-bucket "fsr-autonomous-training" \
    --s3-prefix "mnist-training" \
    --self-terminate \
    "$@"
