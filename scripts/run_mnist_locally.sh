#!/usr/bin/env bash
#
# Run MNIST training locally (no Docker, no AWS).
#
# Usage:
#   ./scripts/run_mnist_locally.sh
#   ./scripts/run_mnist_locally.sh --epochs 5 --batch-size 128 --lr 0.0005
#   ./scripts/run_mnist_locally.sh --epochs 10 --wandb --wandb-project my-project

# -e: exit on error, -u: error on undefined vars, -o pipefail: catch errors in pipes
set -euo pipefail

python -m src.mnist.train_mnist --epochs 1 --batch-size 128 --lr 0.0005 --wandb --wandb-project my-project
