#!/usr/bin/env bash
#
# Run MNIST training on AWS via scripts_py/run_on_aws.py.
# Configuration is loaded from configs/aws.yaml (can be overridden with CLI args).
#
# Usage:
#   ./scripts/run_mnist_docker_aws.sh
#   ./scripts/run_mnist_docker_aws.sh --epochs 5 --batch-size 128
#   ./scripts/run_mnist_docker_aws.sh --epochs 10 --wandb --wandb-project my-project
#   ./scripts/run_mnist_docker_aws.sh --instance-type p3.2xlarge --epochs 25

# -e: exit on error, -u: error on undefined vars, -o pipefail: catch errors in pipes
set -euo pipefail

python scripts_py/run_on_aws.py --module src.mnist.train_mnist "$@" 
