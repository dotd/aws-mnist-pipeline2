# Base image: dependencies only. Code is mounted at runtime.
# Rebuild only when requirements.txt or Dockerfile changes.
# No CMD — the module to run must be provided at runtime, e.g.:
#   docker run my-image python -m src.mnist.train_mnist --epochs 5
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt
