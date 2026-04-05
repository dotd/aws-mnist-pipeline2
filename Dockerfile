# Base image: dependencies only. Code is mounted at runtime.
# Rebuild only when requirements.txt or Dockerfile changes.
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["python", "run.py", "mnist"]
