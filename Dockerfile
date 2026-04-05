# Dockerfile
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

CMD ["python", "run.py", "mnist"]
