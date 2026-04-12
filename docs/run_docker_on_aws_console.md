# How to Run the Docker on AWS Using the Console

This tutorial walks through launching an EC2 GPU instance and running your training container from ECR.

## Prerequisites

- Docker image built locally (`my-cv-model:latest`)
- AWS CLI installed and authenticated
- AWS Console access with permissions for EC2 and ECR

## Step 1: Push the Docker Image to ECR

Authenticate Docker with ECR (dotan private id=519372127358, FR id = 148761683501):

```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 148761683501.dkr.ecr.us-east-1.amazonaws.com
```

- `aws ecr get-login-password`: Requests a temporary authentication token from ECR
- `--region us-east-1`: The AWS region where your ECR repository is hosted
- `|`: Pipes the token output directly into the next command
- `docker login`: Authenticates Docker with a container registry
- `--username AWS`: The fixed username required by ECR (not your personal AWS username)
- `--password-stdin`: Reads the password (the token) from stdin instead of prompting interactively
- `519372127358.dkr.ecr.us-east-1.amazonaws.com`: The ECR registry URL, formatted as `<account_id>.dkr.ecr.<region>.amazonaws.com`

Create a repository (only needed once):

```bash
aws ecr create-repository --repository-name dotan-fr-my-cv-model --region us-east-1
```

Tag and push the image:

```bash
docker tag my-cv-model:latest 148761683501.dkr.ecr.us-east-1.amazonaws.com/dotan-fr-my-cv-model:latest
docker push 148761683501.dkr.ecr.us-east-1.amazonaws.com/dotan-fr-my-cv-model:latest
```

## Step 2: Launch an EC2 Instance

1. Go to **EC2 > Instances > Launch instances**
2. Configure the following:
  - **Name**: `mnist-training`
  - **AMI**: Search for `Deep Learning AMI (Ubuntu)` — this comes with CUDA, cuDNN, and NVIDIA drivers pre-installed
  - **Instance type**: `g4dn.xlarge` (1x T4 GPU, cheapest GPU option) or `p3.2xlarge` (1x V100, faster)
  - **Key pair**: Select an existing key pair or create a new one (you'll need this to SSH in)
  - **Network settings**: Allow SSH (port 22) from your IP
  - **Storage**: Increase root volume to at least 50 GB (the Docker image is large)
3. Under **Advanced details**:
  - **IAM instance profile**: Attach a role with `AmazonEC2ContainerRegistryReadOnly` policy (so the instance can pull from ECR)
  - **Purchasing option**: Check **Request Spot Instances** to save up to 90% on cost (optional)
4. Click **Launch instance**

## Step 3: Connect to the Instance

1. Go to **EC2 > Instances**, select your instance
2. Click **Connect** at the top
3. Choose the **SSH client** tab and follow the instructions, or use:

```bash
ssh -i "your-key.pem" ubuntu@<public-ip>
```

## Step 4: Authenticate Docker with ECR

On the EC2 instance, run:

```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 148761683501.dkr.ecr.us-east-1.amazonaws.com
```

## Step 5: Pull the Image

```bash
docker pull 148761683501.dkr.ecr.us-east-1.amazonaws.com/dotan-fr-my-cv-model:latest
```

## Step 6: Run the Training

```bash
# Train MNIST (default)
docker run --rm --gpus all \
  -v /home/ubuntu/data:/workspace/data \
  -v /home/ubuntu/checkpoints:/workspace/checkpoints \
  148761683501.dkr.ecr.us-east-1.amazonaws.com/dotan-fr-my-cv-model:latest

# Train U-Net segmentation
docker run --rm --gpus all \
  -v /home/ubuntu/data:/workspace/data \
  -v /home/ubuntu/checkpoints:/workspace/checkpoints \
  148761683501.dkr.ecr.us-east-1.amazonaws.com/dotan-fr-my-cv-model:latest \
  python -m src.UNET_PascalVOC_simple.train --epochs 25 --batch-size 8 --image-size 256
```

## Step 7: Copy Results to S3

After training completes, sync checkpoints to S3:

```bash
aws s3 sync /home/ubuntu/checkpoints s3://your-bucket-name/mnist-run-001/
```

## Step 8: Terminate the Instance

**Important** — GPU instances are expensive. Always terminate when done:

1. Go to **EC2 > Instances**
2. Select your instance
3. **Instance state > Terminate instance**

## Cost Reference


| Instance    | GPU    | On-Demand ($/hr) | Spot ($/hr approx) |
| ----------- | ------ | ---------------- | ------------------ |
| g4dn.xlarge | T4     | ~$0.53           | ~$0.16             |
| p3.2xlarge  | V100   | ~$3.06           | ~$0.92             |
| p3.8xlarge  | 4xV100 | ~$12.24          | ~$3.67             |


Prices vary by region. Always check the [EC2 pricing page](https://aws.amazon.com/ec2/pricing/).