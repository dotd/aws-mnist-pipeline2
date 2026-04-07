#!/usr/bin/env python3
"""
Automates the full AWS training workflow:
  1. Build base image (only if needed) and push to ECR
  2. Find the latest Deep Learning AMI
  3. Create key pair (if needed)
  4. Create security group (update SSH IP)
  5. Create IAM role + instance profile for ECR access (if needed)
  6. Launch EC2 instance
  7. Sync code to EC2
  8. Launch training (detached) — safe to close laptop

After training, use check_training.py to monitor, sync results, and terminate.

Usage:
  python scripts_py/run_on_aws.py                          # Train MNIST (default)
  python scripts_py/run_on_aws.py mnist --epochs 2 --wandb --wandb-project my-project
  python scripts_py/run_on_aws.py unet                     # Train U-Net
  python scripts_py/run_on_aws.py unet --epochs 25 --wandb # U-Net with wandb
  python scripts_py/run_on_aws.py unet --epochs 25 --wandb --wandb-project my-project
"""

import json
import logging
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import boto3
import requests

# =============================================================================
# Logging setup — writes to stdout and to ./logs/log_YYYYMMDD_HHMMSS.log
# =============================================================================
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"log_{time.strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),
    ],
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration — edit these to match your setup
# =============================================================================
AWS_ACCOUNT_ID = "148761683501"
AWS_REGION = "us-east-1"
ECR_REPO_NAME = "dotan-fr-my-cv-model"
IMAGE_TAG = "latest"
INSTANCE_TYPE = "g4dn.xlarge"
KEY_NAME = "training-key"
KEY_FILE = str(Path.home() / ".ssh" / f"{KEY_NAME}.pem")
SECURITY_GROUP_NAME = "training-sg"
IAM_ROLE_NAME = "ec2-training-role"
IAM_INSTANCE_PROFILE_NAME = "ec2-training-profile"
S3_BUCKET = "fsr-autonomous-training"
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
VOLUME_SIZE_GB = 100

# Derived values
ECR_URI = f"{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com"
IMAGE_URI = f"{ECR_URI}/{ECR_REPO_NAME}:{IMAGE_TAG}"

RUN_INFO_DIR = Path.home() / ".aws-training-runs"


def run(cmd, check=True, capture=False, **kwargs):
    """Run a shell command and optionally capture output."""
    logger.info("$ %s", cmd)
    if capture:
        result = subprocess.run(
            cmd, shell=True, check=check, capture_output=True, text=True, **kwargs
        )
        return result.stdout.strip()
    else:
        subprocess.run(cmd, shell=True, check=check, **kwargs)
        return None


def ssh_cmd(ip, command, key_file=KEY_FILE, capture=False):
    """Run a command on the remote instance via SSH."""
    ssh_opts = f"-o StrictHostKeyChecking=no -o ConnectTimeout=10 -i {key_file}"
    return run(f'ssh {ssh_opts} ubuntu@{ip} "{command}"', capture=capture)


def step_1_build_and_push():
    """Build base image (only if needed) and push to ECR."""
    logger.info("[Step 1/8] Checking if base image rebuild is needed...")

    # Authenticate with ECR
    password = run(f"aws ecr get-login-password --region {AWS_REGION}", capture=True)
    run(f"echo '{password}' | docker login --username AWS --password-stdin {ECR_URI}")

    # Create ECR repo if it doesn't exist
    ecr = boto3.client("ecr", region_name=AWS_REGION)
    try:
        ecr.describe_repositories(repositoryNames=[ECR_REPO_NAME])
    except ecr.exceptions.RepositoryNotFoundException:
        ecr.create_repository(repositoryName=ECR_REPO_NAME)

    # Check if rebuild is needed
    needs_build = False
    result = subprocess.run(
        f'docker image inspect "{ECR_REPO_NAME}:{IMAGE_TAG}" --format "{{{{.Created}}}}"',
        shell=True,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.info("Image not found locally — building.")
        needs_build = True
    else:
        image_created = result.stdout.strip()
        for f in ("Dockerfile", "requirements.txt"):
            if os.path.exists(f) and os.path.getmtime(f) > time.mktime(
                time.strptime(image_created[:19], "%Y-%m-%dT%H:%M:%S")
            ):
                logger.info("%s changed — rebuilding.", f)
                needs_build = True
                break
        if not needs_build:
            logger.info("Base image is up to date — skipping rebuild.")

    if needs_build:
        run(
            f'docker buildx build --platform linux/amd64 -t "{ECR_REPO_NAME}:{IMAGE_TAG}" .'
        )
        run(f'docker tag "{ECR_REPO_NAME}:{IMAGE_TAG}" "{IMAGE_URI}"')
        run(f'docker push "{IMAGE_URI}"')
        logger.info("Image pushed: %s", IMAGE_URI)


def step_2_find_ami():
    """Find the latest Deep Learning AMI."""
    logger.info("[Step 2/8] Finding latest Deep Learning AMI...")

    ec2 = boto3.client("ec2", region_name=AWS_REGION)
    response = ec2.describe_images(
        Owners=["amazon"],
        Filters=[
            {
                "Name": "name",
                "Values": [
                    "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"
                ],
            },
            {"Name": "state", "Values": ["available"]},
            {"Name": "architecture", "Values": ["x86_64"]},
        ],
    )
    images = sorted(response["Images"], key=lambda x: x["CreationDate"])
    if not images:
        logger.error("Could not find a Deep Learning AMI.")
        sys.exit(1)

    ami_id = images[-1]["ImageId"]
    logger.info("Using AMI: %s", ami_id)
    return ami_id


def step_3_key_pair():
    """Create key pair if needed."""
    logger.info("[Step 3/8] Setting up key pair...")

    ec2 = boto3.client("ec2", region_name=AWS_REGION)
    try:
        ec2.describe_key_pairs(KeyNames=[KEY_NAME])
        logger.info("Key pair '%s' already exists.", KEY_NAME)
    except ec2.exceptions.ClientError:
        logger.info("Creating key pair: %s", KEY_NAME)
        response = ec2.create_key_pair(KeyName=KEY_NAME)
        Path(KEY_FILE).parent.mkdir(parents=True, exist_ok=True)
        Path(KEY_FILE).write_text(response["KeyMaterial"])
        os.chmod(KEY_FILE, stat.S_IRUSR)
        logger.info("Key saved to: %s", KEY_FILE)


def step_4_security_group():
    """Create security group and update SSH IP."""
    logger.info("[Step 4/8] Setting up security group...")

    ec2 = boto3.client("ec2", region_name=AWS_REGION)
    my_ip = requests.get("https://checkip.amazonaws.com", timeout=5).text.strip()

    # Find or create security group
    response = ec2.describe_security_groups(
        Filters=[{"Name": "group-name", "Values": [SECURITY_GROUP_NAME]}]
    )

    if response["SecurityGroups"]:
        sg_id = response["SecurityGroups"][0]["GroupId"]
        logger.info("Security group '%s' already exists (%s).", SECURITY_GROUP_NAME, sg_id)

        # Revoke existing SSH rules
        sg = response["SecurityGroups"][0]
        for perm in sg.get("IpPermissions", []):
            if perm.get("FromPort") == 22:
                for ip_range in perm.get("IpRanges", []):
                    try:
                        ec2.revoke_security_group_ingress(
                            GroupId=sg_id,
                            IpProtocol="tcp",
                            FromPort=22,
                            ToPort=22,
                            CidrIp=ip_range["CidrIp"],
                        )
                    except Exception:
                        pass
    else:
        logger.info("Creating security group: %s", SECURITY_GROUP_NAME)
        response = ec2.create_security_group(
            GroupName=SECURITY_GROUP_NAME,
            Description="SSH access for training instances",
        )
        sg_id = response["GroupId"]

    # Always authorize current IP
    try:
        ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpProtocol="tcp",
            FromPort=22,
            ToPort=22,
            CidrIp=f"{my_ip}/32",
        )
    except ec2.exceptions.ClientError:
        pass  # Already exists
    logger.info("SSH allowed from %s", my_ip)
    return sg_id


def step_5_iam_role():
    """Create IAM role and instance profile for ECR access."""
    logger.info("[Step 5/8] Setting up IAM instance profile...")

    iam = boto3.client("iam")

    # Create role if needed
    try:
        iam.get_role(RoleName=IAM_ROLE_NAME)
        logger.info("IAM role '%s' already exists.", IAM_ROLE_NAME)
    except iam.exceptions.NoSuchEntityException:
        logger.info("Creating IAM role: %s", IAM_ROLE_NAME)
        iam.create_role(
            RoleName=IAM_ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "ec2.amazonaws.com"},
                            "Action": "sts:AssumeRole",
                        }
                    ],
                }
            ),
        )
        iam.attach_role_policy(
            RoleName=IAM_ROLE_NAME,
            PolicyArn="arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
        )
        iam.attach_role_policy(
            RoleName=IAM_ROLE_NAME,
            PolicyArn="arn:aws:iam::aws:policy/AmazonS3FullAccess",
        )
        iam.put_role_policy(
            RoleName=IAM_ROLE_NAME,
            PolicyName="self-terminate",
            PolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": "ec2:TerminateInstances",
                            "Resource": "*",
                        }
                    ],
                }
            ),
        )

    # Create instance profile if needed
    try:
        iam.get_instance_profile(InstanceProfileName=IAM_INSTANCE_PROFILE_NAME)
        logger.info("Instance profile '%s' already exists.", IAM_INSTANCE_PROFILE_NAME)
    except iam.exceptions.NoSuchEntityException:
        logger.info("Creating instance profile: %s", IAM_INSTANCE_PROFILE_NAME)
        iam.create_instance_profile(InstanceProfileName=IAM_INSTANCE_PROFILE_NAME)
        iam.add_role_to_instance_profile(
            InstanceProfileName=IAM_INSTANCE_PROFILE_NAME,
            RoleName=IAM_ROLE_NAME,
        )
        logger.info("Waiting for instance profile to propagate...")
        time.sleep(10)


def step_6_launch_instance(ami_id, sg_id, run_name):
    """Launch EC2 instance."""
    logger.info("[Step 6/8] Launching EC2 instance (%s)...", INSTANCE_TYPE)

    ec2 = boto3.client("ec2", region_name=AWS_REGION)
    response = ec2.run_instances(
        ImageId=ami_id,
        InstanceType=INSTANCE_TYPE,
        KeyName=KEY_NAME,
        SecurityGroupIds=[sg_id],
        IamInstanceProfile={"Name": IAM_INSTANCE_PROFILE_NAME},
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {"VolumeSize": VOLUME_SIZE_GB, "VolumeType": "gp3"},
            }
        ],
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [{"Key": "Name", "Value": run_name}],
            }
        ],
        MinCount=1,
        MaxCount=1,
    )

    instance_id = response["Instances"][0]["InstanceId"]
    logger.info("Instance launched: %s", instance_id)
    logger.info("Waiting for instance to be running...")

    waiter = ec2.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])

    desc = ec2.describe_instances(InstanceIds=[instance_id])
    public_ip = desc["Reservations"][0]["Instances"][0]["PublicIpAddress"]
    logger.info("Instance running at: %s", public_ip)

    # Wait for SSH
    logger.info("Waiting for SSH to become available...")
    for i in range(1, 31):
        result = subprocess.run(
            f'ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i {KEY_FILE} ubuntu@{public_ip} "echo ready"',
            shell=True,
            capture_output=True,
        )
        if result.returncode == 0:
            break
        logger.info("  Attempt %d/30 — waiting...", i)
        time.sleep(10)

    return instance_id, public_ip


def step_7_sync_code(public_ip):
    """Sync code to EC2."""
    logger.info("[Step 7/8] Syncing code to EC2...")

    remote_code_dir = "/home/ubuntu/code"
    ssh_opts = f"-o StrictHostKeyChecking=no -i {KEY_FILE}"

    run(
        f'rsync -avz --exclude ".git" --exclude "data" --exclude "checkpoints" '
        f'--exclude "venv" --exclude "__pycache__" --exclude "wandb" --exclude ".DS_Store" '
        f'-e "ssh {ssh_opts}" ./ ubuntu@{public_ip}:{remote_code_dir}/'
    )
    logger.info("Code synced to %s", remote_code_dir)
    return remote_code_dir


def step_8_launch_training(
    public_ip, remote_code_dir, docker_cmd, pipeline, wandb_env_flag
):
    """Launch training detached on the instance."""
    logger.info("[Step 8/8] Launching training on EC2 (detached)...")

    ssh_opts = f"-o StrictHostKeyChecking=no -i {KEY_FILE}"

    remote_script = f"""set -euo pipefail
echo '--- Authenticating with ECR ---'
aws ecr get-login-password --region {AWS_REGION} | docker login --username AWS --password-stdin {ECR_URI}

echo '--- Pulling base image ---'
docker pull {IMAGE_URI}

echo '--- Starting training (detached): {docker_cmd} ---'
mkdir -p /home/ubuntu/data /home/ubuntu/checkpoints

nohup docker run --rm --gpus all \
  --name training-{pipeline} \
  {wandb_env_flag} \
  -v {remote_code_dir}:/workspace \
  -v /home/ubuntu/data:/workspace/data \
  -v /home/ubuntu/checkpoints:/workspace/checkpoints \
  {IMAGE_URI} \
  {docker_cmd} \
  > /home/ubuntu/training.log 2>&1 &

echo "--- Training launched in background (PID: \\$!) ---"
"""
    subprocess.run(
        f"ssh {ssh_opts} ubuntu@{public_ip} bash -s",
        shell=True,
        input=remote_script,
        text=True,
        check=True,
    )


def save_run_info(run_name, instance_id, public_ip, pipeline):
    """Save run info for check_training.py."""
    RUN_INFO_DIR.mkdir(parents=True, exist_ok=True)
    run_info_file = RUN_INFO_DIR / f"{run_name}.json"
    run_info_file.write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "public_ip": public_ip,
                "key_file": KEY_FILE,
                "run_name": run_name,
                "pipeline": pipeline,
                "aws_region": AWS_REGION,
                "s3_bucket": S3_BUCKET,
            },
            indent=2,
        )
    )


def main():
    # Parse args
    args = sys.argv[1:]
    pipeline = "mnist"
    if args and args[0] in ("mnist", "unet"):
        pipeline = args.pop(0)
    extra_args = " ".join(args)

    # Build docker command with S3 and self-terminate flags
    aws_train_flags = ""
    s3_prefix = ""
    if S3_BUCKET:
        s3_prefix = f"{pipeline}-{time.strftime('%Y%m%d-%H%M%S')}"
        aws_train_flags = f"--s3-bucket {S3_BUCKET} --s3-prefix {s3_prefix}"
    aws_train_flags += " --self-terminate"

    docker_cmd = f"python run.py {pipeline} {extra_args} {aws_train_flags}"

    # Check wandb
    wandb_env_flag = ""
    if "--wandb" in extra_args:
        wandb_key = WANDB_API_KEY
        wandb_key_file = Path("api_keys/wandb.txt")
        if not wandb_key and wandb_key_file.exists():
            wandb_key = wandb_key_file.read_text().strip()
        if not wandb_key:
            logger.error("--wandb requested but WANDB_API_KEY is not set.")
            logger.error("Set it via: export WANDB_API_KEY=your_key or save to api_keys/wandb.txt")
            sys.exit(1)
        wandb_env_flag = f"-e WANDB_API_KEY={wandb_key}"

    run_name = f"{pipeline}-{time.strftime('%Y%m%d-%H%M%S')}"

    logger.info("=" * 60)
    logger.info("  AWS Training Pipeline")
    logger.info("  Pipeline:  %s", pipeline)
    logger.info("  Instance:  %s", INSTANCE_TYPE)
    logger.info("  Run name:  %s", run_name)
    logger.info("=" * 60)

    # Execute steps
    step_1_build_and_push()
    ami_id = step_2_find_ami()
    step_3_key_pair()
    sg_id = step_4_security_group()
    step_5_iam_role()
    instance_id, public_ip = step_6_launch_instance(ami_id, sg_id, run_name)
    remote_code_dir = step_7_sync_code(public_ip)
    step_8_launch_training(
        public_ip, remote_code_dir, docker_cmd, pipeline, wandb_env_flag
    )

    save_run_info(run_name, instance_id, public_ip, pipeline)

    logger.info("=" * 60)
    logger.info("  Training launched! Safe to close your laptop.")
    logger.info("  Run name:    %s", run_name)
    logger.info("  Instance:    %s (%s)", instance_id, public_ip)
    logger.info("  Log file:    /home/ubuntu/training.log")
    logger.info("  Check status:  python scripts_py/check_training.py %s", run_name)
    logger.info("  Tail logs:     python scripts_py/check_training.py %s logs", run_name)
    logger.info("  Sync & stop:   python scripts_py/check_training.py %s finish", run_name)
    logger.info("  SSH in:        ssh -i %s ubuntu@%s", KEY_FILE, public_ip)
    logger.info("=" * 60)
    logger.info("Log saved to: %s", LOG_FILE)


if __name__ == "__main__":
    main()
