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
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import boto3
import requests

from src.utils.logging_utils import get_logger
from src.utils.config_utils import load_config_and_parse_args


def run_cmd(cmd, logger, check=True, capture=False, **kwargs):
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


def step_1_build_and_push(cfg, logger):
    """Build base image (only if needed) and push to ECR."""
    logger.info("[Step 1/8] Checking if base image rebuild is needed...")

    ecr_uri = cfg["ecr_uri"]
    ecr_repo_name = cfg["ecr_repo_name"]
    image_tag = cfg["image_tag"]
    image_uri = cfg["image_uri"]
    aws_region = cfg["aws_region"]

    # Authenticate with ECR
    password = run_cmd(f"aws ecr get-login-password --region {aws_region}", logger, capture=True)
    run_cmd(f"echo '{password}' | docker login --username AWS --password-stdin {ecr_uri}", logger)

    # Create ECR repo if it doesn't exist
    ecr = boto3.client("ecr", region_name=aws_region)
    try:
        ecr.describe_repositories(repositoryNames=[ecr_repo_name])
    except ecr.exceptions.RepositoryNotFoundException:
        ecr.create_repository(repositoryName=ecr_repo_name)

    # Check if rebuild is needed
    needs_build = False
    result = subprocess.run(
        f'docker image inspect "{ecr_repo_name}:{image_tag}" --format "{{{{.Created}}}}"',
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
        run_cmd(f'docker buildx build --platform linux/amd64 -t "{ecr_repo_name}:{image_tag}" .', logger)
        run_cmd(f'docker tag "{ecr_repo_name}:{image_tag}" "{image_uri}"', logger)
        run_cmd(f'docker push "{image_uri}"', logger)
        logger.info("Image pushed: %s", image_uri)


def step_2_find_ami(cfg, logger):
    """Find the latest Deep Learning AMI."""
    logger.info("[Step 2/8] Finding latest Deep Learning AMI...")

    ec2 = boto3.client("ec2", region_name=cfg["aws_region"])
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


def step_3_key_pair(cfg, logger):
    """Create key pair if needed."""
    logger.info("[Step 3/8] Setting up key pair...")

    key_name = cfg["key_name"]
    key_file = cfg["key_file"]

    ec2 = boto3.client("ec2", region_name=cfg["aws_region"])
    try:
        ec2.describe_key_pairs(KeyNames=[key_name])
        logger.info("Key pair '%s' already exists.", key_name)
    except ec2.exceptions.ClientError:
        logger.info("Creating key pair: %s", key_name)
        response = ec2.create_key_pair(KeyName=key_name)
        Path(key_file).parent.mkdir(parents=True, exist_ok=True)
        Path(key_file).write_text(response["KeyMaterial"])
        os.chmod(key_file, stat.S_IRUSR)
        logger.info("Key saved to: %s", key_file)


def step_4_security_group(cfg, logger):
    """Create security group and update SSH IP."""
    logger.info("[Step 4/8] Setting up security group...")

    security_group_name = cfg["security_group_name"]

    ec2 = boto3.client("ec2", region_name=cfg["aws_region"])
    my_ip = requests.get("https://checkip.amazonaws.com", timeout=5).text.strip()

    # Find or create security group
    response = ec2.describe_security_groups(
        Filters=[{"Name": "group-name", "Values": [security_group_name]}]
    )

    if response["SecurityGroups"]:
        sg_id = response["SecurityGroups"][0]["GroupId"]
        logger.info("Security group '%s' already exists (%s).", security_group_name, sg_id)

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
        logger.info("Creating security group: %s", security_group_name)
        response = ec2.create_security_group(
            GroupName=security_group_name,
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


def step_5_iam_role(cfg, logger):
    """Create IAM role and instance profile for ECR access."""
    logger.info("[Step 5/8] Setting up IAM instance profile...")

    iam_role_name = cfg["iam_role_name"]
    iam_instance_profile_name = cfg["iam_instance_profile_name"]

    iam = boto3.client("iam")

    # Create role if needed
    try:
        iam.get_role(RoleName=iam_role_name)
        logger.info("IAM role '%s' already exists.", iam_role_name)
    except iam.exceptions.NoSuchEntityException:
        logger.info("Creating IAM role: %s", iam_role_name)
        iam.create_role(
            RoleName=iam_role_name,
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
            RoleName=iam_role_name,
            PolicyArn="arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
        )
        iam.attach_role_policy(
            RoleName=iam_role_name,
            PolicyArn="arn:aws:iam::aws:policy/AmazonS3FullAccess",
        )
        iam.put_role_policy(
            RoleName=iam_role_name,
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
        iam.get_instance_profile(InstanceProfileName=iam_instance_profile_name)
        logger.info("Instance profile '%s' already exists.", iam_instance_profile_name)
    except iam.exceptions.NoSuchEntityException:
        logger.info("Creating instance profile: %s", iam_instance_profile_name)
        iam.create_instance_profile(InstanceProfileName=iam_instance_profile_name)
        iam.add_role_to_instance_profile(
            InstanceProfileName=iam_instance_profile_name,
            RoleName=iam_role_name,
        )
        logger.info("Waiting for instance profile to propagate...")
        time.sleep(10)


def step_6_launch_instance(cfg, logger, ami_id, sg_id, run_name):
    """Launch EC2 instance."""
    instance_type = cfg["instance_type"]
    key_name = cfg["key_name"]
    key_file = cfg["key_file"]
    volume_size_gb = cfg["volume_size_gb"]
    iam_instance_profile_name = cfg["iam_instance_profile_name"]
    aws_region = cfg["aws_region"]

    logger.info("[Step 6/8] Launching EC2 instance (%s)...", instance_type)

    ec2 = boto3.client("ec2", region_name=aws_region)
    response = ec2.run_instances(
        ImageId=ami_id,
        InstanceType=instance_type,
        KeyName=key_name,
        SecurityGroupIds=[sg_id],
        IamInstanceProfile={"Name": iam_instance_profile_name},
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {"VolumeSize": volume_size_gb, "VolumeType": "gp3"},
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
            f'ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i {key_file} ubuntu@{public_ip} "echo ready"',
            shell=True,
            capture_output=True,
        )
        if result.returncode == 0:
            break
        logger.info("  Attempt %d/30 — waiting...", i)
        time.sleep(10)

    return instance_id, public_ip


def step_7_sync_code(cfg, logger, public_ip):
    """Sync code to EC2."""
    logger.info("[Step 7/8] Syncing code to EC2...")

    key_file = cfg["key_file"]
    remote_code_dir = "/home/ubuntu/code"
    ssh_opts = f"-o StrictHostKeyChecking=no -i {key_file}"

    run_cmd(
        f'rsync -avz --filter=":- .gitignore" --exclude ".git" '
        f'-e "ssh {ssh_opts}" ./ ubuntu@{public_ip}:{remote_code_dir}/',
        logger,
    )
    logger.info("Code synced to %s", remote_code_dir)
    return remote_code_dir


def step_8_launch_training(cfg, logger, public_ip, remote_code_dir, docker_cmd, module, wandb_env_flag):
    """Launch training detached on the instance."""
    logger.info("[Step 8/8] Launching training on EC2 (detached)...")

    key_file = cfg["key_file"]
    aws_region = cfg["aws_region"]
    ecr_uri = cfg["ecr_uri"]
    image_uri = cfg["image_uri"]
    ssh_opts = f"-o StrictHostKeyChecking=no -i {key_file}"

    remote_script = f"""set -euo pipefail
echo '--- Authenticating with ECR ---'
aws ecr get-login-password --region {aws_region} | docker login --username AWS --password-stdin {ecr_uri}

echo '--- Pulling base image ---'
docker pull {image_uri}

echo '--- Starting training (detached): {docker_cmd} ---'
mkdir -p /home/ubuntu/data /home/ubuntu/checkpoints

nohup docker run --rm --gpus all --network host \
  --name training-{module} \
  {wandb_env_flag} \
  -v {remote_code_dir}:/workspace \
  -v /home/ubuntu/data:/workspace/data \
  -v /home/ubuntu/checkpoints:/workspace/checkpoints \
  {image_uri} \
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


def save_run_info(cfg, run_info_dir, run_name, instance_id, public_ip, module):
    """Save run info for check_training.py."""
    run_info_dir.mkdir(parents=True, exist_ok=True)
    run_info_file = run_info_dir / f"{run_name}.json"
    run_info_file.write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "public_ip": public_ip,
                "key_file": cfg["key_file"],
                "run_name": run_name,
                "module": module,
                "aws_region": cfg["aws_region"],
                "s3_bucket": cfg["s3_bucket"],
            },
            indent=2,
        )
    )


def main():
    logger = get_logger(__name__)
    cfg, module, extra_args = load_config_and_parse_args()
    run_info_dir = Path.home() / ".aws-training-runs"

    # Build docker command with S3 and self-terminate flags
    aws_train_flags = ""
    s3_bucket = cfg["s3_bucket"]
    if s3_bucket:
        s3_prefix = f"{module}-{time.strftime('%Y%m%d-%H%M%S')}"
        aws_train_flags = f"--s3-bucket {s3_bucket} --s3-prefix {s3_prefix}"
    aws_train_flags += " --self-terminate"

    docker_cmd = f"python -m {module} {extra_args} {aws_train_flags}"

    # Check wandb
    wandb_env_flag = ""
    if "--wandb" in extra_args:
        wandb_key = cfg["wandb_api_key"]
        wandb_key_file = Path("api_keys/wandb.txt")
        if not wandb_key and wandb_key_file.exists():
            wandb_key = wandb_key_file.read_text().strip()
        if not wandb_key:
            logger.error("--wandb requested but WANDB_API_KEY is not set.")
            logger.error("Set it via: export WANDB_API_KEY=your_key or save to api_keys/wandb.txt")
            sys.exit(1)
        wandb_env_flag = f"-e WANDB_API_KEY={wandb_key}"

    run_name = f"{module}-{time.strftime('%Y%m%d-%H%M%S')}"

    logger.info("=" * 60)
    logger.info("  AWS Training Pipeline")
    logger.info("  Module:    %s", module)
    logger.info("  Instance:  %s", cfg["instance_type"])
    logger.info("  Run name:  %s", run_name)
    logger.info("=" * 60)

    # Execute steps
    step_1_build_and_push(cfg, logger)
    ami_id = step_2_find_ami(cfg, logger)
    step_3_key_pair(cfg, logger)
    sg_id = step_4_security_group(cfg, logger)
    step_5_iam_role(cfg, logger)
    instance_id, public_ip = step_6_launch_instance(cfg, logger, ami_id, sg_id, run_name)
    remote_code_dir = step_7_sync_code(cfg, logger, public_ip)
    step_8_launch_training(cfg, logger, public_ip, remote_code_dir, docker_cmd, module, wandb_env_flag)

    save_run_info(cfg, run_info_dir, run_name, instance_id, public_ip, module)

    logger.info("=" * 60)
    logger.info("  Training launched! Safe to close your laptop.")
    logger.info("  Run name:    %s", run_name)
    logger.info("  Instance:    %s (%s)", instance_id, public_ip)
    logger.info("  Log file:    /home/ubuntu/training.log")
    logger.info("  Check status:  python scripts_py/check_training.py %s", run_name)
    logger.info("  Tail logs:     python scripts_py/check_training.py %s logs", run_name)
    logger.info("  Sync & stop:   python scripts_py/check_training.py %s finish", run_name)
    logger.info("  SSH in:        ssh -i %s ubuntu@%s", cfg["key_file"], public_ip)
    logger.info("=" * 60)
    logger.info("Log saved to: %s", logger.log_file)


if __name__ == "__main__":
    main()
