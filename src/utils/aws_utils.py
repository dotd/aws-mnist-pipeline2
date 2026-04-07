import logging
import os

import boto3
import requests

logger = logging.getLogger(__name__)

# AWS EC2 Instance Metadata Service — a link-local endpoint only reachable from within an EC2 instance.
# Used to query instance ID, region, IAM credentials, etc.
EC2_IMDS_URL = "http://169.254.169.254"


def sync_to_s3(local_dir, bucket, prefix):
    """Upload all files in local_dir to s3://bucket/prefix/."""
    s3 = boto3.client("s3")
    for root, _, files in os.walk(local_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            rel_path = os.path.relpath(local_path, local_dir)
            s3_key = f"{prefix}/{rel_path}"
            s3.upload_file(local_path, bucket, s3_key)
            logger.info("Uploaded %s -> s3://%s/%s", local_path, bucket, s3_key)


def terminate_self():
    """Terminate this EC2 instance using the instance metadata endpoint."""
    try:
        # IMDSv2: get token first
        token = requests.put(
            f"{EC2_IMDS_URL}/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
            timeout=2,
        ).text
        instance_id = requests.get(
            f"{EC2_IMDS_URL}/latest/meta-data/instance-id",
            headers={"X-aws-ec2-metadata-token": token},
            timeout=2,
        ).text
        region = requests.get(
            f"{EC2_IMDS_URL}/latest/meta-data/placement/region",
            headers={"X-aws-ec2-metadata-token": token},
            timeout=2,
        ).text
    except requests.RequestException:
        logger.error("Failed to reach EC2 metadata endpoint. Are we running on EC2?")
        return

    logger.info("Terminating instance %s in %s...", instance_id, region)
    ec2 = boto3.client("ec2", region_name=region)
    ec2.terminate_instances(InstanceIds=[instance_id])
