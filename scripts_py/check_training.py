#!/usr/bin/env python3
"""
Check, monitor, and clean up a detached training run launched by run_on_aws.py.

Usage:
  python scripts_py/check_training.py                    # List all runs
  python scripts_py/check_training.py <run-name>         # Check if training is still running
  python scripts_py/check_training.py <run-name> logs    # Tail the training log
  python scripts_py/check_training.py <run-name> finish  # Sync results to S3 and terminate instance
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import boto3

RUN_INFO_DIR = Path.home() / ".aws-training-runs"


def ssh_run(ip, key_file, command, capture=False):
    """Run a command on the remote instance via SSH."""
    ssh_opts = f"-o StrictHostKeyChecking=no -o ConnectTimeout=10 -i {key_file}"
    cmd = f'ssh {ssh_opts} ubuntu@{ip} "{command}"'
    if capture:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else ""
    else:
        subprocess.run(cmd, shell=True)


def load_run_info(run_name):
    """Load run info from JSON file."""
    run_file = RUN_INFO_DIR / f"{run_name}.json"
    if not run_file.exists():
        print(f"Error: Run '{run_name}' not found.")
        available = list(RUN_INFO_DIR.glob("*.json"))
        if available:
            print("Available runs:")
            for f in available:
                print(f"  {f.stem}")
        sys.exit(1)
    return json.loads(run_file.read_text())


def get_instance_state(instance_id, region):
    """Get EC2 instance state."""
    ec2 = boto3.client("ec2", region_name=region)
    try:
        response = ec2.describe_instances(InstanceIds=[instance_id])
        return response["Reservations"][0]["Instances"][0]["State"]["Name"]
    except Exception:
        return "unknown"


def list_runs():
    """List all saved training runs."""
    print("Saved training runs:")
    print()
    if RUN_INFO_DIR.exists():
        run_files = list(RUN_INFO_DIR.glob("*.json"))
        if run_files:
            for f in run_files:
                info = json.loads(f.read_text())
                state = get_instance_state(info["instance_id"], info["aws_region"])
                print(f"  {f.stem:<40s}  {state:<12s}  {info['instance_id']}")
        else:
            print(f"  No runs found in {RUN_INFO_DIR}")
    else:
        print(f"  No runs found in {RUN_INFO_DIR}")
    print()
    print(f"Usage: python {sys.argv[0]} <run-name> [logs|finish]")


def action_status(info):
    """Show run status."""
    print(f"Run:       {info['run_name']}")
    print(f"Instance:  {info['instance_id']} ({info['public_ip']})")
    print(f"Pipeline:  {info['pipeline']}")

    state = get_instance_state(info["instance_id"], info["aws_region"])
    print(f"State:     {state}")

    if state != "running":
        print(f"\nInstance is '{state}' — not running.")
        (RUN_INFO_DIR / f"{info['run_name']}.json").unlink(missing_ok=True)
        return

    print()
    container_status = ssh_run(
        info["public_ip"], info["key_file"],
        f"docker ps --filter name=training-{info['pipeline']} --format '{{{{.Status}}}}'",
        capture=True,
    )

    if container_status:
        print(f"Training:  RUNNING ({container_status})")
    else:
        print("Training:  FINISHED (or not started)")

    print()
    print("Last 5 lines of log:")
    ssh_run(info["public_ip"], info["key_file"], "tail -5 /home/ubuntu/training.log 2>/dev/null")
    print()
    print("Commands:")
    print(f"  python {sys.argv[0]} {info['run_name']} logs     # Tail full log")
    print(f"  python {sys.argv[0]} {info['run_name']} finish   # Sync results & terminate")


def action_logs(info):
    """Tail the training log."""
    print("Tailing training log (Ctrl+C to stop)...\n")
    ssh_opts = f"-o StrictHostKeyChecking=no -o ConnectTimeout=10 -i {info['key_file']}"
    subprocess.run(
        f"ssh {ssh_opts} ubuntu@{info['public_ip']} 'tail -f /home/ubuntu/training.log'",
        shell=True,
    )


def action_finish(info):
    """Sync results and terminate instance."""
    # Check if still training
    container_id = ssh_run(
        info["public_ip"], info["key_file"],
        f"docker ps --filter name=training-{info['pipeline']} --format '{{{{.ID}}}}'",
        capture=True,
    )

    if container_id:
        print("Warning: Training is still running!")
        confirm = input("Stop training and proceed? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return
        ssh_run(info["public_ip"], info["key_file"], f"docker stop training-{info['pipeline']}")

    # Sync results to S3
    if info.get("s3_bucket"):
        print(f"Syncing checkpoints to s3://{info['s3_bucket']}/{info['run_name']}/...")
        ssh_run(
            info["public_ip"], info["key_file"],
            f"aws s3 sync /home/ubuntu/checkpoints s3://{info['s3_bucket']}/{info['run_name']}/",
        )
        print(f"Results saved to s3://{info['s3_bucket']}/{info['run_name']}/")
    else:
        print("No S3 bucket configured — skipping sync.")

    # Download log
    print("Downloading training log...")
    os.makedirs("logs", exist_ok=True)
    ssh_opts = f"-o StrictHostKeyChecking=no -o ConnectTimeout=10 -i {info['key_file']}"
    subprocess.run(
        f"scp {ssh_opts} ubuntu@{info['public_ip']}:/home/ubuntu/training.log logs/{info['run_name']}.log",
        shell=True,
    )
    print(f"Log saved to logs/{info['run_name']}.log")

    # Terminate
    print()
    confirm = input(f"Terminate instance {info['instance_id']}? [y/N] ").strip().lower()
    if confirm == "y":
        ec2 = boto3.client("ec2", region_name=info["aws_region"])
        ec2.terminate_instances(InstanceIds=[info["instance_id"]])
        print(f"Instance {info['instance_id']} terminated.")
        (RUN_INFO_DIR / f"{info['run_name']}.json").unlink(missing_ok=True)
    else:
        print("Instance left running.")
        print(f"  aws ec2 terminate-instances --instance-ids {info['instance_id']} --region {info['aws_region']}")


def main():
    if len(sys.argv) < 2:
        list_runs()
        return

    run_name = sys.argv[1]
    action = sys.argv[2] if len(sys.argv) > 2 else "status"

    info = load_run_info(run_name)

    # Check instance state
    state = get_instance_state(info["instance_id"], info["aws_region"])
    if state != "running" and action != "status":
        print(f"Instance {info['instance_id']} is '{state}' — not running.")
        print("Cleaning up run info.")
        (RUN_INFO_DIR / f"{run_name}.json").unlink(missing_ok=True)
        sys.exit(1)

    if action == "status":
        action_status(info)
    elif action == "logs":
        action_logs(info)
    elif action == "finish":
        action_finish(info)
    else:
        print(f"Unknown action: {action}")
        print(f"Usage: python {sys.argv[0]} <run-name> [status|logs|finish]")
        sys.exit(1)


if __name__ == "__main__":
    main()
