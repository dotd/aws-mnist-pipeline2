import argparse
import os
from pathlib import Path

import yaml

from src.definitions import CONFIGS_DIR

DEFAULT_CONFIG_PATH = CONFIGS_DIR / "aws.yaml"


def _resolve_path(value):
    """Expand ~ and environment variables in path strings."""
    if isinstance(value, str) and ("~" in value or "$" in value):
        return str(Path(os.path.expandvars(value)).expanduser())
    return value


def _resolve_paths_recursive(d):
    """Walk a dict and resolve paths in all string values."""
    for key, value in d.items():
        if isinstance(value, dict):
            _resolve_paths_recursive(value)
        elif isinstance(value, str):
            d[key] = _resolve_path(value)


def _load_yaml(config_path):
    """Load and flatten YAML config into a flat dict."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    _resolve_paths_recursive(cfg)

    flat = {}
    flat["aws_account_id"] = cfg["aws"]["account_id"]
    flat["aws_region"] = cfg["aws"]["region"]
    flat["ecr_repo_name"] = cfg["ecr"]["repo_name"]
    flat["image_tag"] = cfg["ecr"]["image_tag"]
    flat["instance_type"] = cfg["ec2"]["instance_type"]
    flat["key_name"] = cfg["ec2"]["key_name"]
    flat["key_file"] = cfg["ec2"]["key_file"]
    flat["security_group_name"] = cfg["ec2"]["security_group_name"]
    flat["volume_size_gb"] = cfg["ec2"]["volume_size_gb"]
    flat["iam_role_name"] = cfg["iam"]["role_name"]
    flat["iam_instance_profile_name"] = cfg["iam"]["instance_profile_name"]
    flat["s3_bucket"] = cfg["s3"]["bucket"]
    flat["wandb_api_key"] = cfg.get("wandb", {}).get("api_key", "")
    return flat


def _parse_args():
    """Parse CLI arguments. Returns (pipeline, extra_args, cli_overrides)."""
    parser = argparse.ArgumentParser(description="AWS Training Pipeline")
    parser.add_argument("--module", type=str, required=True,
                        help="Training module to run (e.g. src.mnist.train_mnist, src.UNET_PascalVOC_simple.train)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (default: configs/aws.yaml). Precedence: CLI args > YAML > env vars > defaults")
    parser.add_argument("--instance-type", type=str, default=None,
                        help="EC2 instance type (overrides yaml)")
    parser.add_argument("--s3-bucket", type=str, default=None,
                        help="S3 bucket for checkpoints (overrides yaml)")
    parser.add_argument("--volume-size-gb", type=int, default=None,
                        help="EBS volume size in GB (overrides yaml)")
    parser.add_argument("--key-name", type=str, default=None,
                        help="SSH key pair name (overrides yaml)")
    parser.add_argument("--key-file", type=str, default=None,
                        help="Path to SSH key file (overrides yaml)")
    parser.add_argument("--region", type=str, default=None,
                        help="AWS region (overrides yaml)")

    # parse_known_args: parses recognized args, returns unrecognized ones in 'extra'
    # so training args like --epochs, --wandb pass through to the docker command
    args, extra = parser.parse_known_args()

    # Build overrides dict from args that were explicitly set
    cli_overrides = {}
    if args.instance_type is not None:
        cli_overrides["instance_type"] = args.instance_type
    if args.s3_bucket is not None:
        cli_overrides["s3_bucket"] = args.s3_bucket
    if args.volume_size_gb is not None:
        cli_overrides["volume_size_gb"] = args.volume_size_gb
    if args.key_name is not None:
        cli_overrides["key_name"] = args.key_name
    if args.key_file is not None:
        cli_overrides["key_file"] = _resolve_path(args.key_file)
    if args.region is not None:
        cli_overrides["aws_region"] = args.region

    return args.module, " ".join(extra), args.config, cli_overrides


def load_config_and_parse_args():
    """
    Load configuration with precedence: CLI args > YAML > env vars > defaults.

    Returns:
        Tuple of (cfg dict, module str, extra_args str).
    """
    module, extra_args, config_path, cli_overrides = _parse_args()

    # 4th precedence: defaults
    cfg = {"wandb_api_key": ""}

    # 3rd precedence: env vars
    wandb_from_env = os.environ.get("WANDB_API_KEY", "")
    if wandb_from_env:
        cfg["wandb_api_key"] = wandb_from_env

    # 2nd precedence: YAML config
    yaml_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    yaml_cfg = _load_yaml(yaml_path)
    cfg.update(yaml_cfg)

    # 1st precedence: CLI args
    for key, value in cli_overrides.items():
        cfg[key] = value

    # Recompute derived values after overrides
    cfg["ecr_uri"] = f"{cfg['aws_account_id']}.dkr.ecr.{cfg['aws_region']}.amazonaws.com"
    cfg["image_uri"] = f"{cfg['ecr_uri']}/{cfg['ecr_repo_name']}:{cfg['image_tag']}"

    # Recompute key_file if key_name was overridden but key_file wasn't
    if "key_name" in cli_overrides and "key_file" not in cli_overrides:
        cfg["key_file"] = str(Path.home() / ".ssh" / f"{cfg['key_name']}.pem")

    return cfg, module, extra_args
