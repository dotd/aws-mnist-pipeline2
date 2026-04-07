from pathlib import Path

# Project root directory (parent of src/)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Common directories
SRC_DIR = ROOT_DIR / "src"
CONFIGS_DIR = ROOT_DIR / "configs"
SCRIPTS_DIR = ROOT_DIR / "scripts"
SCRIPTS_PY_DIR = ROOT_DIR / "scripts_py"
DATA_DIR = ROOT_DIR / "data"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
LOGS_DIR = ROOT_DIR / "logs"
DOCS_DIR = ROOT_DIR / "docs"
API_KEYS_DIR = ROOT_DIR / "api_keys"
