"""Unified entry point for all training pipelines."""

import sys
import importlib
import os


PIPELINES = {
    "mnist": {
        "module": "src.mnist.train_mnist",
        "description": "MNIST digit classification (CNN)",
    },
    "unet": {
        "module": "src.UNET_PascalVOC_simple.train",
        "description": "U-Net semantic segmentation on Pascal VOC 2012",
    },
}


def print_usage():
    print("Usage: python run.py <pipeline> [args...]\n")
    print("Available pipelines:")
    for name, info in PIPELINES.items():
        print(f"  {name:10s}  {info['description']}")
    print("\nExamples:")
    print("  python run.py mnist --epochs 10 --batch-size 64")
    print("  python run.py unet  --epochs 25 --batch-size 8 --image-size 256")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print_usage()
        sys.exit(0)

    pipeline = sys.argv[1]
    if pipeline not in PIPELINES:
        print(f"Error: unknown pipeline '{pipeline}'")
        print_usage()
        sys.exit(1)

    # Remove the pipeline name from argv so argparse in each module sees only its own args
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    # Add project root to path so relative imports within pipelines work
    project_root = os.path.dirname(os.path.abspath(__file__))
    for subdir in ("src/mnist", "src/UNET_PascalVOC_simple"):
        path = os.path.join(project_root, subdir)
        if path not in sys.path:
            sys.path.insert(0, path)

    module = importlib.import_module(PIPELINES[pipeline]["module"])
    module.main()


if __name__ == "__main__":
    main()
