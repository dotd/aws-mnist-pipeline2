# AWS General Pipeline training 

## Usage

```bash
python -m <module> [args...]
```

### Available modules

| Module | Description |
|---|---|
| `src.mnist.train_mnist` | MNIST digit classification (CNN) |
| `src.UNET_PascalVOC_simple.train` | U-Net semantic segmentation on Pascal VOC 2012 |

### Examples

```bash
# Train MNIST
python -m src.mnist.train_mnist --epochs 10 --batch-size 64

# Train U-Net segmentation
python -m src.UNET_PascalVOC_simple.train --epochs 25 --batch-size 8 --image-size 256

# Docker: pass module at runtime
docker run my-cv-model python -m src.mnist.train_mnist --epochs 25
```

