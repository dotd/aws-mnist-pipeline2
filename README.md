# AWS General Pipeline training 

## Usage

```bash
python run.py <pipeline> [args...]
```

### Available pipelines

| Pipeline | Description |
|---|---|
| `mnist` | MNIST digit classification (CNN) |
| `unet` | U-Net semantic segmentation on Pascal VOC 2012 |

### Examples

```bash
# Train MNIST
python run.py mnist --epochs 10 --batch-size 64

# Train U-Net segmentation
python run.py unet --epochs 25 --batch-size 8 --image-size 256

# Docker: override pipeline at runtime
docker run my-cv-model python run.py unet --epochs 25
```

