import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image


# Pascal VOC 2012 class names (21 classes including background)
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
]

NUM_CLASSES = len(VOC_CLASSES)  # 21

# Pascal VOC color palette for visualization (RGB)
VOC_COLORMAP = [
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
    (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0),
    (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128),
    (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
    (0, 64, 128),
]


class PascalVOCSegmentation(Dataset):
    """
    Wrapper around torchvision's VOCSegmentation that returns
    properly preprocessed image and label tensors.

    - Images are resized, normalized to ImageNet stats.
    - Labels are resized with nearest-neighbor and converted to long tensors.
    - The VOC "border" class (255) is mapped to 0 (background) to keep things simple.
    """

    def __init__(self, root, year="2012", image_set="train", image_size=256):
        self.dataset = datasets.VOCSegmentation(
            root=root, year=year, image_set=image_set, download=True,
        )
        self.image_size = image_size
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]

        # Transform image
        image = self.image_transform(image)

        # Transform mask: resize with nearest-neighbor, convert to class indices
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        mask = torch.from_numpy(np.array(mask)).long()

        # Map the 255 "border/void" label to 0 (background)
        mask[mask == 255] = 0

        return image, mask


def decode_segmentation(mask):
    """Convert a class-index mask (H, W) to an RGB image (H, W, 3) for visualization."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(VOC_COLORMAP):
        rgb[mask == cls_id] = color
    return rgb
