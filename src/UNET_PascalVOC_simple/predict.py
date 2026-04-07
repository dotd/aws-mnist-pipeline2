"""Run inference on a single image and save the segmentation overlay."""

import argparse

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.UNET_PascalVOC_simple.model import UNet
from src.UNET_PascalVOC_simple.dataset import NUM_CLASSES, decode_segmentation


def predict(image_path, checkpoint_path, image_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (W, H)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(dim=1).squeeze(0).cpu().numpy()

    # Resize prediction back to original image size
    pred_resized = np.array(
        Image.fromarray(pred.astype(np.uint8)).resize(original_size, Image.NEAREST)
    )

    # Create colored overlay
    overlay = decode_segmentation(pred_resized)
    overlay_image = Image.fromarray(overlay)

    # Blend with original
    blended = Image.blend(image, overlay_image, alpha=0.5)
    return blended, overlay_image


def main():
    parser = argparse.ArgumentParser(description="U-Net Pascal VOC Prediction")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--output", type=str, default="prediction.png", help="Output file path")
    args = parser.parse_args()

    blended, _ = predict(args.image, args.checkpoint, args.image_size)
    blended.save(args.output)
    print(f"Saved prediction to {args.output}")


if __name__ == "__main__":
    main()
