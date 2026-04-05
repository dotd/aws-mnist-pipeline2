import logging
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class MNISTConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        correct += output.argmax(dim=1).eq(target).sum().item()
        total += data.size(0)

        if (batch_idx + 1) % 100 == 0:
            logger.info(
                "Epoch %d | Batch %d/%d | Loss: %.4f",
                epoch, batch_idx + 1, len(loader), loss.item(),
            )

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item() * data.size(0)
            correct += output.argmax(dim=1).eq(target).sum().item()
            total += data.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="MNIST CNN Training")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MNIST CNN Training")
    logger.info("=" * 60)

    # --- Device setup ---
    device = get_device()
    logger.info("Using device: %s", device)

    # --- Data download and preparation ---
    logger.info("Downloading and preparing MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info("Train samples: %d | Test samples: %d", len(train_dataset), len(test_dataset))

    # --- Model setup ---
    logger.info("Building convolutional network...")
    model = MNISTConvNet().to(device)
    logger.info("Model architecture:\n%s", model)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Total parameters: %d", total_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- Training loop ---
    os.makedirs(args.save_dir, exist_ok=True)
    logger.info("Starting training for %d epochs...", args.epochs)

    best_test_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        logger.info("-" * 40)
        logger.info("Epoch %d/%d", epoch, args.epochs)

        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        elapsed = time.time() - start

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        logger.info(
            "Train Loss: %.4f | Train Acc: %.2f%% | Test Loss: %.4f | Test Acc: %.2f%% | Time: %.1fs",
            train_loss, train_acc, test_loss, test_acc, elapsed,
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            path = os.path.join(args.save_dir, "best_model.pt")
            torch.save(model.state_dict(), path)
            logger.info("New best model saved to %s (acc=%.2f%%)", path, test_acc)

    # --- Final save ---
    final_path = os.path.join(args.save_dir, "final_model.pt")
    torch.save(model.state_dict(), final_path)
    logger.info("Final model saved to %s", final_path)
    logger.info("Training complete. Best test accuracy: %.2f%%", best_test_acc)


if __name__ == "__main__":
    main()
