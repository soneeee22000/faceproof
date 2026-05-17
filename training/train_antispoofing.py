"""Transfer-learn a MobileNetV2 anti-spoofing classifier on CelebA-Spoof.

Fine-tunes an ImageNet-pretrained MobileNetV2 into a 2-class live/spoof
classifier on the CelebA-Spoof subset (see ``training.celeba_spoof``).
MobileNetV2 is chosen for its small CPU inference cost — the FaceProof service
deploys CPU-only on Cloud Run.

Writes the trained weights and a training-curve plot to ``models/``. The test
split is left untouched here for the held-out evaluation (APCER/BPCER/ACER).

Usage (from the repository root, ideally on a GPU):
    python -m training.train_antispoofing [--epochs N] [--batch-size N]
                                          [--subset-size N] [--lr LR]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from training.celeba_spoof import load_celeba_spoof_subset

_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
_WEIGHTS_PATH = _MODELS_DIR / "antispoofing_mobilenetv2.pth"
_CURVE_PATH = _MODELS_DIR / "antispoofing_training_curve.png"
_IMAGE_SIZE = 224
_NUM_CLASSES = 2
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class _FaceCropDataset(Dataset):
    """Adapts a Hugging Face CelebA-Spoof split to a torchvision-style dataset."""

    def __init__(self, hf_split: Any, transform: Any) -> None:
        self._split = hf_split
        self._transform = transform

    def __len__(self) -> int:
        return len(self._split)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        record = self._split[index]
        image = record["cropped_image"].convert("RGB")
        return self._transform(image), int(record["labels"])


def _transforms(*, train: bool) -> Any:
    """Build the preprocessing pipeline; training adds a horizontal flip."""
    from torchvision import transforms

    steps: list[Any] = [transforms.Resize((_IMAGE_SIZE, _IMAGE_SIZE))]
    if train:
        steps.append(transforms.RandomHorizontalFlip())
    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD))
    return transforms.Compose(steps)


def _build_model() -> nn.Module:
    """Return an ImageNet-pretrained MobileNetV2 with a fresh 2-class head."""
    from torchvision import models

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.last_channel, _NUM_CLASSES)
    return model


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one epoch; train when an optimizer is given, else evaluate. Returns (loss, accuracy)."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss, correct, count = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * len(labels)
        correct += int((logits.argmax(dim=1) == labels).sum())
        count += len(labels)
    return total_loss / count, correct / count


def _plot_curve(history: list[dict[str, float]], destination: Path) -> None:
    """Save the per-epoch loss and accuracy curves as a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history) + 1)
    figure, (loss_axes, acc_axes) = plt.subplots(1, 2, figsize=(10, 4))
    loss_axes.plot(epochs, [h["train_loss"] for h in history], label="train")
    loss_axes.plot(epochs, [h["val_loss"] for h in history], label="validation")
    loss_axes.set_title("Loss")
    loss_axes.set_xlabel("epoch")
    loss_axes.legend()
    acc_axes.plot(epochs, [h["train_acc"] for h in history], label="train")
    acc_axes.plot(epochs, [h["val_acc"] for h in history], label="validation")
    acc_axes.set_title("Accuracy")
    acc_axes.set_xlabel("epoch")
    acc_axes.legend()
    figure.suptitle("MobileNetV2 anti-spoofing — training")
    figure.savefig(destination, dpi=150, bbox_inches="tight")


def _parse_args() -> argparse.Namespace:
    """Parse command-line training options."""
    parser = argparse.ArgumentParser(description="Train the anti-spoofing classifier")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--subset-size", type=int, default=None,
                        help="optional cap on images for a fast iteration run")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Load CelebA-Spoof, fine-tune MobileNetV2, and save the best weights."""
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    splits = load_celeba_spoof_subset(subset_size=args.subset_size, seed=args.seed)
    train_loader = DataLoader(
        _FaceCropDataset(splits.train, _transforms(train=True)),
        batch_size=args.batch_size, shuffle=True, num_workers=2,
    )
    val_loader = DataLoader(
        _FaceCropDataset(splits.validation, _transforms(train=False)),
        batch_size=args.batch_size, num_workers=2,
    )
    print(f"train: {len(train_loader.dataset)}  validation: {len(val_loader.dataset)}", flush=True)

    model = _build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict[str, float]] = []
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_loss, train_acc = _run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, device, criterion)
        history.append({"train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc})
        print(
            f"epoch {epoch}/{args.epochs}  train_acc={train_acc:.4f}  "
            f"val_acc={val_acc:.4f}  ({time.perf_counter() - started:.0f}s)",
            flush=True,
        )
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            _MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), _WEIGHTS_PATH)

    _plot_curve(history, _CURVE_PATH)
    print(f"\nbest validation accuracy: {best_val_acc:.4f}", flush=True)
    print(f"weights: {_WEIGHTS_PATH}", flush=True)
    print(f"training curve: {_CURVE_PATH}", flush=True)


if __name__ == "__main__":
    main()
