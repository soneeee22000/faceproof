"""Shared MobileNetV2 anti-spoofing model — architecture, weights, transforms.

Used by both the training script and the evaluation harness so the network and
its preprocessing match exactly. Torch and torchvision are imported lazily, so
this module imports without the optional ``[ml]`` extra.
"""

from __future__ import annotations

from typing import Any

IMAGE_SIZE = 224
NUM_CLASSES = 2
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LIVE_CLASS = 0
SPOOF_CLASS = 1


def build_antispoofing_model(pretrained: bool = True) -> Any:
    """Return a MobileNetV2 with a fresh 2-class (live/spoof) head.

    Args:
        pretrained: Load ImageNet weights into the backbone (for training).
            Pass ``False`` for evaluation, where trained weights overwrite the
            whole model anyway.
    """
    from torch import nn
    from torchvision import models

    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    return model


def load_antispoofing_model(weights_path: Any, device: Any) -> Any:
    """Build the model, load trained weights, and return it in eval mode on ``device``."""
    import torch

    model = build_antispoofing_model(pretrained=False)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def inference_transform() -> Any:
    """Return the eval-time preprocessing transform (resize, to-tensor, normalize)."""
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
