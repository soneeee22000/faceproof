"""Liveness / anti-spoofing detection.

Given an image and a face box, decide whether the face is a live capture or a
presentation attack (a printed photo or a screen replay).

``detect_liveness`` uses the **trained MobileNetV2 classifier** as the primary
detector when its weights are present, and falls back to the Apache-2.0
**Silent-Face** baseline (vendored architecture in ``faceproof._minifasnet``)
otherwise. ``assess_liveness`` always runs Silent-Face — it is the fallback path
and the benchmark baseline used by the evaluation harness.

Torch and the models are imported lazily, so this module imports without the
optional ``[ml]`` extra (keeps CI light).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from faceproof.config import settings
from faceproof.detection import detect_primary_face
from faceproof.errors import LivenessModelMissingError

_INPUT_SIZE = 80
_REAL_CLASS = 1
_MODEL_FILES = ("2.7_80x80_MiniFASNetV2.pth", "4_0_0_80x80_MiniFASNetV1SE.pth")
_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
_TRAINED_WEIGHTS = _MODELS_DIR / "antispoofing_mobilenetv2.pth"


@dataclass(frozen=True)
class LivenessResult:
    """The outcome of an anti-spoofing assessment.

    Attributes:
        score: Probability the face is a live capture, in ``[0, 1]``.
        is_live: ``True`` for a live face, ``False`` for a presentation attack.
        label: ``"live"`` or ``"spoof"`` — the human-readable form of ``is_live``.
    """

    score: float
    is_live: bool
    label: str


def _parse_model_name(filename: str) -> tuple[int, int, str, float]:
    """Parse a Silent-Face weight filename into ``(height, width, type, scale)``."""
    parts = filename.split("_")[:-1]
    height, width = (int(value) for value in parts[-1].split("x"))
    model_type = filename.split(".pth")[0].split("_")[-1]
    return height, width, model_type, float(parts[0])


def _kernel_size(height: int, width: int) -> tuple[int, int]:
    """Return the final depthwise-conv kernel size for a given input resolution."""
    return (height + 15) // 16, (width + 15) // 16


def _crop_box(
    image_width: int,
    image_height: int,
    bbox: tuple[float, float, float, float],
    scale: float,
) -> tuple[int, int, int, int]:
    """Return the scale-expanded face crop box, clamped inside the image."""
    x, y, box_width, box_height = bbox
    scale = min((image_height - 1) / box_height, (image_width - 1) / box_width, scale)
    new_width, new_height = box_width * scale, box_height * scale
    center_x, center_y = x + box_width / 2, y + box_height / 2
    left, top = center_x - new_width / 2, center_y - new_height / 2
    right, bottom = center_x + new_width / 2, center_y + new_height / 2
    if left < 0:
        right, left = right - left, 0.0
    if top < 0:
        bottom, top = bottom - top, 0.0
    if right > image_width - 1:
        left, right = left - (right - image_width + 1), float(image_width - 1)
    if bottom > image_height - 1:
        top, bottom = top - (bottom - image_height + 1), float(image_height - 1)
    return int(left), int(top), int(right), int(bottom)


def _fuse(probabilities: list[NDArray[np.float32]]) -> tuple[float, bool]:
    """Fuse per-model softmax outputs into a ``(live score, is_live)`` decision."""
    summed = np.sum(probabilities, axis=0)
    predicted = int(np.argmax(summed))
    score = float(summed[_REAL_CLASS]) / len(probabilities)
    return score, predicted == _REAL_CLASS


def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Drop the ``module.`` prefix left by ``DataParallel`` training, if present."""
    prefix = "module."
    if not next(iter(state_dict)).startswith(prefix):
        return state_dict
    return {key.removeprefix(prefix): value for key, value in state_dict.items()}


@lru_cache(maxsize=1)
def _silentface_models() -> list[tuple[Any, float]]:
    """Load the Silent-Face MiniFASNet models once: ``(network, crop scale)`` pairs.

    Raises:
        LivenessModelMissingError: If the pretrained weights are not present.
    """
    import torch

    from faceproof import _minifasnet

    factories: dict[str, Any] = {
        "MiniFASNetV1": _minifasnet.MiniFASNetV1,
        "MiniFASNetV2": _minifasnet.MiniFASNetV2,
        "MiniFASNetV1SE": _minifasnet.MiniFASNetV1SE,
        "MiniFASNetV2SE": _minifasnet.MiniFASNetV2SE,
    }
    loaded: list[tuple[Any, float]] = []
    for filename in _MODEL_FILES:
        weights_path = _MODELS_DIR / filename
        if not weights_path.exists():
            raise LivenessModelMissingError(
                f"Anti-spoofing weights not found: {weights_path}. "
                "Run `python models/download_silentface.py` first."
            )
        height, width, model_type, scale = _parse_model_name(filename)
        network = factories[model_type](conv6_kernel=_kernel_size(height, width))
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        network.load_state_dict(_strip_module_prefix(state_dict))
        network.eval()
        loaded.append((network, scale))
    return loaded


@lru_cache(maxsize=1)
def _trained_model() -> tuple[Any, Any]:
    """Load the trained MobileNetV2 classifier once: ``(network, transform)``."""
    from faceproof._mobilenet import inference_transform, load_antispoofing_model

    return load_antispoofing_model(_TRAINED_WEIGHTS, "cpu"), inference_transform()


def assess_liveness(image: NDArray[np.uint8], bbox: NDArray[np.float32]) -> LivenessResult:
    """Assess liveness with the Silent-Face baseline — the fallback detector.

    Args:
        image: BGR ``uint8`` image.
        bbox: Face bounding box ``[x1, y1, x2, y2]``.
    """
    import cv2
    import torch

    left_x, top_y, right_x, bottom_y = (float(value) for value in bbox)
    face_box = (left_x, top_y, right_x - left_x, bottom_y - top_y)
    image_height, image_width = image.shape[:2]
    probabilities: list[NDArray[np.float32]] = []
    for network, scale in _silentface_models():
        left, top, right, bottom = _crop_box(image_width, image_height, face_box, scale)
        crop = cv2.resize(
            image[top : bottom + 1, left : right + 1], (_INPUT_SIZE, _INPUT_SIZE)
        )
        tensor = torch.from_numpy(crop).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        with torch.no_grad():
            logits = network(tensor)
        probabilities.append(torch.softmax(logits, dim=1)[0].numpy())
    score, is_live = _fuse(probabilities)
    return LivenessResult(score=score, is_live=is_live, label="live" if is_live else "spoof")


def _assess_with_trained_model(
    image: NDArray[np.uint8], bbox: NDArray[np.float32]
) -> LivenessResult:
    """Assess liveness with the trained MobileNetV2 — the primary detector."""
    import torch
    from PIL import Image as PILImage

    from faceproof._mobilenet import LIVE_CLASS

    network, transform = _trained_model()
    left, top, right, bottom = (max(int(value), 0) for value in bbox)
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        crop = image
    face = PILImage.fromarray(np.ascontiguousarray(crop[:, :, ::-1]))
    tensor = transform(face).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(network(tensor), dim=1)[0]
    score = float(probabilities[LIVE_CLASS])
    is_live = score >= settings.liveness_threshold
    return LivenessResult(score=score, is_live=is_live, label="live" if is_live else "spoof")


def detect_liveness(image: NDArray[np.uint8]) -> LivenessResult:
    """Detect the primary face in an image and assess its liveness.

    Uses the trained MobileNetV2 classifier when its weights are present, and
    falls back to the Silent-Face baseline otherwise.

    Raises:
        NoFaceDetectedError: If no face is detected.
        LivenessModelMissingError: If neither the trained classifier nor the
            Silent-Face weights are available.
    """
    face = detect_primary_face(image)
    if _TRAINED_WEIGHTS.exists():
        return _assess_with_trained_model(image, face.bbox)
    return assess_liveness(image, face.bbox)
