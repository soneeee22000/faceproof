"""Tests for the image upload decoding/validation module.

Pillow and NumPy are core dependencies, so every test runs without the CV stack.
"""

import io

import numpy as np
import pytest
from PIL import Image

from faceproof.config import settings
from faceproof.errors import ImageTooLargeError, InvalidImageError
from faceproof.images import decode_image


def _png_bytes(image: Image.Image) -> bytes:
    """Encode a Pillow image as PNG bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_decodes_a_valid_png() -> None:
    """A valid PNG decodes to an (H, W, 3) uint8 array."""
    decoded = decode_image(_png_bytes(Image.new("RGB", (64, 48), (10, 20, 30))))
    assert decoded.shape == (48, 64, 3)
    assert decoded.dtype == np.uint8


def test_converts_rgb_to_bgr_channel_order() -> None:
    """A pure-red RGB image decodes to BGR — blue 0, green 0, red 255."""
    decoded = decode_image(_png_bytes(Image.new("RGB", (8, 8), (255, 0, 0))))
    assert tuple(int(channel) for channel in decoded[0, 0]) == (0, 0, 255)


def test_normalizes_grayscale_to_three_channels() -> None:
    """A single-channel image is normalized to 3-channel BGR."""
    decoded = decode_image(_png_bytes(Image.new("L", (16, 16), 128)))
    assert decoded.shape == (16, 16, 3)


def test_rejects_empty_upload() -> None:
    """An empty upload is rejected."""
    with pytest.raises(InvalidImageError):
        decode_image(b"")


def test_rejects_non_image_bytes() -> None:
    """Bytes that are not an image are rejected."""
    with pytest.raises(InvalidImageError):
        decode_image(b"this is plainly not an image file")


def test_rejects_oversized_upload() -> None:
    """An upload above the size limit is rejected before decoding."""
    with pytest.raises(ImageTooLargeError):
        decode_image(b"\x00" * (settings.max_upload_bytes + 1))
