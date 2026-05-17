"""Image upload decoding and validation.

Decodes an uploaded image into the BGR ``uint8`` array the FaceProof pipeline
expects: it applies any EXIF orientation, normalizes to 3-channel RGB, and
rejects oversized or undecodable uploads. Converting to a NumPy array inherently
drops EXIF and other metadata. Pillow and NumPy are core dependencies, so this
module needs no optional extra.
"""

from __future__ import annotations

import io

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps, UnidentifiedImageError

from faceproof.config import settings
from faceproof.errors import ImageTooLargeError, InvalidImageError


def decode_image(data: bytes) -> NDArray[np.uint8]:
    """Decode uploaded image bytes into a BGR ``uint8`` array.

    Raises:
        ImageTooLargeError: If the upload exceeds ``settings.max_upload_bytes``.
        InvalidImageError: If the bytes are empty or not a decodable image.
    """
    if not data:
        raise InvalidImageError("Uploaded file is empty.")
    if len(data) > settings.max_upload_bytes:
        raise ImageTooLargeError(
            f"Upload is {len(data)} bytes; the limit is {settings.max_upload_bytes}."
        )
    try:
        with Image.open(io.BytesIO(data)) as image:
            rgb_image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise InvalidImageError("Uploaded file is not a decodable image.") from error

    rgb = np.asarray(rgb_image, dtype=np.uint8)
    bgr: NDArray[np.uint8] = np.ascontiguousarray(rgb[:, :, ::-1])
    return bgr
