"""Image upload decoding and validation.

Decodes an uploaded file into the BGR ``uint8`` array the FaceProof pipeline
expects. Standard image formats are decoded by Pillow with EXIF orientation
applied; a PDF (identity documents are often scanned to PDF) has its first page
rendered to a raster image. Either way the result is normalized to 3-channel
RGB, then to BGR. Converting to a NumPy array inherently drops EXIF metadata.
"""

from __future__ import annotations

import io

import numpy as np
import pypdfium2 as pdfium
from numpy.typing import NDArray
from PIL import Image, ImageOps, UnidentifiedImageError

from faceproof.config import settings
from faceproof.errors import ImageTooLargeError, InvalidImageError

# A PDF file always begins with this signature.
_PDF_MAGIC = b"%PDF-"
# Render scale for PDF pages — ~144 DPI, enough resolution for a document portrait.
_PDF_RENDER_SCALE = 2.0


def _render_pdf_first_page(data: bytes) -> Image.Image:
    """Render the first page of a PDF into an RGB Pillow image.

    Raises:
        InvalidImageError: If the PDF cannot be opened or has no pages.
    """
    try:
        document = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as error:
        raise InvalidImageError("Uploaded PDF could not be opened.") from error
    try:
        if len(document) == 0:
            raise InvalidImageError("Uploaded PDF has no pages.")
        bitmap = document[0].render(scale=_PDF_RENDER_SCALE)
        page_image: Image.Image = bitmap.to_pil()
        return page_image.convert("RGB")
    finally:
        document.close()


def decode_image(data: bytes) -> NDArray[np.uint8]:
    """Decode uploaded image or PDF bytes into a BGR ``uint8`` array.

    Raises:
        ImageTooLargeError: If the upload exceeds ``settings.max_upload_bytes``.
        InvalidImageError: If the bytes are empty or not a decodable image/PDF.
    """
    if not data:
        raise InvalidImageError("Uploaded file is empty.")
    if len(data) > settings.max_upload_bytes:
        raise ImageTooLargeError(
            f"Upload is {len(data)} bytes; the limit is {settings.max_upload_bytes}."
        )

    if data.startswith(_PDF_MAGIC):
        rgb_image = _render_pdf_first_page(data)
    else:
        try:
            with Image.open(io.BytesIO(data)) as image:
                rgb_image = ImageOps.exif_transpose(image).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise InvalidImageError(
                "Uploaded file is not a decodable image or PDF."
            ) from error

    rgb = np.asarray(rgb_image, dtype=np.uint8)
    bgr: NDArray[np.uint8] = np.ascontiguousarray(rgb[:, :, ::-1])
    return bgr
