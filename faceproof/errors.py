"""Domain errors for the FaceProof verification pipeline.

These are raised by the pipeline modules and mapped by the FastAPI layer to a
structured error response: ``{"data": null, "error": {"code", "message"}}``.
"""


class FaceProofError(Exception):
    """Base class for all FaceProof domain errors."""


class NoFaceDetectedError(FaceProofError):
    """Raised when no face can be detected in an input image."""


class LivenessModelMissingError(FaceProofError):
    """Raised when the Silent-Face anti-spoofing weights are not available."""


class InvalidImageError(FaceProofError):
    """Raised when an uploaded file is empty or not a decodable image."""


class ImageTooLargeError(FaceProofError):
    """Raised when an uploaded file exceeds the configured size limit."""
