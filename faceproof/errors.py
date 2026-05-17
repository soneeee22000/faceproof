"""Domain errors for the FaceProof verification pipeline.

These are raised by the pipeline modules and mapped to API error codes
(see ``CLAUDE.md`` API Response Pattern) by the FastAPI layer.
"""


class FaceProofError(Exception):
    """Base class for all FaceProof domain errors."""


class NoFaceDetectedError(FaceProofError):
    """Raised when no face can be detected in an input image."""
