"""Pydantic response schemas for the FaceProof API.

Every endpoint returns the same envelope — ``data`` on success, ``error`` on
failure. The ``*Out`` models mirror the pipeline dataclasses and read straight
from them via ``from_attributes``.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

_T = TypeVar("_T")


class FaceMatchOut(BaseModel):
    """Face-match sub-result: cosine similarity, threshold, and the decision."""

    model_config = ConfigDict(from_attributes=True)

    similarity: float
    threshold: float
    is_match: bool


class LivenessOut(BaseModel):
    """Liveness sub-result: live-score, decision, and a human-readable label."""

    model_config = ConfigDict(from_attributes=True)

    score: float
    is_live: bool
    label: str


class VerifyOut(BaseModel):
    """Full verification result with both sub-scores and explanatory reasons."""

    model_config = ConfigDict(from_attributes=True)

    verified: bool
    face_match: FaceMatchOut
    liveness: LivenessOut
    reasons: list[str]


class ErrorOut(BaseModel):
    """Structured error detail — a stable code and a human-readable message."""

    code: str
    message: str


class ApiResponse(BaseModel, Generic[_T]):
    """The response envelope — exactly one of ``data`` or ``error`` is set."""

    data: _T | None = None
    error: ErrorOut | None = None
