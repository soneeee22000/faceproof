"""Runtime configuration for the FaceProof service."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Service settings, loaded from environment variables (prefix ``FACEPROOF_``)."""

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="FACEPROOF_", extra="ignore"
    )

    service_name: str = "faceproof"
    cors_origins: str = "http://localhost:5173"
    max_upload_bytes: int = 8 * 1024 * 1024
    face_match_threshold: float = 0.2528  # LFW-calibrated — see evaluation/results/
    liveness_threshold: float = 0.1058  # CelebA-Spoof-calibrated — see evaluation/results/
    onnx_providers: str = "CPUExecutionProvider"

    @property
    def onnx_provider_list(self) -> list[str]:
        """ONNX Runtime execution providers, highest priority first.

        Defaults to CPU (the Cloud Run deployment target). Set
        ``FACEPROOF_ONNX_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider``
        to run on a GPU, e.g. for the LFW evaluation on Colab.
        """
        return [name.strip() for name in self.onnx_providers.split(",") if name.strip()]

    @property
    def onnx_ctx_id(self) -> int:
        """InsightFace device id: 0 when a CUDA provider is configured, else -1 (CPU)."""
        if any("CUDA" in name for name in self.onnx_provider_list):
            return 0
        return -1

    @property
    def cors_origin_list(self) -> list[str]:
        """Allowed CORS origins, parsed from the comma-separated setting."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
