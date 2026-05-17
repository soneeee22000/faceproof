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


settings = Settings()
