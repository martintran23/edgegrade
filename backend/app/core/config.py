"""Application settings (extensible for DB URLs, model paths, etc.)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven configuration."""

    model_config = SettingsConfigDict(env_prefix="CARDGRADING_", extra="ignore")

    api_title: str = "Card Grading AI"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    # Canonical warped card size (height); width derived from detected aspect ratio.
    warp_height: int = 1120


settings = Settings()
