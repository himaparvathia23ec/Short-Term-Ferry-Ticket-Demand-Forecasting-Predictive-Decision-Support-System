"""Environment-based configuration using Pydantic Settings."""

from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings loaded from environment and optional `.env` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_path: Path = Field(default=Path("data/ferry_tickets.csv"))
    models_dir: Path = Field(default=Path("models"))
    reports_dir: Path = Field(default=Path("reports/figures"))
    logs_dir: Path = Field(default=Path("logs"))
    api_key: str | None = Field(default=None, description="If set, require X-API-Key header")
    cors_origins: str = Field(
        default="http://localhost:8501,http://127.0.0.1:8501",
        description="Comma-separated CORS origins",
    )

    @field_validator("data_path", "models_dir", "reports_dir", "logs_dir", mode="before")
    @classmethod
    def resolve_paths(cls, v: Path | str) -> Path:
        p = Path(v)
        if not p.is_absolute():
            p = _project_root() / p
        return p

    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return cached settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings_cache() -> None:
    """Clear settings cache (for tests)."""
    global _settings
    _settings = None
