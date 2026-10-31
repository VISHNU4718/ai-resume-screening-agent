"""Application settings, read from environment variables or a local .env file."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    log_level: str = "INFO"

    # Upload limits
    max_file_size_mb: float = Field(default=5.0, gt=0)
    max_files: int = Field(default=20, ge=1)
    min_jd_chars: int = Field(default=40, ge=1)
    max_jd_chars: int = Field(default=20_000, ge=100)

    # Semantic similarity
    embedding_backend: Literal["auto", "tfidf", "sbert"] = "auto"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Optional LLM reasoning (any OpenAI-compatible chat completions API)
    llm_api_key: SecretStr | None = None
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "llama-3.1-8b-instant"
    llm_timeout_seconds: float = Field(default=30.0, gt=0)

    # Scoring
    weight_skills: float = Field(default=0.55, ge=0)
    weight_semantic: float = Field(default=0.30, ge=0)
    weight_experience: float = Field(default=0.15, ge=0)
    shortlist_threshold: float = Field(default=70.0, ge=0, le=100)
    maybe_threshold: float = Field(default=45.0, ge=0, le=100)

    @field_validator("llm_api_key", mode="before")
    @classmethod
    def _blank_key_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("log_level")
    @classmethod
    def _upper_level(cls, value: str) -> str:
        return value.upper()

    @model_validator(mode="after")
    def _check_consistency(self) -> "Settings":
        if self.maybe_threshold > self.shortlist_threshold:
            raise ValueError("MAYBE_THRESHOLD must not exceed SHORTLIST_THRESHOLD")
        if self.weight_skills + self.weight_semantic + self.weight_experience <= 0:
            raise ValueError("At least one scoring weight must be positive")
        return self

    @property
    def max_file_size_bytes(self) -> int:
        return int(self.max_file_size_mb * 1024 * 1024)

    @property
    def llm_enabled(self) -> bool:
        return self.llm_api_key is not None


@lru_cache
def get_settings() -> Settings:
    return Settings()
