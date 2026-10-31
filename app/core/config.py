"""Application settings, read from environment variables or a local .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# provider -> (base URL, default model)
PROVIDER_PRESETS: dict[str, tuple[str, str]] = {
    "none": ("", ""),
    "groq": ("https://api.groq.com/openai/v1", "llama-3.1-8b-instant"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "openrouter": ("https://openrouter.ai/api/v1", "meta-llama/llama-3.1-8b-instruct"),
    "ollama": ("http://localhost:11434/v1", "qwen2.5:1.5b"),
    "custom": ("", ""),
}


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

    # LLM provider (any OpenAI-compatible chat completions API). Base URL and model fall
    # back to the provider preset below unless overridden.
    llm_provider: Literal["none", "groq", "openai", "openrouter", "ollama", "custom"] = "groq"
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=6)
    llm_max_concurrency: int = Field(default=2, ge=1, le=16)
    llm_json_mode: bool = True
    llm_cache_enabled: bool = True
    llm_cache_dir: Path = Path(".cache/llm")
    prompt_version: str = "v1"

    # Question answering over screened resumes
    ask_top_k: int = Field(default=5, ge=1, le=12)
    max_question_chars: int = Field(default=500, ge=10)
    max_screenings_kept: int = Field(default=20, ge=1)

    # Scoring
    weight_skills: float = Field(default=0.55, ge=0)
    weight_semantic: float = Field(default=0.30, ge=0)
    weight_experience: float = Field(default=0.15, ge=0)
    shortlist_threshold: float = Field(default=70.0, ge=0, le=100)
    maybe_threshold: float = Field(default=45.0, ge=0, le=100)

    @field_validator("llm_api_key", "llm_base_url", "llm_model", mode="before")
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
        """True when an LLM can be called: a key is set, or the provider needs none."""
        if self.llm_provider == "none":
            return False
        if self.llm_provider == "ollama":
            return True
        if self.llm_provider == "custom":
            return self.llm_base_url is not None and self.llm_model is not None
        return self.llm_api_key is not None

    @property
    def resolved_llm_base_url(self) -> str:
        return (self.llm_base_url or PROVIDER_PRESETS[self.llm_provider][0]).rstrip("/")

    @property
    def resolved_llm_model(self) -> str:
        return self.llm_model or PROVIDER_PRESETS[self.llm_provider][1]


@lru_cache
def get_settings() -> Settings:
    return Settings()
