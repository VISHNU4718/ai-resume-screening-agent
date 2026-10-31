import logging
from functools import lru_cache

from app.core.config import Settings, get_settings
from app.llm.cache import DiskCache
from app.llm.client import OpenAICompatibleClient
from app.llm.prompts import PromptLibrary
from app.llm.structured import StructuredLLM
from app.services.engine import AnalysisEngine, RuleBasedEngine
from app.services.llm_engine import LLMEngine
from app.services.pipeline import ScreeningPipeline
from app.services.rag import AskService, ScreeningStore
from app.services.similarity import build_backend

logger = logging.getLogger(__name__)


def build_structured_llm(settings: Settings) -> StructuredLLM | None:
    if not settings.llm_enabled:
        return None
    cache = DiskCache(settings.llm_cache_dir) if settings.llm_cache_enabled else None
    return StructuredLLM(
        OpenAICompatibleClient.from_settings(settings),
        PromptLibrary(settings.prompt_version),
        cache,
    )


def build_engine(llm: StructuredLLM | None) -> AnalysisEngine:
    return LLMEngine(llm) if llm else RuleBasedEngine()


@lru_cache
def get_store() -> ScreeningStore:
    return ScreeningStore(get_settings().max_screenings_kept)


@lru_cache
def get_structured_llm() -> StructuredLLM | None:
    settings = get_settings()
    llm = build_structured_llm(settings)
    if llm:
        logger.info(
            "LLM enabled: provider=%s model=%s base_url=%s",
            settings.llm_provider,
            llm.model,
            settings.resolved_llm_base_url,
        )
    else:
        logger.info("No LLM configured: running in rule-based fallback mode")
    return llm


@lru_cache
def get_pipeline() -> ScreeningPipeline:
    settings = get_settings()
    return ScreeningPipeline(
        settings, build_backend(settings), build_engine(get_structured_llm()), get_store()
    )


@lru_cache
def get_ask_service() -> AskService:
    settings = get_settings()
    pipeline = get_pipeline()
    return AskService(get_store(), pipeline.similarity, get_structured_llm(), settings.ask_top_k)
