from functools import lru_cache

from app.core.config import get_settings
from app.services.pipeline import ScreeningPipeline
from app.services.reasoning import build_reasoner
from app.services.similarity import build_backend


@lru_cache
def get_pipeline() -> ScreeningPipeline:
    settings = get_settings()
    return ScreeningPipeline(settings, build_backend(settings), build_reasoner(settings))
