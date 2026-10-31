import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.deps import get_pipeline
from app.api.routes import router
from app.core.config import get_settings
from app.core.errors import register_error_handlers
from app.core.logging import configure_logging

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging(get_settings().log_level)
    get_pipeline()  # load models at startup instead of on the first request
    logger.info("Application ready")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Resume Screening Agent",
        version=__version__,
        description="Ranks resumes against a job description.",
        lifespan=lifespan,
    )
    register_error_handlers(app)
    app.include_router(router)
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


app = create_app()
