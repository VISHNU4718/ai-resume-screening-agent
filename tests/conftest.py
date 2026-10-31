import io
from pathlib import Path

import pytest
from docx import Document
from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from app.api.deps import get_ask_service, get_pipeline
from app.core.config import Settings, get_settings
from app.main import app
from app.services.engine import RuleBasedEngine
from app.services.pipeline import ScreeningPipeline
from app.services.rag import AskService, ScreeningStore
from app.services.similarity import TfidfBackend

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def make_settings(**overrides) -> Settings:
    base = {"embedding_backend": "tfidf", "llm_provider": "none", "llm_api_key": None}
    return Settings(_env_file=None, **{**base, **overrides})


def make_docx(text: str) -> bytes:
    doc = Document()
    for line in text.splitlines():
        doc.add_paragraph(line)
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def make_pdf(text: str) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    y = 800
    for line in text.splitlines():
        pdf.drawString(50, y, line)
        y -= 14
    pdf.save()
    return buffer.getvalue()


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def store() -> ScreeningStore:
    return ScreeningStore(max_items=5)


@pytest.fixture
def pipeline(settings: Settings, store: ScreeningStore) -> ScreeningPipeline:
    """Rule-based fallback pipeline."""
    return ScreeningPipeline(settings, TfidfBackend(), RuleBasedEngine(), store)


@pytest.fixture
def ask_service(settings: Settings, store: ScreeningStore) -> AskService:
    return AskService(store, TfidfBackend(), None, settings.ask_top_k)


def make_client(settings: Settings, pipeline: ScreeningPipeline, ask_service: AskService):
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    app.dependency_overrides[get_ask_service] = lambda: ask_service
    return TestClient(app)


@pytest.fixture
def client(settings: Settings, pipeline: ScreeningPipeline, ask_service: AskService):
    yield make_client(settings, pipeline, ask_service)
    app.dependency_overrides.clear()


@pytest.fixture
def jd_text() -> str:
    return (SAMPLES / "job_description.txt").read_text(encoding="utf-8")


@pytest.fixture
def sample_files() -> dict[str, bytes]:
    names = ["ananya_rao.txt", "kiran_patel.txt", "sneha_reddy.docx", "imran_khan.pdf"]
    return {n: (SAMPLES / n).read_bytes() for n in names}
