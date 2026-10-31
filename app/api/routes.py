from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app import __version__
from app.api.deps import get_pipeline
from app.core.config import Settings, get_settings
from app.core.errors import InvalidRequestError
from app.models.schemas import HealthResponse, ScreenResponse
from app.services.pipeline import ResumeUpload, ScreeningPipeline

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    return HealthResponse(status="ok", version=__version__, llm_enabled=settings.llm_enabled)


@router.post("/screen", response_model=ScreenResponse, tags=["screening"])
def screen(
    pipeline: Annotated[ScreeningPipeline, Depends(get_pipeline)],
    resumes: Annotated[list[UploadFile], File(description="One or more PDF, DOCX or TXT resumes.")],
    job_description: Annotated[str, Form(description="Job description text.")],
) -> ScreenResponse:
    """Screen resumes against a job description and return a ranked list of candidates."""
    settings = pipeline.settings
    jd = job_description.strip()
    if len(jd) < settings.min_jd_chars:
        raise InvalidRequestError(
            f"Job description is too short (minimum {settings.min_jd_chars} characters)."
        )
    if len(jd) > settings.max_jd_chars:
        raise InvalidRequestError(
            f"Job description is too long (maximum {settings.max_jd_chars} characters)."
        )
    if len(resumes) > settings.max_files:
        raise InvalidRequestError(f"Too many files (maximum {settings.max_files} per request).")

    limit = settings.max_file_size_bytes
    uploads = []
    for f in resumes:
        # Read one byte past the limit so oversize files are detected without loading them fully.
        uploads.append(
            ResumeUpload(filename=f.filename or "unnamed", content=f.file.read(limit + 1))
        )
    return pipeline.run(jd, uploads)
