"""Application exceptions and their HTTP mapping."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class AppError(Exception):
    """Base class for errors that should be reported to the API client."""

    status_code = 400
    code = "bad_request"

    def __init__(self, message: str, *, extra: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.extra = extra or {}


class InvalidRequestError(AppError):
    status_code = 422
    code = "invalid_request"


class ResumeParseError(AppError):
    """A single resume could not be read. Handled per file by the pipeline."""

    status_code = 422
    code = "resume_unreadable"


class NoValidResumesError(AppError):
    status_code = 422
    code = "no_valid_resumes"


class ScreeningNotFoundError(AppError):
    status_code = 404
    code = "screening_not_found"


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        body = {"detail": exc.message, "code": exc.code, **exc.extra}
        return JSONResponse(status_code=exc.status_code, content=body)
