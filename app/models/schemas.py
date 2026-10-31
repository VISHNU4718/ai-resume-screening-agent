"""Pydantic models for API responses and internal pipeline data."""

from enum import StrEnum

from pydantic import BaseModel, Field


class Recommendation(StrEnum):
    SHORTLIST = "Shortlist"
    MAYBE = "Maybe"
    REJECT = "Reject"


class ScoreBreakdown(BaseModel):
    """Component scores on a 0-100 scale. A component is null when it did not apply."""

    skill_coverage: float | None = Field(None, ge=0, le=100)
    semantic_similarity: float = Field(ge=0, le=100)
    experience_fit: float | None = Field(None, ge=0, le=100)


class JobProfile(BaseModel):
    required_skills: list[str]
    preferred_skills: list[str]
    required_years: float | None = None


class CandidateResult(BaseModel):
    rank: int = Field(ge=1)
    name: str
    filename: str
    score: float = Field(ge=0, le=100)
    recommendation: Recommendation
    matched_skills: list[str]
    missing_skills: list[str] = Field(description="Required skills not found in the resume.")
    missing_preferred_skills: list[str]
    years_experience: float | None = Field(
        None, description="Estimated from the resume text; null when it could not be determined."
    )
    breakdown: ScoreBreakdown
    strengths: list[str]
    concerns: list[str]
    summary: str
    reasoner: str = Field(
        description="Which reasoning step produced the text: 'llm' or 'rule_based'."
    )


class RejectedFile(BaseModel):
    filename: str
    reason: str


class ScreenResponse(BaseModel):
    job: JobProfile
    candidates: list[CandidateResult]
    rejected_files: list[RejectedFile]
    warnings: list[str]
    similarity_backend: str
    llm_enabled: bool
    processing_ms: int


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_enabled: bool
