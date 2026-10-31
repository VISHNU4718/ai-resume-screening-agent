"""Pydantic models for API responses and internal pipeline data."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from app.models.genai import CandidateProfile, JobRequirements

AnalysisMode = Literal["llm", "fallback"]

__all__ = [
    "AnalysisMode",
    "AskResponse",
    "CandidateProfile",
    "CandidateResult",
    "Citation",
    "EvidenceItem",
    "HealthResponse",
    "JobRequirements",
    "Recommendation",
    "RejectedFile",
    "ScoreBreakdown",
    "ScreenResponse",
]


class Recommendation(StrEnum):
    SHORTLIST = "Shortlist"
    MAYBE = "Maybe"
    REJECT = "Reject"


class ScoreBreakdown(BaseModel):
    """Component scores on a 0-100 scale. A component is null when it did not apply."""

    skill_coverage: float | None = Field(None, ge=0, le=100)
    semantic_similarity: float = Field(ge=0, le=100)
    experience_fit: float | None = Field(None, ge=0, le=100)


class EvidenceItem(BaseModel):
    """How one job requirement is supported by the resume."""

    requirement: str
    status: Literal["met", "partial", "not_found", "unverified"] = Field(
        description="'unverified' means the model claimed support but no quote was found "
        "in the resume text."
    )
    quotes: list[str] = Field(description="Passages confirmed to appear verbatim in the resume.")


class CandidateResult(BaseModel):
    rank: int = Field(ge=1)
    name: str
    filename: str
    score: float = Field(ge=0, le=100)
    recommendation: Recommendation
    analysis_mode: AnalysisMode = Field(
        description="'llm' when the LLM produced the profile, evidence and report for this "
        "candidate; 'fallback' when any step used the rule-based path."
    )
    profile: CandidateProfile
    matched_skills: list[str]
    missing_skills: list[str] = Field(description="Required skills not found in the resume.")
    missing_preferred_skills: list[str]
    unsupported_skills: list[str] = Field(
        description="Skills the extractor reported that do not appear in the resume text. "
        "They are excluded from the score."
    )
    unsupported_claims: list[str] = Field(
        description="Claims the LLM flagged as not backed by any detail in the resume."
    )
    evidence: list[EvidenceItem]
    years_experience: float | None = Field(
        None, description="From the profile; null when it could not be determined."
    )
    breakdown: ScoreBreakdown
    strengths: list[str]
    concerns: list[str]
    summary: str
    interview_questions: list[str]


class RejectedFile(BaseModel):
    filename: str
    reason: str


class ScreenResponse(BaseModel):
    screening_id: str = Field(description="Pass this to /ask to question the screened resumes.")
    analysis_mode: AnalysisMode = Field(
        description="'llm' only if every candidate was analysed by the LLM; otherwise 'fallback' "
        "(see each candidate's analysis_mode and the warnings)."
    )
    llm_model: str | None = Field(description="Model used when an LLM is configured.")
    job: JobRequirements
    candidates: list[CandidateResult]
    rejected_files: list[RejectedFile]
    warnings: list[str]
    similarity_backend: str
    llm_enabled: bool
    processing_ms: int


class Citation(BaseModel):
    ref: int = Field(description="Excerpt number referenced in the answer.")
    candidate: str
    filename: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    analysis_mode: AnalysisMode
    grounded: bool = Field(description="True when the answer cites at least one retrieved excerpt.")


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_enabled: bool
    llm_model: str | None = None
