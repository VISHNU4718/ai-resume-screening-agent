"""LLM-backed analysis steps. Any LLM failure drops that one step to the rule-based engine."""

import json
import logging
from collections.abc import Callable
from typing import TypeVar

from app.llm.errors import LLMError
from app.llm.structured import StructuredLLM
from app.models.genai import (
    CandidateProfile,
    CandidateReport,
    FitAssessment,
    JobRequirements,
)
from app.services.engine import Assessment, RuleBasedEngine, Step
from app.services.evidence import verify_assessment
from app.services.parser import ParsedResume
from app.services.reasoning import CandidateAnalysis, Narrative

logger = logging.getLogger(__name__)

# Keeps prompts inside the context window of small local models (Ollama defaults to 4k tokens).
MAX_PROMPT_CHARS = 6000

T = TypeVar("T")


class LLMEngine:
    def __init__(self, llm: StructuredLLM, fallback: RuleBasedEngine | None = None) -> None:
        self._llm = llm
        self._fallback = fallback or RuleBasedEngine()

    @property
    def model(self) -> str:
        return self._llm.model

    def extract_job(self, job_description: str) -> Step[JobRequirements]:
        def call() -> JobRequirements:
            reqs = self._llm.run(
                "extract_job", JobRequirements, job_description=job_description[:MAX_PROMPT_CHARS]
            )
            if not reqs.must_have and not reqs.nice_to_have:
                raise LLMError("model found no requirements in the job description")
            return reqs

        return self._guard("extract_job", call, lambda: self._fallback.extract_job(job_description))

    def extract_profile(self, resume: ParsedResume) -> Step[CandidateProfile]:
        return self._guard(
            f"extract_resume[{resume.filename}]",
            lambda: self._llm.run(
                "extract_resume", CandidateProfile, resume=resume.text[:MAX_PROMPT_CHARS]
            ),
            lambda: self._fallback.extract_profile(resume),
        )

    def assess(self, requirements: list[str], resume_text: str) -> Step[Assessment]:
        if not requirements:
            return Step(Assessment([], []), "llm")

        def call() -> Assessment:
            result = self._llm.run(
                "assess_fit",
                FitAssessment,
                requirements="\n".join(f"- {r}" for r in requirements),
                resume=resume_text[:MAX_PROMPT_CHARS],
            )
            evidence = verify_assessment(requirements, result, resume_text)
            return Assessment(evidence, result.unsupported_claims)

        return self._guard(
            "assess_fit", call, lambda: self._fallback.assess(requirements, resume_text)
        )

    def report(self, a: CandidateAnalysis) -> Step[Narrative]:
        def call() -> Narrative:
            r = self._llm.run(
                "write_report",
                CandidateReport,
                requirements=json.dumps(
                    {"must_have": a.job.must_have, "nice_to_have": a.job.nice_to_have}
                ),
                facts=json.dumps(_facts(a), indent=2),
                resume=a.resume_text[:MAX_PROMPT_CHARS],
            )
            return Narrative(r.strengths, r.concerns, r.summary, r.interview_questions)

        return self._guard("write_report", call, lambda: self._fallback.report(a))

    def _guard(self, step: str, call: Callable[[], T], fallback: Callable[[], Step[T]]) -> Step[T]:
        try:
            return Step(call(), "llm")
        except LLMError as exc:
            logger.warning("LLM step %s failed (%s); using rule-based fallback", step, exc)
            return fallback()


def _facts(a: CandidateAnalysis) -> dict:
    sk = a.skills
    return {
        "recommendation": a.recommendation.value,
        "score_out_of_100": a.score,
        "matched_required": sk.matched_required,
        "missing_required": sk.missing_required,
        "matched_nice_to_have": sk.matched_preferred,
        "missing_nice_to_have": sk.missing_preferred,
        "years_experience": a.years,
        "years_required": a.required_years,
        "unsupported_claims": a.unsupported_claims,
        "projects": a.profile.projects,
        "roles": a.profile.roles,
    }
