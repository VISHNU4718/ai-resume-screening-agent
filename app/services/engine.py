"""The analysis steps the pipeline needs, behind one interface.

Two implementations exist: LLMEngine (primary) and RuleBasedEngine (explicit offline
fallback). Every step reports which one produced its value so the API can label it.
"""

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from app.models.genai import CandidateProfile, JobRequirements
from app.models.schemas import AnalysisMode, EvidenceItem
from app.services.evidence import rule_based_evidence
from app.services.experience import estimate_years, extract_required_years
from app.services.parser import ParsedResume
from app.services.reasoning import CandidateAnalysis, Narrative, RuleBasedReasoner
from app.services.skills import extract_job_skills, extract_skills

T = TypeVar("T")


@dataclass(frozen=True)
class Step(Generic[T]):
    value: T
    mode: AnalysisMode


@dataclass(frozen=True)
class Assessment:
    evidence: list[EvidenceItem]
    unsupported_claims: list[str]


class AnalysisEngine(Protocol):
    model: str | None

    def extract_job(self, job_description: str) -> Step[JobRequirements]: ...

    def extract_profile(self, resume: ParsedResume) -> Step[CandidateProfile]: ...

    def assess(self, requirements: list[str], resume_text: str) -> Step[Assessment]: ...

    def report(self, analysis: CandidateAnalysis) -> Step[Narrative]: ...


class RuleBasedEngine:
    """Keyword taxonomy and templates. Deterministic, offline, much less flexible."""

    model = None

    def __init__(self) -> None:
        self._reasoner = RuleBasedReasoner()

    def extract_job(self, job_description: str) -> Step[JobRequirements]:
        required, preferred = extract_job_skills(job_description)
        reqs = JobRequirements(
            must_have=required,
            nice_to_have=preferred,
            min_years=extract_required_years(job_description),
        )
        return Step(reqs, "fallback")

    def extract_profile(self, resume: ParsedResume) -> Step[CandidateProfile]:
        profile = CandidateProfile(
            name=resume.name,
            skills=extract_skills(resume.text),
            years_experience=estimate_years(resume.text),
        )
        return Step(profile, "fallback")

    def assess(self, requirements: list[str], resume_text: str) -> Step[Assessment]:
        return Step(Assessment(rule_based_evidence(requirements, resume_text), []), "fallback")

    def report(self, analysis: CandidateAnalysis) -> Step[Narrative]:
        return Step(self._reasoner.explain(analysis), "fallback")
