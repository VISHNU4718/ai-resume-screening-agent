"""Screening pipeline.

parse -> extract job requirements -> per candidate: extract profile, check evidence,
score (deterministic), write report -> rank. The extraction, evidence and report steps
use the LLM when one is configured and the rule-based engine otherwise.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from app.core.config import Settings
from app.core.errors import NoValidResumesError, ResumeParseError
from app.models.genai import JobRequirements
from app.models.schemas import (
    AnalysisMode,
    CandidateResult,
    EvidenceItem,
    RejectedFile,
    ScreenResponse,
)
from app.services import scoring
from app.services.engine import AnalysisEngine, RuleBasedEngine
from app.services.evidence import quote_in_text
from app.services.experience import estimate_years
from app.services.parser import ParsedResume, parse_resume
from app.services.rag import Chunk, ScreeningStore, chunk_text
from app.services.reasoning import CandidateAnalysis
from app.services.similarity import SimilarityBackend
from app.services.skills import canonicalize, mentioned_in

logger = logging.getLogger(__name__)

MAX_EVIDENCE_REQUIREMENTS = 15
YEARS_TOLERANCE = 1.0


@dataclass(frozen=True)
class ResumeUpload:
    filename: str
    content: bytes


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out


class ScreeningPipeline:
    def __init__(
        self,
        settings: Settings,
        similarity: SimilarityBackend,
        engine: AnalysisEngine,
        store: ScreeningStore,
    ) -> None:
        self.settings = settings
        self.similarity = similarity
        self.engine = engine
        self.store = store

    def run(self, job_description: str, uploads: list[ResumeUpload]) -> ScreenResponse:
        started = time.perf_counter()
        warnings: list[str] = []

        resumes, rejected = self._parse(uploads)
        if not resumes:
            raise NoValidResumesError(
                "None of the uploaded files could be read.",
                extra={"rejected_files": [r.model_dump() for r in rejected]},
            )

        engine = self.engine
        job_step = engine.extract_job(job_description)
        if job_step.mode == "fallback" and engine.model is not None:
            # The LLM is unreachable or unusable. Skip it for the rest of the run instead of
            # waiting for a timeout on every candidate.
            engine = RuleBasedEngine()
            warnings.append(
                "The LLM could not process the job description, so the rule-based fallback "
                "was used for this whole run."
            )
        job = self._clean_job(job_step.value, job_description, warnings)
        if not job.must_have and not job.nice_to_have:
            warnings.append(
                "No skills were found in the job description; the score uses text "
                "similarity and experience only."
            )

        cosines = self.similarity.similarities(job_description, [r.text for r in resumes])

        def evaluate(pair: tuple[ParsedResume, float]) -> CandidateResult:
            return self._evaluate(engine, pair[0], pair[1], job, warnings)

        workers = max(1, min(len(resumes), self.settings.llm_max_concurrency))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(evaluate, zip(resumes, cosines, strict=True)))

        fell_back = [r.name for r in results if r.analysis_mode == "fallback"]
        mode: AnalysisMode = "fallback" if fell_back else "llm"
        if fell_back and engine.model is not None:
            warnings.append(
                "Rule-based fallback was used for at least one step for: " + ", ".join(fell_back)
            )

        order = sorted(
            range(len(results)), key=lambda i: (-results[i].score, results[i].name.lower())
        )
        ranked = [results[i].model_copy(update={"rank": n}) for n, i in enumerate(order, start=1)]
        screening_id = self.store.put(
            [
                Chunk(results[i].name, results[i].filename, text)
                for i in order
                for text in chunk_text(resumes[i].text)
            ]
        )

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "Screened %d resume(s), rejected %d, mode=%s, in %d ms",
            len(ranked),
            len(rejected),
            mode,
            elapsed_ms,
        )
        return ScreenResponse(
            screening_id=screening_id,
            analysis_mode=mode,
            llm_model=engine.model,
            job=job,
            candidates=ranked,
            rejected_files=rejected,
            warnings=warnings,
            similarity_backend=self.similarity.name,
            llm_enabled=self.settings.llm_enabled,
            processing_ms=elapsed_ms,
        )

    def _parse(self, uploads: list[ResumeUpload]) -> tuple[list[ParsedResume], list[RejectedFile]]:
        limit = self.settings.max_file_size_bytes
        resumes: list[ParsedResume] = []
        rejected: list[RejectedFile] = []
        for upload in uploads:
            try:
                if len(upload.content) > limit:
                    raise ResumeParseError(
                        f"File exceeds the {self.settings.max_file_size_mb:g} MB limit."
                    )
                resumes.append(parse_resume(upload.filename, upload.content))
            except ResumeParseError as exc:
                logger.info("Rejected %s: %s", upload.filename, exc.message)
                rejected.append(RejectedFile(filename=upload.filename, reason=exc.message))
        return resumes, rejected

    @staticmethod
    def _years(
        reported: float | None, resume: ParsedResume, name: str, warnings: list[str]
    ) -> float | None:
        """Prefer the model's figure, but let date arithmetic overrule a large disagreement.

        Small models are unreliable at adding up date ranges. The rule-based estimate reads
        explicit "N years" statements and employment date ranges, so it wins when the two
        differ by more than a year, and the override is reported.
        """
        computed = estimate_years(resume.text)
        if reported is None:
            return computed
        if computed is not None and abs(reported - computed) > YEARS_TOLERANCE:
            warnings.append(
                f"{name}: the model reported {reported:g} years of experience but the resume's "
                f"dates and statements give {computed:g}; used {computed:g}."
            )
            return computed
        return reported

    @staticmethod
    def _clean_job(
        job: JobRequirements, job_description: str, warnings: list[str]
    ) -> JobRequirements:
        """Canonicalise skill names and drop requirements the job text never mentions."""
        must = _unique([canonicalize(s) for s in job.must_have])
        nice = [s for s in _unique([canonicalize(s) for s in job.nice_to_have]) if s not in must]
        dropped = [s for s in [*must, *nice] if not mentioned_in(s, job_description)]
        if dropped:
            warnings.append(
                "Ignored requirement(s) the job description does not mention: " + ", ".join(dropped)
            )
        return JobRequirements(
            must_have=[s for s in must if s not in dropped],
            nice_to_have=[s for s in nice if s not in dropped],
            min_years=job.min_years,
        )

    def _evaluate(
        self,
        engine: AnalysisEngine,
        resume: ParsedResume,
        cosine: float,
        job: JobRequirements,
        warnings: list[str],
    ) -> CandidateResult:
        profile_step = engine.extract_profile(resume)
        profile = profile_step.value
        name = (
            profile.name
            if profile.name and quote_in_text(profile.name, resume.text)
            else resume.name
        )

        # Skills the extractor reported must literally appear in the resume to count.
        reported = _unique([canonicalize(s) for s in profile.skills])
        grounded = [s for s in reported if mentioned_in(s, resume.text)]
        unsupported_skills = [s for s in reported if s not in grounded]

        requirements = [*job.must_have, *job.nice_to_have][:MAX_EVIDENCE_REQUIREMENTS]
        assess_step = engine.assess(requirements, resume.text)
        assessment = assess_step.value
        grounded += [e.requirement for e in assessment.evidence if _quoted_support(e)]

        skills = scoring.match_skills(grounded, job.must_have, job.nice_to_have)
        years = self._years(profile.years_experience, resume, name, warnings)
        score, breakdown = scoring.combine(
            coverage=scoring.skill_coverage(skills),
            semantic=scoring.semantic_score(cosine, self.similarity.saturation),
            experience=scoring.experience_fit(years, job.min_years),
            settings=self.settings,
        )
        recommendation = scoring.recommend(score, self.settings)

        report_step = engine.report(
            CandidateAnalysis(
                name=name,
                resume_text=resume.text,
                job=job,
                profile=profile,
                skills=skills,
                years=years,
                required_years=job.min_years,
                score=score,
                breakdown=breakdown,
                recommendation=recommendation,
                evidence=assessment.evidence,
                unsupported_claims=assessment.unsupported_claims,
            )
        )
        narrative = report_step.value
        all_llm = all(s.mode == "llm" for s in (profile_step, assess_step, report_step))
        return CandidateResult(
            rank=1,  # assigned after sorting
            name=name,
            filename=resume.filename,
            score=score,
            recommendation=recommendation,
            analysis_mode="llm" if all_llm else "fallback",
            profile=profile.model_copy(update={"name": name}),
            matched_skills=skills.matched_required + skills.matched_preferred,
            missing_skills=skills.missing_required,
            missing_preferred_skills=skills.missing_preferred,
            unsupported_skills=unsupported_skills,
            unsupported_claims=assessment.unsupported_claims,
            evidence=assessment.evidence,
            years_experience=years,
            breakdown=breakdown,
            strengths=narrative.strengths,
            concerns=narrative.concerns,
            summary=narrative.summary,
            interview_questions=narrative.interview_questions,
        )


def _quoted_support(item: EvidenceItem) -> bool:
    """A requirement counts as shown if the model's verified quote itself names it."""
    return item.status == "met" and any(mentioned_in(item.requirement, q) for q in item.quotes)
