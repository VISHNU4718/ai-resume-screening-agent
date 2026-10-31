"""Screening pipeline: parse -> extract -> match -> score -> reason -> rank."""

import logging
import time
from dataclasses import dataclass

from app.core.config import Settings
from app.core.errors import NoValidResumesError, ResumeParseError
from app.models.schemas import (
    CandidateResult,
    JobProfile,
    RejectedFile,
    ScreenResponse,
)
from app.services import scoring
from app.services.experience import estimate_years, extract_required_years
from app.services.parser import ParsedResume, parse_resume
from app.services.reasoning import CandidateAnalysis, Reasoner
from app.services.similarity import SimilarityBackend
from app.services.skills import extract_job_skills, extract_skills

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResumeUpload:
    filename: str
    content: bytes


class ScreeningPipeline:
    def __init__(
        self, settings: Settings, similarity: SimilarityBackend, reasoner: Reasoner
    ) -> None:
        self.settings = settings
        self.similarity = similarity
        self.reasoner = reasoner

    def run(self, job_description: str, uploads: list[ResumeUpload]) -> ScreenResponse:
        started = time.perf_counter()
        warnings: list[str] = []

        # 1. Parse
        resumes, rejected = self._parse(uploads)
        if not resumes:
            raise NoValidResumesError(
                "None of the uploaded files could be read.",
                extra={"rejected_files": [r.model_dump() for r in rejected]},
            )

        # 2. Extract skills and experience
        required, preferred = extract_job_skills(job_description)
        required_years = extract_required_years(job_description)
        if not required and not preferred:
            warnings.append(
                "No known skills were found in the job description; the score uses text "
                "similarity and experience only."
            )
        job = JobProfile(
            required_skills=required, preferred_skills=preferred, required_years=required_years
        )

        # 3. Semantic similarity (one batch so the vectoriser sees every document)
        cosines = self.similarity.similarities(job_description, [r.text for r in resumes])

        # 4-5. Score and reason per candidate
        results: list[CandidateResult] = []
        for resume, cosine in zip(resumes, cosines, strict=True):
            results.append(self._evaluate(resume, cosine, job))

        # 6. Rank
        results.sort(key=lambda c: (-c.score, c.name.lower()))
        ranked = [c.model_copy(update={"rank": i}) for i, c in enumerate(results, start=1)]

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "Screened %d resume(s), rejected %d, in %d ms", len(ranked), len(rejected), elapsed_ms
        )
        return ScreenResponse(
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

    def _evaluate(self, resume: ParsedResume, cosine: float, job: JobProfile) -> CandidateResult:
        s = self.settings
        skills = scoring.match_skills(
            extract_skills(resume.text), job.required_skills, job.preferred_skills
        )
        years = estimate_years(resume.text)
        score, breakdown = scoring.combine(
            coverage=scoring.skill_coverage(skills),
            semantic=scoring.semantic_score(cosine, self.similarity.saturation),
            experience=scoring.experience_fit(years, job.required_years),
            settings=s,
        )
        recommendation = scoring.recommend(score, s)
        narrative = self.reasoner.explain(
            CandidateAnalysis(
                name=resume.name,
                resume_text=resume.text,
                skills=skills,
                years=years,
                required_years=job.required_years,
                score=score,
                breakdown=breakdown,
                recommendation=recommendation,
            )
        )
        return CandidateResult(
            rank=1,  # assigned after sorting
            name=resume.name,
            filename=resume.filename,
            score=score,
            recommendation=recommendation,
            matched_skills=skills.matched_required + skills.matched_preferred,
            missing_skills=skills.missing_required,
            missing_preferred_skills=skills.missing_preferred,
            years_experience=years,
            breakdown=breakdown,
            strengths=narrative.strengths,
            concerns=narrative.concerns,
            summary=narrative.summary,
            reasoner=narrative.source,
        )
