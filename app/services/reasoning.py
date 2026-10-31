"""Rule-based narrative used on the fallback path (and as the reference for the LLM report).

The score and recommendation are computed deterministically before this step. The writer
only explains them; it never changes them.
"""

from dataclasses import dataclass, field

from app.models.genai import CandidateProfile, JobRequirements
from app.models.schemas import EvidenceItem, Recommendation, ScoreBreakdown
from app.services.scoring import SkillMatch

MAX_ITEMS = 4
MAX_QUESTIONS = 5
MIN_QUESTIONS = 3


@dataclass(frozen=True)
class CandidateAnalysis:
    """Everything known about a candidate once scoring is done."""

    name: str
    resume_text: str
    job: JobRequirements
    profile: CandidateProfile
    skills: SkillMatch
    years: float | None
    required_years: float | None
    score: float
    breakdown: ScoreBreakdown
    recommendation: Recommendation
    evidence: list[EvidenceItem] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Narrative:
    strengths: list[str]
    concerns: list[str]
    summary: str
    interview_questions: list[str]


def _join(items: list[str], limit: int = 6) -> str:
    shown = items[:limit]
    extra = len(items) - len(shown)
    text = ", ".join(shown)
    return f"{text} and {extra} more" if extra > 0 else text


class RuleBasedReasoner:
    """Deterministic explanation built from the extracted facts."""

    def explain(self, a: CandidateAnalysis) -> Narrative:
        sk = a.skills
        required_total = len(sk.matched_required) + len(sk.missing_required)
        strengths: list[str] = []
        concerns: list[str] = []

        if required_total:
            if sk.matched_required:
                strengths.append(
                    f"Covers {len(sk.matched_required)} of {required_total} required skills: "
                    f"{_join(sk.matched_required)}."
                )
            if sk.missing_required:
                concerns.append(f"Missing required skills: {_join(sk.missing_required)}.")
        if sk.matched_preferred:
            strengths.append(f"Also lists preferred skills: {_join(sk.matched_preferred)}.")

        if a.required_years is not None:
            asked = f"{a.required_years:g}+"
            if a.years is None:
                concerns.append(
                    f"Years of experience could not be determined (role asks for {asked})."
                )
            elif a.years >= a.required_years:
                strengths.append(
                    f"About {a.years:g} years of experience against {asked} requested."
                )
            else:
                concerns.append(f"About {a.years:g} years of experience against {asked} requested.")
        elif a.years is not None and a.years >= 2:
            strengths.append(f"About {a.years:g} years of stated experience.")

        sim = a.breakdown.semantic_similarity
        if sim >= 60:
            strengths.append("Resume wording is close to the job description overall.")
        elif sim < 30:
            concerns.append("Resume content overlaps little with the job description.")

        if not strengths:
            strengths.append("No clear match to the stated requirements was found.")
        if not concerns:
            concerns.append("No significant gaps found against the stated requirements.")

        return Narrative(
            strengths[:MAX_ITEMS],
            concerns[:MAX_ITEMS],
            self._summary(a, required_total),
            self.questions(a),
        )

    @staticmethod
    def _summary(a: CandidateAnalysis, required_total: int) -> str:
        coverage = (
            f"{len(a.skills.matched_required)}/{required_total} required skills"
            if required_total
            else "no recognised required skills in the job description"
        )
        verdict = {
            Recommendation.SHORTLIST: "Strong match; recommend moving to interview.",
            Recommendation.MAYBE: "Partial match; worth a closer look if the pipeline is thin.",
            Recommendation.REJECT: "Weak match for this role.",
        }[a.recommendation]
        return f"Score {a.score:g}/100 with {coverage}. {verdict}"

    @staticmethod
    def questions(a: CandidateAnalysis) -> list[str]:
        """Template questions driven by the gaps and matches. Less tailored than the LLM's."""
        out: list[str] = []
        for skill in a.skills.missing_required[:2]:
            out.append(
                f"The role needs {skill}, which does not show up on your resume. "
                "Have you used it or something close to it?"
            )
        for skill in a.skills.matched_required[:2]:
            project = next(
                (p for p in a.profile.projects if skill.lower() in p.lower()), "your recent work"
            )
            out.append(f"Walk me through how you used {skill} in {project}.")
        if a.required_years is not None and (a.years is None or a.years < a.required_years):
            out.append(
                f"The role asks for {a.required_years:g}+ years of experience. "
                "What have you owned end to end so far?"
            )
        generic = [
            "Which piece of work are you proudest of, and what was your part in it?",
            "Describe a bug that took you a long time to find and how you found it.",
            "How would you explain the trade-offs of your main project to a non-technical manager?",
        ]
        out.extend(generic)
        return out[:MAX_QUESTIONS] if len(out) >= MIN_QUESTIONS else generic
