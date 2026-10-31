"""Step 4: combine skill coverage, semantic similarity and experience into one score."""

from dataclasses import dataclass

from app.core.config import Settings
from app.models.schemas import Recommendation, ScoreBreakdown

PREFERRED_SKILL_WEIGHT = 0.5


@dataclass(frozen=True)
class SkillMatch:
    matched_required: list[str]
    missing_required: list[str]
    matched_preferred: list[str]
    missing_preferred: list[str]


def match_skills(resume_skills: list[str], required: list[str], preferred: list[str]) -> SkillMatch:
    have = set(resume_skills)
    return SkillMatch(
        matched_required=[s for s in required if s in have],
        missing_required=[s for s in required if s not in have],
        matched_preferred=[s for s in preferred if s in have],
        missing_preferred=[s for s in preferred if s not in have],
    )


def skill_coverage(match: SkillMatch) -> float | None:
    """Weighted share of job skills found. Preferred skills count half as much.

    Returns None when the job description names no recognised skills.
    """
    total = len(match.matched_required) + len(match.missing_required)
    pref_total = len(match.matched_preferred) + len(match.missing_preferred)
    denominator = total + PREFERRED_SKILL_WEIGHT * pref_total
    if denominator == 0:
        return None
    numerator = len(match.matched_required) + PREFERRED_SKILL_WEIGHT * len(match.matched_preferred)
    return numerator / denominator


def experience_fit(years: float | None, required_years: float | None) -> float | None:
    """1.0 when the candidate meets the requirement, scaled down linearly below it.

    None when the job states no requirement. Unknown candidate experience scores 0.
    """
    if required_years is None or required_years <= 0:
        return None
    if years is None:
        return 0.0
    return min(1.0, years / required_years)


def semantic_score(cosine: float, saturation: float) -> float:
    return min(1.0, max(0.0, cosine) / saturation)


def combine(
    *,
    coverage: float | None,
    semantic: float,
    experience: float | None,
    settings: Settings,
) -> tuple[float, ScoreBreakdown]:
    """Weighted average over the components that apply, scaled to 0-100.

    Weights of components that do not apply (no skills or no years in the job
    description) are dropped and the remaining weights are renormalised.
    """
    parts: list[tuple[float, float]] = [(settings.weight_semantic, semantic)]
    if coverage is not None:
        parts.append((settings.weight_skills, coverage))
    if experience is not None:
        parts.append((settings.weight_experience, experience))

    total_weight = sum(w for w, _ in parts)
    # If every applicable weight is configured as zero, fall back to similarity alone.
    value = sum(w * v for w, v in parts) / total_weight if total_weight > 0 else semantic

    breakdown = ScoreBreakdown(
        skill_coverage=None if coverage is None else round(coverage * 100, 1),
        semantic_similarity=round(semantic * 100, 1),
        experience_fit=None if experience is None else round(experience * 100, 1),
    )
    return round(value * 100, 1), breakdown


def recommend(score: float, settings: Settings) -> Recommendation:
    if score >= settings.shortlist_threshold:
        return Recommendation.SHORTLIST
    if score >= settings.maybe_threshold:
        return Recommendation.MAYBE
    return Recommendation.REJECT
