import pytest
from pydantic import ValidationError

from app.models.schemas import Recommendation
from app.services import scoring
from tests.conftest import make_settings


def test_match_skills_partitions_required_and_preferred():
    m = scoring.match_skills(["Python", "AWS"], ["Python", "SQL"], ["AWS", "Docker"])
    assert m.matched_required == ["Python"]
    assert m.missing_required == ["SQL"]
    assert m.matched_preferred == ["AWS"]
    assert m.missing_preferred == ["Docker"]


def test_skill_coverage_full_and_empty():
    full = scoring.match_skills(["A", "B"], ["A", "B"], [])
    none = scoring.match_skills([], ["A", "B"], [])
    assert scoring.skill_coverage(full) == 1.0
    assert scoring.skill_coverage(none) == 0.0


def test_preferred_skills_count_half():
    # 1 of 1 required matched, 0 of 2 preferred: 1 / (1 + 0.5 * 2) = 0.5
    m = scoring.match_skills(["A"], ["A"], ["B", "C"])
    assert scoring.skill_coverage(m) == pytest.approx(0.5)


def test_skill_coverage_none_without_job_skills():
    assert scoring.skill_coverage(scoring.match_skills(["A"], [], [])) is None


@pytest.mark.parametrize(
    ("years", "required", "expected"),
    [(3, 2, 1.0), (1, 2, 0.5), (None, 2, 0.0), (5, None, None), (5, 0, None)],
)
def test_experience_fit(years, required, expected):
    assert scoring.experience_fit(years, required) == expected


def test_semantic_score_scales_and_clamps():
    assert scoring.semantic_score(0.225, 0.45) == pytest.approx(0.5)
    assert scoring.semantic_score(0.9, 0.45) == 1.0
    assert scoring.semantic_score(-0.1, 0.45) == 0.0


def test_combine_uses_configured_weights():
    s = make_settings(weight_skills=0.5, weight_semantic=0.3, weight_experience=0.2)
    score, breakdown = scoring.combine(coverage=1.0, semantic=0.5, experience=0.0, settings=s)
    assert score == 65.0  # 0.5*1 + 0.3*0.5 + 0.2*0
    assert breakdown.skill_coverage == 100.0
    assert breakdown.experience_fit == 0.0


def test_combine_renormalises_when_components_do_not_apply():
    s = make_settings()
    score, breakdown = scoring.combine(coverage=None, semantic=0.8, experience=None, settings=s)
    assert score == 80.0
    assert breakdown.skill_coverage is None
    assert breakdown.experience_fit is None


def test_combine_stays_within_bounds():
    s = make_settings()
    top, _ = scoring.combine(coverage=1.0, semantic=1.0, experience=1.0, settings=s)
    bottom, _ = scoring.combine(coverage=0.0, semantic=0.0, experience=0.0, settings=s)
    assert (top, bottom) == (100.0, 0.0)


def test_combine_with_all_applicable_weights_zero_uses_similarity():
    s = make_settings(weight_skills=0, weight_semantic=0.1, weight_experience=0)
    score, _ = scoring.combine(coverage=1.0, semantic=0.4, experience=1.0, settings=s)
    assert score == 40.0


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (100, Recommendation.SHORTLIST),
        (70, Recommendation.SHORTLIST),
        (69.9, Recommendation.MAYBE),
        (45, Recommendation.MAYBE),
        (44.9, Recommendation.REJECT),
        (0, Recommendation.REJECT),
    ],
)
def test_recommendation_thresholds(score, expected):
    assert scoring.recommend(score, make_settings()) == expected


def test_settings_reject_inverted_thresholds():
    with pytest.raises(ValidationError):
        make_settings(shortlist_threshold=40, maybe_threshold=60)


def test_settings_reject_all_zero_weights():
    with pytest.raises(ValidationError):
        make_settings(weight_skills=0, weight_semantic=0, weight_experience=0)


def test_blank_api_key_is_treated_as_unset():
    assert make_settings(llm_api_key="  ").llm_enabled is False
    assert make_settings(llm_api_key="abc").llm_enabled is True
