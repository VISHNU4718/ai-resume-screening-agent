import pytest
from pydantic import ValidationError

from app.models.genai import (
    CandidateProfile,
    CandidateReport,
    FitAssessment,
    JobRequirements,
    QAAnswer,
    RequirementCheck,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(3, 3.0), (2.5, 2.5), ("3 years", 3.0), ("3+", 3.0), ("about 1.5 yrs", 1.5)]
    + [(None, None), ("n/a", None), (-2, None), (400, None), (True, None)],
)
def test_years_are_parsed_leniently(raw, expected):
    assert CandidateProfile(years_experience=raw).years_experience == expected


def test_profile_cleans_lists_and_drops_junk():
    p = CandidateProfile(
        name="  Jane   Doe ",
        skills=["Python", "python", " SQL ", None, {"x": 1}, ""],
        projects="One project",
        education=None,
    )
    assert p.name == "Jane Doe"
    assert p.skills == ["Python", "SQL"]
    assert p.projects == ["One project"]
    assert p.education == []


def test_profile_limits_list_sizes():
    p = CandidateProfile(
        projects=[f"p{i}" for i in range(50)], skills=[f"s{i}" for i in range(500)]
    )
    assert len(p.projects) == 8 and len(p.skills) == 60


def test_profile_rejects_wrong_container_type():
    with pytest.raises(ValidationError):
        CandidateProfile(skills=42)


def test_blank_name_becomes_none():
    assert CandidateProfile(name="   ").name is None


def test_job_requirements_nice_to_have_excludes_must_have():
    job = JobRequirements(must_have=["Python", "SQL"], nice_to_have=["sql", "AWS"])
    assert job.nice_to_have == ["AWS"]


def test_requirement_status_is_normalised_and_validated():
    assert RequirementCheck(requirement="x", status="Not Found").status == "not_found"
    with pytest.raises(ValidationError):
        RequirementCheck(requirement="x", status="great")


def test_assessment_requires_requirements_key():
    with pytest.raises(ValidationError):
        FitAssessment.model_validate({"unsupported_claims": []})


def report(**overrides) -> dict:
    base = {
        "strengths": ["a"],
        "concerns": ["b"],
        "summary": "Fine.",
        "interview_questions": ["q1", "q2", "q3"],
    }
    return {**base, **overrides}


def test_report_truncates_lists_and_summary():
    r = CandidateReport.model_validate(
        report(
            strengths=[str(i) for i in range(9)],
            interview_questions=[f"q{i}" for i in range(9)],
            summary="x" * 900,
        )
    )
    assert len(r.strengths) == 4
    assert len(r.interview_questions) == 5
    assert len(r.summary) == 500


@pytest.mark.parametrize(
    "bad",
    [{"interview_questions": ["only one", "two"]}, {"summary": "  "}, {"interview_questions": []}],
)
def test_report_rejects_unusable_output(bad):
    with pytest.raises(ValidationError):
        CandidateReport.model_validate(report(**bad))


def test_qa_answer_normalises_citations():
    a = QAAnswer.model_validate({"answer": " Yes. ", "citations": ["[1]", 2, "x", "3"]})
    assert a.answer == "Yes." and a.citations == [1, 2, 3]
    assert QAAnswer(answer="ok", citations=None).citations == []
    with pytest.raises(ValidationError):
        QAAnswer(answer="   ")
