import pytest

from app.services.skills import extract_job_skills, extract_skills, split_job_sections


def test_extracts_in_order_of_appearance():
    assert extract_skills("We use Docker, then Python.") == ["Docker", "Python"]


def test_aliases_map_to_canonical_name():
    assert extract_skills("sklearn and k8s and Postgres") == [
        "scikit-learn",
        "Kubernetes",
        "PostgreSQL",
    ]


def test_java_does_not_match_javascript():
    assert extract_skills("Experienced with JavaScript") == ["JavaScript"]
    assert extract_skills("Java and JavaScript") == ["Java", "JavaScript"]


def test_sql_does_not_match_inside_other_words():
    assert extract_skills("PostgreSQL and MySQL") == ["PostgreSQL", "MySQL"]


@pytest.mark.parametrize(
    ("text", "skill"),
    [
        ("C++ developer", "C++"),
        ("built in C#.", "C#"),
        ("Node.js APIs", "Node.js"),
        ("CI/CD pipelines", "CI/CD"),
    ],
)
def test_symbols_in_skill_names(text, skill):
    assert skill in extract_skills(text)


def test_multiword_skill_tolerates_hyphen_and_spacing():
    assert extract_skills("retrieval-augmented generation") == ["RAG"]
    assert extract_skills("Machine   Learning") == ["Machine Learning"]


def test_case_insensitive():
    assert extract_skills("PYTHON and pytorch") == ["Python", "PyTorch"]


def test_no_skills():
    assert extract_skills("Enthusiastic team player") == []


def test_sections_split_on_nice_to_have():
    required, preferred = split_job_sections("Requirements\n- Python\nNice to have\n- Docker")
    assert "Python" in required and "Docker" not in required
    assert "Docker" in preferred


def test_requirements_heading_after_preferred_returns_to_required():
    jd = "Preferred:\n- AWS\nResponsibilities\n- Build Python services"
    required, preferred = split_job_sections(jd)
    assert "AWS" in preferred
    assert "Python" in required


def test_job_skills_required_wins_over_preferred():
    required, preferred = extract_job_skills("Requirements\n- Python, Docker\nBonus\n- Docker, AWS")
    assert required == ["Python", "Docker"]
    assert preferred == ["AWS"]


def test_job_without_headings_is_all_required():
    required, preferred = extract_job_skills("Looking for Python and SQL")
    assert required == ["Python", "SQL"]
    assert preferred == []
