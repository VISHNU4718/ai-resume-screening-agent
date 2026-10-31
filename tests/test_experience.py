from datetime import date

import pytest

from app.services.experience import estimate_years, extract_required_years

TODAY = date(2026, 6, 1)


def test_explicit_statement():
    assert estimate_years("Engineer with 3 years of experience in Python", TODAY) == 3.0


def test_explicit_statement_with_plus_and_decimal():
    assert estimate_years("5+ years professional experience", TODAY) == 5.0
    assert estimate_years("2.5 yrs of work experience", TODAY) == 2.5


def test_implausible_explicit_value_is_ignored():
    assert estimate_years("99 years of experience", TODAY) is None


def test_date_ranges_with_months():
    text = "Developer, Acme (Jan 2022 - Jan 2024)"
    assert estimate_years(text, TODAY) == 2.0


def test_present_uses_today():
    assert estimate_years("Engineer, Acme (Jun 2025 - Present)", TODAY) == 1.0


def test_year_only_range():
    assert estimate_years("Engineer 2020 - 2023", TODAY) == 3.0


def test_overlapping_ranges_are_merged():
    text = "Role A (Jan 2020 - Jan 2023)\nRole B (Jan 2021 - Jan 2022)"
    assert estimate_years(text, TODAY) == 3.0


def test_disjoint_ranges_are_summed():
    text = "Role A (Jan 2018 - Jan 2019)\nRole B (Jan 2021 - Jan 2023)"
    assert estimate_years(text, TODAY) == 3.0


def test_education_lines_are_ignored():
    text = "B.Tech in Computer Science, 2020 - 2024\nIntern, Acme (Jun 2025 - Aug 2025)"
    assert estimate_years(text, TODAY) == 0.2


def test_no_signal_returns_none():
    assert estimate_years("Fresh graduate with strong Python skills", TODAY) is None


def test_future_start_is_ignored():
    assert estimate_years("Role (2030 - 2031)", TODAY) is None


def test_reversed_range_is_ignored():
    assert estimate_years("Role (2024 - 2020)", TODAY) is None


@pytest.mark.parametrize(
    ("jd", "expected"),
    [
        ("3+ years of experience", 3.0),
        ("2-4 years of experience with Python", 2.0),
        ("at least 5 years", 5.0),
        ("No requirement stated", None),
    ],
)
def test_required_years(jd, expected):
    assert extract_required_years(jd) == expected
