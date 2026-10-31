"""Step 2b: estimate years of experience from free text."""

import re
from datetime import date

_MONTHS = {
    m: i
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        start=1,
    )
}

_EXPLICIT_YEARS = re.compile(
    r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)\b[^.\n]{0,40}?experience", re.IGNORECASE
)
_MONTH_NAME = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
_DATE_RANGE = re.compile(
    rf"(?:(?P<sm>{_MONTH_NAME})\.?\s+)?(?P<sy>(?:19|20)\d{{2}})\s*(?:-|\u2013|\u2014|to)\s*"
    rf"(?:(?P<em>{_MONTH_NAME})\.?\s+)?(?P<ey>(?:19|20)\d{{2}}|present|current|now|ongoing)",
    re.IGNORECASE,
)
_EDUCATION_LINE = re.compile(
    r"\b(b\.?\s?tech|m\.?\s?tech|bachelor|master|degree|university|college|school|"
    r"intermediate|cgpa|gpa|diploma|ssc|hsc)\b",
    re.IGNORECASE,
)
_JD_RANGE = re.compile(
    r"(\d{1,2})\s*(?:-|\u2013|to)\s*(\d{1,2})\s*\+?\s*(?:years?|yrs?)", re.IGNORECASE
)
_JD_SINGLE = re.compile(r"(\d{1,2})\s*\+?\s*(?:years?|yrs?)\b", re.IGNORECASE)


def _month(token: str | None) -> int:
    if token:
        return _MONTHS.get(token[:3].lower(), 1)
    return 1


def _to_fraction(year: int, month: int) -> float:
    return year + (month - 1) / 12


def estimate_years(text: str, today: date | None = None) -> float | None:
    """Estimate total years of experience.

    An explicit statement such as "3 years of experience" wins. Otherwise date ranges
    (for example "Jun 2023 - Present") are merged so overlapping roles are not double
    counted. Lines that look like education entries are ignored. Returns None when
    neither signal is present.
    """
    today = today or date.today()

    explicit = [float(m.group(1)) for m in _EXPLICIT_YEARS.finditer(text)]
    explicit = [v for v in explicit if v <= 50]
    if explicit:
        return max(explicit)

    now = _to_fraction(today.year, today.month)
    spans: list[tuple[float, float]] = []
    for line in text.splitlines():
        if _EDUCATION_LINE.search(line):
            continue
        for m in _DATE_RANGE.finditer(line):
            start = _to_fraction(int(m.group("sy")), _month(m.group("sm")))
            end_token = m.group("ey").lower()
            if end_token.isdigit():
                end = _to_fraction(int(end_token), _month(m.group("em")))
            else:
                end = now
            if start <= end <= now + 1:
                spans.append((start, min(end, now)))
    if not spans:
        return None

    spans.sort()
    total = 0.0
    cur_start, cur_end = spans[0]
    for start, end in spans[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    total += cur_end - cur_start
    return round(total, 1)


def extract_required_years(jd_text: str) -> float | None:
    """Find the minimum years of experience requested in a job description."""
    match = _JD_RANGE.search(jd_text)
    if match:
        return float(match.group(1))
    match = _JD_SINGLE.search(jd_text)
    if match:
        return float(match.group(1))
    return None
