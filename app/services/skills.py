"""Step 2a: keyword-based skill extraction against a curated taxonomy."""

import json
import re
from functools import lru_cache
from pathlib import Path

_TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "data" / "skills.json"

# Section headings in a job description that introduce optional skills.
_PREFERRED_HEADING = re.compile(
    r"^\W*(nice[\s-]to[\s-]haves?|good[\s-]to[\s-]haves?|preferred|bonus|desirable|"
    r"a plus|plus points?)\b",
    re.IGNORECASE,
)
# Any other heading returns the parser to the "required" state.
_OTHER_HEADING = re.compile(
    r"^\W*(requirements?|responsibilit\w+|qualifications?|what you('ll| will) (do|bring)|"
    r"about( the)?( role| us| you)?|who you are|benefits|what we offer|skills|must[\s-]haves?)\b",
    re.IGNORECASE,
)


@lru_cache
def _patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    taxonomy: dict[str, list[str]] = json.loads(_TAXONOMY_PATH.read_text(encoding="utf-8"))
    compiled = []
    for canonical, aliases in taxonomy.items():
        alts = sorted({a.lower() for a in aliases}, key=len, reverse=True)
        body = "|".join(re.escape(a).replace(r"\ ", r"[\s\-]+") for a in alts)
        # Custom boundaries so that "java" does not match "javascript" or "c++" match "c".
        compiled.append(
            (canonical, re.compile(rf"(?<![A-Za-z0-9+#.]){body}(?![A-Za-z0-9+#])", re.IGNORECASE))
        )
    return tuple(compiled)


def extract_skills(text: str) -> list[str]:
    """Return canonical skill names found in text, in order of first appearance."""
    found: list[tuple[int, str]] = []
    for canonical, pattern in _patterns():
        match = pattern.search(text)
        if match:
            found.append((match.start(), canonical))
    return [name for _, name in sorted(found)]


def split_job_sections(jd_text: str) -> tuple[str, str]:
    """Split a job description into (required_text, preferred_text) by section headings."""
    required: list[str] = []
    preferred: list[str] = []
    in_preferred = False
    for line in jd_text.splitlines():
        stripped = line.strip()
        if _PREFERRED_HEADING.match(stripped):
            in_preferred = True
        elif len(stripped) <= 60 and _OTHER_HEADING.match(stripped):
            in_preferred = False
        (preferred if in_preferred else required).append(line)
    return "\n".join(required), "\n".join(preferred)


def extract_job_skills(jd_text: str) -> tuple[list[str], list[str]]:
    """Return (required, preferred) skills. A skill listed in both counts as required."""
    required_text, preferred_text = split_job_sections(jd_text)
    required = extract_skills(required_text)
    preferred = [s for s in extract_skills(preferred_text) if s not in required]
    return required, preferred
