"""Evidence checks: keep an LLM quote only if it really appears in the resume."""

import re

from app.models.genai import FitAssessment
from app.models.schemas import EvidenceItem
from app.services.skills import mentioned_in

MIN_QUOTE_CHARS = 6
MAX_EXCERPT_CHARS = 220


def _squash(text: str) -> str:
    """Lowercase and collapse everything but letters and digits, so line breaks, bullets and
    punctuation differences between the resume and the model's copy do not matter."""
    return " ".join(re.findall(r"[a-z0-9+#]+", text.lower()))


def quote_in_text(quote: str, resume_text: str) -> bool:
    needle = _squash(quote)
    return len(needle) >= MIN_QUOTE_CHARS and needle in _squash(resume_text)


def _first_line_naming(requirement: str, text: str) -> str | None:
    line = next((ln.strip() for ln in text.splitlines() if mentioned_in(requirement, ln)), None)
    return line[:MAX_EXCERPT_CHARS] if line else None


def verify_assessment(
    requirements: list[str], assessment: FitAssessment, resume_text: str
) -> list[EvidenceItem]:
    """Turn the model's verdicts into evidence the resume itself backs up.

    The model proposes a status and quotes; these checks decide what is shown:
    - a quote is kept only if it really appears in the resume;
    - "met" needs a verified quote that names the requirement;
    - if the model's quotes fail but the resume does name the requirement, the first line
      that does is used instead (the model said "met": it stays met, otherwise "partial");
    - a claim with no verifiable support at all is shown as "unverified".
    """
    by_name = {_squash(c.requirement): c for c in assessment.requirements}
    items: list[EvidenceItem] = []
    for requirement in requirements:
        check = by_name.get(_squash(requirement))
        status = check.status if check else "not_found"
        verified = [q for q in (check.quotes if check else []) if quote_in_text(q, resume_text)]
        naming = [q for q in verified if mentioned_in(requirement, q)]

        if status in ("met", "partial") and naming:
            items.append(EvidenceItem(requirement=requirement, status=status, quotes=naming))
        elif line := _first_line_naming(requirement, resume_text):
            items.append(
                EvidenceItem(
                    requirement=requirement,
                    status="met" if status == "met" else "partial",
                    quotes=[line],
                )
            )
        elif status in ("met", "partial") and verified:
            # Real quotes, but none names the requirement: related at best.
            items.append(EvidenceItem(requirement=requirement, status="partial", quotes=verified))
        elif status in ("met", "partial"):
            items.append(EvidenceItem(requirement=requirement, status="unverified", quotes=[]))
        else:
            items.append(EvidenceItem(requirement=requirement, status="not_found", quotes=[]))
    return items


def rule_based_evidence(requirements: list[str], resume_text: str) -> list[EvidenceItem]:
    """Evidence for the fallback path: the first resume line that mentions the skill."""
    items: list[EvidenceItem] = []
    for requirement in requirements:
        line = _first_line_naming(requirement, resume_text)
        if line is None:
            items.append(EvidenceItem(requirement=requirement, status="not_found", quotes=[]))
        else:
            items.append(EvidenceItem(requirement=requirement, status="met", quotes=[line]))
    return items
