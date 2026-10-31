"""Schemas for the structured data the LLM is asked to produce.

Validators are deliberately forgiving about format (numbers as strings, long lists, stray
whitespace) because small models drift, and strict about shape (required keys, types).
Anything that still fails validation goes through one repair attempt and then the fallback.
"""

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field, field_validator

MAX_ITEM_CHARS = 300


def _clean_list(limit: int, item_chars: int = MAX_ITEM_CHARS):
    def clean(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("expected a list of strings")
        seen: set[str] = set()
        out: list[str] = []
        for item in value:
            if isinstance(item, dict | list) or item is None:
                continue  # e.g. {"degree": ...}: drop rather than stringify noise
            text = " ".join(str(item).split())[:item_chars]
            if text and text.lower() not in seen:
                seen.add(text.lower())
                out.append(text)
        return out[:limit]

    return clean


def _years(value: Any) -> float | None:
    """Accept 3, 3.5, "3 years", "3+"; treat null or unparseable text as unknown."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        number = float(value)
    else:
        match = re.search(r"\d+(?:\.\d+)?", str(value))
        if not match:
            return None
        number = float(match.group(0))
    return round(number, 1) if 0 <= number <= 60 else None


Years = Annotated[float | None, BeforeValidator(_years)]
SkillList = Annotated[list[str], BeforeValidator(_clean_list(60, 60))]
ShortList = Annotated[list[str], BeforeValidator(_clean_list(8))]


class CandidateProfile(BaseModel):
    """Structured facts extracted from one resume."""

    name: str | None = None
    skills: SkillList = Field(default_factory=list)
    years_experience: Years = None
    education: ShortList = Field(default_factory=list)
    projects: ShortList = Field(default_factory=list)
    roles: ShortList = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str | None) -> str | None:
        value = " ".join((value or "").split())
        return value[:80] or None


class JobRequirements(BaseModel):
    """Structured requirements extracted from a job description."""

    must_have: SkillList = Field(default_factory=list)
    nice_to_have: SkillList = Field(default_factory=list)
    min_years: Years = None

    @field_validator("nice_to_have")
    @classmethod
    def _disjoint(cls, value: list[str], info) -> list[str]:
        must = {s.lower() for s in info.data.get("must_have", [])}
        return [s for s in value if s.lower() not in must]


class RequirementCheck(BaseModel):
    """The model's verdict on one requirement, with verbatim resume quotes as support."""

    requirement: str
    status: Literal["met", "partial", "not_found"]
    quotes: Annotated[list[str], BeforeValidator(_clean_list(3, 240))] = Field(default_factory=list)

    @field_validator("status", mode="before")
    @classmethod
    def _status(cls, value: Any) -> Any:
        return str(value).strip().lower().replace(" ", "_").replace("-", "_")


class FitAssessment(BaseModel):
    requirements: list[RequirementCheck]
    unsupported_claims: Annotated[list[str], BeforeValidator(_clean_list(5))] = Field(
        default_factory=list
    )


class CandidateReport(BaseModel):
    strengths: Annotated[list[str], BeforeValidator(_clean_list(4))]
    concerns: Annotated[list[str], BeforeValidator(_clean_list(4))]
    summary: str
    interview_questions: Annotated[list[str], BeforeValidator(_clean_list(5))]

    @field_validator("summary")
    @classmethod
    def _summary(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("summary must not be empty")
        return value[:500]

    @field_validator("interview_questions")
    @classmethod
    def _questions(cls, value: list[str]) -> list[str]:
        if len(value) < 3:
            raise ValueError("need 3 to 5 interview questions")
        return value


class QAAnswer(BaseModel):
    answer: str
    citations: list[int] = Field(default_factory=list)

    @field_validator("answer")
    @classmethod
    def _answer(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("answer must not be empty")
        return value[:1500]

    @field_validator("citations", mode="before")
    @classmethod
    def _citations(cls, value: Any) -> Any:
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        out: list[int] = []
        for item in value:
            match = re.search(r"\d+", str(item))
            if match:
                out.append(int(match.group(0)))
        return out
