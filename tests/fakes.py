"""A scripted stand-in for the chat client. No test in this project touches the network."""

import json
import re
from collections.abc import Callable
from typing import Any

from app.services.experience import estimate_years
from app.services.skills import extract_job_skills, extract_skills

# The first words of each system prompt identify which step is calling.
STEP_MARKERS = {
    "hiring requirements": "extract_job",
    "structured profile": "extract_resume",
    "against each job requirement": "assess_fit",
    "short screening report": "write_report",
    "recruiter's question": "answer_question",
}

Reply = str | dict | list | Exception | Callable[[str], Any]


class ScriptedChat:
    """Replies per step. A value may be text, a JSON-able object, an exception to raise,
    a callable receiving the user prompt, or a list of those that is consumed in order
    (the last item repeats)."""

    model = "fake-model"

    def __init__(self, **replies: Reply | list[Reply]) -> None:
        self.replies = replies
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def steps(self) -> list[str]:
        return [step for step, _ in self.calls]

    def chat(self, messages: list[dict[str, str]], *, json_mode: bool = True) -> str:
        step = next(
            name for marker, name in STEP_MARKERS.items() if marker in messages[0]["content"]
        )
        self.calls.append((step, messages))
        reply = self.replies[step]
        if isinstance(reply, list):
            index = min(len([s for s in self.steps() if s == step]) - 1, len(reply) - 1)
            reply = reply[index]
        if callable(reply):
            reply = reply(messages[1]["content"])
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)


def _between(tag: str, text: str) -> str:
    match = re.search(rf"<{tag}>\n?(.*?)\n?</{tag}>", text, re.DOTALL)
    return match.group(1) if match else text


def _faithful_job(prompt: str) -> dict:
    must, nice = extract_job_skills(_between("job_description", prompt))
    return {"must_have": must, "nice_to_have": nice, "min_years": 1}


def _faithful_profile(prompt: str) -> dict:
    """A model that reads the resume correctly, plus one skill it made up."""
    resume = _between("resume", prompt)
    lines = [ln for ln in resume.splitlines() if ln.strip()]
    return {
        "name": lines[0].strip(),
        "skills": [*extract_skills(resume), "Kubernetes"],  # never in the sample resumes
        "years_experience": estimate_years(resume),
        "projects": [],
        "roles": [],
    }


def _faithful_assessment(prompt: str) -> dict:
    resume = _between("resume", prompt)
    requirements = [
        ln[2:].strip() for ln in prompt.split("<resume>")[0].splitlines() if ln.startswith("- ")
    ]
    checks = []
    for requirement in requirements:
        line = next(
            (ln.strip() for ln in resume.splitlines() if requirement.lower() in ln.lower()), None
        )
        checks.append(
            {"requirement": requirement, "status": "met", "quotes": [line]}
            if line
            else {"requirement": requirement, "status": "not_found", "quotes": []}
        )
    return {"requirements": checks, "unsupported_claims": []}


REPORT = {
    "strengths": ["Relevant project work"],
    "concerns": ["Limited production experience"],
    "summary": "A reasonable fit for the role.",
    "interview_questions": ["Describe your RAG project.", "How did you test it?", "What failed?"],
}


def faithful_llm(**overrides: Reply | list[Reply]) -> ScriptedChat:
    """A model that behaves well on every step. Override a step to break it."""
    replies: dict[str, Any] = {
        "extract_job": _faithful_job,
        "extract_resume": _faithful_profile,
        "assess_fit": _faithful_assessment,
        "write_report": REPORT,
        "answer_question": {"answer": "Ananya built a RAG system [1].", "citations": [1]},
    }
    return ScriptedChat(**{**replies, **overrides})
