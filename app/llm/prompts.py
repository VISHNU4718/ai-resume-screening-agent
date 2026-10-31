"""Versioned prompt templates.

Prompts live in app/prompts/<version>/<name>.md. Each file has a [system] and/or [user]
section; $placeholders are filled with string.Template, so braces in JSON examples are safe.
Changing a prompt means adding a new version directory, which also invalidates the cache.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from string import Template

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_SECTION = re.compile(r"^\[(system|user)\]\s*$", re.MULTILINE)


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    system: Template
    user: Template

    def variables(self) -> set[str]:
        return set(self.system.get_identifiers()) | set(self.user.get_identifiers())


class PromptLibrary:
    def __init__(self, version: str, directory: Path = PROMPTS_DIR) -> None:
        self.version = version
        self._dir = directory / version
        if not self._dir.is_dir():
            raise ValueError(f"Unknown prompt version {version!r} (looked in {self._dir})")

    def get(self, name: str) -> PromptTemplate:
        path = self._dir / f"{name}.md"
        if not path.is_file():
            raise ValueError(f"No prompt named {name!r} in version {self.version!r}")
        return _load(path)

    def render(self, name: str, **variables: object) -> tuple[str, str]:
        """Return (system, user) text with all variables substituted."""
        template = self.get(name)
        values = {k: str(v) for k, v in variables.items()}
        return template.system.substitute(values), template.user.substitute(values)


@lru_cache
def _load(path: Path) -> PromptTemplate:
    text = path.read_text(encoding="utf-8")
    parts = _SECTION.split(text)
    sections = dict(zip(parts[1::2], (p.strip() for p in parts[2::2]), strict=True))
    if not sections:
        raise ValueError(f"{path.name} has no [system] or [user] section")
    return PromptTemplate(
        name=path.stem,
        system=Template(sections.get("system", "")),
        user=Template(sections.get("user", "")),
    )
