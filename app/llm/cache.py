"""Small on-disk cache for validated LLM output, keyed by a content hash."""

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def content_key(*parts: str) -> str:
    """SHA-256 over the parts. Each part is length-prefixed so boundaries cannot collide."""
    digest = hashlib.sha256()
    for part in parts:
        data = part.encode("utf-8")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


class DiskCache:
    def __init__(self, directory: Path) -> None:
        self._dir = directory

    def get(self, key: str) -> str | None:
        path = self._path(key)
        try:
            return json.loads(path.read_text(encoding="utf-8"))["value"]
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError):
            logger.warning("Ignoring unreadable cache entry %s", path.name)
            return None

    def set(self, key: str, value: str) -> None:
        path = self._path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temp file first so a concurrent reader never sees a partial entry.
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"value": value}, handle)
            os.replace(tmp, path)
        except OSError as exc:
            logger.warning("Could not write LLM cache: %s", exc)

    def _path(self, key: str) -> Path:
        return self._dir / key[:2] / f"{key}.json"
