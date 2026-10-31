"""Question answering over screened resumes: chunk, retrieve, answer with citations."""

import logging
import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass

from app.core.errors import ScreeningNotFoundError
from app.llm.errors import LLMError
from app.llm.structured import StructuredLLM
from app.models.genai import QAAnswer
from app.models.schemas import AskResponse, Citation
from app.services.similarity import SimilarityBackend

logger = logging.getLogger(__name__)

CHUNK_CHARS = 600
MIN_CHUNK_CHARS = 20
FALLBACK_EXCERPTS = 3


@dataclass(frozen=True)
class Chunk:
    candidate: str
    filename: str
    text: str


def chunk_text(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """Split on lines into chunks of about max_chars, repeating the last line as overlap."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in lines:
        if current and size + len(line) > max_chars:
            chunks.append("\n".join(current))
            current = current[-1:]  # one line of overlap keeps context across the boundary
            size = len(current[0]) if current else 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


class ScreeningStore:
    """Keeps the chunks of the most recent screenings in memory (lost on restart)."""

    def __init__(self, max_items: int) -> None:
        self._max = max_items
        self._items: OrderedDict[str, list[Chunk]] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, chunks: list[Chunk]) -> str:
        screening_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._items[screening_id] = chunks
            while len(self._items) > self._max:
                self._items.popitem(last=False)
        return screening_id

    def get(self, screening_id: str) -> list[Chunk]:
        with self._lock:
            chunks = self._items.get(screening_id)
        if chunks is None:
            raise ScreeningNotFoundError(
                "Unknown or expired screening. Run the screening again before asking."
            )
        return chunks


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", text.lower()))


class AskService:
    def __init__(
        self,
        store: ScreeningStore,
        similarity: SimilarityBackend,
        llm: StructuredLLM | None,
        top_k: int,
    ) -> None:
        self._store = store
        self._similarity = similarity
        self._llm = llm
        self._top_k = top_k

    def ask(self, screening_id: str, question: str) -> AskResponse:
        chunks = self._store.get(screening_id)
        hits = self._retrieve(question, chunks)
        if not hits:
            return AskResponse(
                answer="No text in the screened resumes relates to that question.",
                citations=[],
                analysis_mode="fallback" if self._llm is None else "llm",
                grounded=False,
            )
        if self._llm is not None:
            try:
                return self._answer_with_llm(question, hits)
            except LLMError as exc:
                logger.warning("LLM answer failed (%s); returning excerpts only", exc)
        return self._excerpts_only(hits)

    def _retrieve(self, question: str, chunks: list[Chunk]) -> list[Chunk]:
        # A question that names a candidate is answered from that candidate's resume only.
        asked = _words(question)
        named = [c for c in chunks if asked & _words(c.candidate)]
        pool = named or chunks
        scores = self._similarity.similarities(question, [c.text for c in pool])
        ranked = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)
        return [pool[i] for i in ranked[: self._top_k] if scores[i] > 0]

    def _answer_with_llm(self, question: str, hits: list[Chunk]) -> AskResponse:
        assert self._llm is not None
        excerpts = "\n\n".join(
            f"[{i}] Candidate: {c.candidate}\n{c.text}" for i, c in enumerate(hits, start=1)
        )
        result = self._llm.run("answer_question", QAAnswer, question=question, excerpts=excerpts)
        refs = sorted({r for r in result.citations if 1 <= r <= len(hits)})
        if len(refs) != len(set(result.citations)):
            logger.info("Dropped citations that do not match a retrieved excerpt")
        return AskResponse(
            answer=result.answer,
            citations=[_citation(r, hits[r - 1]) for r in refs],
            analysis_mode="llm",
            grounded=bool(refs),
        )

    @staticmethod
    def _excerpts_only(hits: list[Chunk]) -> AskResponse:
        top = hits[:FALLBACK_EXCERPTS]
        return AskResponse(
            answer="No LLM is available, so these are the resume passages most relevant to "
            "your question.",
            citations=[_citation(i, c) for i, c in enumerate(top, start=1)],
            analysis_mode="fallback",
            grounded=True,
        )


def _citation(ref: int, chunk: Chunk) -> Citation:
    return Citation(ref=ref, candidate=chunk.candidate, filename=chunk.filename, excerpt=chunk.text)
