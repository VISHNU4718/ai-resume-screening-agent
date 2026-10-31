"""Step 3: semantic similarity between the job description and each resume.

Two interchangeable backends are provided. Sentence embeddings (all-MiniLM-L6-v2) capture
meaning beyond shared words; TF-IDF needs no model download and works offline. The
backend is chosen by EMBEDDING_BACKEND and falls back to TF-IDF if the model cannot load.
"""

import logging
from typing import Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import Settings

logger = logging.getLogger(__name__)


class SimilarityBackend(Protocol):
    name: str
    # Cosine value treated as a "full" match when scaling to 0-1; differs per backend.
    saturation: float

    def similarities(self, job_text: str, resume_texts: list[str]) -> list[float]:
        """Return cosine similarity (0-1) of each resume against the job text."""
        ...


class TfidfBackend:
    name = "tfidf"
    saturation = 0.45

    def similarities(self, job_text: str, resume_texts: list[str]) -> list[float]:
        if not resume_texts:
            return []
        vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        )
        try:
            matrix = vectorizer.fit_transform([job_text, *resume_texts])
        except ValueError:  # empty vocabulary, e.g. only stop words
            return [0.0] * len(resume_texts)
        scores = cosine_similarity(matrix[0], matrix[1:])[0]
        return [float(max(0.0, min(1.0, s))) for s in scores]


class SbertBackend:
    name = "sbert"
    saturation = 0.8

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # optional dependency

        self._model = SentenceTransformer(model_name)
        self.model_name = model_name

    def similarities(self, job_text: str, resume_texts: list[str]) -> list[float]:
        if not resume_texts:
            return []
        embeddings = self._model.encode(
            [job_text, *resume_texts], normalize_embeddings=True, show_progress_bar=False
        )
        scores = embeddings[1:] @ embeddings[0]
        return [float(max(0.0, min(1.0, s))) for s in scores]


def build_backend(settings: Settings) -> SimilarityBackend:
    if settings.embedding_backend == "tfidf":
        return TfidfBackend()
    try:
        backend = SbertBackend(settings.embedding_model)
        logger.info("Using sentence embeddings (%s)", settings.embedding_model)
        return backend
    except Exception as exc:  # missing package, no network, bad model name
        if settings.embedding_backend == "sbert":
            raise RuntimeError(
                f"EMBEDDING_BACKEND=sbert but the model failed to load: {exc}"
            ) from exc
        logger.warning("Sentence embeddings unavailable (%s); using TF-IDF", exc)
        return TfidfBackend()
