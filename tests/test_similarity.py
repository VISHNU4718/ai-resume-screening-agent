import pytest

from app.services import similarity
from app.services.similarity import TfidfBackend, build_backend
from tests.conftest import make_settings

JOB = "Python developer building REST APIs with FastAPI and Docker"
RELEVANT = "Python engineer who built REST APIs using FastAPI, deployed in Docker containers"
IRRELEVANT = "Pastry chef specialising in sourdough bread and French desserts"


def test_relevant_resume_scores_higher():
    relevant, irrelevant = TfidfBackend().similarities(JOB, [RELEVANT, IRRELEVANT])
    assert relevant > irrelevant
    assert 0.0 <= irrelevant <= relevant <= 1.0


def test_no_resumes():
    assert TfidfBackend().similarities(JOB, []) == []


def test_stopword_only_text_does_not_crash():
    assert TfidfBackend().similarities("the and of", ["the or a"]) == [0.0]


def test_identical_text_is_near_one():
    (score,) = TfidfBackend().similarities(JOB, [JOB])
    assert score == pytest.approx(1.0)


def test_build_backend_tfidf():
    assert build_backend(make_settings(embedding_backend="tfidf")).name == "tfidf"


def test_auto_falls_back_to_tfidf_when_model_unavailable(monkeypatch):
    def broken(_):
        raise OSError("no network")

    monkeypatch.setattr(similarity, "SbertBackend", broken)
    assert build_backend(make_settings(embedding_backend="auto")).name == "tfidf"


def test_explicit_sbert_fails_loudly_when_model_unavailable(monkeypatch):
    def broken(_):
        raise OSError("no network")

    monkeypatch.setattr(similarity, "SbertBackend", broken)
    with pytest.raises(RuntimeError, match="sbert"):
        build_backend(make_settings(embedding_backend="sbert"))
