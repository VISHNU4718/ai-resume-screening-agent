import pytest

from app.api.deps import get_pipeline
from app.core.config import get_settings
from app.main import app
from app.services.engine import RuleBasedEngine
from app.services.pipeline import ScreeningPipeline
from app.services.rag import ScreeningStore
from app.services.similarity import TfidfBackend
from tests.conftest import make_settings


def post_screen(client, jd, files):
    payload = [("resumes", (name, content)) for name, content in files.items()]
    return client.post("/screen", data={"job_description": jd}, files=payload)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["llm_enabled"] is False
    assert body["llm_model"] is None


def test_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Resume Screening Agent" in response.text


def test_screen_ranks_sample_resumes(client, jd_text, sample_files):
    response = post_screen(client, jd_text, sample_files)
    assert response.status_code == 200
    data = response.json()

    names = [c["name"] for c in data["candidates"]]
    assert names[0] == "Ananya Rao"
    assert names[-1] == "Imran Khan"
    assert [c["rank"] for c in data["candidates"]] == [1, 2, 3, 4]
    scores = [c["score"] for c in data["candidates"]]
    assert scores == sorted(scores, reverse=True)

    top = data["candidates"][0]
    assert top["recommendation"] == "Shortlist"
    assert {"Python", "FastAPI", "RAG"} <= set(top["matched_skills"])
    assert "Docker" not in top["missing_skills"]
    assert data["candidates"][-1]["recommendation"] == "Reject"
    assert data["similarity_backend"] == "tfidf"
    assert data["rejected_files"] == []


def test_screen_response_has_expected_fields(client, jd_text, sample_files):
    data = post_screen(client, jd_text, {"ananya_rao.txt": sample_files["ananya_rao.txt"]}).json()
    candidate = data["candidates"][0]
    assert set(candidate) == {
        "rank",
        "name",
        "filename",
        "score",
        "recommendation",
        "matched_skills",
        "missing_skills",
        "missing_preferred_skills",
        "unsupported_skills",
        "unsupported_claims",
        "evidence",
        "profile",
        "analysis_mode",
        "years_experience",
        "breakdown",
        "strengths",
        "concerns",
        "summary",
        "interview_questions",
    }
    assert 0 <= candidate["score"] <= 100
    assert candidate["analysis_mode"] == "fallback"
    assert data["analysis_mode"] == "fallback"
    assert data["llm_model"] is None
    assert data["screening_id"]
    assert data["job"]["min_years"] == 1.0
    assert 3 <= len(candidate["interview_questions"]) <= 5
    assert candidate["evidence"][0]["quotes"]


def test_bad_file_is_reported_without_failing_request(client, jd_text, sample_files):
    files = {
        "ananya_rao.txt": sample_files["ananya_rao.txt"],
        "broken.pdf": b"nope",
        "photo.png": b"x",
    }
    response = post_screen(client, jd_text, files)
    assert response.status_code == 200
    data = response.json()
    assert len(data["candidates"]) == 1
    rejected = {r["filename"]: r["reason"] for r in data["rejected_files"]}
    assert set(rejected) == {"broken.pdf", "photo.png"}
    assert "Unsupported" in rejected["photo.png"]


def test_all_files_unreadable_returns_422_with_reasons(client, jd_text):
    response = post_screen(client, jd_text, {"a.txt": b"", "b.png": b"x"})
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "no_valid_resumes"
    assert len(body["rejected_files"]) == 2


def test_short_job_description_is_rejected(client, sample_files):
    response = post_screen(client, "Python", {"a.txt": sample_files["ananya_rao.txt"]})
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


def test_whitespace_job_description_is_rejected(client, sample_files):
    response = post_screen(client, " " * 100, {"a.txt": sample_files["ananya_rao.txt"]})
    assert response.status_code == 422


def test_missing_resumes_is_rejected(client, jd_text):
    response = client.post("/screen", data={"job_description": jd_text})
    assert response.status_code == 422


def test_missing_job_description_is_rejected(client, sample_files):
    response = client.post(
        "/screen", files=[("resumes", ("a.txt", sample_files["ananya_rao.txt"]))]
    )
    assert response.status_code == 422


def test_too_many_files(client, jd_text):
    settings = make_settings(max_files=2)
    app.dependency_overrides[get_pipeline] = lambda: ScreeningPipeline(
        settings, TfidfBackend(), RuleBasedEngine(), ScreeningStore(3)
    )
    files = {f"{i}.txt": b"Python developer" for i in range(3)}
    response = post_screen(client, jd_text, files)
    assert response.status_code == 422
    assert "Too many files" in response.json()["detail"]


def test_oversize_file_is_rejected_per_file(client, jd_text, sample_files):
    settings = make_settings(max_file_size_mb=0.001)  # about 1 KB
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_pipeline] = lambda: ScreeningPipeline(
        settings, TfidfBackend(), RuleBasedEngine(), ScreeningStore(3)
    )
    files = {"small.txt": b"Python developer with FastAPI", "big.txt": b"x" * 5000}
    data = post_screen(client, jd_text, files).json()
    assert [c["filename"] for c in data["candidates"]] == ["small.txt"]
    assert "limit" in data["rejected_files"][0]["reason"]


def test_job_description_without_known_skills_warns(client, sample_files):
    jd = "We need an enthusiastic and curious person to join our friendly team soon."
    data = post_screen(client, jd, {"a.txt": sample_files["ananya_rao.txt"]}).json()
    assert data["warnings"]
    assert data["job"]["must_have"] == []
    assert 0 <= data["candidates"][0]["score"] <= 100


def test_single_resume_still_works(client, jd_text, sample_files):
    response = post_screen(client, jd_text, {"kiran_patel.txt": sample_files["kiran_patel.txt"]})
    assert response.status_code == 200
    assert response.json()["candidates"][0]["rank"] == 1


def test_duplicate_filenames_are_both_screened(client, jd_text, sample_files):
    content = sample_files["ananya_rao.txt"]
    payload = [("resumes", ("same.txt", content)), ("resumes", ("same.txt", content))]
    data = client.post("/screen", data={"job_description": jd_text}, files=payload).json()
    assert len(data["candidates"]) == 2


def test_openapi_documents_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/screen", "/ask", "/health"} <= set(paths)


def test_ask_returns_cited_excerpts_in_fallback_mode(client, jd_text, sample_files):
    screening_id = post_screen(client, jd_text, sample_files).json()["screening_id"]
    response = client.post(
        "/ask", json={"screening_id": screening_id, "question": "Who has built a RAG system?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["analysis_mode"] == "fallback"
    assert body["citations"] and {"ref", "candidate", "filename", "excerpt"} <= set(
        body["citations"][0]
    )


def test_ask_unknown_screening_is_404(client):
    response = client.post("/ask", json={"screening_id": "nope", "question": "anything?"})
    assert response.status_code == 404
    assert response.json()["code"] == "screening_not_found"


@pytest.mark.parametrize(
    "payload",
    [
        {"screening_id": "x"},
        {"question": "q"},
        {"screening_id": "x", "question": ""},
        {"screening_id": "x", "question": "   "},
        {"screening_id": "x", "question": "q" * 501},
    ],
)
def test_ask_validates_input(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_llm_mode_through_the_api(settings, store, jd_text, sample_files):
    from app.llm.prompts import PromptLibrary
    from app.llm.structured import StructuredLLM
    from app.services.llm_engine import LLMEngine
    from app.services.rag import AskService
    from tests.conftest import make_client
    from tests.fakes import faithful_llm

    llm = StructuredLLM(faithful_llm(), PromptLibrary("v1"))
    pipeline = ScreeningPipeline(settings, TfidfBackend(), LLMEngine(llm), store)
    ask = AskService(store, TfidfBackend(), llm, 5)
    api = make_client(settings, pipeline, ask)
    try:
        data = post_screen(api, jd_text, sample_files).json()
        assert data["analysis_mode"] == "llm" and data["llm_model"] == "fake-model"
        answer = api.post(
            "/ask", json={"screening_id": data["screening_id"], "question": "Who built RAG?"}
        ).json()
        assert answer["analysis_mode"] == "llm" and answer["grounded"] is True
    finally:
        app.dependency_overrides.clear()
