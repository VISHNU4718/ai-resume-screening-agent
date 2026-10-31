import pytest

from app.core.errors import ScreeningNotFoundError
from app.llm.errors import LLMError
from app.llm.prompts import PromptLibrary
from app.llm.structured import StructuredLLM
from app.services.rag import AskService, Chunk, ScreeningStore, chunk_text
from app.services.similarity import TfidfBackend
from tests.fakes import ScriptedChat

ANANYA = "Built a RAG question answering service with LangChain and FAISS vector search."
KIRAN = "Built Power BI dashboards and trained churn models with scikit-learn."


def make_service(chat: ScriptedChat | None, top_k: int = 5):
    store = ScreeningStore(3)
    sid = store.put(
        [
            Chunk("Ananya Rao", "ananya.txt", ANANYA),
            Chunk("Kiran Patel", "kiran.txt", KIRAN),
            Chunk("Kiran Patel", "kiran.txt", "Studied statistics and wrote SQL reports daily."),
        ]
    )
    llm = StructuredLLM(chat, PromptLibrary("v1")) if chat else None
    return AskService(store, TfidfBackend(), llm, top_k), sid


# --- chunking and store -----------------------------------------------------------------------


def test_chunk_text_splits_on_lines_with_overlap():
    lines = [f"Line number {i} with enough words to count" for i in range(30)]
    chunks = chunk_text("\n".join(lines), max_chars=200)
    assert len(chunks) > 3
    assert all(len(c) <= 260 for c in chunks)
    for earlier, later in zip(chunks, chunks[1:], strict=False):
        assert earlier.splitlines()[-1] == later.splitlines()[0]
    assert {ln for c in chunks for ln in c.splitlines()} == set(lines)


def test_chunk_text_handles_empty_and_tiny_text():
    assert chunk_text("") == []
    assert chunk_text("hi") == []
    assert chunk_text("A single reasonably long line of text.") != []


def test_store_evicts_oldest_screening():
    store = ScreeningStore(2)
    first = store.put([Chunk("a", "a.txt", "text one here for a")])
    store.put([Chunk("b", "b.txt", "text two here for b")])
    store.put([Chunk("c", "c.txt", "text three here for c")])
    with pytest.raises(ScreeningNotFoundError):
        store.get(first)


def test_unknown_screening_is_not_found():
    service, _ = make_service(None)
    with pytest.raises(ScreeningNotFoundError):
        service.ask("does-not-exist", "anything about python?")


# --- answering with an LLM ----------------------------------------------------------------------


def test_llm_answer_with_valid_citations():
    chat = ScriptedChat(
        answer_question={"answer": "Ananya built a RAG service [1].", "citations": [1]}
    )
    service, sid = make_service(chat)
    result = service.ask(sid, "Who has built a RAG service with LangChain?")
    assert result.analysis_mode == "llm" and result.grounded is True
    assert [c.ref for c in result.citations] == [1]
    assert result.citations[0].candidate == "Ananya Rao"
    assert result.citations[0].excerpt == ANANYA
    prompt = chat.calls[0][1][1]["content"]
    assert "[1] Candidate: Ananya Rao" in prompt


def test_citations_to_nonexistent_excerpts_are_dropped():
    chat = ScriptedChat(answer_question={"answer": "Yes.", "citations": [1, 7, 0]})
    service, sid = make_service(chat)
    result = service.ask(sid, "RAG with LangChain?")
    assert [c.ref for c in result.citations] == [1]


def test_answer_without_citations_is_flagged_ungrounded():
    chat = ScriptedChat(answer_question={"answer": "Probably everyone.", "citations": []})
    service, sid = make_service(chat)
    result = service.ask(sid, "RAG with LangChain?")
    assert result.analysis_mode == "llm"
    assert result.grounded is False and result.citations == []


def test_question_naming_a_candidate_is_restricted_to_that_resume():
    chat = ScriptedChat(answer_question={"answer": "SQL [1].", "citations": [1, 2, 3]})
    service, sid = make_service(chat)
    result = service.ask(sid, "What does Kiran know about SQL and scikit-learn reports?")
    assert {c.candidate for c in result.citations} == {"Kiran Patel"}
    excerpts = chat.calls[0][1][1]["content"]
    assert "Ananya Rao" not in excerpts


def test_top_k_limits_excerpts_sent_to_the_model():
    chat = ScriptedChat(answer_question={"answer": "ok", "citations": [1]})
    service, sid = make_service(chat, top_k=1)
    service.ask(sid, "Built dashboards RAG LangChain scikit-learn SQL reports")
    assert chat.calls[0][1][1]["content"].count("Candidate:") == 1


def test_llm_failure_returns_excerpts_in_fallback_mode():
    service, sid = make_service(ScriptedChat(answer_question=LLMError("down")))
    result = service.ask(sid, "Who used LangChain?")
    assert result.analysis_mode == "fallback"
    assert result.citations and result.citations[0].candidate == "Ananya Rao"


def test_malformed_llm_answer_falls_back_after_repair():
    chat = ScriptedChat(answer_question=["not json", "{}"])
    service, sid = make_service(chat)
    assert service.ask(sid, "Who used LangChain?").analysis_mode == "fallback"
    assert len(chat.calls) == 2


def test_no_llm_configured_returns_excerpts():
    service, sid = make_service(None)
    result = service.ask(sid, "scikit-learn churn models")
    assert result.analysis_mode == "fallback"
    assert result.citations[0].candidate == "Kiran Patel"
    assert "No LLM" in result.answer


def test_unrelated_question_returns_no_citations_and_skips_the_model():
    chat = ScriptedChat(answer_question={"answer": "x", "citations": [1]})
    service, sid = make_service(chat)
    result = service.ask(sid, "zxqv wibble")
    assert result.citations == [] and result.grounded is False
    assert not chat.calls
