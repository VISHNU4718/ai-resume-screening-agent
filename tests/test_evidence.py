from app.models.genai import FitAssessment
from app.services.evidence import quote_in_text, rule_based_evidence, verify_assessment
from app.services.skills import canonicalize, mentioned_in

RESUME = """Jane Doe
Built a RAG chatbot with LangChain and FAISS.
Skills: Python, SQL, Git
"""


def test_quote_matching_ignores_case_whitespace_and_punctuation():
    assert quote_in_text("built a RAG chatbot with  LangChain", RESUME)
    assert quote_in_text("- Built a RAG chatbot with LangChain and FAISS", RESUME)


def test_made_up_or_tiny_quotes_do_not_match():
    assert not quote_in_text("Led a team of ten engineers", RESUME)
    assert not quote_in_text("RAG", RESUME)  # too short to prove anything
    assert not quote_in_text("", RESUME)


def assessment(*checks) -> FitAssessment:
    return FitAssessment.model_validate(
        {"requirements": [{"requirement": r, "status": s, "quotes": q} for r, s, q in checks]}
    )


def test_verified_quote_keeps_status():
    items = verify_assessment(
        ["RAG"], assessment(("RAG", "met", ["Built a RAG chatbot with LangChain"])), RESUME
    )
    assert items[0].status == "met"
    assert items[0].quotes == ["Built a RAG chatbot with LangChain"]


def test_claim_without_a_real_quote_becomes_unverified():
    items = verify_assessment(
        ["Docker"], assessment(("Docker", "met", ["Deployed services with Docker Compose"])), RESUME
    )
    assert items[0].status == "unverified"
    assert items[0].quotes == []


def test_unverified_quotes_are_dropped_but_verified_ones_kept():
    a = assessment(("Testing", "met", ["invented line here", "Skills: Python, SQL, Git"]))
    item = verify_assessment(["Testing"], a, RESUME)[0]
    assert item.status == "partial" and item.quotes == ["Skills: Python, SQL, Git"]


def test_not_found_and_missing_requirements():
    a = assessment(("Docker", "not_found", []))
    items = verify_assessment(["Docker", "Kubernetes"], a, RESUME)
    assert [i.status for i in items] == ["not_found", "not_found"]


def test_resume_that_names_the_requirement_overrides_a_model_miss():
    a = assessment(("SQL", "not_found", []))
    item = verify_assessment(["SQL"], a, RESUME)[0]
    assert item.status == "partial"
    assert item.quotes == ["Skills: Python, SQL, Git"]


def test_met_without_a_naming_quote_uses_the_line_that_names_it():
    a = assessment(("Git", "met", ["Built a RAG chatbot with LangChain and FAISS."]))
    item = verify_assessment(["Git"], a, RESUME)[0]
    assert item.status == "met"
    assert item.quotes == ["Skills: Python, SQL, Git"]


def test_requirement_text_is_matched_loosely():
    a = assessment(("  python ", "met", ["Skills: Python, SQL, Git"]))
    assert verify_assessment(["Python"], a, RESUME)[0].status == "met"


def test_rule_based_evidence_quotes_the_first_matching_line():
    items = rule_based_evidence(["LangChain", "Docker"], RESUME)
    assert items[0].status == "met"
    assert items[0].quotes == ["Built a RAG chatbot with LangChain and FAISS."]
    assert items[1].status == "not_found"


def test_canonicalize_maps_aliases_and_keeps_unknown_names():
    assert canonicalize("fast api") == "FastAPI"
    assert canonicalize("  PYTHON ") == "Python"
    assert canonicalize("Quantum   Basket Weaving") == "Quantum Basket Weaving"


def test_mentioned_in_uses_word_boundaries_and_aliases():
    assert mentioned_in("Python", "I write python daily")
    assert not mentioned_in("Java", "Experienced in JavaScript only")
    assert mentioned_in("Quantum Basket Weaving", "level 2 quantum-basket weaving course")
    assert not mentioned_in("Docker", RESUME)
