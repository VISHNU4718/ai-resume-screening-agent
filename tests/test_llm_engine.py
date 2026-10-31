import pytest

from app.llm.errors import LLMError
from app.llm.prompts import PromptLibrary
from app.llm.structured import StructuredLLM
from app.models.genai import CandidateProfile, JobRequirements
from app.models.schemas import Recommendation, ScoreBreakdown
from app.services.llm_engine import LLMEngine
from app.services.parser import ParsedResume
from app.services.reasoning import CandidateAnalysis
from app.services.scoring import SkillMatch
from tests.fakes import REPORT, ScriptedChat, faithful_llm

RESUME = ParsedResume(
    filename="jane.txt",
    name="Jane Doe",
    text="Jane Doe\nBuilt a RAG chatbot with LangChain and FAISS.\nSkills: Python, SQL, Git\n",
)
JD = "Requirements\n- Python and FastAPI\n- RAG\nNice to have\n- AWS\n"


def engine(chat: ScriptedChat) -> LLMEngine:
    return LLMEngine(StructuredLLM(chat, PromptLibrary("v1")))


def analysis() -> CandidateAnalysis:
    return CandidateAnalysis(
        name="Jane Doe",
        resume_text=RESUME.text,
        job=JobRequirements(must_have=["Python", "RAG"], nice_to_have=["AWS"]),
        profile=CandidateProfile(name="Jane Doe", skills=["Python"]),
        skills=SkillMatch(["Python"], ["RAG"], [], ["AWS"]),
        years=2.0,
        required_years=1.0,
        score=60.0,
        breakdown=ScoreBreakdown(skill_coverage=50, semantic_similarity=40, experience_fit=100),
        recommendation=Recommendation.MAYBE,
    )


def test_all_steps_report_llm_mode_on_success():
    eng = engine(faithful_llm())
    assert eng.model == "fake-model"
    job = eng.extract_job(JD)
    profile = eng.extract_profile(RESUME)
    assess = eng.assess(["Python", "RAG"], RESUME.text)
    report = eng.report(analysis())
    assert {s.mode for s in (job, profile, assess, report)} == {"llm"}
    assert "FastAPI" in job.value.must_have
    assert report.value.interview_questions == REPORT["interview_questions"]


def test_each_step_falls_back_on_its_own():
    broken = LLMError("down")
    eng = engine(
        ScriptedChat(
            extract_job=broken, extract_resume=broken, assess_fit=broken, write_report=broken
        )
    )
    job = eng.extract_job(JD)
    profile = eng.extract_profile(RESUME)
    assess = eng.assess(["Python", "RAG"], RESUME.text)
    report = eng.report(analysis())
    assert [s.mode for s in (job, profile, assess, report)] == ["fallback"] * 4
    assert "Python" in job.value.must_have
    assert profile.value.name == "Jane Doe"
    assert assess.value.evidence[1].status == "met"
    assert len(report.value.interview_questions) >= 3


def test_one_failing_step_does_not_affect_the_others():
    eng = engine(faithful_llm(write_report=LLMError("down")))
    assert eng.extract_profile(RESUME).mode == "llm"
    assert eng.report(analysis()).mode == "fallback"


def test_malformed_output_after_repair_falls_back():
    eng = engine(faithful_llm(extract_resume=["garbage", "more garbage"]))
    step = eng.extract_profile(RESUME)
    assert step.mode == "fallback"
    assert "Python" in step.value.skills


def test_job_with_no_requirements_from_the_model_falls_back():
    eng = engine(faithful_llm(extract_job={"must_have": [], "nice_to_have": []}))
    step = eng.extract_job(JD)
    assert step.mode == "fallback"
    assert "Python" in step.value.must_have


def test_assess_with_nothing_to_check_makes_no_call():
    chat = faithful_llm()
    step = engine(chat).assess([], RESUME.text)
    assert step.mode == "llm" and step.value.evidence == []
    assert not chat.calls


def test_assessment_is_verified_against_the_resume():
    reply = {
        "requirements": [
            {
                "requirement": "RAG",
                "status": "met",
                "quotes": ["Built a RAG chatbot with LangChain"],
            },
            {
                "requirement": "Docker",
                "status": "met",
                "quotes": ["Containerised everything with Docker"],
            },
        ],
        "unsupported_claims": ["expert in everything"],
    }
    step = engine(faithful_llm(assess_fit=reply)).assess(["RAG", "Docker"], RESUME.text)
    rag, docker = step.value.evidence
    assert rag.status == "met" and rag.quotes
    assert docker.status == "unverified" and docker.quotes == []
    assert step.value.unsupported_claims == ["expert in everything"]


def test_report_prompt_carries_the_computed_facts():
    chat = faithful_llm()
    engine(chat).report(analysis())
    prompt = chat.calls[0][1][1]["content"]
    assert '"score_out_of_100": 60.0' in prompt
    assert "Recommendation" not in prompt or '"recommendation": "Maybe"' in prompt


@pytest.mark.parametrize("step", ["extract_job", "extract_resume"])
def test_long_inputs_are_truncated_before_prompting(step):
    chat = faithful_llm()
    eng = engine(chat)
    long_text = "Python " * 5000
    if step == "extract_job":
        eng.extract_job(long_text)
    else:
        eng.extract_profile(ParsedResume("a.txt", "A", long_text))
    assert len(chat.calls[0][1][1]["content"]) < 7000
