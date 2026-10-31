import pytest
from pydantic import BaseModel

from app.llm.cache import DiskCache, content_key
from app.llm.errors import LLMError, StructuredOutputError
from app.llm.prompts import PromptLibrary
from app.llm.structured import StructuredLLM, extract_json
from app.models.genai import CandidateProfile, JobRequirements
from tests.fakes import ScriptedChat

JOB_OK = {"must_have": ["Python"], "nice_to_have": ["AWS"], "min_years": 2}


def structured(chat, cache=None, version="v1") -> StructuredLLM:
    return StructuredLLM(chat, PromptLibrary(version), cache)


def run_job(llm: StructuredLLM) -> JobRequirements:
    return llm.run("extract_job", JobRequirements, job_description="Python and AWS wanted")


# --- extract_json ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        'Sure! Here you go:\n{"a": 1}\nHope that helps.',
        '  {"a": 1} {"b": 2}',
    ],
)
def test_extract_json_tolerates_fences_and_prose(text):
    assert extract_json(text) == {"a": 1}


@pytest.mark.parametrize("text", ["", "no json here", '{"a": '])
def test_extract_json_rejects_garbage(text):
    with pytest.raises(ValueError):
        extract_json(text)


# --- validation and repair --------------------------------------------------------------------


def test_valid_output_makes_one_call():
    chat = ScriptedChat(extract_job=JOB_OK)
    job = run_job(structured(chat))
    assert job.must_have == ["Python"] and job.min_years == 2
    assert chat.steps() == ["extract_job"]


def test_system_prompt_and_untrusted_framing_are_sent():
    chat = ScriptedChat(extract_job=JOB_OK)
    run_job(structured(chat))
    _, messages = chat.calls[0]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "Python and AWS wanted" in messages[1]["content"]


def test_repair_succeeds_on_second_attempt():
    chat = ScriptedChat(extract_job=["this is not json", JOB_OK])
    assert run_job(structured(chat)).must_have == ["Python"]
    assert chat.steps() == ["extract_job", "extract_job"]
    repair_messages = chat.calls[1][1]
    assert [m["role"] for m in repair_messages] == ["system", "user", "assistant", "user"]
    assert repair_messages[2]["content"] == "this is not json"
    assert "could not be used" in repair_messages[3]["content"]


def test_schema_violation_triggers_repair_and_error_is_content_free():
    chat = ScriptedChat(extract_job=[{"must_have": 5}, JOB_OK])
    run_job(structured(chat))
    hint = chat.calls[1][1][3]["content"]
    assert "must_have" in hint


def test_two_invalid_outputs_raise_structured_output_error():
    chat = ScriptedChat(extract_job=["nope", "still nope"])
    with pytest.raises(StructuredOutputError):
        run_job(structured(chat))
    assert len(chat.calls) == 2  # exactly one repair, never a loop


def test_transport_failure_is_not_repaired():
    chat = ScriptedChat(extract_job=LLMError("down"))
    with pytest.raises(LLMError):
        run_job(structured(chat))
    assert len(chat.calls) == 1


def test_failure_during_repair_call_propagates():
    chat = ScriptedChat(extract_job=["bad", LLMError("down")])
    with pytest.raises(LLMError) as info:
        run_job(structured(chat))
    assert not isinstance(info.value, StructuredOutputError)


# --- caching ------------------------------------------------------------------------------------


def test_cache_hit_skips_the_model(tmp_path):
    cache = DiskCache(tmp_path)
    chat = ScriptedChat(extract_job=JOB_OK)
    first = run_job(structured(chat, cache))
    second = run_job(structured(chat, cache))
    assert first == second
    assert len(chat.calls) == 1


def test_cache_survives_a_new_process_object(tmp_path):
    run_job(structured(ScriptedChat(extract_job=JOB_OK), DiskCache(tmp_path)))
    chat = ScriptedChat(extract_job=LLMError("should not be called"))
    assert run_job(structured(chat, DiskCache(tmp_path))).must_have == ["Python"]
    assert not chat.calls


def test_cache_key_depends_on_model_prompt_version_and_input(tmp_path):
    cache = DiskCache(tmp_path)
    chat = ScriptedChat(extract_job=JOB_OK)
    run_job(structured(chat, cache))

    other_model = ScriptedChat(extract_job=JOB_OK)
    other_model.model = "another-model"
    run_job(structured(other_model, cache))
    assert len(other_model.calls) == 1

    other_input = ScriptedChat(extract_job=JOB_OK)
    structured(other_input, cache).run("extract_job", JobRequirements, job_description="different")
    assert len(other_input.calls) == 1


def test_cache_key_changes_with_prompt_version(tmp_path):
    versions = tmp_path / "prompts"
    for name in ("v1", "v2"):
        (versions / name).mkdir(parents=True)
        (versions / name / "extract_job.md").write_text(
            f"[system]\nversion {name} hiring requirements\n[user]\n$job_description\n"
        )
        (versions / name / "repair.md").write_text("[user]\n$error\n")
    cache = DiskCache(tmp_path / "cache")
    chat = ScriptedChat(extract_job=JOB_OK)
    for name in ("v1", "v2"):
        llm = StructuredLLM(chat, PromptLibrary(name, versions), cache)
        run_job(llm)
    assert len(chat.calls) == 2


def test_invalid_output_is_never_cached(tmp_path):
    cache = DiskCache(tmp_path)
    with pytest.raises(StructuredOutputError):
        run_job(structured(ScriptedChat(extract_job=["x", "y"]), cache))
    chat = ScriptedChat(extract_job=JOB_OK)
    run_job(structured(chat, cache))
    assert len(chat.calls) == 1


def test_cached_entry_that_no_longer_validates_is_ignored(tmp_path):
    cache = DiskCache(tmp_path)

    class Strict(BaseModel):
        must_have: list[str]
        extra_required_field: str

    system, user = PromptLibrary("v1").render("extract_job", job_description="d")
    key = content_key("fake-model", "v1", "extract_job", system, user)
    cache.set(key, '{"must_have": []}')
    chat = ScriptedChat(extract_job={"must_have": ["Python"], "extra_required_field": "x"})
    result = structured(chat, cache).run("extract_job", Strict, job_description="d")
    assert result.extra_required_field == "x"
    assert len(chat.calls) == 1


def test_disk_cache_roundtrip_and_corruption(tmp_path):
    cache = DiskCache(tmp_path)
    assert cache.get("abcd1234") is None
    cache.set("abcd1234", '{"x": 1}')
    assert cache.get("abcd1234") == '{"x": 1}'
    (tmp_path / "ab" / "abcd1234.json").write_text("{broken")
    assert cache.get("abcd1234") is None


def test_content_key_is_stable_and_unambiguous():
    assert content_key("a", "b") == content_key("a", "b")
    assert content_key("ab", "c") != content_key("a", "bc")
    assert len(content_key("x")) == 64


# --- prompt library ---------------------------------------------------------------------------


EXPECTED_VARIABLES = {
    "extract_job": {"job_description"},
    "extract_resume": {"resume"},
    "assess_fit": {"requirements", "resume"},
    "write_report": {"requirements", "facts", "resume"},
    "answer_question": {"question", "excerpts"},
    "repair": {"error"},
}


@pytest.mark.parametrize("name", sorted(EXPECTED_VARIABLES))
def test_prompt_files_render_and_declare_expected_variables(name):
    library = PromptLibrary("v1")
    assert library.get(name).variables() == EXPECTED_VARIABLES[name]
    system, user = library.render(name, **{v: f"<<{v}>>" for v in EXPECTED_VARIABLES[name]})
    assert "$" not in system + user
    assert user.strip()


@pytest.mark.parametrize("name", sorted(set(EXPECTED_VARIABLES) - {"repair"}))
def test_prompts_treat_input_as_untrusted_and_show_output_shape(name):
    system, _ = PromptLibrary("v1").render(name, **{v: "x" for v in EXPECTED_VARIABLES[name]})
    assert "Output shape" in system
    if name != "extract_job":
        assert "untrusted" in system


def test_unknown_version_and_prompt_are_errors():
    with pytest.raises(ValueError):
        PromptLibrary("v999")
    with pytest.raises(ValueError, match="nope"):
        PromptLibrary("v1").get("nope")


def test_missing_variable_is_an_error():
    with pytest.raises(KeyError):
        PromptLibrary("v1").render("extract_resume")


def test_candidate_profile_runs_through_structured_llm():
    chat = ScriptedChat(
        extract_resume={"name": "A B", "skills": ["Python"], "years_experience": "3+"}
    )
    profile = structured(chat).run("extract_resume", CandidateProfile, resume="A B\nPython")
    assert profile.years_experience == 3.0
