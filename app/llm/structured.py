"""Structured output: render a prompt, call the model, validate JSON, repair once, cache."""

import json
import logging
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.llm.cache import DiskCache, content_key
from app.llm.client import ChatClient, Message
from app.llm.errors import LLMError, StructuredOutputError
from app.llm.prompts import PromptLibrary

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def extract_json(text: str) -> object:
    """Parse the first JSON value in model output, tolerating code fences and trailing prose."""
    start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
    if start == -1:
        raise ValueError("no JSON found in model output")
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    return value


class StructuredLLM:
    """Runs named prompts and returns validated Pydantic objects."""

    def __init__(
        self, client: ChatClient, prompts: PromptLibrary, cache: DiskCache | None = None
    ) -> None:
        self.client = client
        self.prompts = prompts
        self._cache = cache

    @property
    def model(self) -> str:
        return self.client.model

    def run(self, prompt: str, schema: type[T], **variables: object) -> T:
        system, user = self.prompts.render(prompt, **variables)
        messages: list[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        key = content_key(self.client.model, self.prompts.version, prompt, system, user)
        if self._cache is not None and (hit := self._cache.get(key)) is not None:
            try:
                logger.debug("LLM cache hit for prompt %s", prompt)
                return schema.model_validate_json(hit)
            except ValidationError:
                logger.warning("Cached output for %s no longer matches the schema", prompt)

        result = self._call_with_repair(prompt, schema, messages)
        if self._cache is not None:
            self._cache.set(key, result.model_dump_json())
        return result

    def _call_with_repair(self, prompt: str, schema: type[T], messages: list[Message]) -> T:
        raw = self.client.chat(messages)
        try:
            return _parse(raw, schema)
        except (ValueError, ValidationError) as first_error:
            logger.info(
                "Invalid %s output (%s); requesting one repair", prompt, _brief(first_error)
            )
            _, repair_user = self.prompts.render("repair", error=_brief(first_error))
            repair_messages = [
                *messages,
                {"role": "assistant", "content": raw},
                {"role": "user", "content": repair_user},
            ]
            try:
                return _parse(self.client.chat(repair_messages), schema)
            except LLMError:
                raise
            except (ValueError, ValidationError) as second_error:
                raise StructuredOutputError(
                    f"Model output for {prompt!r} stayed invalid after one repair: "
                    f"{_brief(second_error)}"
                ) from second_error


def _parse(raw: str, schema: type[T]) -> T:
    return schema.model_validate(extract_json(raw))


def _brief(error: Exception) -> str:
    """Short, content-free description of a validation problem, safe to log and to send back."""
    if isinstance(error, ValidationError):
        problems = [
            f"{'.'.join(str(p) for p in e['loc']) or 'root'}: {e['msg']}" for e in error.errors()
        ]
        return "; ".join(problems[:5])
    return str(error)
