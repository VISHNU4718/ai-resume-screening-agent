class LLMError(Exception):
    """The LLM could not produce a usable answer (transport, HTTP or output-format failure)."""


class StructuredOutputError(LLMError):
    """The model answered, but not with valid JSON for the requested schema, even after repair."""
