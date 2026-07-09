"""Shared test stubs for the LLM client seam."""

from playtest.llm import ROLES, LLMClient


class StubLLMClient(LLMClient):
    """Plays back canned completion strings in order; records every call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.models = {role: "stub-model" for role in ROLES}
        self.calls: list[dict] = []

    def complete(self, messages, *, role, json_schema=None):
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "role": role,
                "json_schema": json_schema,
            }
        )
        return self._responses.pop(0)
