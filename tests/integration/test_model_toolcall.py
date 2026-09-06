from __future__ import annotations

import os

import pytest

from refaat.domain.models import ToolSpec
from refaat.infrastructure.generation import OpenAICompatibleGenerator
from refaat.programs import validate_tool_arguments

pytestmark = pytest.mark.integration

_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("REFAAT_OPENAI_API_KEY")

_SPEC = ToolSpec(
    name="lookup_policy",
    description="Look up a rollback policy by topic.",
    parameters={
        "type": "object",
        "properties": {"topic": {"type": "string"}},
        "required": ["topic"],
        "additionalProperties": False,
    },
)


@pytest.mark.skipif(not _KEY, reason="no model API key configured; the offline gate never needs a live model")
def test_real_model_proposes_a_schema_valid_tool_call() -> None:
    assert _KEY is not None
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("REFAAT_OPENAI_BASE_URL") or "https://api.openai.com/v1"
    model = os.getenv("REFAAT_TOOLCALL_MODEL") or "gpt-4o-mini"
    generator = OpenAICompatibleGenerator(base_url=base_url, api_key=_KEY, model=model)
    try:
        proposal = generator.propose_tool_call("Look up the rollback policy for deployments.", (_SPEC,), frozenset())
    finally:
        generator.close()

    assert proposal is not None
    assert proposal.tool == "lookup_policy"
    ok, reason = validate_tool_arguments(proposal.arguments, _SPEC.parameters)
    assert ok, reason
