from __future__ import annotations

import os

import pytest

from refaat.domain.models import Chunk, ContextPacket, GenerationRequest, SearchHit
from refaat.infrastructure.generation import OpenAICompatibleGenerator
from refaat.programs import validate_structured
from refaat.programs.program import GroundedRagOutput

pytestmark = pytest.mark.integration

_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("REFAAT_OPENAI_API_KEY")


def _context() -> ContextPacket:
    chunk = Chunk(
        id="chunk_rollback",
        corpus_id="corpus",
        document_id="doc",
        ordinal=0,
        text="Rollback restores the previously approved harness snapshot before deployment.",
        token_count=10,
        checksum="sum",
        metadata={},
        embedding=(),
    )
    hit = SearchHit(chunk=chunk, score=1.0)
    return ContextPacket(
        query="how does rollback work",
        hits=(hit,),
        graph_facts=(),
        session_summary=None,
        profile={},
        supplemental_harness={},
        token_count=10,
        routes=("semantic",),
    )


@pytest.mark.skipif(not _KEY, reason="no model API key configured; the offline gate never needs a live model")
def test_real_model_returns_schema_valid_structured_output() -> None:
    assert _KEY is not None
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("REFAAT_OPENAI_BASE_URL") or "https://api.openai.com/v1"
    model = os.getenv("REFAAT_STRUCTURED_MODEL") or "gpt-4o-mini"
    schema = GroundedRagOutput.json_schema()
    generator = OpenAICompatibleGenerator(base_url=base_url, api_key=_KEY, model=model)
    try:
        result = generator.generate(
            GenerationRequest(
                query="how does rollback work",
                context=_context(),
                system_prompt="Answer from the evidence and cite chunks with [chunk:<id>].",
                response_schema=schema,
            )
        )
    finally:
        generator.close()

    assert isinstance(result.answer, str)
    assert result.answer
    ok, reason = validate_structured({"answer": result.answer, "citations": list(result.cited_chunk_ids)}, schema)
    assert ok, reason
