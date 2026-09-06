from __future__ import annotations

import pytest

from refaat.application.runtime import Refaat
from refaat.domain.models import DocumentInput
from refaat.evaluation import Dataset
from refaat.optimization import GraphOptimizer

pytestmark = pytest.mark.integration

_TEXT = (
    "Oronts builds Refaat for Production Intelligence. Refaat powers Oronts platforms. "
    "The Release Manager verifies evidence before deployment to production."
)


def test_graph_optimizer_uses_shadow_and_leaves_active_graph_untouched(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    for index in range(3):
        runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id=f"g{index}.txt",
                title=f"G{index}",
                media_type="text/plain",
                content=_TEXT.encode(),
            ),
        )
    before_docs = len(project.list_documents())
    before_entities, before_relations = runtime.store.list_graph(project.workspace_id, project.corpus_id)
    dataset = Dataset.from_records("graph", [{"query": "who builds Refaat", "expected_terms": ["Oronts"]}])

    result = GraphOptimizer(project).optimize(dataset, max_entities=(2, 200), min_token_lengths=(4,))

    assert result.candidate_count == 2
    assert {candidate.max_entities for candidate in result.candidates} == {2, 200}
    assert all(candidate.mean_tokens >= 0.0 for candidate in result.candidates)

    by_cap = {candidate.max_entities: candidate.entity_count for candidate in result.candidates}
    assert by_cap[2] <= 2 * before_docs
    assert by_cap[2] <= by_cap[200]

    assert len(project.list_documents()) == before_docs
    after_entities, after_relations = runtime.store.list_graph(project.workspace_id, project.corpus_id)
    assert len(after_entities) == len(before_entities)
    assert len(after_relations) == len(before_relations)
