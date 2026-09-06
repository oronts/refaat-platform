from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from refaat.domain.models import JsonObject
from refaat.evaluation.bundle import MetricBundle, default_bundle
from refaat.evaluation.dataset import Dataset
from refaat.optimization.candidates import pareto_front
from refaat.optimization.shadow import active_documents, shadow_index

if TYPE_CHECKING:
    from refaat.application.composition import ShadowComposition
    from refaat.application.runtime import RefaatProject


@dataclass(frozen=True, slots=True)
class GraphCandidate:
    max_entities: int
    min_token_length: int
    entity_count: int
    relation_count: int
    hard_constraints_passed: bool
    score: float
    mean_tokens: float

    def to_json(self) -> JsonObject:
        return {
            "max_entities": self.max_entities,
            "min_token_length": self.min_token_length,
            "entity_count": self.entity_count,
            "relation_count": self.relation_count,
            "hard_constraints_passed": self.hard_constraints_passed,
            "score": self.score,
            "mean_tokens": self.mean_tokens,
        }


@dataclass(frozen=True, slots=True)
class GraphResult:
    program: str
    dataset_version: str
    candidate_count: int
    best: GraphCandidate
    candidates: tuple[GraphCandidate, ...]
    pareto_front: tuple[GraphCandidate, ...]
    shadow_composition: ShadowComposition | None = None

    def to_json(self) -> JsonObject:
        payload: JsonObject = {
            "program": self.program,
            "dataset_version": self.dataset_version,
            "candidate_count": self.candidate_count,
            "best": self.best.to_json(),
            "candidates": [candidate.to_json() for candidate in self.candidates],
            "pareto_front": [candidate.to_json() for candidate in self.pareto_front],
        }
        if self.shadow_composition is not None and not self.shadow_composition.is_empty():
            payload["shadow_composition"] = self.shadow_composition.to_json()
        return payload


class GraphOptimizer:
    """Evaluate graph-extractor configurations on an ephemeral shadow index, never touching the active graph.

    Each candidate re-ingests the corpus documents into a throwaway runtime built with a different
    co-occurrence graph configuration, evaluates the program there, records structural graph stats,
    and discards the runtime. The active corpus and its graph are never modified, so the winning
    configuration is a recommendation to re-ingest deliberately. Answer-quality F1 over labeled gold
    graphs is a follow-up: the offline corpus has no labeled entities or relations to score against.
    """

    def __init__(self, project: RefaatProject) -> None:
        self._project = project

    def optimize(
        self,
        dataset: Dataset,
        *,
        max_entities: Sequence[int] = (40, 80, 120),
        min_token_lengths: Sequence[int] = (4, 6),
        program_name: str = "grounded-rag",
        bundle: MetricBundle | None = None,
    ) -> GraphResult:
        self._project.runtime.require_principal(None)
        if self._project.runtime.composition.graph_extractor is not None:
            raise ValueError(
                "graph optimization tunes the built-in co-occurrence extractor, so it cannot run with a "
                "custom graph extractor injected; remove it or optimize retrieval instead"
            )
        active_bundle = bundle or default_bundle()
        documents = active_documents(self._project)
        if not documents:
            raise ValueError("The corpus has no documents to optimize graph extraction for")
        configs = [
            (entities, tokens)
            for entities in max_entities
            for tokens in min_token_lengths
            if entities > 0 and tokens > 0
        ]
        if not configs:
            raise ValueError("No valid graph configuration; max entities and min token length must be positive")
        candidates: list[GraphCandidate] = []
        shadow_composition: ShadowComposition | None = None
        for entities, tokens in configs:
            overrides: dict[str, object] = {
                "graph_provider": "cooccurrence",
                "graph_max_entities": entities,
                "graph_min_token_length": tokens,
            }
            with shadow_index(self._project, documents, overrides) as (shadow_project, config_composition):
                # The composition is identical across configs, so capturing the first is enough.
                if shadow_composition is None:
                    shadow_composition = config_composition
                report = shadow_project.programs.get(program_name).evaluate(dataset, active_bundle)
                graph_entities, graph_relations = shadow_project.runtime.store.list_graph(
                    shadow_project.workspace_id, shadow_project.corpus_id
                )
            candidates.append(
                GraphCandidate(
                    max_entities=entities,
                    min_token_length=tokens,
                    entity_count=len(graph_entities),
                    relation_count=len(graph_relations),
                    hard_constraints_passed=report.hard_constraints_passed,
                    score=report.score,
                    mean_tokens=report.mean_tokens,
                )
            )
        ranked = sorted(
            candidates,
            key=lambda candidate: (candidate.hard_constraints_passed, candidate.score, -candidate.mean_tokens),
            reverse=True,
        )
        return GraphResult(
            program=program_name,
            dataset_version=dataset.version,
            candidate_count=len(candidates),
            best=ranked[0],
            candidates=tuple(ranked),
            pareto_front=pareto_front(candidates),
            shadow_composition=shadow_composition,
        )
