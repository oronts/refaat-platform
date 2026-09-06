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
class ChunkingCandidate:
    chunk_size: int
    overlap: int
    strategy: str
    hard_constraints_passed: bool
    score: float
    mean_tokens: float

    def to_json(self) -> JsonObject:
        return {
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "strategy": self.strategy,
            "hard_constraints_passed": self.hard_constraints_passed,
            "score": self.score,
            "mean_tokens": self.mean_tokens,
        }


@dataclass(frozen=True, slots=True)
class ChunkingResult:
    program: str
    dataset_version: str
    candidate_count: int
    best: ChunkingCandidate
    candidates: tuple[ChunkingCandidate, ...]
    pareto_front: tuple[ChunkingCandidate, ...]
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


class ChunkingOptimizer:
    """Evaluate chunking configurations on an ephemeral shadow index, never touching the active corpus.

    Each candidate re-ingests the corpus documents into a throwaway runtime built with a different
    chunk size and overlap, evaluates the program there, and discards the runtime. The active corpus
    and its index are never modified, so the winning configuration is a recommendation to re-ingest
    deliberately rather than an in-place mutation.
    """

    def __init__(self, project: RefaatProject) -> None:
        self._project = project

    def optimize(
        self,
        dataset: Dataset,
        *,
        chunk_sizes: Sequence[int] = (256, 420, 600),
        overlaps: Sequence[int] = (32, 64),
        strategies: Sequence[str] = ("flat",),
        program_name: str = "grounded-rag",
        bundle: MetricBundle | None = None,
    ) -> ChunkingResult:
        self._project.runtime.require_principal(None)
        if self._project.runtime.composition.chunking_strategy is not None:
            raise ValueError(
                "chunking optimization tunes the built-in chunking strategies, so it cannot run with a "
                "custom chunking strategy injected; remove it to sweep chunk size and overlap"
            )
        active_bundle = bundle or default_bundle()
        documents = active_documents(self._project)
        if not documents:
            raise ValueError("The corpus has no documents to optimize chunking for")
        configs = [
            (size, overlap, strategy)
            for size in chunk_sizes
            for overlap in overlaps
            for strategy in strategies
            if overlap < size
        ]
        if not configs:
            raise ValueError("No valid chunking configuration; overlap must be smaller than chunk size")
        candidates: list[ChunkingCandidate] = []
        shadow_composition: ShadowComposition | None = None
        for size, overlap, strategy in configs:
            with shadow_index(
                self._project,
                documents,
                {"default_chunk_tokens": size, "chunk_overlap_tokens": overlap, "chunking_strategy": strategy},
            ) as (shadow_project, config_composition):
                # The composition is identical across configs, so capturing the first is enough.
                if shadow_composition is None:
                    shadow_composition = config_composition
                report = shadow_project.programs.get(program_name).evaluate(dataset, active_bundle)
            candidates.append(
                ChunkingCandidate(
                    chunk_size=size,
                    overlap=overlap,
                    strategy=strategy,
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
        return ChunkingResult(
            program=program_name,
            dataset_version=dataset.version,
            candidate_count=len(candidates),
            best=ranked[0],
            candidates=tuple(ranked),
            pareto_front=pareto_front(candidates),
            shadow_composition=shadow_composition,
        )
