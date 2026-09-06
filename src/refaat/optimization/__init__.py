from __future__ import annotations

from refaat.application.composition import OmittedComponent, ReproducedComponent, ShadowComposition
from refaat.core.router import RouterConfig
from refaat.optimization.bootstrap import (
    BOOTSTRAP,
    RANDOM_SEARCH,
    BootstrapFewShotOptimizer,
    BootstrapTrace,
    bootstrap_demonstrations,
    random_demonstration_subsets,
)
from refaat.optimization.candidates import (
    Candidate,
    OptimizationResult,
    assemble_result,
    build_ranked_result,
    evaluate_candidate,
)
from refaat.optimization.chunking import ChunkingCandidate, ChunkingOptimizer, ChunkingResult
from refaat.optimization.durable import resume_optimization_run
from refaat.optimization.fewshot import LabeledFewShotOptimizer, select_demonstrations
from refaat.optimization.graph import GraphCandidate, GraphOptimizer, GraphResult
from refaat.optimization.instructions import (
    InstructionOptimizer,
    propose_instructions,
    reflect_instruction,
)
from refaat.optimization.optimizer import RetrievalOptimizer
from refaat.optimization.router import RouterOptimizer, default_router_grid
from refaat.optimization.search import (
    context_packing_search_space,
    default_search_space,
    fusion_search_space,
    retrieval_search_space,
)
from refaat.optimization.telemetry import ShadowComparison, compare_results

__all__ = [
    "BOOTSTRAP",
    "RANDOM_SEARCH",
    "BootstrapFewShotOptimizer",
    "BootstrapTrace",
    "Candidate",
    "ChunkingCandidate",
    "ChunkingOptimizer",
    "ChunkingResult",
    "GraphCandidate",
    "GraphOptimizer",
    "GraphResult",
    "InstructionOptimizer",
    "LabeledFewShotOptimizer",
    "OmittedComponent",
    "OptimizationResult",
    "ReproducedComponent",
    "RetrievalOptimizer",
    "RouterConfig",
    "RouterOptimizer",
    "ShadowComparison",
    "ShadowComposition",
    "assemble_result",
    "bootstrap_demonstrations",
    "build_ranked_result",
    "compare_results",
    "context_packing_search_space",
    "default_router_grid",
    "default_search_space",
    "evaluate_candidate",
    "fusion_search_space",
    "propose_instructions",
    "random_demonstration_subsets",
    "reflect_instruction",
    "resume_optimization_run",
    "retrieval_search_space",
    "select_demonstrations",
]
