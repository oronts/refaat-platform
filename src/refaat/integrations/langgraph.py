from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, TypedDict

from refaat.core.text import sha256_text
from refaat.integrations.base import GroundedClient, GroundedRetrieverClient, RetrieverClient

if TYPE_CHECKING:
    from refaat.programs.registry import BoundProgram


class RefaatGraphState(TypedDict, total=False):
    question: str
    retrieved: list[dict[str, Any]]
    routes: list[str]
    answer: str
    citations: list[dict[str, Any]]
    react_steps: list[dict[str, Any]]
    grounded: bool
    grounded_digest: str


def _answer_binding_digest(answer: str) -> str:
    return sha256_text(answer)


def refaat_grounded(
    state: Mapping[str, Any],
    *,
    answer_key: str = "answer",
    grounded_key: str = "grounded",
    grounded_digest_key: str = "grounded_digest",
) -> bool:
    """True only when the grounded flag and the answer-binding digest match the current answer.

    A host node that rewrites the answer without re-running the grounded pipeline leaves the digest
    stale, so this reads as not grounded: Refaat never labels externally rewritten text as grounded.
    """
    if not state.get(grounded_key):
        return False
    recorded = state.get(grounded_digest_key)
    answer = state.get(answer_key)
    if not isinstance(recorded, str) or not isinstance(answer, str):
        return False
    return _answer_binding_digest(answer) == recorded


_ANSWER_STATE_KEY = "answer"
_CITATIONS_STATE_KEY = "citations"


def _reject_answer_key_collision(protected: Mapping[str, str], data: Mapping[str, str]) -> None:
    """Fail construction when a data key would overwrite the answer or its citations.

    The answer and citation keys carry the grounded, cited output. A branching or
    trajectory data key aliased onto either one would let unverified state replace the
    answer or its citations, so an overlap is rejected at construction rather than
    silently corrupting the grounded result at invoke time.
    """
    reserved = set(protected.values())
    for label, key in data.items():
        if key in reserved:
            collides = next(name for name, value in protected.items() if value == key)
            raise ValueError(f"{label} must not alias the {collides} state key: {key!r}")


def as_langgraph_node(
    client: GroundedClient,
    corpus_id: str,
    *,
    question_key: str = "question",
    answer_key: str = "answer",
    citations_key: str = "citations",
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Build a LangGraph-compatible node that answers from a Refaat corpus.

    A LangGraph node is a plain callable that receives the graph state and returns a
    partial state update, so this needs no LangGraph dependency to construct. Add the
    returned callable to a StateGraph with add_node.
    """

    def refaat_node(state: Mapping[str, Any]) -> dict[str, Any]:
        answer = client.query(corpus_id, str(state[question_key]))
        return {answer_key: answer.answer, citations_key: list(answer.citations)}

    return refaat_node


def as_langgraph_retrieve_node(
    client: RetrieverClient,
    corpus_id: str,
    *,
    top_k: int = 8,
    question_key: str = "question",
    retrieved_key: str = "retrieved",
    routes_key: str = "routes",
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Build a node that retrieves branching evidence without answering.

    The node writes the retrieved chunks and the retrieval routes so a router can
    branch on the evidence. It never writes an answer or a citation: only the reason
    node (client.query) or a program node (BoundProgram.run) produces those, through
    the grounded pipeline, so retrieved evidence can never leak into a citation.
    """
    _reject_answer_key_collision(
        {"answer_key": _ANSWER_STATE_KEY, "citations_key": _CITATIONS_STATE_KEY},
        {"retrieved_key": retrieved_key, "routes_key": routes_key},
    )

    def refaat_retrieve_node(state: Mapping[str, Any]) -> dict[str, Any]:
        result = client.retrieve(corpus_id, str(state[question_key]), top_k=top_k)
        return {
            retrieved_key: [
                {
                    "chunk_id": document.chunk_id,
                    "document_id": document.document_id,
                    "text": document.text,
                    "score": document.score,
                    "metadata": dict(document.metadata),
                    "reasons": list(document.reasons),
                }
                for document in result.documents
            ],
            routes_key: list(result.routes),
        }

    return refaat_retrieve_node


def refaat_evidence_router(
    state: Mapping[str, Any],
    *,
    retrieved_key: str = "retrieved",
    min_documents: int = 1,
    sufficient_label: str = "reason",
    insufficient_label: str = "abstain",
) -> str:
    """Route to the reason branch when enough evidence was retrieved, else abstain."""

    if min_documents < 1:
        raise ValueError("min_documents must be at least 1 so the abstain branch stays reachable")
    documents = state.get(retrieved_key) or ()
    return sufficient_label if len(documents) >= min_documents else insufficient_label


def as_langgraph_abstain_node(
    *,
    answer_key: str = "answer",
    citations_key: str = "citations",
    message: str = "",
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Build an honest terminal node that abstains with an empty citation list."""
    if answer_key == citations_key:
        raise ValueError(f"answer_key and citations_key must differ: {answer_key!r}")

    def refaat_abstain_node(state: Mapping[str, Any]) -> dict[str, Any]:
        del state
        return {answer_key: message, citations_key: []}

    return refaat_abstain_node


def as_langgraph_program_node(
    program: BoundProgram,
    *,
    question_key: str = "question",
    answer_key: str = "answer",
    citations_key: str = "citations",
    steps_key: str = "react_steps",
    grounded_key: str = "grounded",
    grounded_digest_key: str = "grounded_digest",
    user_id: str | None = None,
    session_id: str | None = None,
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Build a node that runs a governed program and surfaces its trajectory.

    The answer and citations come only from BoundProgram.run, so they route through
    the same grounded pipeline as client.query; the governance trajectory becomes
    graph-observable state for a downstream router.
    """
    _reject_answer_key_collision(
        {"answer_key": answer_key, "citations_key": citations_key},
        {"steps_key": steps_key, "grounded_key": grounded_key, "grounded_digest_key": grounded_digest_key},
    )

    def refaat_program_node(state: Mapping[str, Any]) -> dict[str, Any]:
        result = program.run(question=str(state[question_key]), user_id=user_id, session_id=session_id)
        return {
            answer_key: result.answer,
            citations_key: [
                {"chunk_id": citation.chunk_id, "document_id": citation.document_id, "label": citation.label}
                for citation in result.citations
            ],
            steps_key: list(result.outputs.get("react_steps", [])),
            grounded_key: result.grounded,
            grounded_digest_key: _answer_binding_digest(result.answer),
        }

    return refaat_program_node


def refaat_trajectory_router(
    state: Mapping[str, Any],
    *,
    steps_key: str = "react_steps",
    authorized_label: str = "done",
    review_label: str = "review",
) -> str:
    """Route to review when any recorded governance step was not authorized."""

    steps = state.get(steps_key) or ()
    if any(not step.get("authorized", True) for step in steps):
        return review_label
    return authorized_label


def build_grounded_graph(
    client: GroundedRetrieverClient,
    corpus_id: str,
    *,
    top_k: int = 8,
    min_documents: int = 1,
) -> Any:
    """Assemble a retrieve then reason-or-abstain state machine and compile it.

    The graph branches on retrieved evidence: enough documents route to the reason
    node (a grounded client.query answer), too few route to an honest abstain node
    with an empty citation list. This is the only place besides build_react_subgraph
    that imports langgraph.
    """
    if min_documents < 1:
        raise ValueError("min_documents must be at least 1 so the abstain branch stays reachable")

    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("Install langgraph for the LangGraph state-machine integration") from exc

    graph: Any = StateGraph(RefaatGraphState)
    graph.add_node("retrieve", as_langgraph_retrieve_node(client, corpus_id, top_k=top_k))
    graph.add_node("reason", as_langgraph_node(client, corpus_id))
    graph.add_node("abstain", as_langgraph_abstain_node())
    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        partial(refaat_evidence_router, min_documents=min_documents),
        {"reason": "reason", "abstain": "abstain"},
    )
    graph.add_edge("reason", END)
    graph.add_edge("abstain", END)
    return graph.compile()


def build_react_subgraph(program: BoundProgram, *, review: bool = False) -> Any:
    """Assemble a governed-program subgraph and compile it for in-process composition.

    A compiled graph is usable as a node in a larger StateGraph. With review enabled
    the trajectory router splits an authorized run from one carrying an unauthorized
    step, routing the latter to a review terminal that preserves the grounded answer.
    """

    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("Install langgraph for the LangGraph state-machine integration") from exc

    graph: Any = StateGraph(RefaatGraphState)
    graph.add_node("react", as_langgraph_program_node(program))
    graph.add_edge(START, "react")
    if review:

        def refaat_review_node(state: Mapping[str, Any]) -> dict[str, Any]:
            del state
            return {}

        graph.add_node("review", refaat_review_node)
        graph.add_conditional_edges("react", refaat_trajectory_router, {"done": END, "review": "review"})
        graph.add_edge("review", END)
    else:
        graph.add_edge("react", END)
    return graph.compile()
