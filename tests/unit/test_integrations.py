from __future__ import annotations

from typing import Any

import pytest

from refaat.domain.models import Citation
from refaat.integrations import (
    RefaatRetriever,
    as_crewai_tool,
    as_langgraph_abstain_node,
    as_langgraph_node,
    as_langgraph_program_node,
    as_langgraph_retrieve_node,
    refaat_evidence_router,
    refaat_trajectory_router,
)
from refaat.integrations.langchain import as_langchain_retriever
from refaat.integrations.llamaindex import as_llamaindex_retriever
from refaat.programs.program import ProgramResult, ValidationOutcome
from refaat.sdk.models import Answer, RetrievalResult, RetrievedDocument


def retrieval_result() -> RetrievalResult:
    return RetrievalResult(
        query="release",
        documents=(
            RetrievedDocument(
                chunk_id="chunk_1",
                document_id="document_1",
                text="Production changes require peer review.",
                score=0.91,
                metadata={"section": "release"},
                reasons=("semantic",),
            ),
        ),
        routes=("semantic",),
        graph_facts=(),
        token_count=7,
    )


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def retrieve(self, corpus_id: str, query: str, **options: object) -> RetrievalResult:
        self.calls.append((corpus_id, query, options))
        return retrieval_result()


def test_framework_neutral_retriever_and_crewai_tool() -> None:
    client = FakeClient()
    retriever = RefaatRetriever(client=client, corpus_id="corpus_1", top_k=5)

    assert retriever("release") == retrieval_result()
    tool = as_crewai_tool(client, "corpus_1", top_k=3)  # type: ignore[arg-type]
    output = tool("policy")

    assert tool.__name__ == "retrieve_refaat_knowledge"
    assert "[chunk:chunk_1]" in output
    assert client.calls == [
        ("corpus_1", "release", {"top_k": 5}),
        ("corpus_1", "policy", {"top_k": 3}),
    ]


class _FakeProgram:
    def __init__(self, result: ProgramResult) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []

    def run(self, *, user_id: str | None = None, session_id: str | None = None, **inputs: Any) -> ProgramResult:
        self.calls.append({"user_id": user_id, "session_id": session_id, **inputs})
        return self._result


def _program_result(citations: tuple[Citation, ...], react_steps: list[dict[str, Any]]) -> ProgramResult:
    return ProgramResult(
        program="governed-react",
        artifact_id="artifact-1",
        answer="grounded answer",
        citations=citations,
        routes=("vector",),
        cached=False,
        corpus_version=1,
        trace_id="trace-1",
        outputs={"answer": "grounded answer", "react_steps": react_steps},
        validations=(ValidationOutcome(validator="grounding", passed=True, detail="1 citation(s)"),),
    )


class _LeakGuardClient:
    def retrieve(self, corpus_id: str, query: str, **options: object) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            documents=(
                RetrievedDocument(chunk_id="A", document_id="da", text="a", score=0.9),
                RetrievedDocument(chunk_id="B", document_id="db", text="b", score=0.8),
            ),
            routes=("semantic",),
            graph_facts=(),
            token_count=2,
        )

    def query(self, corpus_id: str, query: str, **options: object) -> Answer:
        return Answer(
            answer="grounded",
            citations=({"chunk_id": "C", "document_id": "dc", "label": "[1]", "quote": "q"},),
            routes=("vector",),
            cached=False,
            corpus_version=1,
            trace_id="trace",
            usage={},
        )


def test_retrieve_node_writes_evidence_and_routes_without_answering() -> None:
    client = FakeClient()
    node = as_langgraph_retrieve_node(client, "corpus_1", top_k=5)

    state = node({"question": "release"})

    assert state["retrieved"][0]["chunk_id"] == "chunk_1"
    assert state["retrieved"][0]["document_id"] == "document_1"
    assert state["routes"] == ["semantic"]
    assert "answer" not in state
    assert "citations" not in state
    assert client.calls == [("corpus_1", "release", {"top_k": 5})]


@pytest.mark.parametrize(
    ("documents", "min_documents", "expected"),
    [([], 1, "abstain"), ([{"chunk_id": "chunk_1"}], 1, "reason"), ([{"chunk_id": "chunk_1"}], 2, "abstain")],
)
def test_evidence_router_branches_on_document_count(
    documents: list[dict[str, str]], min_documents: int, expected: str
) -> None:
    assert refaat_evidence_router({"retrieved": documents}, min_documents=min_documents) == expected


def test_abstain_node_emits_empty_citations() -> None:
    node = as_langgraph_abstain_node(message="no supporting evidence")

    assert node({"question": "x"}) == {"answer": "no supporting evidence", "citations": []}


def test_program_node_lands_answer_citations_steps_grounded() -> None:
    citations = (Citation(chunk_id="chunk_9", document_id="doc_9", label="[1]", quote="q"),)
    steps = [{"tool": "digest", "authorized": True}]
    program = _FakeProgram(_program_result(citations, steps))
    node = as_langgraph_program_node(program, user_id="u1")  # type: ignore[arg-type]

    state = node({"question": "how?"})

    assert state["answer"] == "grounded answer"
    assert state["citations"] == [{"chunk_id": "chunk_9", "document_id": "doc_9", "label": "[1]"}]
    assert state["react_steps"] == steps
    assert state["grounded"] is True
    assert program.calls == [{"user_id": "u1", "session_id": None, "question": "how?"}]


def test_refaat_grounded_is_bound_to_the_answer_text() -> None:
    from refaat.integrations.langgraph import refaat_grounded

    citations = (Citation(chunk_id="chunk_9", document_id="doc_9", label="[1]", quote="q"),)
    program = _FakeProgram(_program_result(citations, []))
    state = as_langgraph_program_node(program)({"question": "how?"})  # type: ignore[arg-type]

    assert refaat_grounded(state) is True
    # A host node that rewrites the answer without re-running the pipeline reads as not grounded.
    assert refaat_grounded({**state, "answer": "externally rewritten"}) is False
    # A false grounded flag or a missing digest also reads as not grounded.
    assert refaat_grounded({**state, "grounded": False}) is False
    assert refaat_grounded({k: v for k, v in state.items() if k != "grounded_digest"}) is False


@pytest.mark.parametrize(
    ("steps", "expected"),
    [
        ([{"authorized": True}, {"authorized": True}], "done"),
        ([{"authorized": True}, {"authorized": False}], "review"),
        ([], "done"),
    ],
)
def test_trajectory_router_flags_unauthorized_steps(steps: list[dict[str, bool]], expected: str) -> None:
    assert refaat_trajectory_router({"react_steps": steps}) == expected


def test_nodes_honor_custom_state_keys() -> None:
    client = FakeClient()
    retrieve = as_langgraph_retrieve_node(client, "c", question_key="q", retrieved_key="ev", routes_key="paths")
    retrieved = retrieve({"q": "release"})
    assert set(retrieved) == {"ev", "paths"}

    program = _FakeProgram(_program_result((), [{"authorized": True}]))
    node = as_langgraph_program_node(
        program,  # type: ignore[arg-type]
        question_key="q",
        answer_key="a",
        citations_key="cites",
        steps_key="trace",
        grounded_key="ok",
        grounded_digest_key="okdigest",
    )
    assert set(node({"q": "hi"})) == {"a", "cites", "trace", "ok", "okdigest"}


def test_retrieve_node_rejects_data_key_aliasing_answer_or_citations() -> None:
    client = FakeClient()
    with pytest.raises(ValueError, match="retrieved_key"):
        as_langgraph_retrieve_node(client, "c", retrieved_key="citations")
    with pytest.raises(ValueError, match="routes_key"):
        as_langgraph_retrieve_node(client, "c", routes_key="answer")


def test_program_node_rejects_data_key_aliasing_answer_or_citations() -> None:
    program = _FakeProgram(_program_result((), [{"authorized": True}]))
    with pytest.raises(ValueError, match="steps_key"):
        as_langgraph_program_node(program, steps_key="citations")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="grounded_key"):
        as_langgraph_program_node(program, grounded_key="answer")  # type: ignore[arg-type]


def test_abstain_node_rejects_answer_key_equal_to_citations_key() -> None:
    with pytest.raises(ValueError, match="answer_key and citations_key must differ"):
        as_langgraph_abstain_node(answer_key="citations", citations_key="citations")


def test_evidence_router_rejects_non_positive_min_documents() -> None:
    with pytest.raises(ValueError, match="min_documents must be at least 1"):
        refaat_evidence_router({"retrieved": []}, min_documents=0)


def test_retrieved_evidence_never_leaks_into_citations() -> None:
    client = _LeakGuardClient()
    retrieve = as_langgraph_retrieve_node(client, "corpus")  # type: ignore[arg-type]
    reason = as_langgraph_node(client, "corpus")  # type: ignore[arg-type]

    state: dict[str, Any] = {"question": "who verifies releases?"}
    state.update(retrieve(state))
    assert [document["chunk_id"] for document in state["retrieved"]] == ["A", "B"]

    state.update(reason(state))
    assert [citation["chunk_id"] for citation in state["citations"]] == ["C"]


def test_optional_framework_errors_are_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    client: Any = FakeClient()
    real_import = __import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> Any:
        if name.startswith(("langchain_core", "llama_index")):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked_import)
    with pytest.raises(RuntimeError, match=r"refaat\[integrations\].*LangChain"):
        as_langchain_retriever(client, "corpus_1")
    with pytest.raises(RuntimeError, match=r"refaat\[integrations\].*LlamaIndex"):
        as_llamaindex_retriever(client, "corpus_1")
