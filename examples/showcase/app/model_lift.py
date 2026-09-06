"""Model-lift seams exercised offline through clearly labeled deterministic stand-ins.

Four refaat capabilities only reach their full quality with a real model: the
cross-encoder reranker, model-driven function calling, streaming tool calls, and the
LLM-as-judge metric. Offline the deterministic stack cannot honestly produce a model
result, so each capability is exercised at its real seam by injecting a deterministic
stand-in, exactly as the shipped unit tests do. The stand-in proves the machinery and
the path run; the quality lift itself activates only when a real model is configured.
Nothing here fabricates a model answer or claims a lift a real model would provide.

Every runner asserts a discriminating effect: an outcome that would flip if the seam
were a no-op. The reranker moves a target chunk from last to first, the governed tool
is proposed and admitted and executed and its observation is fed back, the streaming
path emits tool-call fragments before a completed step, and the judge verdict gates a
candidate that would otherwise pass. The citation contract holds throughout: every
answer cites only real stored chunks.

The last runner exercises the shipped LangChain and LlamaIndex adapters for real over
the in-process grounded backend, so both run the actual integration code, not the
missing-dependency fallback.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from refaat import Refaat, RuntimeBuilder
from refaat.domain.enums import SideEffectClass
from refaat.domain.models import (
    ContextPacket,
    DocumentInput,
    JsonObject,
    JudgeVerdict,
    ReActPolicy,
    ToolCallDelta,
    ToolGrant,
    ToolProposal,
    ToolSpec,
)
from refaat.evaluation import Dataset, ModelJudge, ProgramEvaluator, judge_bundle
from refaat.evaluation.bundle import MetricBundle
from refaat.evaluation.metrics import AnswerContains, CitationRecall, Grounded, TokenF1
from refaat.integrations import as_langchain_retriever, as_langchain_tool, as_llamaindex_retriever
from refaat.programs import (
    GovernedReActProgram,
    GroundedRagProgram,
    ModelToolPlanner,
    StreamingModelToolPlanner,
    build_tool_manifest,
)

from .agents import InProcessGroundedClient
from .config import ShowcaseConfig

_CHUNK_ID = re.compile(r"chunk_[0-9a-f]+")

_TOOL_ARGS: JsonObject = {
    "type": "object",
    "properties": {"topic": {"type": "string"}},
    "required": ["topic"],
    "additionalProperties": False,
}


def _isolated_settings(config: ShowcaseConfig, directory: Path, **overrides: Any) -> Any:
    directory.mkdir(parents=True, exist_ok=True)
    update: dict[str, Any] = {
        "database_url": f"sqlite+pysqlite:///{directory / 'refaat.db'}",
        "blob_root": directory / "blobs",
        "connector_file_root": directory / "connectors",
        "skill_root": directory / "skills",
        "hype_enabled": False,
        "raptor_enabled": False,
        "contextual_retrieval": False,
    }
    update.update(overrides)
    return config.settings.model_copy(update=update)


def _stored_chunk_ids(runtime: Refaat, project: Any) -> set[str]:
    return {chunk.id for chunk in runtime.store.list_chunks(project.workspace_id, project.corpus_id)}


class KeywordCrossEncoderStandIn:
    """Deterministic stand-in for a real cross-encoder reranker.

    A trained cross-encoder scores each candidate passage against the query. Offline
    there is no model, so this scores the passage that answers the rollback query at
    the top and every other candidate at a small positive relevance, which is enough
    to exercise the reordering seam. It mirrors the keyword scorer in the shipped
    cross-encoder unit tests and is injected through the CrossEncoderScorer port.
    """

    keyword = "rollback"

    @property
    def model_id(self) -> str:
        return "stand-in-cross-encoder-v1"

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        del query
        return [1.0 if self.keyword in document.lower() else 0.1 for document in documents]


_CE_DOCS = (
    ("alpha.txt", "Peer review is mandatory before any production deployment and staging sign off."),
    ("beta.txt", "A production deployment follows staging sign off and a change review window."),
    ("gamma.txt", "Rollback restores the previously approved artifact atomically after a failed change."),
    ("delta.txt", "The change review window covers staging sign off and production deployment gating."),
)
_CE_QUERY = "how does a production deployment staging review window work"
_CE_TARGET = "gamma.txt"


def _cross_encoder_order(config: ShowcaseConfig, directory: Path, provider: str) -> tuple[list[str], list[str], bool]:
    builder = RuntimeBuilder(_isolated_settings(config, directory, reranker_provider=provider))
    if provider == "cross_encoder":
        builder.with_cross_encoder_scorer(KeywordCrossEncoderStandIn())
    runtime = builder.build()
    try:
        project = runtime.project(workspace="oronts", corpus="cross-encoder")
        for source_id, body in _CE_DOCS:
            runtime.ingest(
                project.workspace_id,
                project.corpus_id,
                DocumentInput(source_id=source_id, title=source_id, media_type="text/plain", content=body.encode()),
            )
        context = project.retrieve(_CE_QUERY)
        answer = project.ask(_CE_QUERY)
        stored = _stored_chunk_ids(runtime, project)
        order = [str(hit.chunk.metadata.get("source_id")) for hit in context.hits]
        resolves = bool(answer.citations) and all(citation.chunk_id in stored for citation in answer.citations)
        return order, list(context.routes), resolves
    finally:
        runtime.close()


def run_cross_encoder(config: ShowcaseConfig, root: Path) -> dict[str, Any]:
    """Exercise the cross-encoder reranker seam and prove it reorders the fused list.

    The bi-encoder baseline ranks the rollback passage last; the injected stand-in lifts
    it to the top and the rerank route fires. A no-op scorer would leave the baseline
    order and the target chunk at the bottom.
    """
    flat_order, flat_routes, _ = _cross_encoder_order(config, root / "cross-encoder-off", "none")
    rerank_order, rerank_routes, resolves = _cross_encoder_order(config, root / "cross-encoder-on", "cross_encoder")
    flat_target_index = flat_order.index(_CE_TARGET) if _CE_TARGET in flat_order else -1
    reordered = flat_order[0] != rerank_order[0]
    ok = (
        flat_order[0] != _CE_TARGET
        and flat_target_index > 0
        and rerank_order[0] == _CE_TARGET
        and reordered
        and "rerank" in rerank_routes
        and "rerank" not in flat_routes
        and resolves
    )
    return {
        "flat_top": flat_order[0] if flat_order else "",
        "flat_target_index": flat_target_index,
        "rerank_top": rerank_order[0] if rerank_order else "",
        "reordered": reordered,
        "rerank_route": "rerank" in rerank_routes,
        "baseline_no_rerank_route": "rerank" not in flat_routes,
        "citations_resolve": resolves,
        "ok": ok,
    }


class _RecordingSkill:
    """A read-only tool that records the arguments it was invoked with."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"Read-only {name} over the retrieved Oronts evidence."
        self.parameters = _TOOL_ARGS
        self.calls: list[JsonObject] = []

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del context
        self.calls.append(arguments)
        return {"echoed": arguments}


class _PagerSkill:
    """An external paging side effect, correctly denied under a read-only ceiling."""

    name = "pager_escalation"
    description = "External paging side effect."

    def __init__(self) -> None:
        self.calls = 0

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del context, arguments
        self.calls += 1
        return {"paged": True}


class ScriptedToolCallGenerator:
    """Deterministic stand-in for a function-calling model.

    A real model reads the tool manifest and emits the next tool call with arguments.
    Offline this replays a fixed queue of proposals, so the governed loop runs its real
    validate, authorize, and execute body against a model-shaped plan without a model.
    Injected through ModelToolPlanner at the ToolCallingGenerator seam.
    """

    def __init__(self, proposals: Sequence[ToolProposal]) -> None:
        self._queue = list(proposals)

    @property
    def model_id(self) -> str:
        return "stand-in-tool-model-v1"

    def propose_tool_call(self, query: str, tools: Sequence[ToolSpec], used: frozenset[str]) -> ToolProposal | None:
        del query, tools, used
        return self._queue.pop(0) if self._queue else None


class ScriptedStreamingToolGenerator:
    """Deterministic stand-in for a streaming function-calling model.

    A real streaming model emits tool-call fragments incrementally before the loop can
    know the full call. Offline this replays fixed fragment scripts, so the streaming
    reducer, the completion boundary, and the governed execution all run without a
    model. Injected through StreamingModelToolPlanner at the streaming seam.
    """

    def __init__(self, scripts: Sequence[Sequence[ToolCallDelta]]) -> None:
        self._scripts = [list(script) for script in scripts]

    @property
    def model_id(self) -> str:
        return "stand-in-tool-model-v1"

    def stream_tool_call(self, query: str, tools: Sequence[ToolSpec], used: frozenset[str]) -> Iterator[ToolCallDelta]:
        del query, tools, used
        if self._scripts:
            yield from self._scripts.pop(0)


_TOOL_DOC = b"Rollback restores the previously approved harness snapshot before deployment."
_TOOL_QUESTION = "how does rollback work"
_LOOKUP_TOOL = "evidence_lookup"
_PAGER_TOOL = "pager_escalation"


def _tool_policy() -> ReActPolicy:
    return ReActPolicy(
        max_iterations=4,
        granted_scopes=("read", "write:crm"),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(
            ToolGrant(_LOOKUP_TOOL, required_scopes=("read",), side_effect=SideEffectClass.READ_ONLY),
            ToolGrant(_PAGER_TOOL, required_scopes=("write:crm",), side_effect=SideEffectClass.EXTERNAL),
        ),
    )


def _tool_skills() -> dict[str, Any]:
    return {_LOOKUP_TOOL: _RecordingSkill(_LOOKUP_TOOL), _PAGER_TOOL: _PagerSkill()}


def _seed_tool_corpus(config: ShowcaseConfig, directory: Path) -> Refaat:
    runtime = RuntimeBuilder(_isolated_settings(config, directory)).build()
    project = runtime.project(workspace="oronts", corpus="toolcall")
    runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(source_id="ops.txt", title="Operations", media_type="text/plain", content=_TOOL_DOC),
    )
    return runtime


def run_model_function_calling(config: ShowcaseConfig, root: Path) -> dict[str, Any]:
    """Drive the governed ReAct loop with a model-shaped tool planner stand-in.

    The admitted read-only tool is proposed with arguments, validated, authorized, and
    executed, and its observation is fed back into the step record. The external pager is
    proposed but denied by the read-only ceiling and never executes. The final answer is
    grounded. A no-op planner would propose nothing and no tool step would appear.
    """
    runtime = _seed_tool_corpus(config, root / "model-function-calling")
    try:
        project = runtime.project(workspace="oronts", corpus="toolcall")
        skills = _tool_skills()
        policy = _tool_policy()
        manifest = build_tool_manifest(skills, policy.grants)
        generator = ScriptedToolCallGenerator(
            [
                ToolProposal(tool=_LOOKUP_TOOL, arguments={"topic": "rollback"}),
                ToolProposal(tool=_PAGER_TOOL, arguments={"topic": "rollback"}),
            ]
        )
        program = GovernedReActProgram(policy=policy, planner=ModelToolPlanner(generator, manifest), skills=skills)
        result = program.execute(project, inputs={"question": _TOOL_QUESTION}, user_id=None, session_id=None)
        steps = {step["tool"]: step for step in result.outputs["react_steps"]}
        stored = _stored_chunk_ids(runtime, project)
        lookup = skills[_LOOKUP_TOOL]
        pager = skills[_PAGER_TOOL]
        assert isinstance(lookup, _RecordingSkill)
        assert isinstance(pager, _PagerSkill)
        grounded = (
            result.grounded
            and bool(result.citations)
            and all(citation.chunk_id in stored for citation in result.citations)
        )
        lookup_step = steps.get(_LOOKUP_TOOL, {})
        pager_step = steps.get(_PAGER_TOOL, {})
        tool_executed = (
            lookup_step.get("authorized") is True
            and lookup_step.get("arguments") == {"topic": "rollback"}
            and lookup.calls == [{"topic": "rollback"}]
        )
        observation_present = lookup_step.get("observation") == {"echoed": {"topic": "rollback"}}
        denied_external = (
            pager_step.get("authorized") is False
            and "exceeds ceiling" in str(pager_step.get("reason", ""))
            and pager.calls == 0
        )
        return {
            "tool_executed": tool_executed,
            "observation_present": observation_present,
            "denied_external": denied_external,
            "grounded": grounded,
            "ok": tool_executed and observation_present and denied_external and grounded,
        }
    finally:
        runtime.close()


def run_streaming_tool_calls(config: ShowcaseConfig, root: Path) -> dict[str, Any]:
    """Drive the streaming governed ReAct loop with a streaming tool planner stand-in.

    Tool-call fragments stream before the first answer delta, then exactly one completed
    tool step is emitted for the finished call. A separate incomplete script that never
    terminates executes nothing, proving a partial argument is never run. The streamed
    answer is byte-identical to execute() and folds into the same answer cache, so the
    streaming path is a faithful sibling of the non-streaming one.
    """
    runtime = _seed_tool_corpus(config, root / "streaming-tool-calls")
    try:
        project = runtime.project(workspace="oronts", corpus="toolcall")
        policy = _tool_policy()

        complete_skills = _tool_skills()
        complete_manifest = build_tool_manifest(complete_skills, policy.grants)
        complete_script = [
            ToolCallDelta(tool=_LOOKUP_TOOL),
            ToolCallDelta(arguments_fragment='{"topic": '),
            ToolCallDelta(arguments_fragment='"rollback"}'),
            ToolCallDelta(done=True),
        ]
        streaming = GovernedReActProgram(
            policy=policy,
            planner=StreamingModelToolPlanner(ScriptedStreamingToolGenerator([complete_script]), complete_manifest),
            skills=complete_skills,
        )
        events = list(streaming.stream(project, inputs={"question": _TOOL_QUESTION}, user_id=None, session_id=None))
        types = [event["type"] for event in events]
        deltas = [event for event in events if event["type"] == "tool_call_delta"]
        steps = [event for event in events if event["type"] == "tool_step"]
        streamed_answer = "".join(event["text"] for event in events if event["type"] == "delta")
        delta_before_answer = (
            "tool_call_delta" in types and "delta" in types and types.index("tool_call_delta") < types.index("delta")
        )

        incomplete_skills = _tool_skills()
        incomplete_manifest = build_tool_manifest(incomplete_skills, policy.grants)
        incomplete_script = [
            ToolCallDelta(tool=_LOOKUP_TOOL),
            ToolCallDelta(arguments_fragment='{"topic": "rollback"}'),
        ]
        incomplete = GovernedReActProgram(
            policy=policy,
            planner=StreamingModelToolPlanner(ScriptedStreamingToolGenerator([incomplete_script]), incomplete_manifest),
            skills=incomplete_skills,
        )
        incomplete_events = list(
            incomplete.stream(project, inputs={"question": _TOOL_QUESTION}, user_id=None, session_id=None)
        )
        incomplete_lookup = incomplete_skills[_LOOKUP_TOOL]
        assert isinstance(incomplete_lookup, _RecordingSkill)
        partial_not_executed = (
            not [event for event in incomplete_events if event["type"] == "tool_step"] and incomplete_lookup.calls == []
        )

        executed = GroundedRagProgram().execute(
            project, inputs={"question": _TOOL_QUESTION}, user_id=None, session_id=None
        )
        follow_up = GroundedRagProgram().execute(
            project, inputs={"question": _TOOL_QUESTION}, user_id=None, session_id=None
        )
        streamed_matches_execute = streamed_answer == executed.answer
        folds_into_cache = follow_up.cached is True and "cag:exact" in follow_up.routes

        streamed_step_ok = [step["tool"] for step in steps] == [_LOOKUP_TOOL] and steps[0]["authorized"] is True
        return {
            "delta_before_answer": delta_before_answer,
            "delta_count": len(deltas),
            "tool_step_count": len(steps),
            "streamed_step_ok": streamed_step_ok,
            "partial_not_executed": partial_not_executed,
            "streamed_matches_execute": streamed_matches_execute,
            "folds_into_cache": folds_into_cache,
            "ok": (
                delta_before_answer
                and len(deltas) > 1
                and streamed_step_ok
                and partial_not_executed
                and streamed_matches_execute
                and folds_into_cache
            ),
        }
    finally:
        runtime.close()


class ScriptedJudge:
    """Deterministic stand-in for an LLM-as-judge.

    A real judge reads the answer and its cited evidence and returns a bounded support
    verdict. Offline this returns a high, passing verdict when the answer contains the
    sentinel term the judge was scripted to require and a low, failing verdict otherwise,
    so the judge decision is real and the metric machinery runs without a model. Injected
    through the JudgeModel seam of the ModelJudge metric.
    """

    def __init__(self, sentinel: str) -> None:
        self._sentinel = sentinel
        self.model_id = "stand-in-judge-v1"

    def assess(self, query: str, answer: str, evidence: Sequence[str]) -> JudgeVerdict:
        del query, evidence
        supported = self._sentinel in answer.lower()
        return JudgeVerdict(
            score=0.9 if supported else 0.2,
            passed=supported,
            rationale="stand-in judge verdict",
        )


def _judge_hard_bundle(judge: ScriptedJudge) -> MetricBundle:
    """The grounded-rag metrics with the model judge promoted to a hard constraint."""
    return MetricBundle(
        name="grounded-rag-judge-hard",
        metrics=(Grounded(), AnswerContains(), CitationRecall(), TokenF1(), ModelJudge(judge=judge)),
        hard_constraints=frozenset({"grounded", "model_judge"}),
        objectives={"answer_contains": 0.3, "citation_recall": 0.2, "token_f1": 0.2, "model_judge": 0.3},
    )


_JUDGE_STRONG_DOC = b"Rollback restores the previously approved artifact atomically after a failed change."
_JUDGE_WEAK_DOC = b"Peer review is required before any production deployment and staging sign off."
_JUDGE_STRONG_QUERY = "how does rollback restore the prior artifact"
_JUDGE_WEAK_QUERY = "what is the peer review policy"


def run_llm_judge(config: ShowcaseConfig, root: Path) -> dict[str, Any]:
    """Run the model-judge metric inside a real evaluation and prove it gates a candidate.

    Two candidates run on their own single-document corpus so the answer sentinel stays
    clean. With the judge as a hard constraint the strong candidate passes and the weak
    one is gated out, so candidate selection picks the strong one. A control run with the
    same judge as a plain objective leaves the weak candidate ungated, which isolates the
    judge verdict as the cause: a no-op judge would gate neither.
    """
    runtime = RuntimeBuilder(_isolated_settings(config, root / "llm-judge")).build()
    try:
        strong_project = runtime.project(workspace="oronts", corpus="judge-strong")
        runtime.ingest(
            strong_project.workspace_id,
            strong_project.corpus_id,
            DocumentInput(
                source_id="rollback.txt", title="Rollback", media_type="text/plain", content=_JUDGE_STRONG_DOC
            ),
        )
        weak_project = runtime.project(workspace="oronts", corpus="judge-weak")
        runtime.ingest(
            weak_project.workspace_id,
            weak_project.corpus_id,
            DocumentInput(source_id="review.txt", title="Review", media_type="text/plain", content=_JUDGE_WEAK_DOC),
        )
        strong_program = strong_project.programs.get("grounded-rag")
        weak_program = weak_project.programs.get("grounded-rag")
        strong_dataset = Dataset.from_records(
            "judge-strong", [{"id": "s1", "query": _JUDGE_STRONG_QUERY, "expected_terms": ["rollback"]}]
        )
        weak_dataset = Dataset.from_records(
            "judge-weak", [{"id": "w1", "query": _JUDGE_WEAK_QUERY, "expected_terms": ["review"]}]
        )

        judge = ScriptedJudge("rollback")
        strong = ProgramEvaluator(strong_program).evaluate(strong_dataset, _judge_hard_bundle(judge))
        weak = ProgramEvaluator(weak_program).evaluate(weak_dataset, _judge_hard_bundle(judge))
        control_weak = ProgramEvaluator(weak_program).evaluate(weak_dataset, judge_bundle(judge))

        candidates = [("strong", strong), ("weak", weak)]
        selected = max(candidates, key=lambda item: (item[1].hard_constraints_passed, item[1].score))[0]

        strong_score = strong.aggregate["model_judge"]
        weak_score = weak.aggregate["model_judge"]
        bounded = all(0.0 <= score <= 1.0 for score in (strong_score, weak_score))
        return {
            "strong_score": strong_score,
            "weak_score": weak_score,
            "bounded": bounded,
            "in_aggregate": "model_judge" in strong.aggregate and "model_judge" in weak.aggregate,
            "weak_gated": weak.hard_constraints_passed is False,
            "strong_passed": strong.hard_constraints_passed is True,
            "objective_control_not_gated": control_weak.hard_constraints_passed is True,
            "selected": selected,
            "ok": (
                bounded
                and strong_score > weak_score
                and strong.hard_constraints_passed is True
                and weak.hard_constraints_passed is False
                and control_weak.hard_constraints_passed is True
                and selected == "strong"
            ),
        }
    finally:
        runtime.close()


_INTEGRATION_QUERY = "How does OGuardAI protect sensitive data before it reaches an LLM provider?"


def run_real_integrations(project: Any, chunk_ids: set[str]) -> dict[str, Any]:
    """Exercise the LangChain and LlamaIndex adapters for real over the grounded backend.

    Both adapters run their actual integration code in process, not the missing-dependency
    fallback. The LangChain retriever returns real documents whose chunk ids resolve to
    stored chunks, the LangChain grounded tool answers through the query pipeline and its
    returned text carries resolving citation ids, and the LlamaIndex retriever returns
    nodes whose ids resolve. A stubbed or no-op adapter would carry no resolving id.
    """
    client = InProcessGroundedClient(project)

    retriever = as_langchain_retriever(client, "in-process", top_k=4)
    documents = retriever.invoke(_INTEGRATION_QUERY)
    langchain_doc_ids = [document.metadata.get("refaat_chunk_id") for document in documents]
    langchain_docs_resolve = bool(langchain_doc_ids) and all(
        str(chunk_id) in chunk_ids for chunk_id in langchain_doc_ids
    )

    tool = as_langchain_tool(client, "in-process")
    tool_text = str(tool.invoke({"question": _INTEGRATION_QUERY}))
    tool_cited = _CHUNK_ID.findall(tool_text)
    langchain_tool_grounded = (
        "Citations:" in tool_text and bool(tool_cited) and all(chunk_id in chunk_ids for chunk_id in tool_cited)
    )

    llamaindex_retriever = as_llamaindex_retriever(client, "in-process", top_k=4)
    nodes = llamaindex_retriever.retrieve(_INTEGRATION_QUERY)
    llamaindex_node_ids = [node.node.id_ for node in nodes]
    llamaindex_nodes_resolve = (
        bool(llamaindex_node_ids)
        and all(str(node_id) in chunk_ids for node_id in llamaindex_node_ids)
        and all(node.score is not None for node in nodes)
    )

    return {
        "langchain_documents": len(documents),
        "langchain_docs_resolve": langchain_docs_resolve,
        "langchain_tool_grounded": langchain_tool_grounded,
        "llamaindex_nodes": len(nodes),
        "llamaindex_nodes_resolve": llamaindex_nodes_resolve,
        "ok": langchain_docs_resolve and langchain_tool_grounded and llamaindex_nodes_resolve,
    }


__all__ = [
    "KeywordCrossEncoderStandIn",
    "ScriptedJudge",
    "ScriptedStreamingToolGenerator",
    "ScriptedToolCallGenerator",
    "run_cross_encoder",
    "run_llm_judge",
    "run_model_function_calling",
    "run_real_integrations",
    "run_streaming_tool_calls",
]
