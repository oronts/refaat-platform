from __future__ import annotations

import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import httpx
from _shared import bootstrap_workspace, demo_settings, emit, live_api_server

from refaat import RuntimeBuilder
from refaat.domain.enums import SideEffectClass
from refaat.domain.models import ContextPacket, DocumentInput, JsonObject, ReActPolicy, ToolGrant
from refaat.evaluation import Dataset, ProgramEvaluator, trajectory_bundle
from refaat.integrations import (
    as_langchain_retriever,
    as_langchain_tool,
    as_langgraph_node,
    as_langgraph_program_node,
    build_grounded_graph,
    build_react_subgraph,
)
from refaat.optimization import RetrievalOptimizer, retrieval_search_space
from refaat.programs import (
    GovernedReActProgram,
    GroundedRagProgram,
    KeywordToolPlanner,
    StructuredOutputProgram,
    Tunables,
)
from refaat.sdk import RefaatClient
from refaat.skills import ProgramSkill, SkillRegistry

PLAYBOOK_TOOL = "incident_playbook"
DIGEST_TOOL = "evidence_digest"
PAGER_TOOL = "pager_escalation"

RAG_QUESTION = "What does the On-Call Dispatcher do during a Severity One incident on the Cargo Gateway?"
GRAPH_QUESTION = "How do the Platform Team and the Cargo Gateway work together during a Severity One incident?"
INCIDENT_QUESTION = (
    "During a Severity One incident on the Cargo Gateway, what rollback and escalation steps "
    "does the On-Call Dispatcher follow?"
)
STRUCTURED_QUESTION = "Where are Manifest Store secrets stored and how often are they rotated?"


def northwind_documents() -> list[dict[str, str]]:
    """A fabricated Northwind Logistics operations and product knowledge base.

    Recurring named entities (the Cargo Gateway, the Freight Ledger, the Depot Console, the
    Manifest Store, the Vault Service, the On-Call Dispatcher, the Incident Commander, the
    Platform Team, the Reliability Team, the Security Guild, the Rollback Runbook, the
    Escalation Runbook, SKU NW-100, and SKU NW-220) recur across teams, systems, severities,
    and document types so vector, lexical, graph, parent_child, contextual retrieval, and
    citations are all genuinely exercised over one corpus.
    """
    return [
        {
            "source_id": "runbook/rollback.md",
            "team": "Platform Team",
            "system": "Cargo Gateway",
            "severity": "sev1",
            "doc_type": "runbook",
            "title": "Rollback Runbook",
            "text": (
                "# Rollback Runbook\n\n"
                "## Trigger\n"
                "The On-Call Dispatcher starts the Rollback Runbook when the Cargo Gateway error rate "
                "exceeds the alert threshold during a Severity One incident.\n\n"
                "## Steps\n"
                "The On-Call Dispatcher restores the previously approved deployment, verifies the Cargo "
                "Gateway health check, and records the rollback in the Freight Ledger."
            ),
        },
        {
            "source_id": "runbook/escalation.md",
            "team": "Reliability Team",
            "system": "Cargo Gateway",
            "severity": "sev1",
            "doc_type": "runbook",
            "title": "Escalation Runbook",
            "text": (
                "# Escalation Runbook\n\n"
                "## Chain\n"
                "The On-Call Dispatcher runs the Escalation Runbook and pages the Incident Commander when a "
                "rollback does not recover the Cargo Gateway within fifteen minutes.\n\n"
                "## Breach\n"
                "The Incident Commander escalates a suspected breach to the Security Guild immediately."
            ),
        },
        {
            "source_id": "role/on-call-dispatcher.md",
            "team": "Reliability Team",
            "system": "On Call",
            "severity": "sev2",
            "doc_type": "role",
            "title": "On-Call Dispatcher Charter",
            "text": (
                "# On-Call Dispatcher Charter\n\n"
                "## Duties\n"
                "The On-Call Dispatcher holds the pager, runs the Rollback Runbook, and posts an update every "
                "fifteen minutes during a Severity One incident on the Cargo Gateway."
            ),
        },
        {
            "source_id": "role/incident-commander.md",
            "team": "Reliability Team",
            "system": "On Call",
            "severity": "sev1",
            "doc_type": "role",
            "title": "Incident Commander Charter",
            "text": (
                "# Incident Commander Charter\n\n"
                "## Duties\n"
                "The Incident Commander coordinates a Severity One incident, owns the incident bridge, and "
                "decides when to escalate to the Security Guild."
            ),
        },
        {
            "source_id": "service/cargo-gateway.md",
            "team": "Platform Team",
            "system": "Cargo Gateway",
            "severity": "sev1",
            "doc_type": "service",
            "title": "Cargo Gateway Service",
            "text": (
                "# Cargo Gateway Service\n\n"
                "## Role\n"
                "The Cargo Gateway routes carrier traffic and enforces the tenant rate limit. "
                "The Platform Team owns the Cargo Gateway and supports it during a Severity One incident.\n\n"
                "## Recovery\n"
                "A saturated Cargo Gateway is recovered with the Rollback Runbook before any new deploy."
            ),
        },
        {
            "source_id": "service/freight-ledger.md",
            "team": "Platform Team",
            "system": "Freight Ledger",
            "severity": "sev2",
            "doc_type": "service",
            "title": "Freight Ledger Service",
            "text": (
                "# Freight Ledger Service\n\n"
                "## Role\n"
                "The Freight Ledger records every shipment event and the incident log the Incident Commander "
                "reviews during a Severity One incident on the Cargo Gateway."
            ),
        },
        {
            "source_id": "service/depot-console.md",
            "team": "Reliability Team",
            "system": "Depot Console",
            "severity": "sev2",
            "doc_type": "service",
            "title": "Depot Console Service",
            "text": (
                "# Depot Console Service\n\n"
                "## Role\n"
                "The Depot Console renders the dashboards the Reliability Team watches. "
                "The Depot Console reads shipment events from the Freight Ledger."
            ),
        },
        {
            "source_id": "service/manifest-store.md",
            "team": "Security Guild",
            "system": "Manifest Store",
            "severity": "sev1",
            "doc_type": "service",
            "title": "Manifest Store Service",
            "text": (
                "# Manifest Store Service\n\n"
                "## Secrets\n"
                "The Manifest Store keeps carrier secrets in the Vault Service and rotates the keys every "
                "ninety days. The Security Guild grants Manifest Store access only to a named owner."
            ),
        },
        {
            "source_id": "policy/change-freeze.md",
            "team": "Platform Team",
            "system": "Cargo Gateway",
            "severity": "sev2",
            "doc_type": "policy",
            "title": "Change Freeze Policy",
            "text": (
                "# Change Freeze Policy\n\n"
                "## Rule\n"
                "The Platform Team enforces a change freeze on the Cargo Gateway during any active Severity One "
                "incident until the Incident Commander clears it."
            ),
        },
        {
            "source_id": "policy/breach-response.md",
            "team": "Security Guild",
            "system": "Manifest Store",
            "severity": "sev1",
            "doc_type": "policy",
            "title": "Breach Response Policy",
            "text": (
                "# Breach Response Policy\n\n"
                "## Response\n"
                "The Security Guild leads a breach response, rotates every Manifest Store key in the Vault "
                "Service, and briefs the Incident Commander before the incident closes."
            ),
        },
        {
            "source_id": "policy/comms.md",
            "team": "Reliability Team",
            "system": "On Call",
            "severity": "sev1",
            "doc_type": "policy",
            "title": "Incident Communications Policy",
            "text": (
                "# Incident Communications Policy\n\n"
                "## Cadence\n"
                "The On-Call Dispatcher posts an update every fifteen minutes during a Severity One incident and "
                "a final summary when the Cargo Gateway is confirmed healthy."
            ),
        },
        {
            "source_id": "incident/postmortem-gateway.md",
            "team": "Reliability Team",
            "system": "Cargo Gateway",
            "severity": "sev1",
            "doc_type": "incident",
            "title": "Postmortem Gateway Saturation",
            "text": (
                "# Postmortem Gateway Saturation\n\n"
                "## Summary\n"
                "The Cargo Gateway saturated, the On-Call Dispatcher ran the Rollback Runbook, and the Incident "
                "Commander coordinated the Severity One incident.\n\n"
                "## Follow up\n"
                "A corrective action and a named owner are recorded in the Freight Ledger within two business days."
            ),
        },
        {
            "source_id": "incident/postmortem-breach.md",
            "team": "Security Guild",
            "system": "Manifest Store",
            "severity": "sev1",
            "doc_type": "incident",
            "title": "Postmortem Access Breach",
            "text": (
                "# Postmortem Access Breach\n\n"
                "## Summary\n"
                "The Security Guild detected a breach, rotated the Manifest Store keys in the Vault Service, and "
                "the Incident Commander posted an update every fifteen minutes."
            ),
        },
        {
            "source_id": "product/sku-nw-100.md",
            "team": "Warehouse Crew",
            "system": "Depot Console",
            "severity": "sev3",
            "doc_type": "product",
            "title": "SKU NW-100 Pallet Wrapper",
            "text": (
                "# SKU NW-100 Pallet Wrapper\n\n"
                "## Handling\n"
                "SKU NW-100 is the Pallet Wrapper the Warehouse Crew scans into the Depot Console before it "
                "leaves the dock. A damaged SKU NW-100 is logged in the Freight Ledger."
            ),
        },
        {
            "source_id": "product/sku-nw-220.md",
            "team": "Warehouse Crew",
            "system": "Depot Console",
            "severity": "sev3",
            "doc_type": "product",
            "title": "SKU NW-220 Forklift",
            "text": (
                "# SKU NW-220 Forklift\n\n"
                "## Handling\n"
                "SKU NW-220 is the Forklift the Warehouse Crew inspects each shift and records in the Depot "
                "Console. A grounded SKU NW-220 blocks the dock until the Warehouse Crew clears it."
            ),
        },
        {
            "source_id": "policy/data-residency.md",
            "team": "Security Guild",
            "system": "Manifest Store",
            "severity": "sev2",
            "doc_type": "policy",
            "title": "Data Residency Policy",
            "text": (
                "# Data Residency Policy\n\n"
                "## Region\n"
                "Carrier personal data stays inside the European Region only. "
                "The Security Guild audits the Manifest Store residency every ninety days."
            ),
        },
    ]


_HEADING_LINE = re.compile(r"^(#+ .+)$", re.MULTILINE)


def _space_headings(text: str) -> str:
    """Put a blank line after every ATX heading so a heading is its own paragraph.

    The co-occurrence graph extractor treats a paragraph (a ``\\n\\n`` block) as one context
    window and reads capitalized runs across single newlines as one entity. Separating the
    heading from its body keeps entity names clean ("Cargo Gateway", not "Trigger\\nThe Cargo
    Gateway") without changing any sentence text.
    """
    return _HEADING_LINE.sub(lambda match: f"{match.group(1)}\n", text)


def ingest_in_process(project: Any, documents: list[dict[str, str]]) -> None:
    for document in documents:
        project.runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id=document["source_id"],
                title=document["title"],
                media_type="text/markdown",
                content=_space_headings(document["text"]).encode(),
                metadata={
                    "team": document["team"],
                    "system": document["system"],
                    "severity": document["severity"],
                    "doc_type": document["doc_type"],
                },
            ),
        )


def stored_chunk_ids(project: Any) -> set[str]:
    identifiers: set[str] = set()
    for document in project.list_documents():
        for chunk in project.runtime.store.list_document_chunks(
            project.workspace_id, project.corpus_id, str(document["id"])
        ):
            identifiers.add(chunk.id)
    return identifiers


class EvidenceDigestSkill:
    name = DIGEST_TOOL
    description = "Read-only digest of the top retrieved evidence for a governed step."

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del arguments
        return {
            "systems": sorted({str(hit.chunk.metadata.get("system")) for hit in context.hits}),
            "evidence": [hit.chunk.text.split(".", maxsplit=1)[0] for hit in context.hits[:3]],
        }


class PagerEscalationSkill:
    name = PAGER_TOOL
    description = "External paging side effect, correctly denied under a read-only ceiling."

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del context, arguments
        return {"paged": True}


def run_live_service(base_url: str, api_key: str, documents: list[dict[str, str]], report: dict[str, Any]) -> None:
    """Face 1: drive the real HTTP API and SDK, then LangChain and LangGraph over the live client."""
    with RefaatClient(base_url, api_key) as client:
        corpus_id = str(client.create_corpus("northwind-operations")["id"])
        report["corpus_id"] = corpus_id
        for document in documents:
            client.ingest_text(
                corpus_id,
                source_id=document["source_id"],
                title=document["title"],
                content=_space_headings(document["text"]),
                media_type="text/markdown",
                metadata={
                    "team": document["team"],
                    "system": document["system"],
                    "severity": document["severity"],
                    "doc_type": document["doc_type"],
                },
            )

        known = client.retrieve(corpus_id, RAG_QUESTION, top_k=25)
        known_ids = {value.chunk_id for value in known.documents}
        report["retrieve_routes"] = list(known.routes)

        answer = client.query(corpus_id, RAG_QUESTION)
        report["rag_answer"] = answer.answer
        report["rag_citations"] = [str(citation["chunk_id"]) for citation in answer.citations]
        report["rag_citations_resolve"] = bool(answer.citations) and all(
            str(citation["chunk_id"]) in known_ids for citation in answer.citations
        )

        cached = client.query(corpus_id, RAG_QUESTION)
        report["cache_hit"] = bool(cached.cached) and cached.routes[0] == "cag:exact"

        stream_events = list(client.stream_query(corpus_id, INCIDENT_QUESTION))
        stream_final = next(event for event in stream_events if event["type"] == "final")
        report["stream_delta_count"] = sum(1 for event in stream_events if event["type"] == "delta")
        report["stream_grounded"] = bool(stream_final["citations"])

        graph = client.retrieve(corpus_id, GRAPH_QUESTION)
        report["graph_facts"] = list(graph.graph_facts[:5])
        graph_answer = client.query(corpus_id, GRAPH_QUESTION)
        report["graph_answer"] = graph_answer.answer
        report["graph_answer_grounded"] = bool(graph_answer.citations)

        run_program_http(base_url, api_key, corpus_id, report)
        report["langchain_answer"] = run_langchain(client, corpus_id)
        report["langgraph_final"] = run_langgraph(client, corpus_id)


def run_program_http(base_url: str, api_key: str, corpus_id: str, report: dict[str, Any]) -> None:
    """Exercise the program and registry HTTP surface: list, run, evaluate, approve, promote, resolve."""
    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(base_url=base_url, headers=headers, timeout=30, trust_env=False) as http:
        listed = http.get(f"/v1/corpora/{corpus_id}/programs")
        listed.raise_for_status()
        report["http_programs_listed"] = [program["program"] for program in listed.json()]

        run = http.post(
            f"/v1/corpora/{corpus_id}/programs/grounded-rag/run",
            json={"question": RAG_QUESTION},
        )
        run.raise_for_status()
        report["http_program_grounded"] = bool(run.json()["citations"])

        evaluate = http.post(
            f"/v1/corpora/{corpus_id}/programs/grounded-rag/evaluate",
            json={
                "cases": [
                    {"id": "h1", "query": RAG_QUESTION, "expected_terms": ["on-call dispatcher", "rollback runbook"]},
                    {"id": "h2", "query": GRAPH_QUESTION, "expected_terms": ["platform team", "cargo gateway"]},
                ]
            },
        )
        evaluate.raise_for_status()
        report["http_evaluation_score"] = round(float(evaluate.json()["score"]), 4)

        environment = "production"
        approve = http.post(
            f"/v1/corpora/{corpus_id}/programs/grounded-rag/approve",
            json={"environment": environment},
        )
        approve.raise_for_status()
        promote = http.post(
            f"/v1/corpora/{corpus_id}/programs/grounded-rag/promote",
            json={"environment": environment},
        )
        promote.raise_for_status()
        resolved = http.get(f"/v1/programs/grounded-rag/deployments/{environment}")
        resolved.raise_for_status()
        report["http_registry_env"] = environment
        report["http_registry_resolved"] = bool(resolved.json()["definition_hash"])


def _langgraph_missing(exc: RuntimeError) -> bool:
    # The assemblers raise exactly this signal when langgraph is not installed; any
    # other RuntimeError is a real graph-compile failure that must not be swallowed.
    return "Install langgraph" in str(exc)


def run_langchain(client: RefaatClient, corpus_id: str) -> Any:
    # Executes for real wherever langchain is installed (the gate env has it); degrades
    # to a recorded status on a minimal install rather than crashing.
    try:
        retriever = as_langchain_retriever(client, corpus_id, top_k=4)
        tool = as_langchain_tool(client, corpus_id)
    except RuntimeError as exc:
        return {"status": "optional_dependency_missing", "message": str(exc)}
    documents = retriever.invoke("On-Call Dispatcher rollback during a Severity One incident")
    answer = str(tool.invoke({"question": RAG_QUESTION}))
    return {"status": "executed", "documents": len(documents), "answer": answer}


def run_langgraph(client: RefaatClient, corpus_id: str) -> Any:
    # The node primitives are dependency-free, so on a minimal install a grounded answer
    # is still produced; a real branching StateGraph is built and invoked wherever
    # langgraph is installed. The grounded graph genuinely routes retrieve to reason or
    # abstain: a normal question reaches the grounded reason branch, and forcing an
    # unreachable evidence threshold reaches the honest empty-citation abstain branch.
    node = as_langgraph_node(client, corpus_id)
    try:
        reason_graph: Any = build_grounded_graph(client, corpus_id)
    except RuntimeError as exc:
        if not _langgraph_missing(exc):
            raise
        skipped = node({"question": RAG_QUESTION})
        return {
            "status": "optional_dependency_missing",
            "message": str(exc),
            "answer": skipped["answer"],
            "grounded": bool(skipped["citations"]),
        }
    reasoned = reason_graph.invoke({"question": RAG_QUESTION})
    abstained = build_grounded_graph(client, corpus_id, min_documents=10_000).invoke({"question": RAG_QUESTION})
    return {
        "status": "executed",
        "answer": reasoned["answer"],
        "grounded": bool(reasoned["citations"]),
        "reason_branch_cited": bool(reasoned["citations"]),
        "abstain_branch_empty": abstained["answer"] == "" and abstained["citations"] == [],
    }


def run_agentic_core(root: Path, documents: list[dict[str, str]], report: dict[str, Any]) -> None:
    """Face 2: the in-process agentic core over the same deterministic corpus."""
    settings = demo_settings(root).model_copy(update={"reranker_provider": "lexical"})
    with RuntimeBuilder(settings).build() as runtime:
        project = runtime.project(workspace="northwind", workspace_name="Northwind Logistics", corpus="operations")
        ingest_in_process(project, documents)
        identifiers = stored_chunk_ids(project)
        report["core_documents_ingested"] = project.stats()["documents"]

        reranked = project.retrieve(RAG_QUESTION)
        report["core_retrieve_routes"] = list(reranked.routes)

        registry = project.registry
        artifact = GroundedRagProgram().artifact()
        promoted_id = registry.save(artifact)
        registry.approve("grounded-rag", "production", promoted_id)
        registry.promote("grounded-rag", "production", promoted_id)

        playbook = ProgramSkill(
            project.programs.get("grounded-rag"),
            name=PLAYBOOK_TOOL,
            description="Run the promoted grounded incident program as one governed tool.",
        )
        report["core_playbook_artifact"] = playbook.artifact_id
        report["core_playbook_matches_promotion"] = playbook.artifact_id == promoted_id

        skills = SkillRegistry()
        skills.register(playbook)
        skills.register(EvidenceDigestSkill())
        skills.register(PagerEscalationSkill())

        policy = ReActPolicy(
            max_iterations=4,
            granted_scopes=("read",),
            side_effect_ceiling=SideEffectClass.READ_ONLY,
            grants=(
                ToolGrant(PLAYBOOK_TOOL, required_scopes=("read",), side_effect=SideEffectClass.READ_ONLY),
                ToolGrant(DIGEST_TOOL, required_scopes=("read",), side_effect=SideEffectClass.READ_ONLY),
                ToolGrant(PAGER_TOOL, required_scopes=("read",), side_effect=SideEffectClass.EXTERNAL),
            ),
        )
        planner = KeywordToolPlanner(
            {PLAYBOOK_TOOL: ("incident",), DIGEST_TOOL: ("incident",), PAGER_TOOL: ("incident",)}
        )
        react = GovernedReActProgram.from_registry(policy=policy, planner=planner, registry=skills)
        bound_react = project.programs.register(react)
        result = react.execute(project, inputs={"question": INCIDENT_QUESTION}, user_id=None, session_id=None)
        steps = result.outputs["react_steps"]
        report["agent_authorized_tools"] = sorted(step["tool"] for step in steps if step["authorized"])
        report["agent_denied_tools"] = sorted(step["tool"] for step in steps if not step["authorized"])
        report["agent_answer_grounded"] = result.grounded and bool(result.citations)

        try:
            react_subgraph = build_react_subgraph(bound_react)
            trajectory = react_subgraph.invoke({"question": INCIDENT_QUESTION})
        except RuntimeError as exc:
            if not _langgraph_missing(exc):
                raise
            # The program node is dependency-free, so a minimal install still exercises
            # the grounded trajectory without compiling a StateGraph.
            trajectory = as_langgraph_program_node(bound_react)({"question": INCIDENT_QUESTION})
        report["react_subgraph_grounded"] = (
            bool(trajectory["react_steps"])
            and bool(trajectory["citations"])
            and trajectory["grounded"] is True
            and all(citation["chunk_id"] in identifiers for citation in trajectory["citations"])
        )

        structured = StructuredOutputProgram(GroundedRagProgram())
        structured_result = project.programs.register(structured).run(question=STRUCTURED_QUESTION)
        report["structured_valid"] = structured_result.grounded and all(
            citation.chunk_id in identifiers for citation in structured_result.citations
        )

        active_before = project.programs.get("grounded-rag").program.artifact().artifact_id
        optimization = RetrievalOptimizer(project).optimize(
            GroundedRagProgram(),
            Dataset.from_records(
                "northwind-opt",
                [
                    {"id": "o1", "query": RAG_QUESTION, "expected_terms": ["on-call dispatcher"]},
                    {"id": "o2", "query": GRAPH_QUESTION, "expected_terms": ["cargo gateway"]},
                    {"id": "o3", "query": STRUCTURED_QUESTION, "expected_terms": ["vault service"]},
                ],
            ),
            search_space=retrieval_search_space(top_k=(4, 8, 12)),
        )
        active_after = project.programs.get("grounded-rag").program.artifact().artifact_id
        report["optimize_pareto_size"] = len(optimization.pareto_front)
        report["optimize_candidate_count"] = optimization.candidate_count
        report["optimize_no_mutation"] = active_before == active_after

        evaluation = ProgramEvaluator(bound_react).evaluate(
            Dataset.from_records(
                "northwind-trajectory",
                [
                    {"id": "t1", "query": INCIDENT_QUESTION},
                    {"id": "t2", "query": GRAPH_QUESTION},
                    {"id": "t3", "query": RAG_QUESTION},
                ],
            ),
            trajectory_bundle(),
        )
        report["eval_grounded"] = evaluation.aggregate["grounded"]
        report["eval_tool_safety"] = evaluation.aggregate["tool_safety"]
        report["eval_hard_passed"] = evaluation.hard_constraints_passed

        base = GroundedRagProgram().artifact()
        tuned = GroundedRagProgram(tunables=Tunables(top_k=4)).artifact()
        registry.save(base)
        registry.save(tuned)
        registry.promote("grounded-rag", "staging", base.artifact_id)
        registry.promote("grounded-rag", "staging", tuned.artifact_id)
        resolved_tuned = registry.resolve("grounded-rag", "staging")["definition_hash"] == tuned.definition_hash
        rolled = registry.rollback("grounded-rag", "staging")
        resolved_base = registry.resolve("grounded-rag", "staging")["definition_hash"] == base.definition_hash
        report["registry_env"] = rolled["environment"]
        report["registry_promoted"] = resolved_tuned
        report["registry_rolled_back"] = resolved_base and rolled["artifact_id"] == base.artifact_id


def langgraph_carries_answer(value: Any) -> bool:
    # The gate installs langgraph, so this asserts a real StateGraph run there; an
    # optional-dependency skip is accepted only on a minimal install.
    if not isinstance(value, dict) or value.get("status") not in ("executed", "optional_dependency_missing"):
        return False
    return bool(value.get("answer")) and bool(value.get("grounded"))


def langgraph_branches(value: Any) -> bool:
    # Where langgraph is installed, both the reason and abstain branches must have run
    # for real; a minimal-install skip is accepted only on the degraded status.
    if not isinstance(value, dict):
        return False
    if value.get("status") == "optional_dependency_missing":
        return True
    return value.get("reason_branch_cited") is True and value.get("abstain_branch_empty") is True


def langchain_executed(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("status") == "optional_dependency_missing":
        return True
    return value.get("status") == "executed" and int(value.get("documents", 0)) > 0 and bool(value.get("answer"))


def verify(report: dict[str, Any], documents: list[dict[str, str]]) -> None:
    checks: dict[str, bool] = {
        "rag_answer": bool(report["rag_answer"].strip()),
        "rag_citations": bool(report["rag_citations"]),
        "rag_citations_resolve": report["rag_citations_resolve"] is True,
        "retrieve_routes": bool(report["retrieve_routes"]),
        "cache_hit": report["cache_hit"] is True,
        "stream_delta_count": report["stream_delta_count"] > 1,
        "stream_grounded": report["stream_grounded"] is True,
        "graph_facts": len(report["graph_facts"]) > 0,
        "graph_answer": bool(report["graph_answer"].strip()) and report["graph_answer_grounded"] is True,
        "http_programs_listed": "grounded-rag" in report["http_programs_listed"],
        "http_program_grounded": report["http_program_grounded"] is True,
        "http_registry_resolved": report["http_registry_resolved"] is True,
        "langchain_answer": langchain_executed(report["langchain_answer"]),
        "langgraph_final": langgraph_carries_answer(report["langgraph_final"]),
        "langgraph_branches": langgraph_branches(report["langgraph_final"]),
        "core_documents_ingested": report["core_documents_ingested"] == len(documents),
        "core_retrieve_routes": "rerank" in report["core_retrieve_routes"],
        "core_playbook_matches_promotion": report["core_playbook_matches_promotion"] is True,
        "agent_authorized_tools": report["agent_authorized_tools"] == [DIGEST_TOOL, PLAYBOOK_TOOL],
        "agent_denied_tools": report["agent_denied_tools"] == [PAGER_TOOL],
        "agent_answer_grounded": report["agent_answer_grounded"] is True,
        "react_subgraph_grounded": report["react_subgraph_grounded"] is True,
        "structured_valid": report["structured_valid"] is True,
        "optimize_pareto_size": report["optimize_pareto_size"] >= 1,
        "optimize_no_mutation": report["optimize_no_mutation"] is True,
        "eval_grounded": isinstance(report["eval_grounded"], (int, float)),
        "eval_tool_safety": report["eval_tool_safety"] == 1.0,
        "eval_hard_passed": report["eval_hard_passed"] is True,
        "registry_promoted": report["registry_promoted"] is True,
        "registry_rolled_back": report["registry_rolled_back"] is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"Reference application assertions failed: {', '.join(failed)}")


def main() -> None:
    documents = northwind_documents()
    with TemporaryDirectory(prefix="refaat-reference-app-") as directory:
        root = Path(directory)
        report: dict[str, Any] = {}
        api_root = root / "api"
        api_root.mkdir(parents=True, exist_ok=True)
        with live_api_server(api_root) as (base_url, bootstrap_token):
            bootstrap = bootstrap_workspace(base_url, bootstrap_token)
            run_live_service(base_url, bootstrap["api_key"], documents, report)
        core_root = root / "core"
        core_root.mkdir(parents=True, exist_ok=True)
        run_agentic_core(core_root, documents, report)
        verify(report, documents)
        emit(report)


if __name__ == "__main__":
    main()
