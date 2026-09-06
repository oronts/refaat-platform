from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from _shared import emit
from fastapi.testclient import TestClient
from showcase.app.agents import agent_layer_grounded, run_agent_layer
from showcase.app.api import QUERY_MODES
from showcase.app.config import load_config
from showcase.app.ingest import IMAGE_QUERY, _png_bytes, ingest_image_asset, ingest_knowledge, open_project
from showcase.app.knowledge import PRODUCT_CATEGORY, load_documents
from showcase.app.main import create_app
from showcase.app.mcp import mcp_layer_ok, run_mcp_layer
from showcase.app.model_lift import (
    run_cross_encoder,
    run_llm_judge,
    run_model_function_calling,
    run_real_integrations,
    run_streaming_tool_calls,
)
from showcase.app.programs import (
    build_governed_react,
    governed_fine_tuning,
    program_layer_grounded,
    run_program_layer,
)
from showcase.app.retrieval import (
    ShowcaseRetrieval,
    _code_policy,
    _governed_requests,
    citations_resolve,
    stored_chunk_ids,
)
from showcase.app.runtime import build_runtime

from refaat import RuntimeBuilder
from refaat.application.services import ABSTAIN_MESSAGE
from refaat.domain.models import DocumentInput
from refaat.domain.sandbox import CodeRequest
from refaat.infrastructure.sandbox import SafeExpressionSandbox

MULTI_HOP = "How do OGuardAI and PimTx fit into Oronts GmbH under Refaat Al Ktifan?"
PRODUCT_QUESTION = "Which Oronts products exist and what does each one do?"
PRINCIPLES_QUESTION = "What engineering principles and framework native patterns does Oronts GmbH follow?"
OFFLINE_QUESTION = "What is the 90-Day Production Pilot and what does it deliver?"

# Offline the deterministic transforms for sub-query, HyDE, and step-back actively add a
# retrieval variant; multi-query and query rewriting are deterministic identity offline
# and light up only when a real model writes the alternative phrasings.
TRANSFORM_ACTIVE_OFFLINE = ("sub_query", "hyde", "step_back")
TRANSFORM_MODEL_ACTIVATED = ("multi_query", "query_rewrite")


def run_primary_modes(config: Any, report: dict[str, Any], citation_ok: list[bool]) -> None:
    runtime = RuntimeBuilder(config.settings).build()
    try:
        project = open_project(runtime)
        report["documents_ingested"] = ingest_knowledge(project) or project.stats()["documents"]
        svc = ShowcaseRetrieval(project)

        plain = svc.plain_rag("What does Oronts GmbH build and what does it promise clients?")
        report["plain_rag_grounded"] = plain.grounded
        citation_ok.append(plain.grounded)

        cache = svc.cache_hit("Who founded Oronts GmbH and where is it based?")
        report["cache_hit"] = bool(cache.extra["exact"]) and not cache.extra["first_cached"]

        graph = svc.graphrag(MULTI_HOP)
        report["graphrag_ok"] = bool(graph.extra["graph_route"]) and len(graph.extra["graph_facts"]) > 0
        report["graph_facts_sample"] = graph.extra["graph_facts"]

        kag = svc.kag(MULTI_HOP)
        report["kag_grounded"] = kag.grounded and bool(kag.extra["graph_route"])
        citation_ok.append(kag.grounded)

        features = svc.feature_routes(MULTI_HOP)
        report["feature_routes"] = {
            "hype": bool(features.extra["hype_route"]),
            "raptor": bool(features.extra["raptor_route"]),
            "rerank": bool(features.extra["rerank_route"]),
        }

        reranked = svc.reranked(OFFLINE_QUESTION)
        report["reranked_ok"] = bool(reranked.extra["rerank_route"])

        autocut = svc.autocut("What does OGuardAI protect and how does it work?")
        report["autocut_ok"] = bool(autocut.extra["autocut_route"])

        crag = svc.corrective_rag("How does Oronts GmbH guarantee clients own and control their code?")
        report["crag_grounded"] = crag.grounded and bool(crag.extra["assessment_present"])
        citation_ok.append(crag.grounded)

        self_rag = svc.self_rag("Where is Oronts GmbH based and who leads the company?")
        report["self_rag_grounded"] = self_rag.grounded and bool(self_rag.extra["reflection_present"])
        citation_ok.append(self_rag.grounded)

        rlm = svc.rlm("How do OGuardAI, PimTx, and the Oronts AI Assistant fit together at Oronts GmbH?")
        report["rlm_grounded"] = rlm.grounded
        report["rlm_nodes"] = int(rlm.extra["nodes"])
        citation_ok.append(rlm.grounded)

        run_program_and_agent_layers(project, config, report, citation_ok)
        run_scoped_modes(runtime, report)
    finally:
        runtime.close()


def run_program_and_agent_layers(project: Any, config: Any, report: dict[str, Any], citation_ok: list[bool]) -> None:
    """Drive the program, agent, and MCP layers over the already-ingested corpus.

    All three reuse the primary project so the whole layer runs on one deterministic
    corpus. The grounded flags feed the shared citation-contract check, so an agent or a
    program that fabricated a citation would fail the same invariant as plain RAG.
    """
    chunk_ids = stored_chunk_ids(project)
    bound_react = build_governed_react(project)

    programs = run_program_layer(project, config, chunk_ids, bound_react)
    report["programs"] = programs
    report["program_layer_grounded"] = program_layer_grounded(programs)
    report["registry_register_not_promote"] = programs["retrieval_optimization"]["no_mutation"] is True
    report["registry_rollback_atomic"] = programs["registry_lifecycle"]["rolled_back"] is True
    report["denied_tool_recorded"] = programs["governed_react"]["denied"] == ["pager_escalation"]
    report["pareto_non_empty"] = programs["retrieval_optimization"]["pareto_size"] >= 1
    report["durable_optimization_ok"] = (
        programs["durable_optimization"]["status"] == "succeeded"
        and programs["durable_optimization"]["matches_synchronous"] is True
    )
    report["program_evaluation_ok"] = (
        programs["program_evaluation"]["hard_passed"] is True
        and programs["program_evaluation"]["cases"] > 0
        and bool(programs["program_evaluation"]["grounded"])
    )
    report["model_cache_budget_ok"] = (
        programs["model_cache_and_budget"]["byte_identical"] is True
        and programs["model_cache_and_budget"]["budget_active"] is True
        and programs["model_cache_and_budget"]["within_budget"] is True
        and programs["model_cache_and_budget"]["budget_enforces"] is True
    )
    report["shadow_canary_ok"] = (
        programs["shadow_canary"]["comparison_present"] is True
        and programs["shadow_canary"]["shadow_recorded"] is True
        and programs["shadow_canary"]["served"] == "active"
    )
    report["multi_agent_routes_differ"] = programs["multi_agent"]["routes_differ"] is True
    report["program_as_skill_ok"] = (
        programs["program_as_skill"]["skill_matches_promotion"] is True
        and programs["program_as_skill"]["trajectory_hard_passed"] is True
        and programs["program_as_skill"]["trajectory_tool_safety"] == 1.0
    )
    report["optimizer_output_registered"] = (
        programs["durable_optimization"]["registered_best_program"] == "grounded-rag"
        and programs["durable_optimization"]["best_promoted_to_any_env"] is False
    )
    citation_ok.append(report["program_layer_grounded"])

    agents = run_agent_layer(project, bound_react, chunk_ids)
    report["agents"] = agents
    report["agent_layer_grounded"] = agent_layer_grounded(agents)
    citation_ok.append(report["agent_layer_grounded"])

    mcp = run_mcp_layer(project, chunk_ids)
    report["mcp"] = mcp
    report["mcp_layer_ok"] = mcp_layer_ok(mcp)
    citation_ok.append(mcp["consume"]["grounded"])

    real_integrations = run_real_integrations(project, chunk_ids)
    report["real_integrations"] = real_integrations
    report["real_integrations_ok"] = real_integrations["ok"]
    citation_ok.append(real_integrations["ok"])

    fine_tuning = governed_fine_tuning(project)
    report["fine_tuning"] = fine_tuning
    report["fine_tuning_ok"] = (
        fine_tuning["refused_without_approval"] is True
        and fine_tuning["run_status"] == "succeeded"
        and fine_tuning["model_ref"].startswith("ft:refaat:")
        and fine_tuning["provenance_trained"] is False
        and bool(fine_tuning["provenance_dataset_version"])
        and fine_tuning["artifact_registered"] is True
        and fine_tuning["not_auto_promoted"] is True
    )


def run_scoped_modes(runtime: Any, report: dict[str, Any]) -> None:
    """Metadata hard-filter and context compression on dedicated corpora.

    Both are corpus-level policies applied through an active harness snapshot, so
    each runs on its own corpus to leave the primary corpus and its answer cache
    untouched.
    """
    filtered = runtime.project(workspace="oronts", corpus="oronts-products")
    ingest_knowledge(filtered)
    filtered.create_harness_snapshot(
        supplemental={"retrieval": {"filters": {"category": PRODUCT_CATEGORY}}},
        evidence={"reason": "scope retrieval to product documents"},
        activate=True,
    )
    scoped = ShowcaseRetrieval(filtered).metadata_filtered(PRODUCT_QUESTION, PRODUCT_CATEGORY)
    report["metadata_filter_ok"] = bool(scoped.extra["filter_route"]) and bool(scoped.extra["only_category"])

    compress = runtime.project(workspace="oronts", corpus="oronts-compress")
    ingest_knowledge(compress)
    baseline = compress.retrieve(PRINCIPLES_QUESTION)
    compress.create_harness_snapshot(
        supplemental={"retrieval": {"compression_enabled": True, "compression_min_salience": 0.1}},
        evidence={"reason": "prune low-salience sentences and cross-chunk duplicates"},
        activate=True,
    )
    compressed = compress.retrieve(PRINCIPLES_QUESTION)
    report["compression_tokens_dropped"] = baseline.token_count - compressed.token_count
    report["compression_route"] = "compress" in compressed.routes


def run_transform_catalog(config: Any, root: Path, report: dict[str, Any]) -> None:
    transforms: dict[str, dict[str, Any]] = {}
    for provider in (*TRANSFORM_ACTIVE_OFFLINE, *TRANSFORM_MODEL_ACTIVATED):
        directory = root / f"transform-{provider}"
        directory.mkdir(parents=True, exist_ok=True)
        settings = config.settings.model_copy(
            update={
                "query_transform_provider": provider,
                "hype_enabled": False,
                "raptor_enabled": False,
                "database_url": f"sqlite+pysqlite:///{directory / 'refaat.db'}",
                "blob_root": directory / "blobs",
                "connector_file_root": directory / "connectors",
                "skill_root": directory / "skills",
            }
        )
        with RuntimeBuilder(settings).build() as runtime:
            project = open_project(runtime)
            ingest_knowledge(project)
            result = ShowcaseRetrieval(project).transform(
                "How does OGuardAI tokenize sensitive data, and how does PimTx keep Pimcore edits safe?",
                provider,
            )
            transforms[provider] = {
                "expansion_route": bool(result.extra["expansion_route"]),
                "grounded": result.grounded,
            }
    report["transforms"] = transforms
    report["transforms_active_offline_ok"] = all(
        transforms[name]["expansion_route"] for name in TRANSFORM_ACTIVE_OFFLINE
    )
    report["transforms_model_activated_ok"] = all(
        transforms[name]["grounded"] and not transforms[name]["expansion_route"] for name in TRANSFORM_MODEL_ACTIVATED
    )


def run_hierarchical(config: Any, root: Path, report: dict[str, Any]) -> None:
    results: dict[str, bool] = {}
    for strategy in ("parent_child", "sentence_window"):
        directory = root / f"hierarchy-{strategy}"
        directory.mkdir(parents=True, exist_ok=True)
        settings = config.settings.model_copy(
            update={
                "chunking_strategy": strategy,
                "hype_enabled": False,
                "raptor_enabled": False,
                "database_url": f"sqlite+pysqlite:///{directory / 'refaat.db'}",
                "blob_root": directory / "blobs",
                "connector_file_root": directory / "connectors",
                "skill_root": directory / "skills",
            }
        )
        with RuntimeBuilder(settings).build() as runtime:
            project = open_project(runtime)
            ingest_knowledge(project)
            identifiers = stored_chunk_ids(project)
            answer = project.ask(OFFLINE_QUESTION)
            results[strategy] = "expand" in answer.routes and citations_resolve(answer.citations, identifiers)
    report["hierarchical"] = results
    report["hierarchical_ok"] = all(results.values())


CONTEXTUAL_QUERY = "basalt codename brief report"
CONTEXTUAL_TARGET = "brief/basalt.md"


def _contextual_documents() -> list[DocumentInput]:
    """Briefs whose match depends on the title, not the body.

    Each codename lives only in the title, never in the body, so a flat body-only
    embedding cannot see it. Contextual retrieval prefixes the title before embedding,
    so the codename in the query lands on the right brief. The decoy token "polar"
    sits only in the Aurora body, which pulls a flat baseline toward a different brief
    and proves a real ranking difference rather than a no-op flag.
    """
    briefs = [
        (
            "brief/aurora.md",
            "Aurora Codename Brief",
            "The report describes polar sky glow and solar wind across a long winter.",
        ),
        (
            "brief/basalt.md",
            "Basalt Codename Brief",
            "The report covers cooled lava formations and columnar jointing across ancient flows.",
        ),
        (
            "brief/cinnamon.md",
            "Cinnamon Codename Brief",
            "The report explains bark harvesting, drying, and grinding for culinary use.",
        ),
    ]
    return [
        DocumentInput(
            source_id=source_id,
            title=title,
            media_type="text/markdown",
            content=text.encode("utf-8"),
            metadata={"category": "brief"},
        )
        for source_id, title, text in briefs
    ]


def run_contextual(config: Any, root: Path, report: dict[str, Any], citation_ok: list[bool]) -> None:
    """Prove contextual retrieval genuinely changes the result over identical bodies.

    Each codename lives only in the title, so a flat baseline (body-only embedding)
    ranks a different brief first, while the contextual run (title prefixed before
    embedding) lifts the target to the top. The stored and cited chunk text is
    identical across both runs by design, so the gate targets the differing ranking,
    not merely that the contextual answer still grounds. Kept deterministic and offline
    with the hash embedder.
    """
    documents = _contextual_documents()

    def _run(directory: Path, flag: bool) -> tuple[list[str], Any, set[str]]:
        directory.mkdir(parents=True, exist_ok=True)
        settings = config.settings.model_copy(
            update={
                "contextual_retrieval": flag,
                "hype_enabled": False,
                "raptor_enabled": False,
                "database_url": f"sqlite+pysqlite:///{directory / 'refaat.db'}",
                "blob_root": directory / "blobs",
                "connector_file_root": directory / "connectors",
                "skill_root": directory / "skills",
            }
        )
        with RuntimeBuilder(settings).build() as runtime:
            project = runtime.project(workspace="showcase-contextual", corpus="briefs")
            for document in documents:
                project.runtime.ingest(project.workspace_id, project.corpus_id, document)
            identifiers = stored_chunk_ids(project)
            context = project.retrieve(CONTEXTUAL_QUERY)
            sources = [str(hit.chunk.metadata.get("source_id")) for hit in context.hits]
            answer = project.ask(CONTEXTUAL_QUERY)
            return sources, answer, identifiers

    flat_sources, _, _ = _run(root / "contextual-off", False)
    contextual_sources, answer, identifiers = _run(root / "contextual-on", True)

    resolved = citations_resolve(answer.citations, identifiers)
    citation_ok.append(resolved)
    report["contextual_flat_top"] = flat_sources[0] if flat_sources else ""
    report["contextual_top"] = contextual_sources[0] if contextual_sources else ""
    report["contextual_grounded"] = resolved
    report["contextual_effect"] = (
        bool(flat_sources)
        and bool(contextual_sources)
        and flat_sources[0] != CONTEXTUAL_TARGET
        and contextual_sources[0] == CONTEXTUAL_TARGET
        and contextual_sources != flat_sources
        and resolved
    )


ABSTAIN_QUESTION = "What is Google and how does its search ranking algorithm work?"


def run_abstain(config: Any, root: Path, report: dict[str, Any]) -> None:
    """Prove the opt-in abstention path offline and deterministically.

    Enabled with the maximum floor (1.0), every real hit for a non-identical query is
    strictly below it, so the query short circuits to an honest no-answer with no
    citations and a distinguishable abstain route, never reaching the extractive
    fallback that would stitch the nearest chunks into a confident wrong answer. With
    the default offline config (abstain disabled) the same query takes the normal
    grounded flow unchanged, which is the byte-identical disabled path. Offline uses
    hash embeddings, so the 1.01 floor is the deterministic forcing device; the
    showcase enables a real cosine floor only on the online path.
    """
    enabled_dir = root / "abstain-on"
    disabled_dir = root / "abstain-off"
    enabled_dir.mkdir(parents=True, exist_ok=True)
    disabled_dir.mkdir(parents=True, exist_ok=True)

    def _settings(directory: Path, floor: float) -> Any:
        return config.settings.model_copy(
            update={
                "abstain_below_score": floor,
                "database_url": f"sqlite+pysqlite:///{directory / 'refaat.db'}",
                "blob_root": directory / "blobs",
                "connector_file_root": directory / "connectors",
                "skill_root": directory / "skills",
            }
        )

    with RuntimeBuilder(_settings(disabled_dir, 0.0)).build() as runtime:
        project = open_project(runtime)
        ingest_knowledge(project)
        grounded = project.ask(ABSTAIN_QUESTION)

    with RuntimeBuilder(_settings(enabled_dir, 1.0)).build() as runtime:
        project = open_project(runtime)
        ingest_knowledge(project)
        abstained = project.ask(ABSTAIN_QUESTION)

    report["abstain_disabled_unchanged"] = grounded.abstained is False and "abstain" not in grounded.routes
    report["abstain"] = {
        "abstained": abstained.abstained,
        "citations": len(abstained.citations),
        "route": "abstain" in abstained.routes,
        "message_ok": abstained.answer == ABSTAIN_MESSAGE,
        "not_stitch": abstained.answer != grounded.answer,
    }
    report["abstain_ok"] = (
        abstained.abstained is True
        and len(abstained.citations) == 0
        and "abstain" in abstained.routes
        and abstained.answer == ABSTAIN_MESSAGE
        and abstained.answer != grounded.answer
        and report["abstain_disabled_unchanged"] is True
    )


def run_multimodal(config: Any, root: Path, report: dict[str, Any], citation_ok: list[bool]) -> None:
    """Ingest a deterministic in-process image as one citable descriptor chunk.

    The image lands on its own corpus so the primary document-count gates stay honest.
    Only the descriptor built from the caption and title is embedded and citable; the raw
    PNG bytes are stored as the chunk image reference and never appear in a citation, which
    the raw-bytes-not-in-quote check proves.
    """
    del config
    mm_config = load_config(root / "multimodal")
    with RuntimeBuilder(mm_config.settings).build() as runtime:
        project = runtime.project(workspace="oronts", corpus="assets")
        ingest = ingest_image_asset(project)
        descriptor = runtime.store.list_chunks(project.workspace_id, project.corpus_id)[0]
        answer = project.ask(IMAGE_QUERY)
        cited = [citation.chunk_id for citation in answer.citations]
        identifiers = stored_chunk_ids(project)
        resolves = citations_resolve(answer.citations, identifiers)
        raw = _png_bytes(220, 20, 60).decode("latin-1")
        raw_not_cited = all(raw not in citation.quote for citation in answer.citations)
        image_cited = bool(answer.citations) and all(cid == descriptor.id for cid in cited)
        report["multimodal"] = {
            "descriptor_text": descriptor.text,
            "image_ref_present": bool(descriptor.image_ref),
            "image_chunk_cited": image_cited,
            "citation_resolves": resolves,
            "raw_image_not_cited": raw_not_cited,
        }
        report["multimodal_ok"] = (
            ingest.changed
            and ingest.chunks == 1
            and bool(descriptor.image_ref)
            and descriptor.media_type == "image/png"
            and image_cited
            and resolves
            and raw_not_cited
        )
    citation_ok.append(report["multimodal_ok"])


SANDBOX_QUERY = OFFLINE_QUESTION


def run_sandboxed_rlm(config: Any, root: Path, report: dict[str, Any]) -> None:
    """Deny-by-default RLM code, then a governed SafeExpressionSandbox computation.

    The default runtime installs no sandbox, so a code request is refused and nothing
    executes: the answer and citations are byte-identical to the same RLM run with no code
    requests, which is the honesty claim. The opt-in runtime installs the pure-AST
    SafeExpressionSandbox, so a governed compute step runs, the next step feeds a prior
    result forward, an over-ceiling request is denied, and an escape attempt is refused.
    """
    del config
    with RuntimeBuilder(load_config(root / "sandbox-deny").settings).build() as runtime:
        project = open_project(runtime)
        ingest_knowledge(project)
        baseline = project.rlm(SANDBOX_QUERY)
        denied = project.rlm(
            SANDBOX_QUERY,
            code_policy=_code_policy(),
            code_requests=(CodeRequest(code="a + b", inputs={"a": 1, "b": 2}, required_scopes=("compute",)),),
        )
        deny_step = denied.usage["code_steps"][0]

    safe_runtime = build_runtime(load_config(root / "sandbox-safe"), code_sandbox=SafeExpressionSandbox())
    try:
        project = open_project(safe_runtime)
        ingest_knowledge(project)
        governed = ShowcaseRetrieval(project).governed_code_rlm(SANDBOX_QUERY, _code_policy(), _governed_requests())
    finally:
        safe_runtime.close()
    steps = governed.extra["steps"]

    report["sandboxed_rlm"] = {
        "default_deny_answer_identical": denied.answer == baseline.answer,
        "default_deny_sandbox": denied.usage["code_sandbox"],
        "default_deny_step_denied": deny_step["admitted"] is False and deny_step["ok"] is False,
        "default_deny_nothing_executed": deny_step["value"] is None,
        "sandbox_name": governed.extra["sandbox"],
        "governed_value": steps[0]["value"],
        "fed_forward_value": steps[1]["value"],
        "over_ceiling_denied": steps[2]["admitted"] is False,
        "escape_refused": steps[3]["admitted"] is True and steps[3]["ok"] is False,
        "citations_present": bool(governed.citations),
    }
    report["sandboxed_rlm_ok"] = (
        denied.answer == baseline.answer
        and denied.citations == baseline.citations
        and deny_step["admitted"] is False
        and deny_step["ok"] is False
        and deny_step["value"] is None
        and denied.usage["code_sandbox"] == "deny"
        and governed.extra["sandbox"] == "safe_expression"
        and steps[0]["value"] == 5
        and steps[0]["admitted"]
        and steps[0]["ok"]
        and steps[1]["value"] == 50
        and steps[2]["admitted"] is False
        and "exceeds ceiling" in steps[2]["reason"]
        and steps[3]["admitted"] is True
        and steps[3]["ok"] is False
        and len(steps) == 4
        and bool(governed.citations)
    )


def run_model_lift(config: Any, root: Path, report: dict[str, Any], citation_ok: list[bool]) -> None:
    """Exercise the four model-lift seams offline through deterministic stand-ins.

    Each seam runs on its own isolated runtime so the primary corpus and its document
    counts stay untouched. Every runner asserts a discriminating effect that would flip
    if the seam were a no-op, and each grounded outcome feeds the shared citation-contract
    check.
    """
    cross_encoder = run_cross_encoder(config, root)
    report["cross_encoder"] = cross_encoder
    report["cross_encoder_ok"] = cross_encoder["ok"]
    citation_ok.append(cross_encoder["citations_resolve"])

    function_calling = run_model_function_calling(config, root)
    report["model_function_calling"] = function_calling
    report["model_function_calling_ok"] = function_calling["ok"]
    citation_ok.append(function_calling["grounded"])

    streaming = run_streaming_tool_calls(config, root)
    report["streaming_tool_calls"] = streaming
    report["streaming_tool_calls_ok"] = streaming["ok"]

    judge = run_llm_judge(config, root)
    report["llm_judge"] = judge
    report["llm_judge_ok"] = judge["ok"]


HTTP_INCIDENT = "In a security incident, how does OGuardAI protect sensitive data before it reaches an LLM provider?"


def _no_external_host(text: str) -> bool:
    """The page and its assets never load from an external CDN, font, or script host.

    The favicon is an inline data URI, so its SVG namespace is not a resource load; a
    real external load would appear as ``src="http`` or ``href="http`` or a known CDN
    host, none of which may be present.
    """
    lowered = text.lower()
    forbidden = ('src="http', 'href="http', "cdn.", "googleapis", "gstatic", "unpkg", "jsdelivr")
    return not any(token in lowered for token in forbidden)


def _sse_events(lines: Iterator[str]) -> list[dict[str, Any]]:
    return [json.loads(line[5:].strip()) for line in lines if line.startswith("data:")]


def run_http_surface(root: Path, report: dict[str, Any], citation_ok: list[bool]) -> None:
    """Drive the showcase FastAPI app offline through an in-process TestClient.

    The app is the showcase's own FastAPI instance, not the refaat API. It is built with
    the offline deterministic stack, so the whole HTTP surface runs with no network and no
    paid model. Every asserted citation is checked to resolve to a real stored chunk, so
    the grounding contract is proven over HTTP exactly as it is in process.
    """
    http_config = load_config(root / "http")
    if http_config.online:
        raise SystemExit("HTTP surface check must run on the offline deterministic stack")
    with TestClient(create_app(http_config)) as client:
        page = client.get("/")
        style = client.get("/static/style.css")
        script = client.get("/static/app.js")
        report["http_ui_ok"] = (
            page.status_code == 200
            and "Oronts Knowledge Terminal" in page.text
            and '/static/app.js"' in page.text
            and '/static/style.css"' in page.text
            and style.status_code == 200
            and script.status_code == 200
        )
        report["http_ui_no_login"] = not any(
            token in page.text.lower() for token in ('type="password"', "<form", 'name="password"', 'name="username"')
        )
        report["http_ui_no_external_host"] = (
            _no_external_host(page.text) and _no_external_host(style.text) and _no_external_host(script.text)
        )
        report["http_healthz"] = client.get("/healthz").json()["status"] == "ok"

        mode_grounded: dict[str, bool] = {}
        for mode in QUERY_MODES:
            payload = client.post("/api/query", json={"question": HTTP_INCIDENT, "mode": mode}).json()
            resolves = bool(payload["citations"]) and all(citation["resolves"] for citation in payload["citations"])
            mode_grounded[mode] = bool(payload["grounded"]) and resolves and bool(payload["routes"])
        report["http_modes_grounded"] = mode_grounded
        report["http_modes_all_grounded"] = all(mode_grounded.values())
        citation_ok.append(report["http_modes_all_grounded"])

        react = client.post("/api/query", json={"question": HTTP_INCIDENT, "mode": "react_agent"}).json()
        report["http_react_governed"] = react["extra"]["authorized"] == ["evidence_digest"] and react["extra"][
            "denied"
        ] == ["pager_escalation"]

        graphrag = client.post("/api/query", json={"question": MULTI_HOP, "mode": "graphrag"}).json()
        report["http_graphrag_route"] = (
            bool(graphrag["extra"]["graph_route"]) and len(graphrag["extra"]["graph_facts"]) > 0
        )

        corpus = client.get("/api/corpus").json()
        report["http_corpus_count"] = corpus["document_count"]

        features = client.get("/api/features").json()
        report["http_features_ok"] = (
            features["mode"] == "offline"
            and any(feature["model_lift"] for feature in features["features"])
            and any(not feature["model_lift"] for feature in features["features"])
        )

        graph = client.get("/api/graph", params={"question": MULTI_HOP}).json()
        report["http_graph_ok"] = (
            len(graph["nodes"]) > 0
            and len(graph["edges"]) > 0
            and all({"id", "label", "kind"} <= set(node) for node in graph["nodes"])
            and all({"source", "target", "relation"} <= set(edge) for edge in graph["edges"])
        )

        stats = client.get("/api/stats").json()
        report["http_stats_ok"] = (
            stats["documents"] == len(load_documents())
            and stats["chunks"] > 0
            and stats["entities"] > 0
            and stats["relations"] > 0
            and len(stats["capabilities"]) > 0
            and stats["capability_counts"]["model_lift"] > 0
        )

        # A fresh question asked twice through the same mode is a real miss then a real
        # cache hit: the enriched payload carries the server-measured elapsed time, the
        # cached flag, and the ordered pipeline stages the query fired.
        cache_question = "How does Oronts GmbH make sure clients own and control their software?"
        first = client.post("/api/query", json={"question": cache_question, "mode": "rag"}).json()
        second = client.post("/api/query", json={"question": cache_question, "mode": "rag"}).json()
        report["http_query_enriched"] = (
            "cached" in first
            and isinstance(first["elapsed_ms"], (int, float))
            and bool(first["stages"])
            and "usage" in first
            and first["retrieval"]["retrieved"] > 0
        )
        report["http_query_cache_hit"] = first["cached"] is False and second["cached"] is True

        # Economics: offline the extractive generator reports no tokens and there is no
        # price map, so tokens and USD cost are honestly zero; a cache hit carries a
        # saved block whose keys are always present and whose token and USD savings are
        # zero offline (the fresh run spent nothing to avoid).
        report["http_query_cost_zero_offline"] = (
            first["tokens"]["prompt_tokens"] == 0
            and first["tokens"]["completion_tokens"] == 0
            and first["tokens"]["total_tokens"] == 0
            and first["usd_cost"] == 0.0
        )
        report["http_query_saved_keys"] = set(first["saved"]) == {"tokens", "usd", "latency_ms"} and set(
            second["saved"]
        ) == {"tokens", "usd", "latency_ms"}
        report["http_cache_saved_zero_offline"] = second["saved"]["tokens"] == 0 and second["saved"]["usd"] == 0.0

        # Observability: a real captured OpenTelemetry trace from a genuine program run,
        # the real run-budget meter, the session economics, and the runtime event log.
        # The span names are asserted (deterministic), the durations are not (timing).
        obs = client.get("/api/observability").json()
        trace = obs["trace"]
        report["http_observability"] = {
            "source": trace["source"],
            "span_count": trace["span_count"],
            "span_names": trace["span_names"],
        }
        report["http_observability_ok"] = (
            trace["span_count"] >= 2
            and {"refaat.program.run", "refaat.program.evaluate"} <= set(trace["span_names"])
            and set(obs["budget"]) >= {"token_ceiling", "tokens_used", "remaining_tokens", "cost_used_usd"}
            and obs["budget"]["tokens_used"] == 0
            and obs["budget"]["cost_used_usd"] == 0.0
            and set(obs["economics"]) >= {"total_tokens", "total_usd", "total_saved_tokens", "total_saved_latency_ms"}
            and obs["economics"]["total_tokens"] == 0
            and obs["economics"]["total_usd"] == 0.0
            and len(obs["events"]) > 0
            and all(event["topic"] == "document.indexed" for event in obs["events"])
        )

        program = client.post(
            "/api/program/grounded-rag/run",
            json={"question": "Who founded Oronts GmbH and where is it based?"},
        ).json()
        report["http_program_grounded"] = bool(program["grounded"]) and all(
            citation["resolves"] for citation in program["citations"]
        )
        citation_ok.append(report["http_program_grounded"])

        registry = client.get("/api/registry").json()
        report["http_registry_ok"] = "grounded-rag" in registry["programs"]

        with client.stream("GET", "/api/query/stream", params={"question": OFFLINE_QUESTION}) as response:
            events = _sse_events(response.iter_lines())
        deltas = [event for event in events if event["type"] == "delta"]
        final = next(event for event in events if event["type"] == "final")
        report["http_stream_delta_count"] = len(deltas)
        report["http_stream_grounded"] = bool(final["citations"]) and final["citations_resolve"] is True
        citation_ok.append(report["http_stream_grounded"])


def verify(report: dict[str, Any]) -> None:
    checks: dict[str, bool] = {
        "mode": report["mode"] == "offline",
        "documents_ingested": report["documents_ingested"] == len(load_documents()),
        "plain_rag_grounded": report["plain_rag_grounded"] is True,
        "cache_hit": report["cache_hit"] is True,
        "graphrag_ok": report["graphrag_ok"] is True,
        "kag_grounded": report["kag_grounded"] is True,
        "feature_routes": all(report["feature_routes"].values()),
        "reranked_ok": report["reranked_ok"] is True,
        "autocut_ok": report["autocut_ok"] is True,
        "crag_grounded": report["crag_grounded"] is True,
        "self_rag_grounded": report["self_rag_grounded"] is True,
        "rlm_grounded": report["rlm_grounded"] is True and report["rlm_nodes"] > 0,
        "metadata_filter_ok": report["metadata_filter_ok"] is True,
        "compression_tokens_dropped": report["compression_tokens_dropped"] > 0 and report["compression_route"] is True,
        "transforms_active_offline_ok": report["transforms_active_offline_ok"] is True,
        "transforms_model_activated_ok": report["transforms_model_activated_ok"] is True,
        "hierarchical_ok": report["hierarchical_ok"] is True,
        "contextual_grounded": report["contextual_grounded"] is True,
        "contextual_effect": report["contextual_effect"] is True,
        "abstain_ok": report["abstain_ok"] is True,
        "abstain_disabled_unchanged": report["abstain_disabled_unchanged"] is True,
        "citation_contract_ok": report["citation_contract_ok"] is True,
        "program_layer_grounded": report["program_layer_grounded"] is True,
        "agent_layer_grounded": report["agent_layer_grounded"] is True,
        "mcp_layer_ok": report["mcp_layer_ok"] is True,
        "registry_register_not_promote": report["registry_register_not_promote"] is True,
        "registry_rollback_atomic": report["registry_rollback_atomic"] is True,
        "denied_tool_recorded": report["denied_tool_recorded"] is True,
        "pareto_non_empty": report["pareto_non_empty"] is True,
        "durable_optimization_ok": report["durable_optimization_ok"] is True,
        "program_evaluation_ok": report["program_evaluation_ok"] is True,
        "model_cache_budget_ok": report["model_cache_budget_ok"] is True,
        "shadow_canary_ok": report["shadow_canary_ok"] is True,
        "fine_tuning_ok": report["fine_tuning_ok"] is True,
        "multimodal_ok": report["multimodal_ok"] is True,
        "sandboxed_rlm_ok": report["sandboxed_rlm_ok"] is True,
        "multi_agent_routes_differ": report["multi_agent_routes_differ"] is True,
        "program_as_skill_ok": report["program_as_skill_ok"] is True,
        "optimizer_output_registered": report["optimizer_output_registered"] is True,
        "cross_encoder_ok": report["cross_encoder_ok"] is True,
        "model_function_calling_ok": report["model_function_calling_ok"] is True,
        "streaming_tool_calls_ok": report["streaming_tool_calls_ok"] is True,
        "llm_judge_ok": report["llm_judge_ok"] is True,
        "real_integrations_ok": report["real_integrations_ok"] is True,
        "http_ui_ok": report["http_ui_ok"] is True,
        "http_ui_no_login": report["http_ui_no_login"] is True,
        "http_ui_no_external_host": report["http_ui_no_external_host"] is True,
        "http_healthz": report["http_healthz"] is True,
        "http_modes_all_grounded": report["http_modes_all_grounded"] is True,
        "http_react_governed": report["http_react_governed"] is True,
        "http_graphrag_route": report["http_graphrag_route"] is True,
        "http_corpus_count": report["http_corpus_count"] == len(load_documents()),
        "http_features_ok": report["http_features_ok"] is True,
        "http_graph_ok": report["http_graph_ok"] is True,
        "http_stats_ok": report["http_stats_ok"] is True,
        "http_query_enriched": report["http_query_enriched"] is True,
        "http_query_cache_hit": report["http_query_cache_hit"] is True,
        "http_query_cost_zero_offline": report["http_query_cost_zero_offline"] is True,
        "http_query_saved_keys": report["http_query_saved_keys"] is True,
        "http_cache_saved_zero_offline": report["http_cache_saved_zero_offline"] is True,
        "http_observability_ok": report["http_observability_ok"] is True,
        "http_program_grounded": report["http_program_grounded"] is True,
        "http_registry_ok": report["http_registry_ok"] is True,
        "http_stream_delta_count": report["http_stream_delta_count"] > 1,
        "http_stream_grounded": report["http_stream_grounded"] is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"Showcase assertions failed: {', '.join(failed)}")


def main() -> None:
    with TemporaryDirectory(prefix="refaat-showcase-") as directory:
        root = Path(directory)
        config = load_config(root / "primary")
        report: dict[str, Any] = {"mode": config.mode}
        citation_ok: list[bool] = []

        run_primary_modes(config, report, citation_ok)
        run_transform_catalog(config, root, report)
        run_hierarchical(config, root, report)
        run_contextual(config, root, report, citation_ok)
        run_abstain(config, root, report)
        run_multimodal(config, root, report, citation_ok)
        run_sandboxed_rlm(config, root, report)
        run_model_lift(config, root, report, citation_ok)
        run_http_surface(root, report, citation_ok)

        report["citation_contract_ok"] = all(citation_ok)
        report["features_exercised"] = [
            "rag_citations",
            "graphrag_multi_hop",
            "cag_cache_hit",
            "kag_graph_answer",
            "query_transforms",
            "reranking",
            "hype",
            "raptor",
            "contextual_retrieval",
            "hierarchical_parent_child",
            "hierarchical_sentence_window",
            "rrf_autocut",
            "compression",
            "metadata_filter_bm25",
            "abstention",
            "corrective_rag",
            "self_rag",
            "recursive_language_method",
            "typed_program",
            "bounded_inference_policy",
            "governed_react",
            "structured_output",
            "multi_agent_orchestrator",
            "program_evaluation",
            "retrieval_optimization",
            "immutable_registry",
            "program_as_skill",
            "model_call_cache_and_budget",
            "durable_optimization",
            "shadow_canary",
            "governed_fine_tuning",
            "multimodal_descriptor",
            "sandboxed_code_rlm",
            "cross_encoder_reranker",
            "model_function_calling",
            "streaming_tool_calls",
            "llm_as_judge",
            "llamaindex_retriever",
            "langchain_agent",
            "langgraph_state_graph",
            "governed_react_subgraph",
            "crewai_answer_tool",
            "mcp_serve_bridge",
            "mcp_remote_skill",
            "http_api_surface",
            "streaming_web_ui",
        ]
        verify(report)
        emit(report)


if __name__ == "__main__":
    main()
