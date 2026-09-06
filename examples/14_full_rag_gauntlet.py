from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from _shared import demo_settings, emit

from refaat import RuntimeBuilder
from refaat.domain.enums import SideEffectClass
from refaat.domain.models import Citation, ContextPacket, Demonstration, DocumentInput, ReActPolicy, ToolGrant
from refaat.evaluation import Dataset
from refaat.optimization import LabeledFewShotOptimizer, retrieval_search_space
from refaat.programs import (
    BEST_OF,
    REFINE,
    BoundedInferencePolicy,
    EnsembleProgram,
    GovernedReActProgram,
    GroundedRagProgram,
    InputField,
    KeywordToolPlanner,
    KNNFewShotProgram,
    Signature,
    Tunables,
)

BOILERPLATE = "Every Meridian service inherits the shared reliability baseline and audit trail."


def handbook_documents() -> list[dict[str, str]]:
    """A fabricated Meridian Robotics knowledge base with repeated entities and headings."""
    return [
        {
            "source_id": "eng/release-policy.md",
            "category": "engineering",
            "team": "Platform Team",
            "title": "Production Release Policy",
            "text": (
                "# Production Release Policy\n\n"
                "## Overview\n"
                "The Release Manager verifies peer review, automated tests, and a tested rollback plan "
                "before a production release. " + BOILERPLATE + "\n\n"
                "## Approval\n"
                "The Release Manager approves a deployment only after the Platform Team completes a canary. "
                "High-risk changes to the Atlas Gateway require sign off from the Security Guild."
            ),
        },
        {
            "source_id": "eng/deploy-runbook.md",
            "category": "engineering",
            "team": "Platform Team",
            "title": "Deployment Runbook",
            "text": (
                "# Deployment Runbook\n\n"
                "## Canary\n"
                "The Platform Team runs a canary deployment and watches error rate, latency, and saturation. "
                "A failed canary must be rolled back before another attempt begins.\n\n"
                "## Ownership\n"
                "Nadia Farouk leads the Platform Team and owns the Atlas Gateway roadmap."
            ),
        },
        {
            "source_id": "eng/branching.md",
            "category": "engineering",
            "team": "Platform Team",
            "title": "Branching Standards",
            "text": (
                "# Branching Standards\n\n"
                "## Trunk\n"
                "Engineers merge into the trunk behind a feature flag. "
                "The Platform Team requires two approvals for a change to the Atlas Gateway."
            ),
        },
        {
            "source_id": "eng/observability.md",
            "category": "engineering",
            "team": "Reliability Team",
            "title": "Observability Handbook",
            "text": (
                "# Observability Handbook\n\n"
                "## Signals\n"
                "The Beacon Pipeline ingests telemetry and feeds the Orbit Console dashboards. "
                "The Reliability Team reviews the Orbit Console before every release."
            ),
        },
        {
            "source_id": "eng/data-retention.md",
            "category": "engineering",
            "team": "Reliability Team",
            "title": "Data Retention",
            "text": (
                "# Data Retention\n\n"
                "## Windows\n"
                "The Nimbus Store persists tenant data inside the European Region only. " + BOILERPLATE
            ),
        },
        {
            "source_id": "product/atlas-gateway.md",
            "category": "product",
            "team": "Platform Team",
            "title": "Atlas Gateway Specification",
            "text": (
                "# Atlas Gateway Specification\n\n"
                "## Routing\n"
                "The Atlas Gateway routes external traffic and enforces the rate limit for every tenant. "
                "Nadia Farouk and the Platform Team support the Atlas Gateway during a Severity One incident.\n\n"
                "## Limits\n"
                "The Atlas Gateway rejects a request that exceeds the tenant rate limit."
            ),
        },
        {
            "source_id": "product/beacon-pipeline.md",
            "category": "product",
            "team": "Reliability Team",
            "title": "Beacon Pipeline Specification",
            "text": (
                "# Beacon Pipeline Specification\n\n"
                "## Ingestion\n"
                "The Beacon Pipeline ingests telemetry from every service and feeds the Orbit Console. "
                "Tomas Berg maintains the Beacon Pipeline for the Reliability Team."
            ),
        },
        {
            "source_id": "product/orbit-console.md",
            "category": "product",
            "team": "Reliability Team",
            "title": "Orbit Console Specification",
            "text": (
                "# Orbit Console Specification\n\n"
                "## Dashboards\n"
                "The Orbit Console renders the dashboards fed by the Beacon Pipeline. "
                "The Reliability Team pins the incident dashboard for a Severity One incident."
            ),
        },
        {
            "source_id": "product/nimbus-store.md",
            "category": "product",
            "team": "Platform Team",
            "title": "Nimbus Store Specification",
            "text": (
                "# Nimbus Store Specification\n\n"
                "## Residency\n"
                "The Nimbus Store keeps tenant data inside the European Region only. "
                "Priya Raman designed the Nimbus Store residency model."
            ),
        },
        {
            "source_id": "incident/postmortem-101.md",
            "category": "incident",
            "team": "Reliability Team",
            "title": "Postmortem Gateway Saturation",
            "text": (
                "# Postmortem Gateway Saturation\n\n"
                "## Summary\n"
                "The Atlas Gateway saturated and the Reliability Team ran the incident bridge. "
                "The Incident Commander coordinated a Severity One incident and posted updates every "
                "fifteen minutes.\n\n"
                "## Follow up\n"
                "Repeated alerts require a corrective action and a documented owner within two business days."
            ),
        },
        {
            "source_id": "incident/postmortem-102.md",
            "category": "incident",
            "team": "Reliability Team",
            "title": "Postmortem Pipeline Lag",
            "text": (
                "# Postmortem Pipeline Lag\n\n"
                "## Summary\n"
                "The Beacon Pipeline lagged and the Orbit Console showed stale data. "
                "Tomas Berg and the Reliability Team restored the Beacon Pipeline."
            ),
        },
        {
            "source_id": "incident/postmortem-103.md",
            "category": "incident",
            "team": "Security Guild",
            "title": "Postmortem Access Breach",
            "text": (
                "# Postmortem Access Breach\n\n"
                "## Summary\n"
                "The Security Guild detected a breach and escalated from the Reliability Team. "
                "The Incident Commander opened a Severity One incident and posted updates every fifteen minutes."
            ),
        },
        {
            "source_id": "incident/oncall.md",
            "category": "incident",
            "team": "Reliability Team",
            "title": "On Call Rotation",
            "text": (
                "# On Call Rotation\n\n"
                "## Escalation\n"
                "The Reliability Team runs the incident bridge and escalates to the Security Guild for a breach. "
                + BOILERPLATE
            ),
        },
        {
            "source_id": "security/secrets.md",
            "category": "security",
            "team": "Security Guild",
            "title": "Secret Management Policy",
            "text": (
                "# Secret Management Policy\n\n"
                "## Storage\n"
                "Secrets and encryption keys are stored in the Vault Service and rotated every ninety days. "
                + BOILERPLATE
                + "\n\n"
                "## Access\n"
                "The Security Guild grants Vault Service access only to a named owner."
            ),
        },
        {
            "source_id": "security/access-control.md",
            "category": "security",
            "team": "Security Guild",
            "title": "Access Control Policy",
            "text": (
                "# Access Control Policy\n\n"
                "## Roles\n"
                "The Security Guild reviews every role grant to the Atlas Gateway each quarter. "
                "Marco Silva approves a privileged role for the Security Guild."
            ),
        },
        {
            "source_id": "security/data-residency.md",
            "category": "security",
            "team": "Security Guild",
            "title": "Data Residency Policy",
            "text": (
                "# Data Residency Policy\n\n"
                "## Region\n"
                "Personal Data must remain inside the European Region. "
                "The Security Guild audits the Nimbus Store residency every ninety days."
            ),
        },
        {
            "source_id": "security/encryption.md",
            "category": "security",
            "team": "Security Guild",
            "title": "Encryption Standard",
            "text": (
                "# Encryption Standard\n\n"
                "## Transport\n"
                "The Atlas Gateway enforces transport encryption for every tenant. "
                "Encryption keys are stored in the Vault Service and rotated every ninety days."
            ),
        },
        {
            "source_id": "people/nadia-farouk.md",
            "category": "people",
            "team": "Platform Team",
            "title": "Nadia Farouk Profile",
            "text": (
                "# Nadia Farouk Profile\n\n"
                "## Role\n"
                "Nadia Farouk leads the Platform Team and owns the Atlas Gateway roadmap. "
                "Nadia Farouk partners with the Security Guild on gateway hardening."
            ),
        },
        {
            "source_id": "people/tomas-berg.md",
            "category": "people",
            "team": "Reliability Team",
            "title": "Tomas Berg Profile",
            "text": (
                "# Tomas Berg Profile\n\n"
                "## Role\n"
                "Tomas Berg maintains the Beacon Pipeline and the Orbit Console for the Reliability Team."
            ),
        },
        {
            "source_id": "people/priya-raman.md",
            "category": "people",
            "team": "Platform Team",
            "title": "Priya Raman Profile",
            "text": (
                "# Priya Raman Profile\n\n"
                "## Role\n"
                "Priya Raman designed the Nimbus Store and records acceptance criteria for onboarding."
            ),
        },
        {
            "source_id": "people/onboarding-crew.md",
            "category": "people",
            "team": "Onboarding Crew",
            "title": "Onboarding Crew Charter",
            "text": (
                "# Onboarding Crew Charter\n\n"
                "## Activation\n"
                "The Onboarding Crew records acceptance criteria and a rollback path before production activation. "
                "The Onboarding Crew hands a new tenant to the Platform Team."
            ),
        },
    ]


def compression_documents() -> list[dict[str, str]]:
    """Near-duplicate documents that share boilerplate so cross-chunk dedup removes tokens."""
    shared = (
        "The shared reliability baseline requires a rollback plan and an audit trail. "
        "Every Meridian service publishes the same reliability baseline and audit trail."
    )
    unique = {
        "atlas": "The Atlas Gateway records the reliability baseline for external traffic.",
        "beacon": "The Beacon Pipeline records the reliability baseline for telemetry.",
        "nimbus": "The Nimbus Store records the reliability baseline for tenant data.",
        "orbit": "The Orbit Console records the reliability baseline for dashboards.",
        "vault": "The Vault Service records the reliability baseline for secrets.",
    }
    return [
        {
            "source_id": f"baseline/{key}.md",
            "category": "engineering",
            "team": "Reliability Team",
            "title": f"{key.title()} Reliability Baseline",
            "text": f"# {key.title()} Reliability Baseline\n\n## Baseline\n{shared} {sentence}",
        }
        for key, sentence in unique.items()
    ]


def hierarchical_documents() -> list[dict[str, str]]:
    """Multi-sentence sectioned documents for parent_child and sentence_window chunking."""
    return [
        {
            "source_id": "atlas.md",
            "category": "product",
            "team": "Platform Team",
            "title": "Atlas Gateway Guide",
            "text": (
                "# Atlas Gateway Guide\n\n"
                "## Routing\n"
                "The Atlas Gateway routes external traffic for every tenant. "
                "The Atlas Gateway enforces a per tenant rate limit. "
                "The Platform Team owns the Atlas Gateway routing table. "
                "A failed route is retried against a healthy region before it errors.\n\n"
                "## Security\n"
                "The Atlas Gateway enforces transport encryption. "
                "The Security Guild reviews every Atlas Gateway role grant. "
                "Secrets used by the Atlas Gateway live in the Vault Service."
            ),
        },
        {
            "source_id": "beacon.md",
            "category": "product",
            "team": "Reliability Team",
            "title": "Beacon Pipeline Guide",
            "text": (
                "# Beacon Pipeline Guide\n\n"
                "## Ingestion\n"
                "The Beacon Pipeline ingests telemetry from every service. "
                "The Beacon Pipeline batches events before it writes them. "
                "The Reliability Team tunes the Beacon Pipeline batch size. "
                "A dropped event is replayed from the durable buffer.\n\n"
                "## Dashboards\n"
                "The Beacon Pipeline feeds the Orbit Console. "
                "The Orbit Console renders the incident dashboard. "
                "Tomas Berg maintains the Orbit Console dashboards."
            ),
        },
        {
            "source_id": "incident.md",
            "category": "incident",
            "team": "Reliability Team",
            "title": "Incident Handling Guide",
            "text": (
                "# Incident Handling Guide\n\n"
                "## Bridge\n"
                "The Reliability Team opens an incident bridge for a Severity One incident. "
                "The Incident Commander coordinates the incident bridge. "
                "The Incident Commander posts updates every fifteen minutes. "
                "A breach escalates to the Security Guild immediately.\n\n"
                "## Recovery\n"
                "The Reliability Team restores the Beacon Pipeline first. "
                "The Platform Team restores the Atlas Gateway next. "
                "A corrective action is recorded within two business days."
            ),
        },
    ]


def contextual_documents() -> list[dict[str, str]]:
    """Documents whose match depends on the title context, not the body.

    Each codename ("Aurora", "Basalt", "Cinnamon") lives only in the title, never in
    the body, so the flat embedding (body only) cannot see it. Contextual retrieval
    prefixes the title before embedding, so the codename in the query then lands on
    the right document. The decoy token "polar" sits only in the Aurora body, which
    pulls a flat baseline away from the Basalt target and proves a real difference.
    """
    return [
        {
            "source_id": "brief/aurora.md",
            "category": "brief",
            "team": "Reliability Team",
            "title": "Aurora Codename Brief",
            "text": "The report describes polar sky glow and solar wind across a long winter.",
        },
        {
            "source_id": "brief/basalt.md",
            "category": "brief",
            "team": "Platform Team",
            "title": "Basalt Codename Brief",
            "text": "The report covers cooled lava formations and columnar jointing across ancient flows.",
        },
        {
            "source_id": "brief/cinnamon.md",
            "category": "brief",
            "team": "Onboarding Crew",
            "title": "Cinnamon Codename Brief",
            "text": "The report explains bark harvesting, drying, and grinding for culinary use.",
        },
    ]


def _space_headings(text: str) -> str:
    """Put a blank line after every ATX heading so a heading is its own paragraph.

    The co-occurrence graph extractor treats a paragraph (a ``\\n\\n`` block) as one
    context window and reads capitalized runs across single newlines as one entity.
    Separating the heading from its body keeps entity names clean ("Atlas Gateway",
    not "Routing\\nThe Atlas Gateway") without changing any sentence text.
    """
    lines = text.split("\n")
    spaced: list[str] = []
    for index, line in enumerate(lines):
        spaced.append(line)
        following = lines[index + 1] if index + 1 < len(lines) else ""
        if line.lstrip().startswith("#") and following.strip():
            spaced.append("")
    return "\n".join(spaced)


def ingest_documents(project: Any, documents: list[dict[str, str]]) -> None:
    for document in documents:
        project.runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id=document["source_id"],
                title=document["title"],
                media_type="text/markdown",
                content=_space_headings(document["text"]).encode(),
                metadata={"category": document["category"], "team": document["team"]},
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


def citations_resolve(citations: tuple[Citation, ...], identifiers: set[str]) -> bool:
    return bool(citations) and all(citation.chunk_id in identifiers for citation in citations)


def retrieved_sources(context: ContextPacket) -> list[str]:
    return [str(hit.chunk.metadata.get("source_id")) for hit in context.hits]


class HandbookQuestion(Signature):
    question: str = InputField(description="An operations question answered from the Meridian handbook.")


class HandbookAssistant(GroundedRagProgram):
    name = "handbook-assistant"
    input_signature = HandbookQuestion


class EvidenceChecklistSkill:
    name = "evidence_checklist"
    description = "Summarize the top evidence sentences for a governed ReAct step."

    def run(self, context: ContextPacket, arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            "items": [hit.chunk.text.split(".", maxsplit=1)[0] for hit in context.hits[:3]],
            "routes": list(context.routes),
        }


DEMONSTRATION_POOL = (
    Demonstration(question="Who verifies release evidence?", answer="The Release Manager verifies the evidence."),
    Demonstration(question="Who coordinates a Severity One incident?", answer="The Incident Commander coordinates it."),
    Demonstration(question="Where does tenant data stay?", answer="Tenant data stays in the European Region."),
    Demonstration(question="Where are secrets stored?", answer="Secrets are stored in the Vault Service."),
)

EVALUATION_RECORDS = [
    {
        "id": "c1",
        "query": "What does the Release Manager verify before a production release?",
        "expected_terms": ["release manager", "rollback plan"],
    },
    {
        "id": "c2",
        "query": "Who coordinates a Severity One incident and how often are updates posted?",
        "expected_terms": ["incident commander", "fifteen minutes"],
    },
    {
        "id": "c3",
        "query": "Where are secrets and encryption keys stored?",
        "expected_terms": ["vault service", "ninety days"],
    },
    {
        "id": "c4",
        "query": "What does the Atlas Gateway enforce for every tenant?",
        "expected_terms": ["atlas gateway", "rate limit"],
    },
    {
        "id": "c5",
        "query": "What does the Beacon Pipeline feed?",
        "expected_terms": ["beacon pipeline", "orbit console"],
    },
    {"id": "c6", "query": "Who leads the Platform Team?", "expected_terms": ["nadia farouk", "platform team"]},
    {
        "id": "c7",
        "query": "What does the Onboarding Crew record before production activation?",
        "expected_terms": ["acceptance criteria", "rollback path"],
    },
    {
        "id": "c8",
        "query": "Where does the Nimbus Store keep tenant data?",
        "expected_terms": ["nimbus store", "european region"],
    },
    {
        "id": "c9",
        "query": "How does the Reliability Team escalate a breach?",
        "expected_terms": ["reliability team", "security guild"],
    },
    {
        "id": "c10",
        "query": "What must repeated alerts produce?",
        "expected_terms": ["corrective action", "two business days"],
    },
]

LABELED_RECORDS = [
    {
        "id": "l1",
        "query": "Who verifies release evidence?",
        "expected_terms": ["release manager"],
        "reference_answer": "The Release Manager verifies release evidence.",
    },
    {
        "id": "l2",
        "query": "Who coordinates a Severity One incident?",
        "expected_terms": ["incident commander"],
        "reference_answer": "The Incident Commander coordinates it.",
    },
    {
        "id": "l3",
        "query": "Where are secrets stored?",
        "expected_terms": ["vault service"],
        "reference_answer": "Secrets are stored in the Vault Service.",
    },
    {
        "id": "l4",
        "query": "Where does the Nimbus Store keep tenant data?",
        "expected_terms": ["european region"],
        "reference_answer": "The Nimbus Store keeps data in the European Region.",
    },
    {
        "id": "l5",
        "query": "What does the Atlas Gateway enforce?",
        "expected_terms": ["rate limit"],
        "reference_answer": "The Atlas Gateway enforces a tenant rate limit.",
    },
    {
        "id": "l6",
        "query": "What does the Beacon Pipeline feed?",
        "expected_terms": ["orbit console"],
        "reference_answer": "The Beacon Pipeline feeds the Orbit Console.",
    },
]


def run_handbook_steps(runtime: Any, report: dict[str, Any]) -> None:
    handbook = runtime.project(workspace="meridian", workspace_name="Meridian Robotics", corpus="handbook")
    ingest_documents(handbook, handbook_documents())
    handbook_ids = stored_chunk_ids(handbook)
    report["documents_ingested"] = handbook.stats()["documents"]

    rag_query = "What does the Release Manager verify before a production release?"
    rag = handbook.ask(rag_query, fusion="rrf")
    report["rag_grounded"] = citations_resolve(rag.citations, handbook_ids) and "rrf" in rag.routes

    cached = handbook.ask(rag_query, fusion="rrf")
    report["cache_hit"] = cached.cached and cached.routes[0] == "cag:exact"

    linear = handbook.ask(rag_query, fusion="linear")
    report["linear_uncached"] = (not linear.cached) and "linear" in linear.routes

    autocut = handbook.retrieve(
        "How does the Reliability Team run the incident bridge for a Severity One incident?",
        autocut_enabled=True,
        autocut_threshold=0.05,
    )
    report["autocut_ok"] = "autocut" in autocut.routes and bool(autocut.hits)

    subquery = handbook.retrieve(
        "How does the Release Manager approve a deployment, "
        "and how does the Reliability Team escalate a Severity One incident?",
        query_expansion=True,
        max_query_variants=4,
    )
    report["subquery_routes"] = list(subquery.routes)

    graph = handbook.retrieve("How do Nadia Farouk and the Platform Team support the Atlas Gateway?")
    report["graph_facts"] = list(graph.graph_facts[:5])

    typed = handbook.programs.register(
        HandbookAssistant(tunables=Tunables(top_k=6, minimum_citation_coverage=0.5))
    ).run(question="How does the Release Manager approve a production deployment?")
    report["program_grounded"] = typed.grounded and citations_resolve(typed.citations, handbook_ids)

    fewshot = LabeledFewShotOptimizer(handbook).optimize(
        GroundedRagProgram(),
        Dataset.from_records("handbook-labeled", LABELED_RECORDS),
        shots=(1, 2),
    )
    report["fewshot_candidates"] = len({candidate.artifact_id for candidate in fewshot.candidates})

    knn = KNNFewShotProgram(GroundedRagProgram(), DEMONSTRATION_POOL, k=2).execute(
        handbook, inputs={"question": "Who coordinates a Severity One incident?"}, user_id=None, session_id=None
    )
    report["knn_grounded"] = knn.grounded and citations_resolve(knn.citations, handbook_ids)

    ensemble = EnsembleProgram(
        [GroundedRagProgram(tunables=Tunables(top_k=6)), GroundedRagProgram(tunables=Tunables(top_k=10))]
    ).execute(handbook, inputs={"question": "What does the Atlas Gateway enforce?"}, user_id=None, session_id=None)
    report["ensemble_grounded"] = ensemble.grounded and citations_resolve(ensemble.citations, handbook_ids)

    policy = ReActPolicy(
        max_iterations=3,
        granted_scopes=("read",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(ToolGrant("evidence_checklist", required_scopes=("read",), side_effect=SideEffectClass.READ_ONLY),),
    )
    react = GovernedReActProgram(
        policy=policy,
        planner=KeywordToolPlanner({"evidence_checklist": ("evidence", "release", "incident")}),
        skills={"evidence_checklist": EvidenceChecklistSkill()},
    ).execute(
        handbook,
        inputs={"question": "What evidence does the Release Manager review for a production release?"},
        user_id=None,
        session_id=None,
    )
    react_steps = react.outputs.get("react_steps", [])
    report["react_steps"] = len(react_steps)
    report["react_authorized"] = any(step["authorized"] for step in react_steps) and react.grounded

    best_of = BoundedInferencePolicy(GroundedRagProgram(), mode=BEST_OF, attempts=3).execute(
        handbook, inputs={"question": "Who leads the Platform Team?"}, user_id=None, session_id=None
    )
    refine = BoundedInferencePolicy(GroundedRagProgram(), mode=REFINE, attempts=3).execute(
        handbook, inputs={"question": "Who leads the Platform Team?"}, user_id=None, session_id=None
    )
    report["policy_best_of_ok"] = best_of.grounded and refine.grounded

    evaluation = handbook.programs.get("grounded-rag").evaluate(
        Dataset.from_records("handbook-eval", EVALUATION_RECORDS)
    )
    report["evaluation_score"] = round(evaluation.score, 4)
    report["evaluation_cases"] = evaluation.case_count
    report["evaluation_hard_passed"] = evaluation.hard_constraints_passed
    report["evaluation_answer_contains"] = evaluation.aggregate["answer_contains"]

    optimization = handbook.programs.get("grounded-rag").optimize(
        Dataset.from_records("handbook-opt", EVALUATION_RECORDS[:6]),
        search_space=retrieval_search_space(top_k=(4, 8, 12)),
    )
    report["optimization_candidates"] = optimization.candidate_count
    report["pareto_size"] = len(optimization.pareto_front)

    registry = handbook.registry
    base = GroundedRagProgram().artifact()
    tuned = GroundedRagProgram(tunables=Tunables(top_k=4)).artifact()
    registry.save(base)
    registry.save(tuned)
    registry.promote("grounded-rag", "development", base.artifact_id)
    registry.promote("grounded-rag", "staging", base.artifact_id)
    registry.promote("grounded-rag", "staging", tuned.artifact_id)
    resolved_tuned = registry.resolve("grounded-rag", "staging")["definition_hash"] == tuned.definition_hash
    rolled = registry.rollback("grounded-rag", "staging")
    resolved_base = registry.resolve("grounded-rag", "staging")["definition_hash"] == base.definition_hash
    report["registry_promoted"] = resolved_tuned
    report["registry_rolled_back"] = resolved_base and rolled["artifact_id"] == base.artifact_id


def run_metadata_scope_step(runtime: Any, report: dict[str, Any]) -> None:
    scoped = runtime.project(workspace="meridian", corpus="scoped")
    ingest_documents(scoped, handbook_documents())
    scoped.create_harness_snapshot(
        supplemental={"retrieval": {"filters": {"category": "security"}}},
        evidence={"reason": "scope retrieval to security policies"},
        activate=True,
    )
    context = scoped.retrieve("How are secrets and encryption keys protected?")
    only_security = bool(context.hits) and all(hit.chunk.metadata.get("category") == "security" for hit in context.hits)
    answer = scoped.ask("How are secrets and encryption keys protected?")
    security_sources = {
        document["source_id"] for document in handbook_documents() if document["category"] == "security"
    }
    security_documents = {
        str(document["id"]) for document in scoped.list_documents() if document.get("source_id") in security_sources
    }
    cited_scoped = bool(answer.citations) and all(
        citation.document_id in security_documents for citation in answer.citations
    )
    report["metadata_scoped"] = only_security and "filter" in context.routes and cited_scoped


def run_compression_step(runtime: Any, report: dict[str, Any]) -> None:
    compress = runtime.project(workspace="meridian", corpus="compress")
    ingest_documents(compress, compression_documents())
    query = "What does the shared reliability baseline and audit trail require?"
    baseline = compress.retrieve(query)
    compress.create_harness_snapshot(
        supplemental={"retrieval": {"compression_enabled": True, "compression_min_salience": 0.0}},
        evidence={"reason": "drop redundant boilerplate sentences"},
        activate=True,
    )
    compressed = compress.retrieve(query)
    answer = compress.ask(query)
    dropped = baseline.token_count - compressed.token_count
    report["compression_tokens_dropped"] = dropped
    report["compression_grounded"] = "compress" in compressed.routes and bool(answer.citations)


def run_hierarchical_steps(root: Path, report: dict[str, Any]) -> None:
    for strategy, key in (("parent_child", "parent_child"), ("sentence_window", "sentence_window")):
        directory = root / key
        directory.mkdir(parents=True, exist_ok=True)
        settings = demo_settings(directory).model_copy(update={"chunking_strategy": strategy})
        with RuntimeBuilder(settings).build() as runtime:
            project = runtime.project(workspace="meridian", corpus=f"hierarchy-{key}")
            ingest_documents(project, hierarchical_documents())
            identifiers = stored_chunk_ids(project)
            answer = project.ask("How does the Incident Commander coordinate a Severity One incident?")
            report[f"{key}_citations_ok"] = (
                citations_resolve(answer.citations, identifiers) and "expand" in answer.routes
            )
    report["hierarchical_citations_ok"] = report["parent_child_citations_ok"] and report["sentence_window_citations_ok"]


def run_contextual_step(root: Path, report: dict[str, Any]) -> None:
    """Prove contextual retrieval genuinely changes the result over identical docs.

    The Basalt codename lives only in the title, so a flat baseline (body-only
    embedding) ranks a different brief first, while the contextual run (title
    prefixed before embedding) lifts the target to the top. The difference is a real
    library output, and the contextual answer's citations resolve to real stored
    chunk ids.
    """
    query = "basalt codename brief report"
    target = "brief/basalt.md"

    flat_dir = root / "contextual-flat"
    flat_dir.mkdir(parents=True, exist_ok=True)
    with RuntimeBuilder(demo_settings(flat_dir)).build() as flat_runtime:
        flat = flat_runtime.project(workspace="meridian", corpus="contextual-flat")
        ingest_documents(flat, contextual_documents())
        flat_sources = retrieved_sources(flat.retrieve(query))

    contextual_dir = root / "contextual-on"
    contextual_dir.mkdir(parents=True, exist_ok=True)
    settings = demo_settings(contextual_dir).model_copy(update={"contextual_retrieval": True})
    with RuntimeBuilder(settings).build() as contextual_runtime:
        contextual = contextual_runtime.project(workspace="meridian", corpus="contextual-on")
        ingest_documents(contextual, contextual_documents())
        contextual_ids = stored_chunk_ids(contextual)
        contextual_sources = retrieved_sources(contextual.retrieve(query))
        answer = contextual.ask(query)

    report["contextual_flat_top"] = flat_sources[0] if flat_sources else ""
    report["contextual_top"] = contextual_sources[0] if contextual_sources else ""
    report["contextual_retrieval_ok"] = (
        bool(flat_sources)
        and bool(contextual_sources)
        and flat_sources[0] != target
        and contextual_sources[0] == target
        and contextual_sources != flat_sources
        and contextual_sources.index(target) < flat_sources.index(target)
        and citations_resolve(answer.citations, contextual_ids)
    )


def verify(report: dict[str, Any]) -> None:
    checks: dict[str, bool] = {
        "documents_ingested": report["documents_ingested"] == len(handbook_documents()),
        "rag_grounded": report["rag_grounded"] is True,
        "cache_hit": report["cache_hit"] is True,
        "linear_uncached": report["linear_uncached"] is True,
        "metadata_scoped": report["metadata_scoped"] is True,
        "autocut_ok": report["autocut_ok"] is True,
        "subquery_routes": "expansion" in report["subquery_routes"] and "rrf" in report["subquery_routes"],
        "compression_tokens_dropped": report["compression_tokens_dropped"] > 0,
        "compression_grounded": report["compression_grounded"] is True,
        "graph_facts": len(report["graph_facts"]) > 0,
        "hierarchical_citations_ok": report["hierarchical_citations_ok"] is True,
        "contextual_retrieval_ok": report["contextual_retrieval_ok"] is True,
        "program_grounded": report["program_grounded"] is True,
        "fewshot_candidates": report["fewshot_candidates"] >= 2,
        "knn_grounded": report["knn_grounded"] is True,
        "ensemble_grounded": report["ensemble_grounded"] is True,
        "react_steps": report["react_steps"] > 0 and report["react_authorized"] is True,
        "policy_best_of_ok": report["policy_best_of_ok"] is True,
        "evaluation_score": report["evaluation_score"] >= 0.9 and report["evaluation_hard_passed"] is True,
        "evaluation_answer_contains": report["evaluation_answer_contains"] == 1.0,
        "evaluation_cases": report["evaluation_cases"] == len(EVALUATION_RECORDS),
        "optimization_candidates": report["optimization_candidates"] == 3,
        "pareto_size": report["pareto_size"] >= 1,
        "registry_promoted": report["registry_promoted"] is True,
        "registry_rolled_back": report["registry_rolled_back"] is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"Full RAG gauntlet assertions failed: {', '.join(failed)}")


def main() -> None:
    with TemporaryDirectory(prefix="refaat-gauntlet-") as directory:
        root = Path(directory)
        report: dict[str, Any] = {}
        main_root = root / "main"
        main_root.mkdir(parents=True, exist_ok=True)
        with RuntimeBuilder(demo_settings(main_root)).build() as runtime:
            run_handbook_steps(runtime, report)
            run_metadata_scope_step(runtime, report)
            run_compression_step(runtime, report)
        run_hierarchical_steps(root, report)
        run_contextual_step(root, report)
        verify(report)
        emit(report)


if __name__ == "__main__":
    main()
