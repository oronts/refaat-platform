from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from _shared import demo_settings, emit

from refaat import RuntimeBuilder
from refaat.domain.enums import SideEffectClass
from refaat.domain.models import ContextPacket, DocumentInput, JsonObject, ReActPolicy, ToolGrant
from refaat.evaluation import Dataset, ProgramEvaluator, trajectory_bundle
from refaat.programs import GovernedReActProgram, GroundedRagProgram, KeywordToolPlanner
from refaat.skills import ProgramSkill, SkillRegistry

PLAYBOOK_TOOL = "incident_playbook"
DIGEST_TOOL = "evidence_digest"
PAGER_TOOL = "pager_escalation"


def operations_documents() -> list[dict[str, str]]:
    """A fabricated Helios Robotics operations knowledge base.

    Recurring named entities (Atlas Gateway, Beacon Pipeline, Vault Service, the
    Incident Commander, the On-Call Engineer, the Reliability Team, the Security
    Guild) appear across teams, severities, and systems so retrieval, GraphRAG,
    parent_child chunking, and citations are all genuinely exercised.
    """
    return [
        {
            "source_id": "runbook/rollback.md",
            "team": "Platform Team",
            "severity": "sev1",
            "system": "Atlas Gateway",
            "title": "Rollback Runbook",
            "text": (
                "# Rollback Runbook\n\n"
                "## Trigger\n"
                "The On-Call Engineer starts a rollback when the Atlas Gateway error rate exceeds the alert "
                "threshold during a Severity One incident.\n\n"
                "## Steps\n"
                "The On-Call Engineer restores the previously approved deployment, verifies the Atlas Gateway "
                "health check, and records the rollback in the incident log."
            ),
        },
        {
            "source_id": "runbook/escalation.md",
            "team": "Reliability Team",
            "severity": "sev1",
            "system": "Atlas Gateway",
            "title": "Escalation Runbook",
            "text": (
                "# Escalation Runbook\n\n"
                "## Chain\n"
                "The On-Call Engineer escalates to the Incident Commander when a rollback does not recover the "
                "Atlas Gateway within fifteen minutes.\n\n"
                "## Breach\n"
                "The Incident Commander escalates a suspected breach to the Security Guild immediately."
            ),
        },
        {
            "source_id": "role/on-call-engineer.md",
            "team": "Reliability Team",
            "severity": "sev2",
            "system": "On Call",
            "title": "On-Call Engineer Charter",
            "text": (
                "# On-Call Engineer Charter\n\n"
                "## Duties\n"
                "The On-Call Engineer holds the pager, runs the Rollback Runbook, and posts an update every "
                "fifteen minutes during a Severity One incident."
            ),
        },
        {
            "source_id": "role/incident-commander.md",
            "team": "Reliability Team",
            "severity": "sev1",
            "system": "On Call",
            "title": "Incident Commander Charter",
            "text": (
                "# Incident Commander Charter\n\n"
                "## Duties\n"
                "The Incident Commander coordinates a Severity One incident, owns the incident bridge, and "
                "decides when to escalate to the Security Guild."
            ),
        },
        {
            "source_id": "service/atlas-gateway.md",
            "team": "Platform Team",
            "severity": "sev1",
            "system": "Atlas Gateway",
            "title": "Atlas Gateway Service",
            "text": (
                "# Atlas Gateway Service\n\n"
                "## Role\n"
                "The Atlas Gateway routes external traffic and enforces the tenant rate limit. "
                "The Platform Team owns the Atlas Gateway and supports it during a Severity One incident.\n\n"
                "## Recovery\n"
                "A saturated Atlas Gateway is recovered with the Rollback Runbook before any new deploy."
            ),
        },
        {
            "source_id": "service/beacon-pipeline.md",
            "team": "Reliability Team",
            "severity": "sev2",
            "system": "Beacon Pipeline",
            "title": "Beacon Pipeline Service",
            "text": (
                "# Beacon Pipeline Service\n\n"
                "## Role\n"
                "The Beacon Pipeline ingests telemetry and feeds the incident dashboard the Incident Commander "
                "watches during a Severity One incident."
            ),
        },
        {
            "source_id": "service/vault-service.md",
            "team": "Security Guild",
            "severity": "sev1",
            "system": "Vault Service",
            "title": "Vault Service",
            "text": (
                "# Vault Service\n\n"
                "## Secrets\n"
                "The Vault Service stores secrets and rotates encryption keys every ninety days. "
                "The Security Guild grants Vault Service access only to a named owner."
            ),
        },
        {
            "source_id": "policy/change-freeze.md",
            "team": "Platform Team",
            "severity": "sev2",
            "system": "Atlas Gateway",
            "title": "Change Freeze Policy",
            "text": (
                "# Change Freeze Policy\n\n"
                "## Rule\n"
                "The Platform Team enforces a change freeze on the Atlas Gateway during any active Severity One "
                "incident until the Incident Commander clears it."
            ),
        },
        {
            "source_id": "policy/breach-response.md",
            "team": "Security Guild",
            "severity": "sev1",
            "system": "Vault Service",
            "title": "Breach Response Policy",
            "text": (
                "# Breach Response Policy\n\n"
                "## Response\n"
                "The Security Guild leads a breach response, rotates every Vault Service key, and briefs the "
                "Incident Commander before the incident closes."
            ),
        },
        {
            "source_id": "incident/postmortem-gateway.md",
            "team": "Reliability Team",
            "severity": "sev1",
            "system": "Atlas Gateway",
            "title": "Postmortem Gateway Saturation",
            "text": (
                "# Postmortem Gateway Saturation\n\n"
                "## Summary\n"
                "The Atlas Gateway saturated, the On-Call Engineer ran the Rollback Runbook, and the Incident "
                "Commander coordinated the Severity One incident.\n\n"
                "## Follow up\n"
                "A corrective action and a named owner are recorded within two business days."
            ),
        },
        {
            "source_id": "incident/postmortem-breach.md",
            "team": "Security Guild",
            "severity": "sev1",
            "system": "Vault Service",
            "title": "Postmortem Access Breach",
            "text": (
                "# Postmortem Access Breach\n\n"
                "## Summary\n"
                "The Security Guild detected a breach, rotated the Vault Service keys, and the Incident Commander "
                "posted an update every fifteen minutes."
            ),
        },
        {
            "source_id": "incident/postmortem-pipeline.md",
            "team": "Reliability Team",
            "severity": "sev2",
            "system": "Beacon Pipeline",
            "title": "Postmortem Pipeline Lag",
            "text": (
                "# Postmortem Pipeline Lag\n\n"
                "## Summary\n"
                "The Beacon Pipeline lagged, the incident dashboard showed stale data, and the Reliability Team "
                "restored the Beacon Pipeline before the next release."
            ),
        },
        {
            "source_id": "policy/comms.md",
            "team": "Reliability Team",
            "severity": "sev1",
            "system": "On Call",
            "title": "Incident Communications Policy",
            "text": (
                "# Incident Communications Policy\n\n"
                "## Cadence\n"
                "The Incident Commander posts an update every fifteen minutes during a Severity One incident and "
                "a final summary when the On-Call Engineer confirms the Atlas Gateway is healthy."
            ),
        },
    ]


def ingest_documents(project: Any, documents: list[dict[str, str]]) -> None:
    for document in documents:
        project.runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id=document["source_id"],
                title=document["title"],
                media_type="text/markdown",
                content=document["text"].encode(),
                metadata={
                    "team": document["team"],
                    "severity": document["severity"],
                    "system": document["system"],
                },
            ),
        )


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
    description = "External paging side effect, denied under a read-only ceiling."

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del context, arguments
        return {"paged": True}


INCIDENT_QUESTION = (
    "During a Severity One incident on the Atlas Gateway, what rollback and escalation steps "
    "does the On-Call Engineer follow?"
)

TRAJECTORY_RECORDS = [
    {"id": "e1", "query": INCIDENT_QUESTION},
    {
        "id": "e2",
        "query": "During a Severity One incident, who coordinates the bridge and when is a breach escalated?",
    },
    {"id": "e3", "query": "During a Severity One incident, how often does the On-Call Engineer post an update?"},
]


def run_trajectory(runtime: Any, report: dict[str, Any]) -> None:
    project = runtime.project(workspace="helios", workspace_name="Helios Robotics", corpus="operations")
    ingest_documents(project, operations_documents())
    report["documents_ingested"] = project.stats()["documents"]

    registry = project.registry
    artifact = GroundedRagProgram().artifact()
    promoted_id = registry.save(artifact)
    registry.approve("grounded-rag", "production", promoted_id)
    registry.promote("grounded-rag", "production", promoted_id)
    resolved = registry.resolve("grounded-rag", "production")
    report["promoted_artifact"] = promoted_id
    report["registry_resolves_promotion"] = resolved["definition_hash"] == artifact.definition_hash

    playbook = ProgramSkill(
        project.programs.get("grounded-rag"),
        name=PLAYBOOK_TOOL,
        description="Run the promoted grounded incident program as one governed tool.",
    )
    report["pinned_artifact_id"] = playbook.artifact_id
    report["pinned_matches_promotion"] = playbook.artifact_id == promoted_id

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
    planner = KeywordToolPlanner({PLAYBOOK_TOOL: ("incident",), DIGEST_TOOL: ("incident",), PAGER_TOOL: ("incident",)})
    react = GovernedReActProgram.from_registry(policy=policy, planner=planner, registry=skills)

    result = react.execute(project, inputs={"question": INCIDENT_QUESTION}, user_id=None, session_id=None)
    steps = result.outputs["react_steps"]
    by_tool = {step["tool"]: step for step in steps}
    report["authorized_tools"] = sorted(step["tool"] for step in steps if step["authorized"])
    report["denied_tools"] = sorted(step["tool"] for step in steps if not step["authorized"])

    playbook_step = by_tool[PLAYBOOK_TOOL]
    observation = playbook_step["observation"]
    report["playbook_observation_grounded"] = bool(observation["answer"].strip()) and bool(observation["citations"])
    report["playbook_observation_artifact"] = observation["artifact_id"]
    report["pager_denied_empty"] = by_tool[PAGER_TOOL]["observation"] == {} and (
        "exceeds ceiling" in by_tool[PAGER_TOOL]["reason"]
    )
    report["answer_grounded"] = result.grounded and bool(result.citations)

    bound_react = project.programs.register(react)
    evaluation = ProgramEvaluator(bound_react).evaluate(
        Dataset.from_records("helios-trajectory", TRAJECTORY_RECORDS),
        trajectory_bundle(),
    )
    report["trajectory_score"] = round(evaluation.score, 4)
    report["hard_constraints_passed"] = evaluation.hard_constraints_passed
    report["trajectory_tool_safety"] = evaluation.aggregate["tool_safety"]
    report["trajectory_tool_authorization"] = evaluation.aggregate["tool_authorization"]
    report["trajectory_mean_tokens"] = round(evaluation.mean_tokens, 2)
    report["trajectory_cases"] = evaluation.case_count


def verify(report: dict[str, Any]) -> None:
    checks: dict[str, bool] = {
        "documents_ingested": report["documents_ingested"] == len(operations_documents()),
        "registry_resolves_promotion": report["registry_resolves_promotion"] is True,
        "pinned_matches_promotion": report["pinned_matches_promotion"] is True,
        "authorized_tools": report["authorized_tools"] == [DIGEST_TOOL, PLAYBOOK_TOOL],
        "denied_tools": report["denied_tools"] == [PAGER_TOOL],
        "playbook_observation_grounded": report["playbook_observation_grounded"] is True,
        "playbook_observation_artifact": report["playbook_observation_artifact"] == report["pinned_artifact_id"],
        "pager_denied_empty": report["pager_denied_empty"] is True,
        "answer_grounded": report["answer_grounded"] is True,
        "hard_constraints_passed": report["hard_constraints_passed"] is True,
        "trajectory_score": report["trajectory_score"] > 0.0,
        "trajectory_tool_safety": report["trajectory_tool_safety"] == 1.0,
        "trajectory_tool_authorization": report["trajectory_tool_authorization"] == 1.0,
        "trajectory_mean_tokens": report["trajectory_mean_tokens"] > 0.0,
        "trajectory_cases": report["trajectory_cases"] == len(TRAJECTORY_RECORDS),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"Program-as-skill trajectory assertions failed: {', '.join(failed)}")


def main() -> None:
    with TemporaryDirectory(prefix="refaat-trajectory-") as directory:
        root = Path(directory) / "main"
        root.mkdir(parents=True, exist_ok=True)
        report: dict[str, Any] = {}
        with RuntimeBuilder(demo_settings(root)).build() as runtime:
            run_trajectory(runtime, report)
        verify(report)
        emit(report)


if __name__ == "__main__":
    main()
