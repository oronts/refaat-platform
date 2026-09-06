from __future__ import annotations

import json
import mimetypes
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from refaat.application.derivation import DerivationService, compare_manifests
from refaat.application.kill_switch import KillSwitchService
from refaat.application.legal_hold import LegalHoldService
from refaat.application.runtime import ProjectRef, Refaat, RefaatProject, RuntimeBuilder
from refaat.application.status import StatusService
from refaat.application.worker import DurableWorker
from refaat.config import Settings, get_settings
from refaat.domain.kill_switch import KillSwitchScope
from refaat.domain.models import DocumentInput, QueryRequest, RlmLimits

app = typer.Typer(name="refaat", no_args_is_help=True, help="Refaat adaptive context runtime")
corpus_app = typer.Typer(no_args_is_help=True)
derivation_app = typer.Typer(no_args_is_help=True)
document_app = typer.Typer(no_args_is_help=True)
evaluation_app = typer.Typer(no_args_is_help=True)
skill_app = typer.Typer(no_args_is_help=True)
program_app = typer.Typer(no_args_is_help=True)
canary_app = typer.Typer(no_args_is_help=True)
fine_tune_app = typer.Typer(no_args_is_help=True)
legal_hold_app = typer.Typer(no_args_is_help=True)
kill_switch_app = typer.Typer(no_args_is_help=True)
plugin_app = typer.Typer(no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")
corpus_app.add_typer(derivation_app, name="derivations")
app.add_typer(document_app, name="document")
app.add_typer(evaluation_app, name="evaluation")
app.add_typer(skill_app, name="skill")
app.add_typer(program_app, name="program")
program_app.add_typer(canary_app, name="canary")
app.add_typer(fine_tune_app, name="fine-tune")
app.add_typer(legal_hold_app, name="legal-hold")
app.add_typer(kill_switch_app, name="kill-switch")
app.add_typer(plugin_app, name="plugin")


@contextmanager
def runtime_context(*, legal_hold_management: bool = False, kill_switch_management: bool = False) -> Iterator[Refaat]:
    settings = Settings(kill_switch_backend="database") if kill_switch_management else None
    builder = RuntimeBuilder(settings)
    if legal_hold_management:
        builder = builder.with_legal_hold_registry()
    runtime = builder.build()
    try:
        yield runtime
    finally:
        runtime.close()


def emit(value: object) -> None:
    typer.echo(json.dumps(value, indent=2, default=str, ensure_ascii=False))


@app.command("init")
def initialize(
    slug: Annotated[str, typer.Option(prompt=True)],
    name: Annotated[str, typer.Option(prompt=True)],
) -> None:
    """Create a workspace and its first administrator key."""
    with runtime_context() as runtime:
        workspace_id = runtime.create_workspace(slug, name)
        key_id, raw = runtime.issue_api_key(
            workspace_id,
            "local-admin",
            scopes=("read", "write", "admin"),
        )
        emit({"workspace_id": workspace_id, "api_key_id": key_id, "api_key": raw})


@app.command()
def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """Run the FastAPI process."""
    import uvicorn

    # Launch the configured worker count so REFAAT_API_WORKERS is honored, not just
    # validated. uvicorn ignores workers under reload, so keep the default single
    # process there; the shared rate limiter is enforced at app construction.
    settings = get_settings()
    workers = None if reload else settings.api_workers
    uvicorn.run(
        "refaat.api.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
        workers=workers,
    )


@app.command()
def worker(once: bool = False) -> None:
    """Run the durable background worker."""
    with runtime_context() as runtime:
        durable = DurableWorker(runtime)
        if once:
            emit({"worked": durable.run_once()})
        else:
            durable.run_forever()


@app.command()
def doctor() -> None:
    """Validate configuration and core dependencies."""
    settings = get_settings()
    with runtime_context() as runtime:
        runtime.store.ping()
        emit(
            {
                "status": "ok",
                "environment": settings.environment,
                "database": settings.database_url.split(":", 1)[0],
                "blob_backend": settings.blob_backend,
                "embedding_provider": settings.embedding_provider,
                "generation_provider": settings.generation_provider,
                "graph_provider": settings.graph_provider,
                "media_types": sorted(runtime.ingestion.extractors.media_types),
                "connector_schemes": sorted(runtime.connectors.schemes),
            }
        )


@corpus_app.command("create")
def create_corpus(workspace_id: str, name: str) -> None:
    with runtime_context() as runtime:
        corpus_id = runtime.create_corpus(workspace_id, name)
        emit(asdict(runtime.store.get_corpus(workspace_id, corpus_id)))


def _derivation_service(runtime: Refaat) -> DerivationService:
    if runtime.derivation is None:
        raise typer.BadParameter("Derivation manifests are not enabled")
    return runtime.derivation


@derivation_app.command("list")
def list_derivations(workspace_id: str, corpus_id: str) -> None:
    with runtime_context() as runtime:
        emit([asdict(manifest) for manifest in _derivation_service(runtime).manifests(workspace_id, corpus_id)])


@derivation_app.command("get")
def get_derivation(workspace_id: str, corpus_id: str, derivation_key: str) -> None:
    with runtime_context() as runtime:
        manifest = _derivation_service(runtime).manifest(workspace_id, corpus_id, derivation_key)
        if manifest is None:
            raise typer.BadParameter("No derivation manifest for that key")
        emit(asdict(manifest))


@derivation_app.command("verify")
def verify_derivation(workspace_id: str, corpus_id: str) -> None:
    with runtime_context() as runtime:
        emit({"corpus_id": corpus_id, "status": _derivation_service(runtime).verify_detail(workspace_id, corpus_id)})


@derivation_app.command("compare")
def compare_derivations(workspace_id: str, corpus_id: str, left: str, right: str) -> None:
    with runtime_context() as runtime:
        service = _derivation_service(runtime)
        left_manifest = service.manifest(workspace_id, corpus_id, left)
        right_manifest = service.manifest(workspace_id, corpus_id, right)
        if left_manifest is None or right_manifest is None:
            raise typer.BadParameter("Both derivation keys must exist to compare")
        emit(compare_manifests(left_manifest, right_manifest))


@derivation_app.command("verify-document")
def verify_document_derivation(workspace_id: str, corpus_id: str, document_id: str) -> None:
    with runtime_context() as runtime:
        status = _derivation_service(runtime).verify_document(workspace_id, corpus_id, document_id)
        emit({"document_id": document_id, "status": status})


@corpus_app.command("list")
def list_corpora(workspace_id: str) -> None:
    with runtime_context() as runtime:
        emit([asdict(corpus) for corpus in runtime.store.list_corpora(workspace_id)])


@document_app.command("erase")
def erase_document(workspace_id: str, corpus_id: str, document_id: str) -> None:
    """Completely erase a document and every auto-derived evidence surface it produced."""
    with runtime_context() as runtime:
        receipt = runtime.erasure.erase(workspace_id, corpus_id, document_id)
        emit(asdict(receipt))


def _legal_hold_service(runtime: Refaat, workspace_id: str) -> LegalHoldService:
    return LegalHoldService(
        runtime.store,
        workspace_id,
        audit_sink=runtime.audit_sink,
        audit_fail_closed=runtime.settings.audit_fail_closed,
        deferred_processor=runtime.erasure.deferred_processor,
    )


@legal_hold_app.command("place")
def place_hold(
    workspace_id: str,
    corpus_id: str,
    placed_by: str,
    document_id: str | None = None,
    reason: str | None = None,
) -> None:
    """Place a legal hold on a document, or corpus-wide when document_id is omitted."""
    with runtime_context(legal_hold_management=True) as runtime:
        service = _legal_hold_service(runtime, workspace_id)
        hold_id = service.place(corpus_id=corpus_id, document_id=document_id, placed_by=placed_by, reason=reason)
        emit({"hold_id": hold_id})


@legal_hold_app.command("lift")
def lift_hold(workspace_id: str, hold_id: str, lifted_by: str) -> None:
    """Lift an active legal hold."""
    with runtime_context(legal_hold_management=True) as runtime:
        service = _legal_hold_service(runtime, workspace_id)
        emit({"lifted": service.lift(hold_id, lifted_by=lifted_by)})


@legal_hold_app.command("list")
def list_holds(workspace_id: str, corpus_id: str | None = None) -> None:
    """List the active legal holds for a workspace, optionally scoped to one corpus."""
    with runtime_context(legal_hold_management=True) as runtime:
        service = _legal_hold_service(runtime, workspace_id)
        emit([asdict(record) for record in service.list_active(corpus_id)])


def _kill_switch_service(runtime: Refaat, workspace_id: str) -> KillSwitchService:
    return KillSwitchService(
        runtime.store,
        workspace_id,
        audit_sink=runtime.audit_sink,
        audit_fail_closed=runtime.settings.audit_fail_closed,
    )


def _kill_switch_scope(scope: str) -> KillSwitchScope:
    try:
        return KillSwitchScope(scope)
    except ValueError:
        raise typer.BadParameter(f"Unknown kill-switch scope: {scope}") from None


@kill_switch_app.command("engage")
def engage_kill_switch(scope: str, engaged_by: str, workspace_id: str = "", reason: str | None = None) -> None:
    """Engage an incident kill switch for a workspace, or globally when workspace_id is omitted."""
    with runtime_context(kill_switch_management=True) as runtime:
        service = _kill_switch_service(runtime, workspace_id)
        switch_id = service.engage(scope=_kill_switch_scope(scope), engaged_by=engaged_by, reason=reason)
        emit({"switch_id": switch_id})


@kill_switch_app.command("disengage")
def disengage_kill_switch(scope: str, disengaged_by: str, workspace_id: str = "") -> None:
    """Disengage an incident kill switch. This is the operator path that lifts a workspace freeze."""
    with runtime_context(kill_switch_management=True) as runtime:
        service = _kill_switch_service(runtime, workspace_id)
        emit({"disengaged": service.disengage(scope=_kill_switch_scope(scope), disengaged_by=disengaged_by)})


@kill_switch_app.command("list")
def list_kill_switches(workspace_id: str = "") -> None:
    """List the active incident kill switches for a workspace (an empty workspace lists globals)."""
    with runtime_context(kill_switch_management=True) as runtime:
        service = _kill_switch_service(runtime, workspace_id)
        emit([asdict(record) for record in service.list_engaged()])


@app.command("status")
def status(
    workspace_id: str = "",
    operator: bool = typer.Option(False, "--operator/--no-operator"),
) -> None:
    """Operator status plane: the posture of every governance subsystem. Reads in-process and is
    freeze-exempt, so it stays available during an incident. An empty workspace or --operator reads
    the process-wide inventory and the global kill-switch state."""
    with runtime_context(legal_hold_management=True, kill_switch_management=True) as runtime:
        op = operator or not workspace_id
        plane = StatusService(runtime).snapshot(workspace_id=(workspace_id or None), operator=op)
        emit(plane.to_json())


@app.command()
def ingest(
    workspace_id: str,
    corpus_id: str,
    path: Path,
    source_id: str | None = None,
    title: str | None = None,
    media_type: str | None = None,
) -> None:
    """Ingest a local document through the same production pipeline."""
    if not path.is_file():
        raise typer.BadParameter("path must be an existing file")
    detected = media_type or ({".md": "text/markdown"}.get(path.suffix.casefold()))
    detected = detected or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    with runtime_context() as runtime:
        result = runtime.ingest(
            workspace_id,
            corpus_id,
            DocumentInput(
                source_id=source_id or path.name,
                title=title or path.name,
                media_type=detected,
                content=path.read_bytes(),
                metadata={"file_name": path.name},
            ),
        )
        emit(asdict(result))


@app.command("ingest-image")
def ingest_image(
    workspace_id: str,
    corpus_id: str,
    path: Path,
    source_id: str | None = None,
    title: str | None = None,
    media_type: str | None = None,
    caption: str | None = None,
    alt_text: str | None = None,
) -> None:
    """Ingest a local image as a citable document backed by a descriptor chunk."""
    if not path.is_file():
        raise typer.BadParameter("path must be an existing file")
    detected = media_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    with runtime_context() as runtime:
        result = runtime.ingest_image(
            workspace_id,
            corpus_id,
            content=path.read_bytes(),
            media_type=detected,
            source_id=source_id or path.name,
            title=title or path.name,
            caption=caption,
            alt_text=alt_text,
            metadata={"file_name": path.name},
        )
        emit(asdict(result))


@app.command("ingest-source")
def ingest_source(
    workspace_id: str,
    corpus_id: str,
    source: str,
    source_id: str | None = None,
    title: str | None = None,
) -> None:
    """Ingest a registered file://, http://, https://, or plugin source."""
    with runtime_context() as runtime:
        result = runtime.ingest_source(
            workspace_id,
            corpus_id,
            source,
            source_id=source_id,
            title=title,
        )
        emit(asdict(result))


@app.command()
def query(
    workspace_id: str,
    corpus_id: str,
    text: str,
    user_id: str | None = None,
    session_id: str | None = None,
) -> None:
    """Query a corpus and print the grounded response."""
    with runtime_context() as runtime:
        result = runtime.query(
            QueryRequest(
                workspace_id=workspace_id,
                corpus_id=corpus_id,
                query=text,
                user_id=user_id,
                session_id=session_id,
            )
        )
        emit(asdict(result))


@app.command()
def rlm(
    workspace_id: str,
    corpus_id: str,
    text: str,
    user_id: str | None = None,
    session_id: str | None = None,
    max_depth: int = 2,
    max_nodes: int = 16,
    branch_factor: int = 4,
    leaf_token_budget: int = 1_500,
    background: bool = False,
) -> None:
    """Run bounded recursive inference over externalized corpus context."""
    limits = RlmLimits(
        max_depth=max_depth,
        max_nodes=max_nodes,
        branch_factor=branch_factor,
        leaf_token_budget=leaf_token_budget,
    )
    with runtime_context() as runtime:
        if background:
            job_id = runtime.enqueue_rlm(
                workspace_id,
                corpus_id,
                text,
                user_id=user_id,
                session_id=session_id,
                limits=limits,
            )
            emit({"job_id": job_id})
            return
        result = runtime.run_rlm(
            workspace_id,
            corpus_id,
            text,
            user_id=user_id,
            session_id=session_id,
            limits=limits,
        )
        emit(asdict(result))


@evaluation_app.command("create")
def create_evaluation(workspace_id: str, name: str, cases_file: Path) -> None:
    cases = json.loads(cases_file.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise typer.BadParameter("cases_file must contain a JSON array")
    with runtime_context() as runtime:
        suite_id = runtime.store.create_evaluation_suite(workspace_id, name, cases)
        emit({"suite_id": suite_id})


@evaluation_app.command("run")
def run_evaluation(workspace_id: str, suite_id: str, corpus_id: str) -> None:
    with runtime_context() as runtime:
        run_id = runtime.evaluations.run(workspace_id, suite_id, corpus_id)
        emit(runtime.store.get_evaluation_run(workspace_id, run_id))


@skill_app.command("list")
def list_skills() -> None:
    with runtime_context() as runtime:
        emit(runtime.skill_registry.list())


@skill_app.command("create")
def create_skill(name: str, description: str) -> None:
    with runtime_context() as runtime:
        emit({"path": str(runtime.skill_creator.create(name, description))})


@plugin_app.command("ports")
def list_ports() -> None:
    """List every typed port a plugin can implement, with its tier and injection seam."""
    from refaat.devkit.catalog import PORT_CATALOG

    emit(
        [
            {"port": spec.name, "tier": spec.tier, "seam": spec.seam, "members": list(spec.members)}
            for spec in PORT_CATALOG
        ]
    )


@plugin_app.command("describe")
def describe_port(port: str) -> None:
    """Show how to implement and install an adapter for a single port."""
    from refaat.devkit.catalog import get_port

    spec = get_port(port)
    if spec is None:
        raise typer.BadParameter(f"Unknown port {port!r}")
    emit(
        {
            "port": spec.name,
            "members": list(spec.members),
            "tier": spec.tier,
            "seam": spec.seam,
            "default": spec.default,
            "conformance_base": spec.conformance_base,
            "notes": spec.notes,
        }
    )


@plugin_app.command("new")
def new_plugin(
    port: str,
    name: str,
    out: Annotated[Path, typer.Option(help="Directory to write the plugin project into")] = Path("."),
) -> None:
    """Scaffold an installable plugin package implementing a port."""
    from refaat.devkit.scaffold import PortScaffolder

    emit({"path": str(PortScaffolder(out).create(port, name))})


@plugin_app.command("verify")
def verify_plugin(
    target: Annotated[str, typer.Argument(help="The adapter as module.path:ClassName")],
    port: Annotated[str, typer.Option(help="The port the adapter implements")],
) -> None:
    """Check that an adapter satisfies a port's declared members; exits non-zero if it does not."""
    import importlib

    from refaat.conformance import structural_conformance
    from refaat.devkit.catalog import get_port

    if get_port(port) is None:
        raise typer.BadParameter(f"Unknown port {port!r}")
    module_name, _, attr = target.partition(":")
    if not module_name or not attr:
        raise typer.BadParameter("target must be 'module.path:ClassName'")
    try:
        adapter = getattr(importlib.import_module(module_name), attr)
    except Exception as exc:
        raise typer.BadParameter(f"could not import {target!r}: {exc}") from exc
    violations = structural_conformance(adapter, port)
    emit(
        {
            "port": port,
            "conformant": not violations,
            "violations": [{"member": v.member, "problem": v.problem} for v in violations],
        }
    )
    if violations:
        raise typer.Exit(code=1)


@plugin_app.command("pin")
def pin_plugin(
    target: Annotated[str, typer.Argument(help="The entry point as group:name, e.g. refaat.skills:my_skill")],
) -> None:
    """Compute the content-hash pin for an installed plugin and print the REFAAT_PLUGIN_PINS fragment.

    Reads the distribution's shipped source without importing the plugin, so an untrusted entry point is
    never executed to pin it.
    """
    from refaat.errors import ConfigurationError
    from refaat.infrastructure.plugins import PLUGIN_PIN_GROUPS, pin_for_entry_point

    group, _, name = target.partition(":")
    if not name:
        raise typer.BadParameter("target must be 'group:name'")
    if group not in PLUGIN_PIN_GROUPS:
        raise typer.BadParameter(f"unknown entry-point group {group!r}; one of {', '.join(PLUGIN_PIN_GROUPS)}")
    try:
        descriptor = pin_for_entry_point(group, name)
    except ConfigurationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    emit(
        {
            f"{group}:{name}": {
                "value": descriptor.value,
                "dist_name": descriptor.dist_name,
                "dist_version": descriptor.dist_version,
                "code_hash": descriptor.code_hash,
            }
        }
    )


@program_app.command("list")
def list_programs(workspace_id: str, corpus_id: str) -> None:
    """List typed programs available to a workspace and corpus."""
    with runtime_context() as runtime:
        project = RefaatProject(runtime, ProjectRef(workspace_id, corpus_id))
        emit(project.programs.list())


@program_app.command("run")
def run_program(
    workspace_id: str,
    corpus_id: str,
    name: str,
    question: str,
    user_id: str | None = None,
    session_id: str | None = None,
) -> None:
    """Run a typed program and print its grounded, typed result."""
    with runtime_context() as runtime:
        project = RefaatProject(runtime, ProjectRef(workspace_id, corpus_id))
        result = project.programs.get(name).run(question=question, user_id=user_id, session_id=session_id)
        emit(result.to_json())


@program_app.command("eval")
def eval_program(workspace_id: str, corpus_id: str, name: str, cases_file: Path) -> None:
    """Evaluate a program over a JSON dataset with the default metric bundle."""
    from refaat.evaluation import Dataset, default_bundle

    records = json.loads(cases_file.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise typer.BadParameter("cases_file must contain a JSON array")
    with runtime_context() as runtime:
        project = RefaatProject(runtime, ProjectRef(workspace_id, corpus_id))
        dataset = Dataset.from_records(cases_file.stem, records)
        emit(project.programs.get(name).evaluate(dataset, default_bundle()).to_json())


@program_app.command("optimize")
def optimize_program(workspace_id: str, corpus_id: str, name: str, cases_file: Path) -> None:
    """Search a program's tunable parameters against a JSON dataset."""
    from refaat.evaluation import Dataset

    records = json.loads(cases_file.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise typer.BadParameter("cases_file must contain a JSON array")
    with runtime_context() as runtime:
        project = RefaatProject(runtime, ProjectRef(workspace_id, corpus_id))
        dataset = Dataset.from_records(cases_file.stem, records)
        emit(project.programs.get(name).optimize(dataset).to_json())


@program_app.command("optimize-job")
def optimize_job(workspace_id: str, corpus_id: str, name: str, cases_file: Path) -> None:
    """Enqueue a durable, resumable optimization run against a JSON dataset."""
    from refaat.evaluation import Dataset

    records = json.loads(cases_file.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise typer.BadParameter("cases_file must contain a JSON array")
    with runtime_context() as runtime:
        project = RefaatProject(runtime, ProjectRef(workspace_id, corpus_id))
        dataset = Dataset.from_records(cases_file.stem, records)
        run_id = project.programs.get(name).enqueue_optimization(dataset)
        emit({"run_id": run_id})


@program_app.command("optimization")
def show_optimization(workspace_id: str, run_id: str) -> None:
    """Show a durable optimization run's status, best artifact, and result."""
    with runtime_context() as runtime:
        emit(runtime.store.get_optimization_run(workspace_id, run_id))


@program_app.command("promote")
def promote_program(workspace_id: str, corpus_id: str, name: str, environment: str) -> None:
    """Save the current definition of a program and promote it in an environment."""
    with runtime_context() as runtime:
        project = RefaatProject(runtime, ProjectRef(workspace_id, corpus_id))
        artifact = project.programs.get(name).program.artifact()
        project.registry.save(artifact)
        emit(project.registry.promote(name, environment, artifact.artifact_id))


@program_app.command("approve")
def approve_program(workspace_id: str, corpus_id: str, name: str, environment: str) -> None:
    """Save the current definition of a program and approve it for an environment."""
    with runtime_context() as runtime:
        project = RefaatProject(runtime, ProjectRef(workspace_id, corpus_id))
        artifact = project.programs.get(name).program.artifact()
        project.registry.save(artifact)
        emit(project.registry.approve(name, environment, artifact.artifact_id))


@program_app.command("resolve")
def resolve_program(workspace_id: str, name: str, environment: str) -> None:
    """Show the artifact currently deployed for a program in an environment."""
    with runtime_context() as runtime:
        emit(runtime.registry.resolve(workspace_id, name, environment))


@program_app.command("rollback")
def rollback_program(workspace_id: str, name: str, environment: str) -> None:
    """Roll an environment back to the previously deployed artifact."""
    with runtime_context() as runtime:
        emit(runtime.registry.rollback(workspace_id, name, environment))


@canary_app.command("set")
def canary_set(
    workspace_id: str,
    corpus_id: str,
    name: str,
    environment: str,
    candidate_artifact_id: str,
    percent: int = 0,
    compare: bool = True,
) -> None:
    """Attach a canary candidate (percent 0 is pure shadow) to an environment."""
    with runtime_context() as runtime:
        project = RefaatProject(runtime, ProjectRef(workspace_id, corpus_id))
        emit(project.deployment.set_canary(name, environment, candidate_artifact_id, percent=percent, compare=compare))


@canary_app.command("clear")
def canary_clear(workspace_id: str, corpus_id: str, name: str, environment: str) -> None:
    """Detach the canary candidate from an environment."""
    with runtime_context() as runtime:
        project = RefaatProject(runtime, ProjectRef(workspace_id, corpus_id))
        emit(project.deployment.clear_canary(name, environment))


@canary_app.command("status")
def canary_status(workspace_id: str, corpus_id: str, name: str, environment: str) -> None:
    """Show the attached canary and recent shadow telemetry for an environment."""
    with runtime_context() as runtime:
        project = RefaatProject(runtime, ProjectRef(workspace_id, corpus_id))
        emit(
            {
                "canary": project.deployment.current_canary(name, environment),
                "shadow_runs": project.deployment.shadow_runs(name, environment),
            }
        )


@program_app.command("deploy-run")
def deploy_run(
    workspace_id: str,
    corpus_id: str,
    name: str,
    environment: str,
    question: str,
    user_id: str | None = None,
    session_id: str | None = None,
) -> None:
    """Serve a question through the environment's active artifact and optional canary."""
    with runtime_context() as runtime:
        project = RefaatProject(runtime, ProjectRef(workspace_id, corpus_id))
        result = project.deployment.run(
            name, environment, inputs={"question": question}, user_id=user_id, session_id=session_id
        )
        emit(result.to_json())


@fine_tune_app.command("request")
def fine_tune_request(
    workspace_id: str,
    base_model: str,
    cases_file: Path,
    requester: str = "cli",
) -> None:
    """Create a PENDING, unapproved fine-tune run over a JSON training dataset."""
    from refaat.evaluation import Dataset

    records = json.loads(cases_file.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise typer.BadParameter("cases_file must contain a JSON array")
    with runtime_context() as runtime:
        dataset = Dataset.from_records(cases_file.stem, records)
        run_id = runtime.fine_tuning.request(workspace_id, dataset, base_model=base_model, requester=requester)
        emit({"run_id": run_id})


@fine_tune_app.command("approve")
def fine_tune_approve(
    workspace_id: str,
    run_id: str,
    approver: str = typer.Option(..., help="A distinct approver; four-eyes forbids the requester approving."),
) -> None:
    """Approve the training data of a fine-tune run so it may execute."""
    with runtime_context() as runtime:
        emit(runtime.fine_tuning.approve(workspace_id, run_id, approver))


@fine_tune_app.command("enqueue")
def fine_tune_enqueue(workspace_id: str, run_id: str) -> None:
    """Enqueue a durable fine-tune job for an approved run."""
    with runtime_context() as runtime:
        emit({"job_id": runtime.fine_tuning.enqueue(workspace_id, run_id)})


@fine_tune_app.command("run")
def fine_tune_run(workspace_id: str, run_id: str) -> None:
    """Execute an approved fine-tune run inline (refuses an unapproved run)."""
    with runtime_context() as runtime:
        emit(runtime.fine_tuning.run(workspace_id, run_id))


@fine_tune_app.command("status")
def fine_tune_status(workspace_id: str, run_id: str) -> None:
    """Show a fine-tune run's status, approval, model reference, and provenance."""
    with runtime_context() as runtime:
        emit(runtime.store.get_fine_tune_run(workspace_id, run_id))


if __name__ == "__main__":
    app()
