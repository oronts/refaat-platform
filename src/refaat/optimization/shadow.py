from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from refaat.domain.kill_switch import ensure_workspace_active
from refaat.domain.models import DocumentInput

if TYPE_CHECKING:
    from refaat.application.composition import ShadowComposition
    from refaat.application.runtime import RefaatProject


def active_documents(project: RefaatProject) -> list[DocumentInput]:
    """Read the corpus documents with their raw content so a shadow runtime can re-ingest them."""

    ensure_workspace_active(project.runtime.kill_switches, project.workspace_id)
    runtime = project.runtime
    documents: list[DocumentInput] = []
    for row in runtime.store.list_documents(project.workspace_id, project.corpus_id):
        with runtime.ingestion.blob_store.open(str(row["blob_uri"])) as handle:
            content = handle.read()
        documents.append(
            DocumentInput(
                source_id=str(row["source_id"]),
                title=str(row["title"]),
                media_type=str(row["media_type"]),
                content=content,
            )
        )
    return documents


@contextmanager
def shadow_index(
    project: RefaatProject,
    documents: Sequence[DocumentInput],
    overrides: dict[str, object],
) -> Iterator[tuple[RefaatProject, ShadowComposition]]:
    """Build a throwaway runtime that reproduces the production composition, re-ingest, yield, discard.

    The shadow reproduces the stateless model and text components the production runtime was built
    with and refuses the stateful, store-bound, governance, and side-effecting ones, yielding both the
    shadow project and a report of what it reproduced and left out. The active corpus and its index are
    never touched: the shadow uses a separate temporary SQLite database and blob directory, and the
    runtime is closed and the directory removed on exit.
    """

    from refaat.application.runtime import RuntimeBuilder

    base = project.runtime.settings
    with TemporaryDirectory(prefix="refaat-shadow-") as directory:
        root = Path(directory)
        shadow_settings = base.model_copy(
            update={
                "environment": "test",
                "database_url": f"sqlite:///{root / 'shadow.db'}",
                "auto_create_schema": True,
                "blob_backend": "local",
                "blob_root": root / "blobs",
                "connector_file_root": root / "connectors",
                "skill_root": root / "skills",
                "otel_enabled": False,
                # The shadow refuses the injected audit sink, so it must not run fail-closed or durable
                # audit: build() raises when audit_fail_closed is set with no sink, and durable audit
                # would persist events for a worker that never runs.
                "audit_fail_closed": False,
                "audit_durable": False,
                # The shadow re-ingests already-authorized documents into a throwaway runtime, so it
                # carries no tenant quota: otherwise a base ingest_requests rule would throttle an
                # internal optimization run once the corpus exceeds the limit.
                "quota_backend": "none",
                "quota_rules": [],
                **overrides,
                # After overrides so a caller can never point the shadow at a production Redis cache.
                "answer_cache_backend": "sql",
                "model_call_cache_backend": "sql",
            }
        )
        shadow_builder = RuntimeBuilder(shadow_settings)
        report = shadow_builder.reproduce_from(project.runtime.composition)
        shadow = shadow_builder.build()
        try:
            shadow_project = shadow.project(workspace="shadow", corpus="shadow")
            for document in documents:
                shadow.ingest(shadow_project.workspace_id, shadow_project.corpus_id, document)
            yield shadow_project, report
        finally:
            shadow.close()
