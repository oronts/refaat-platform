from __future__ import annotations

import pytest

from refaat.application.runtime import Refaat
from refaat.domain.models import DocumentInput

pytestmark = pytest.mark.integration


def _ingest(runtime: Refaat, *, workspace: str, source_id: str, content: str) -> tuple[str, str, set[str]]:
    project = runtime.project(workspace=workspace, corpus="handbook")
    runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(
            source_id=source_id,
            title=source_id,
            media_type="text/markdown",
            content=content.encode(),
        ),
    )
    stored = {
        chunk.id
        for document in project.list_documents()
        for chunk in runtime.store.list_document_chunks(project.workspace_id, project.corpus_id, str(document["id"]))
    }
    return project.workspace_id, project.corpus_id, stored


def test_two_tenants_never_see_each_others_evidence(runtime: Refaat) -> None:
    _, _, alpha_chunks = _ingest(
        runtime,
        workspace="tenant-alpha",
        source_id="alpha.md",
        content="Alpha Corp deploys the payment gateway with a mandatory dual-approval release gate.",
    )
    _, _, beta_chunks = _ingest(
        runtime,
        workspace="tenant-beta",
        source_id="beta.md",
        content="Beta Industries ships the logistics scheduler behind a single-owner rollback plan.",
    )
    assert alpha_chunks.isdisjoint(beta_chunks)

    alpha = runtime.project(workspace="tenant-alpha", corpus="handbook")
    beta = runtime.project(workspace="tenant-beta", corpus="handbook")

    alpha_answer = alpha.ask("How is the release gated?")
    assert alpha_answer.citations
    assert {citation.chunk_id for citation in alpha_answer.citations} <= alpha_chunks
    assert {citation.chunk_id for citation in alpha_answer.citations}.isdisjoint(beta_chunks)

    beta_answer = beta.ask("How is the release gated?")
    assert beta_answer.citations
    assert {citation.chunk_id for citation in beta_answer.citations} <= beta_chunks
    assert {citation.chunk_id for citation in beta_answer.citations}.isdisjoint(alpha_chunks)


def test_answer_cache_does_not_cross_tenants(runtime: Refaat) -> None:
    _ingest(
        runtime,
        workspace="cache-alpha",
        source_id="alpha.md",
        content="Alpha Corp release evidence is signed by the release manager before approval.",
    )
    _ingest(
        runtime,
        workspace="cache-beta",
        source_id="beta.md",
        content="Beta Industries release evidence is archived by the platform team after approval.",
    )
    question = "Who handles the release evidence?"

    alpha = runtime.project(workspace="cache-alpha", corpus="handbook")
    beta = runtime.project(workspace="cache-beta", corpus="handbook")

    first_alpha = alpha.ask(question)
    first_beta = beta.ask(question)
    second_alpha = alpha.ask(question)

    assert not first_alpha.cached
    # Beta's first ask is a fresh compute, never a serve of Alpha's cached answer.
    assert not first_beta.cached
    assert second_alpha.cached
    assert first_alpha.answer != first_beta.answer


def test_registry_and_api_keys_are_workspace_scoped(runtime: Refaat) -> None:
    alpha_ws = runtime.create_workspace("scope-alpha", "Alpha")
    beta_ws = runtime.create_workspace("scope-beta", "Beta")
    _, alpha_key = runtime.issue_api_key(alpha_ws, "alpha-admin", scopes=("read", "write", "admin"))
    _, beta_key = runtime.issue_api_key(beta_ws, "beta-admin", scopes=("read", "write", "admin"))

    alpha_workspace, alpha_scopes, _ = runtime.authentication.authenticate(alpha_key)
    beta_workspace, _, _ = runtime.authentication.authenticate(beta_key)

    assert alpha_workspace == alpha_ws
    assert beta_workspace == beta_ws
    assert alpha_workspace != beta_workspace
    assert "admin" in alpha_scopes
    # A key never authenticates into another tenant's workspace.
    assert runtime.authentication.authenticate(beta_key)[0] != alpha_ws
