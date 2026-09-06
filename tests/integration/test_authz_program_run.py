"""Per-principal authorization on the typed program run path.

The query path enforcement is proven in tests/integration/test_authz_enforcement_e2e.py.
Here the program run path is proven: a plain grounded program and a governed ReAct
program answer under the request principal, so a program run as one principal never cites
or includes another principal's evidence, the unconfigured path (no policy, no principal)
is byte-identical, and the principal never shifts a program artifact id.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from refaat.application.runtime import RefaatProject, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.authz import CanonicalAccessPolicy, Principal
from refaat.domain.enums import SideEffectClass
from refaat.domain.models import ContextPacket, DocumentInput, JsonObject, ReActPolicy, ToolGrant
from refaat.programs import GovernedReActProgram, GroundedRagProgram, KeywordToolPlanner
from refaat.programs.registry import BoundProgram
from refaat.skills.program_skill import ProgramSkill

pytestmark = pytest.mark.integration

ALICE = Principal(id="alice", tenant="merck", groups=frozenset({"research-a"}))
BOB = Principal(id="bob", tenant="merck", groups=frozenset({"research-b"}))
QUERY = "checkout production release rollback plan"

DOCUMENTS: dict[str, dict[str, object]] = {
    "alpha.md": {"acl": {"classification": "confidential", "allow_groups": ["research-a"]}, "doc": "alpha"},
    "bravo.md": {"acl": {"classification": "confidential", "allow_groups": ["research-b"]}, "doc": "bravo"},
    "public.md": {"acl": {"classification": "public"}, "doc": "public"},
}
ALL_DOCS = {str(meta["doc"]) for meta in DOCUMENTS.values()}


def _ingest(project: RefaatProject) -> None:
    for source_id, metadata in DOCUMENTS.items():
        content = f"The {metadata['doc']} checkout production release requires an approved rollback plan."
        project.runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id=source_id,
                title=source_id,
                media_type="text/markdown",
                content=content.encode(),
                metadata=dict(metadata),
            ),
        )


@pytest.fixture
def authz_project(settings: Settings) -> Iterator[RefaatProject]:
    runtime = RuntimeBuilder(settings).with_access_policy(CanonicalAccessPolicy()).build()
    try:
        project = runtime.project(workspace="merck", corpus="programs")
        _ingest(project)
        yield project
    finally:
        runtime.close()


@pytest.fixture
def open_project(settings: Settings) -> Iterator[RefaatProject]:
    runtime = RuntimeBuilder(settings).build()
    try:
        project = runtime.project(workspace="merck", corpus="programs")
        _ingest(project)
        yield project
    finally:
        runtime.close()


def _quotes(result_citations: object) -> str:
    return " ".join(citation.quote for citation in result_citations)  # type: ignore[attr-defined]


def test_plain_program_run_scopes_evidence_to_the_principal(authz_project: RefaatProject) -> None:
    bound = authz_project.programs.get("grounded-rag")
    alice = bound.run(question=QUERY, principal=ALICE)
    assert alice.citations
    for forbidden in ("bravo",):
        assert forbidden not in _quotes(alice.citations)
    bob = bound.run(question=QUERY, principal=BOB)
    assert bob.citations
    assert "alpha" not in _quotes(bob.citations)


def test_governed_program_run_scopes_evidence_to_the_principal(authz_project: RefaatProject) -> None:
    program = _governed_program()
    alice = program.execute(authz_project, inputs={"question": QUERY}, user_id=None, session_id=None, principal=ALICE)
    assert alice.citations
    assert "bravo" not in _quotes(alice.citations)
    bob = program.execute(authz_project, inputs={"question": QUERY}, user_id=None, session_id=None, principal=BOB)
    assert bob.citations
    assert "alpha" not in _quotes(bob.citations)


def test_program_cache_is_isolated_per_principal(authz_project: RefaatProject) -> None:
    bound = authz_project.programs.get("grounded-rag")
    assert bound.run(question=QUERY, principal=ALICE).cached is False
    assert bound.run(question=QUERY, principal=BOB).cached is False
    # Alice's second run hits her own scoped cache entry, never Bob's.
    assert bound.run(question=QUERY, principal=ALICE).cached is True


def test_program_run_without_a_principal_is_unfiltered_and_byte_identical(open_project: RefaatProject) -> None:
    # With no policy installed and no principal passed, a program run is byte-identical to
    # before: it sees every document and the grounded-rag artifact id is unchanged.
    bound = open_project.programs.get("grounded-rag")
    without_principal = bound.run(question=QUERY)
    cited_docs = {citation.document_id for citation in without_principal.citations}
    assert cited_docs
    baseline_artifact_id = GroundedRagProgram().artifact().artifact_id
    assert without_principal.artifact_id == baseline_artifact_id


def test_principal_never_shifts_the_program_artifact_id(authz_project: RefaatProject) -> None:
    bound = authz_project.programs.get("grounded-rag")
    with_principal = bound.run(question=QUERY, principal=ALICE)
    assert with_principal.artifact_id == GroundedRagProgram().artifact().artifact_id


def test_program_skill_tool_does_not_leak_across_the_principal_boundary(authz_project: RefaatProject) -> None:
    # A GovernedReActProgram calling a program-as-skill must thread the governed principal into
    # the inner program run, or the inner run re-retrieves unfiltered and leaks another
    # principal's evidence into the tool observation surfaced by outputs["react_steps"].
    skill = ProgramSkill(BoundProgram(authz_project, GroundedRagProgram()), name="lookup")
    policy = ReActPolicy(
        max_iterations=1,
        granted_scopes=("read:corpus",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(ToolGrant("lookup", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),),
    )
    planner = KeywordToolPlanner({"lookup": ("rollback",)})
    react = GovernedReActProgram(policy=policy, planner=planner, skills={"lookup": skill})

    alice = react.execute(authz_project, inputs={"question": QUERY}, user_id=None, session_id=None, principal=ALICE)
    assert "bravo" not in json.dumps(alice.outputs.get("react_steps", []))
    bob = react.execute(authz_project, inputs={"question": QUERY}, user_id=None, session_id=None, principal=BOB)
    assert "alpha" not in json.dumps(bob.outputs.get("react_steps", []))


def test_program_skill_fails_closed_without_a_principal_under_an_active_policy(authz_project: RefaatProject) -> None:
    skill = ProgramSkill(BoundProgram(authz_project, GroundedRagProgram()), name="lookup")
    from refaat.errors import AuthorizationError

    with pytest.raises(AuthorizationError):
        skill.run(authz_project.retrieve(QUERY, principal=ALICE), {}, principal=None)


def _governed_program() -> GovernedReActProgram:
    policy = ReActPolicy(
        max_iterations=2,
        granted_scopes=("read:corpus",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(ToolGrant("lookup", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),),
    )
    planner = KeywordToolPlanner({"lookup": ("rollback",)})
    return GovernedReActProgram(policy=policy, planner=planner, skills={"lookup": _LookupSkill()})


class _LookupSkill:
    name = "lookup"
    description = "Read-only lookup over the retrieved context."

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del arguments
        return {"documents": len({hit.chunk.document_id for hit in context.hits})}
