from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
import tempfile
from pathlib import Path
from types import ModuleType

# Optional or dev-only packages that a core-only install would NOT have. The
# program layer and its offline default (KeywordToolPlanner) must import and run
# without any of these, so we block them before importing refaat and prove the
# keyword governed ReAct path stays core-clean. jsonschema is the specific risk:
# it lives only in the dev extra and must never leak onto the runtime path.
BLOCKED = frozenset({"jsonschema", "reportlab", "redis"})


class _BlockedFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        root = fullname.split(".", 1)[0]
        if root in BLOCKED:
            raise ImportError(f"core-only gate blocks optional module {fullname!r}")
        return None


def _install_blocker() -> None:
    for name in list(sys.modules):
        if name.split(".", 1)[0] in BLOCKED:
            del sys.modules[name]
    sys.meta_path.insert(0, _BlockedFinder())


def _settings(base: Path) -> object:
    from refaat.config import Settings

    return Settings(
        environment="test",
        database_url=f"sqlite:///{base / 'refaat.db'}",
        auto_create_schema=True,
        blob_root=base / "blobs",
        connector_file_root=base / "connectors",
        skill_root=base / "skills",
        api_key_pepper="core-import-pepper-with-entropy",
        bootstrap_token="core-import-bootstrap-token",
        default_chunk_tokens=64,
        chunk_overlap_tokens=12,
        query_token_budget=600,
        log_json=False,
    )


def main() -> int:
    _install_blocker()

    from refaat.application.runtime import RuntimeBuilder
    from refaat.domain.enums import SideEffectClass
    from refaat.domain.models import DocumentInput, ReActPolicy, ToolGrant
    from refaat.programs import GovernedReActProgram, KeywordToolPlanner
    from refaat.skills.builtin import CorpusHealthSkill

    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        runtime = RuntimeBuilder(_settings(base)).build()  # type: ignore[arg-type]
        try:
            project = runtime.project(workspace="core", corpus="import")
            runtime.ingest(
                project.workspace_id,
                project.corpus_id,
                DocumentInput(
                    source_id="ops.txt",
                    title="Ops",
                    media_type="text/plain",
                    content=b"Rollback restores the previously approved harness snapshot before deployment.",
                ),
            )
            policy = ReActPolicy(
                granted_scopes=("read:corpus",),
                side_effect_ceiling=SideEffectClass.READ_ONLY,
                grants=(ToolGrant("corpus_health", required_scopes=("read:corpus",)),),
            )
            planner = KeywordToolPlanner({"corpus_health": ("rollback",)})
            program = GovernedReActProgram(
                policy=policy, planner=planner, skills={"corpus_health": CorpusHealthSkill()}
            )
            result = program.execute(
                project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None
            )
        finally:
            runtime.close()

    steps = result.outputs["react_steps"]
    if not steps or not steps[0]["authorized"]:
        print("core-only import check failed: keyword governed ReAct did not authorize its tool", file=sys.stderr)
        return 1
    print(f"Core-only import check passed: keyword GovernedReActProgram ran {len(steps)} governed step(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
