from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

SMOKE_PROGRAM = """\
from pathlib import Path

from refaat import Refaat, RlmLimits
from refaat.config import Settings

root = Path(__file__).parent
source = root / "knowledge"
source.mkdir()
(source / "policy.md").write_text(
    "Production changes require peer review and rollback evidence.",
    encoding="utf-8",
)
settings = Settings(
    environment="test",
    database_url=f"sqlite:///{root / 'smoke.db'}",
    blob_root=root / "blobs",
    connector_file_root=root / "connectors",
    skill_root=root / "skills",
    api_key_pepper="wheel-smoke-test-pepper",
    log_json=False,
)
with Refaat(settings) as runtime:
    project = runtime.project(workspace="wheel-smoke", corpus="knowledge")
    project.ingest(source)
    answer = project.ask("What does a production change require?")
    recursive = project.rlm(
        "What evidence is required?",
        limits=RlmLimits(
            max_depth=1,
            max_nodes=3,
            branch_factor=2,
            leaf_token_budget=64,
        ),
    )
    assert answer.citations and recursive.citations
    print({
        "query_citations": len(answer.citations),
        "rlm_citations": len(recursive.citations),
    })
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    if not wheel.is_file():
        raise FileNotFoundError(wheel)
    with tempfile.TemporaryDirectory(prefix="refaat-wheel-") as raw:
        root = Path(raw)
        target = root / "site"
        target.mkdir()
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(target)
        script = root / "smoke.py"
        script.write_text(SMOKE_PROGRAM, encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(target)
        result = subprocess.run(
            [sys.executable, str(script)],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        print(json.dumps({"wheel": wheel.name, "output": result.stdout.strip()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
