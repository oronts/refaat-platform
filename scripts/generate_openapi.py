from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from refaat.api.app import create_app
from refaat.application.runtime import RuntimeBuilder
from refaat.config import Settings

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "docs" / "reference" / "openapi.json"


def schema() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="refaat-openapi-") as directory:
        root = Path(directory)
        settings = Settings(
            environment="test",
            database_url=f"sqlite+pysqlite:///{root / 'openapi.db'}",
            auto_create_schema=True,
            blob_root=root / "blobs",
            connector_file_root=root / "connectors",
            skill_root=root / "skills",
            api_key_pepper="openapi-api-key-pepper-with-enough-entropy",
            bootstrap_token="openapi-bootstrap-token-with-enough-entropy",
            log_json=False,
        )
        runtime = RuntimeBuilder(settings).build()
        try:
            return create_app(settings, runtime).openapi()
        finally:
            runtime.close()


def rendered_schema() -> str:
    return json.dumps(schema(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate or verify Refaat OpenAPI JSON")
    parser.add_argument("--check", action="store_true", help="Fail when the committed schema is stale")
    args = parser.parse_args()
    rendered = rendered_schema()
    if args.check:
        if not TARGET.is_file() or TARGET.read_text(encoding="utf-8") != rendered:
            print("OpenAPI schema is stale. Run: PYTHONPATH=src python scripts/generate_openapi.py")
            return 1
        print(f"OpenAPI schema is current: {TARGET.relative_to(ROOT)}")
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(rendered, encoding="utf-8")
    print(f"Wrote {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
