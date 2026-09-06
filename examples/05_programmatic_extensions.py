from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from _shared import demo_settings, emit
from plugins.custom_extractor import MarkdownFrontMatterExtractor

from refaat import RuntimeBuilder
from refaat.application.worker import DurableWorker
from refaat.domain.models import ContextPacket, JsonObject


@dataclass(slots=True)
class MemoryConnector:
    documents: dict[str, str]
    scheme: str = "memory"

    def fetch(self, source: str) -> tuple[bytes, str, JsonObject]:
        key = source.removeprefix("memory://")
        try:
            content = self.documents[key]
        except KeyError as exc:
            raise ValueError(f"Unknown in-memory document: {key}") from exc
        return content.encode(), "text/x-markdown-frontmatter", {"memory_key": key}


class EvidenceChecklistSkill:
    name = "evidence_checklist"
    description = "Return a concise checklist from the highest-ranked evidence chunks."

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        limit = max(1, min(int(arguments.get("limit", 3)), 10))
        return {
            "query": context.query,
            "items": [hit.chunk.text.split(".", maxsplit=1)[0] for hit in context.hits[:limit]],
            "routes": list(context.routes),
        }


@dataclass(slots=True)
class CapturingEventSink:
    events: list[tuple[str, JsonObject]] = field(default_factory=list)

    def emit(self, topic: str, payload: JsonObject) -> None:
        self.events.append((topic, payload))


def main() -> None:
    with TemporaryDirectory(prefix="refaat-extensions-") as directory:
        root = Path(directory)
        connector = MemoryConnector(
            {
                "release": (
                    "---\nowner: Release Manager\nkind: runbook\n---\n# Release Runbook\n"
                    "Production Release evidence includes peer review, automated tests, and a rollback plan. "
                    "The Release Manager verifies the change record."
                )
            }
        )
        event_sink = CapturingEventSink()
        builder = (
            RuntimeBuilder(demo_settings(root))
            .with_connector(connector)
            .with_extractor(MarkdownFrontMatterExtractor())
            .with_skill(EvidenceChecklistSkill())
            .with_event_sink(event_sink)
        )
        with builder.build() as runtime:
            project = runtime.project(workspace="extensions", corpus="runbooks")
            ingestion = project.ingest_source("memory://release", source_id="release.runbook")
            checklist = project.run_skill(
                "evidence_checklist",
                "What evidence is required for a Production Release?",
                arguments={"limit": 2},
            )
            DurableWorker(runtime).run_once()

            assert ingestion.changed
            assert checklist["items"]
            assert any(topic == "document.indexed" for topic, _ in event_sink.events)

            emit(
                {
                    "ingestion": ingestion,
                    "checklist": checklist,
                    "events": [{"topic": topic, "payload": payload} for topic, payload in event_sink.events],
                }
            )


if __name__ == "__main__":
    main()
