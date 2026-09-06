from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from refaat.application.runtime import RuntimeBuilder
from refaat.application.worker import DurableWorker
from refaat.domain.models import ContextPacket, ExtractedDocument, JsonObject

pytestmark = pytest.mark.integration


@dataclass(slots=True)
class MemoryConnector:
    content: bytes
    scheme: str = "memory"

    def fetch(self, source: str) -> tuple[bytes, str, JsonObject]:
        return self.content, "application/x-test-runbook", {"source": source}


class RunbookExtractor:
    media_types = frozenset({"application/x-test-runbook"})

    def extract(self, content: bytes, *, metadata: JsonObject) -> ExtractedDocument:
        return ExtractedDocument(text=content.decode(), metadata={"connector": metadata["source"]})


class RouteSkill:
    name = "route_summary"
    description = "Return routes and evidence count."

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        return {
            "routes": list(context.routes),
            "evidence": len(context.hits),
            "label": arguments.get("label"),
        }


@dataclass(slots=True)
class CapturingSink:
    events: list[tuple[str, JsonObject]] = field(default_factory=list)

    def emit(self, topic: str, payload: JsonObject) -> None:
        self.events.append((topic, payload))


def test_builder_registers_programmatic_extensions(settings) -> None:
    sink = CapturingSink()
    runtime = (
        RuntimeBuilder(settings)
        .with_connector(MemoryConnector(b"A release requires review and rollback evidence."))
        .with_extractor(RunbookExtractor())
        .with_skill(RouteSkill())
        .with_event_sink(sink)
        .build()
    )
    try:
        project = runtime.project(workspace="extensions", corpus="runbooks")
        ingestion = project.ingest_source("memory://release", source_id="release.runbook")
        result = project.run_skill(
            "route_summary",
            "What evidence is required?",
            arguments={"label": "release"},
        )
        DurableWorker(runtime).run_once()

        assert ingestion.changed
        assert result["label"] == "release"
        assert result["evidence"] == 1
        assert {topic for topic, _ in sink.events} == {"document.indexed"}
    finally:
        runtime.close()
