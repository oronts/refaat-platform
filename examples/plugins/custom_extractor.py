from __future__ import annotations

from dataclasses import dataclass

from refaat.domain.models import ExtractedDocument, JsonObject


@dataclass(frozen=True, slots=True)
class MarkdownFrontMatterExtractor:
    """Extract Markdown and expose simple YAML-style front matter as metadata."""

    media_types = frozenset({"text/x-markdown-frontmatter"})

    def extract(self, content: bytes, *, metadata: JsonObject) -> ExtractedDocument:
        text = content.decode(str(metadata.get("encoding", "utf-8")))
        extracted: JsonObject = {}
        body = text
        if text.startswith("---\n") and "\n---\n" in text[4:]:
            header, body = text[4:].split("\n---\n", maxsplit=1)
            for line in header.splitlines():
                key, separator, value = line.partition(":")
                if separator and key.strip():
                    extracted[key.strip()] = value.strip()
        return ExtractedDocument(text=body.strip(), metadata=extracted)
