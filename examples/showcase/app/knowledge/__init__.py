"""The curated Oronts knowledge base and its metadata.

Each entry pairs a markdown file of factual Oronts prose with retrieval metadata
(title, category, source, tags). The prose recurs named entities across files so
graph extraction, recursive summaries, and multi-hop questions are meaningful. The
metadata lives here rather than in frontmatter so the markdown stays pure prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

KNOWLEDGE_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class KnowledgeDocument:
    source_id: str
    title: str
    category: str
    tags: tuple[str, ...]
    text: str
    metadata: dict[str, str] = field(default_factory=dict)

    def as_metadata(self) -> dict[str, object]:
        return {
            "title": self.title,
            "category": self.category,
            "source": self.source_id,
            "tags": list(self.tags),
        }


_CATALOG: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("company-overview.md", "Oronts GmbH Company Overview", "company", ("company", "munich", "founder")),
    ("methodology.md", "The 90-Day Production Pilot", "company", ("methodology", "90-day-pilot", "delivery")),
    ("founder.md", "Refaat Al Ktifan, Founder and CEO", "company", ("founder", "refaat", "leadership")),
    ("contact-and-trust.md", "Contact, Company Facts, and Trust", "company", ("contact", "trust", "legal")),
    ("services.md", "Oronts Services and Engagement Packages", "services", ("services", "packages", "engagements")),
    ("commerce-and-pim.md", "Commerce, PIM, and Data Platforms", "services", ("commerce", "pim", "pimcore")),
    (
        "technology-ecosystem.md",
        "The Oronts Technology Ecosystem",
        "platform",
        ("technology", "stack", "infrastructure"),
    ),
    (
        "engineering-principles.md",
        "Oronts Engineering Principles",
        "company",
        ("engineering", "principles", "ownership"),
    ),
    ("ai-and-rag.md", "Production AI and the RAG Practice", "ai", ("ai", "rag", "retrieval")),
    ("oguardai.md", "OGuardAI", "product", ("product", "security", "privacy")),
    ("pimtx.md", "PimTx", "product", ("product", "pimcore", "concurrency")),
    ("brand.md", "The Oronts Brand", "company", ("brand", "voice", "visual")),
    ("products-open-source.md", "Oronts Products and Open Source", "product", ("product", "open-source", "assistant")),
    ("exfinity.md", "Exfinity, an Oronts Client Build", "work", ("work", "commerce", "client")),
    ("guides.md", "The Oronts Technical Guides Library", "content", ("content", "guides", "engineering")),
)


def load_documents() -> list[KnowledgeDocument]:
    documents: list[KnowledgeDocument] = []
    for filename, title, category, tags in _CATALOG:
        path = KNOWLEDGE_ROOT / filename
        text = path.read_text(encoding="utf-8")
        documents.append(
            KnowledgeDocument(
                source_id=filename,
                title=title,
                category=category,
                tags=tags,
                text=text,
            )
        )
    return documents


PRODUCT_CATEGORY = "product"


__all__ = ["KNOWLEDGE_ROOT", "PRODUCT_CATEGORY", "KnowledgeDocument", "load_documents"]
