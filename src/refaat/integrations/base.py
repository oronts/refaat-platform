from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from refaat.sdk.models import Answer, RetrievalResult


class RetrieverClient(Protocol):
    def retrieve(self, corpus_id: str, query: str, **options: object) -> RetrievalResult: ...


class GroundedClient(Protocol):
    def query(self, corpus_id: str, query: str, **options: object) -> Answer: ...


class GroundedRetrieverClient(RetrieverClient, GroundedClient, Protocol):
    """A client that both retrieves evidence and answers a grounded query.

    The graph assemblers take this so one client backs the retrieve node's
    branching evidence and the reason node's grounded answer; ``RefaatClient``
    satisfies it.
    """


@dataclass(slots=True)
class RefaatRetriever:
    client: RetrieverClient
    corpus_id: str
    top_k: int = 8

    def __call__(self, query: str) -> RetrievalResult:
        return self.client.retrieve(self.corpus_id, query, top_k=self.top_k)


@dataclass(slots=True)
class RefaatAnswerer:
    client: GroundedClient
    corpus_id: str

    def __call__(self, question: str) -> Answer:
        return self.client.query(self.corpus_id, question)
