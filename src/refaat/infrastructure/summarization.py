from __future__ import annotations

from collections.abc import Sequence

from refaat.core.text import content_terms, split_sentences
from refaat.domain.models import ContextPacket, GenerationRequest
from refaat.errors import DependencyUnavailableError
from refaat.ports import Generator

_RAPTOR_SYSTEM_PROMPT = (
    "You are a retrieval assistant. Write one short factual summary of the passages below. Do not add "
    "numbering, sources, citation markers, or commentary. Reply with the summary only."
)


class DeterministicExtractiveSummarizer:
    """Model-free RAPTOR summarizer: the most salient sentences of the child texts.

    Reproducible and network-free, so the offline tree is fully deterministic: the
    same cluster always yields the same summary, which keeps the node ids and their
    embeddings stable across runs and re-ingestion idempotent. Sentences are ranked by
    the summed corpus-cluster frequency of their content terms, the top sentences are
    kept, and they are emitted in their original order so the summary reads naturally
    and embeds near a query about the cluster's shared subject.
    """

    def __init__(self, max_sentences: int = 3) -> None:
        if max_sentences < 1:
            raise ValueError("max_sentences must be at least 1")
        self._max_sentences = max_sentences

    @property
    def provider_id(self) -> str:
        return f"raptor-extractive-v1:{self._max_sentences}"

    def summarize(self, texts: Sequence[str], *, workspace_id: str = "") -> str:
        del workspace_id
        sentences: list[str] = []
        for text in texts:
            sentences.extend(split_sentences(text))
        if not sentences:
            return " ".join(text.strip() for text in texts).strip()
        frequency: dict[str, int] = {}
        for sentence in sentences:
            for term in content_terms(sentence):
                frequency[term] = frequency.get(term, 0) + 1
        scored = [
            (sum(frequency.get(term, 0) for term in content_terms(sentence)), index, sentence)
            for index, sentence in enumerate(sentences)
        ]
        top = sorted(scored, key=lambda item: (-item[0], item[1]))[: self._max_sentences]
        chosen = sorted(top, key=lambda item: item[1])
        return " ".join(sentence for _score, _index, sentence in chosen)


class ModelSummarizer:
    """Model-backed RAPTOR summarizer with a deterministic fallback.

    A real generator writes the node summary; the call sees only the passage text (an
    empty retrieval context), so no stored chunk id reaches the model and no summary
    can carry or forge a citation. A failed dependency or a blank generation degrades
    to the deterministic extractive summary, so a tree build never fails or stalls on
    the model path.
    """

    def __init__(self, generator: Generator, fallback: DeterministicExtractiveSummarizer) -> None:
        self._generator = generator
        self._fallback = fallback

    @property
    def provider_id(self) -> str:
        return f"raptor-model:{self._generator.model_id}"

    def summarize(self, texts: Sequence[str], *, workspace_id: str = "") -> str:
        joined = "\n\n".join(text.strip() for text in texts if text.strip())
        if not joined:
            return self._fallback.summarize(texts)
        request = GenerationRequest(
            query=joined,
            context=ContextPacket(
                query=joined,
                hits=(),
                graph_facts=(),
                session_summary=None,
                profile={},
                supplemental_harness={},
                token_count=0,
                routes=(),
            ),
            system_prompt=_RAPTOR_SYSTEM_PROMPT,
            workspace_id=workspace_id,
        )
        try:
            answer = self._generator.generate(request).answer
        except DependencyUnavailableError:
            return self._fallback.summarize(texts)
        return answer.strip() or self._fallback.summarize(texts)
