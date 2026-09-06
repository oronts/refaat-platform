from __future__ import annotations

import re

from refaat.core.text import content_terms, normalize_text, normalized_key, split_clauses
from refaat.domain.models import ContextPacket, GenerationRequest, TransformedQuery
from refaat.errors import DependencyUnavailableError
from refaat.ports import Generator

_HYDE_SYSTEM_PROMPT = (
    "You are a retrieval assistant. Write one short, factual, hypothetical passage that could answer "
    "the question. Do not add citation markers, sources, or commentary. Reply with the passage only."
)
_QUERY_REWRITE_SYSTEM_PROMPT = (
    "You are a retrieval assistant. Rewrite the user question into a single clear search query that "
    "maximizes retrieval recall. Reply with the rewritten query only, on one line, no commentary."
)
_MULTI_QUERY_SYSTEM_PROMPT = (
    "You are a retrieval assistant. Produce several alternative phrasings of the user question, one per "
    "line, each a standalone search query. Do not add numbering, sources, or commentary."
)
_STEP_BACK_SYSTEM_PROMPT = (
    "You are a retrieval assistant. Write one broader, more general background question that steps back "
    "from the specifics of the user question. Reply with the single question only, no commentary."
)

_LIST_PREFIX = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")


def _dedupe_key(text: str) -> str:
    return normalized_key(text.rstrip(" \t?!.,;:"))


def _empty_context(query: str) -> ContextPacket:
    return ContextPacket(
        query=query,
        hits=(),
        graph_facts=(),
        session_summary=None,
        profile={},
        supplemental_harness={},
        token_count=0,
        routes=(),
    )


def _generate_text(generator: Generator, query: str, system_prompt: str, workspace_id: str) -> str:
    """Run one model generation over an empty context and return the stripped answer.

    Shared by every model-backed transform. The empty context keeps the call a
    pure query-to-text rewrite: no retrieved chunk ever reaches the model, so the
    output can never carry or forge a citation. A failed dependency degrades to
    the empty string, which each caller maps to its offline-safe fallback. The
    ``workspace_id`` scopes the round-trip in the model-call cache so a transform
    generation for one tenant is never served to another.
    """
    request = GenerationRequest(
        query=query,
        context=_empty_context(query),
        system_prompt=system_prompt,
        workspace_id=workspace_id,
    )
    try:
        result = generator.generate(request)
    except DependencyUnavailableError:
        return ""
    return result.answer.strip()


def _split_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        cleaned = _LIST_PREFIX.sub("", raw).strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def _templated_hypothesis(query: str) -> str:
    """Deterministic model-free pseudo-answer whose embedding differs from the query's.

    Built from the query's content terms so a hash embedder maps it to a
    measurably different point than the raw query, giving HyDE genuine structure
    offline without a model.
    """
    terms = content_terms(query)
    if not terms:
        return ""
    subject = " ".join(terms)
    return f"{subject}. This passage explains {subject} and describes how {subject} works in detail."


def _hyde_variants(query: str, max_variants: int, hypothesis: str) -> tuple[TransformedQuery, ...]:
    original = TransformedQuery(text=query, kind="original")
    if max_variants <= 1 or not hypothesis.strip():
        return (original,)
    hypothetical = TransformedQuery(text=query, kind="hypothetical", embed_text=hypothesis)
    return (original, hypothetical)


def _multi_variants(query: str, max_variants: int, generated: list[str]) -> tuple[TransformedQuery, ...]:
    """Original query plus deduped generated phrasings, total capped at max_variants.

    Every variant is an ordinary ``original``-kind query with no embed_text, so it
    feeds both semantic and lexical search over stored chunks. The generated
    phrasings never carry embed_text and are never cited; the runtime RRF-merges
    the per-variant rankings by chunk id.
    """
    original = TransformedQuery(text=query, kind="original")
    if max_variants <= 1:
        return (original,)
    seen = {_dedupe_key(query)}
    variants = [original]
    for text in generated:
        if len(variants) >= max_variants:
            break
        key = _dedupe_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        variants.append(TransformedQuery(text=text, kind="original"))
    return tuple(variants)


def _deterministic_stepback(query: str) -> str:
    terms = content_terms(query)
    if not terms:
        return ""
    return " ".join(terms)


def _stepback_variants(query: str, max_variants: int, broader: str) -> tuple[TransformedQuery, ...]:
    """Original query plus one broader concept variant, RRF-merged.

    The broader query is an ordinary variant text (no embed_text), so it drives
    both semantic and lexical search over stored chunks; it is never a
    hypothetical and is never cited. A single-variant budget, a blank broadening,
    or a broadening that collapses back to the original yields the byte-identical
    original-only path.
    """
    original = TransformedQuery(text=query, kind="original")
    if max_variants <= 1 or not broader.strip():
        return (original,)
    if _dedupe_key(broader) == _dedupe_key(query):
        return (original,)
    return (original, TransformedQuery(text=broader, kind="stepback"))


class DeterministicQueryTransform:
    """Model-free query transform: sub-query decomposition, RRF-merged.

    For an ordinary single-clause question it returns only the original variant,
    keeping the single-query retrieval path byte-identical to the pre-slice
    behavior (sentence-final punctuation does not spawn a redundant subquery).
    A compound query (two or more distinct, content-bearing clauses) additionally
    yields one subquery per clause, deduped by a punctuation-stripped normalized
    key and capped at max_variants. A model-backed transform can later implement
    the same neutral QueryTransform port.
    """

    @property
    def cache_id(self) -> str:
        return "deterministic-v1"

    def expand(self, query: str, max_variants: int, *, workspace_id: str = "") -> tuple[TransformedQuery, ...]:
        del workspace_id
        variants: list[TransformedQuery] = [TransformedQuery(text=query, kind="original")]
        if max_variants <= 1:
            return tuple(variants)

        clauses = split_clauses(query)
        if len(clauses) < 2:
            return tuple(variants)

        seen: set[str] = set()
        original_key = _dedupe_key(query)
        if original_key:
            seen.add(original_key)
        original_terms = set(content_terms(query))
        for clause in clauses:
            if len(variants) >= max_variants:
                break
            clause_terms = content_terms(clause)
            if not clause_terms:
                continue
            key = _dedupe_key(clause)
            if not key or key in seen:
                continue
            if set(clause_terms) == original_terms:
                continue
            seen.add(key)
            variants.append(TransformedQuery(text=clause, kind="subquery"))

        return tuple(variants)


class DeterministicHydeQueryTransform:
    """Model-free HyDE: pairs the original query with a templated pseudo-answer.

    The hypothetical variant keeps the original query text for lexical search and
    carries a deterministic pseudo-answer in embed_text, so only the semantic
    embedding target moves. It is the offline structure of HyDE: no model, fully
    deterministic, and the hypothetical is never lexically searched or cited. A
    single-variant budget or a blank hypothesis yields the byte-identical
    original-only path.
    """

    @property
    def cache_id(self) -> str:
        return "hyde-deterministic-v1"

    def expand(self, query: str, max_variants: int, *, workspace_id: str = "") -> tuple[TransformedQuery, ...]:
        del workspace_id
        return _hyde_variants(query, max_variants, _templated_hypothesis(query))


class ModelHydeQueryTransform:
    """Model-backed HyDE: a real generator writes the hypothetical answer.

    The generated passage is used only as embed_text on the hypothetical variant,
    so it moves the semantic embedding target while the original query still
    drives lexical search. It is never parsed for citations, never retrieved, and
    never becomes a chunk. Blank or failed generation degrades to the
    original-only path.
    """

    def __init__(self, generator: Generator) -> None:
        self._generator = generator

    @property
    def cache_id(self) -> str:
        return f"hyde-model:{self._generator.model_id}"

    def expand(self, query: str, max_variants: int, *, workspace_id: str = "") -> tuple[TransformedQuery, ...]:
        hypothesis = (
            _generate_text(self._generator, query, _HYDE_SYSTEM_PROMPT, workspace_id) if max_variants > 1 else ""
        )
        return _hyde_variants(query, max_variants, hypothesis)


class DeterministicRewriteQueryTransform:
    """Model-free query rewriting: the shipped deterministic normalize, single variant.

    Returns exactly one original-kind variant carrying the normalized query, so
    the offline single-query retrieval path is byte-identical to today. It is the
    offline default for the query_rewrite provider; the model-backed sibling
    replaces the text with a real rewrite.
    """

    @property
    def cache_id(self) -> str:
        return "rewrite-deterministic-v1"

    def expand(self, query: str, max_variants: int, *, workspace_id: str = "") -> tuple[TransformedQuery, ...]:
        del workspace_id
        return (TransformedQuery(text=normalize_text(query), kind="original"),)


class ModelRewriteQueryTransform:
    """Model-backed query rewriting: a real generator restates the query for retrieval.

    Returns the original query plus the model rewrite; both feed semantic and
    lexical search over stored chunks and are RRF-merged, so the rewrite actively
    contributes to retrieval (a single variant would be short-circuited by the
    pre-retrieval seam) while never being treated as a chunk or cited. Blank or
    failed generation degrades to the normalized original only.
    """

    def __init__(self, generator: Generator) -> None:
        self._generator = generator

    @property
    def cache_id(self) -> str:
        return f"rewrite-model:{self._generator.model_id}"

    def expand(self, query: str, max_variants: int, *, workspace_id: str = "") -> tuple[TransformedQuery, ...]:
        lines = _split_lines(_generate_text(self._generator, query, _QUERY_REWRITE_SYSTEM_PROMPT, workspace_id))
        if not lines:
            return (TransformedQuery(text=normalize_text(query), kind="original"),)
        return (TransformedQuery(text=query, kind="original"), TransformedQuery(text=lines[0], kind="rewrite"))


class DeterministicMultiQueryTransform:
    """Model-free multi-query expansion: the single original variant, identity.

    Returns exactly one original-kind variant, so the offline retrieval path is
    byte-identical to single-query. It is the offline default for the multi_query
    provider; the model-backed sibling fans out to several phrasings merged by
    RRF.
    """

    @property
    def cache_id(self) -> str:
        return "multiquery-deterministic-v1"

    def expand(self, query: str, max_variants: int, *, workspace_id: str = "") -> tuple[TransformedQuery, ...]:
        del workspace_id
        return (TransformedQuery(text=query, kind="original"),)


class ModelMultiQueryTransform:
    """Model-backed multi-query expansion: a real generator writes alternative phrasings.

    Each generated phrasing plus the original query becomes an ordinary
    original-kind variant with no embed_text, so every variant feeds semantic and
    lexical search over stored chunks and the runtime RRF-merges the per-variant
    rankings by chunk id. A phrasing is never a chunk and is never cited. The
    total variant count is capped at max_variants; blank or failed generation
    degrades to the original-only path.
    """

    def __init__(self, generator: Generator) -> None:
        self._generator = generator

    @property
    def cache_id(self) -> str:
        return f"multiquery-model:{self._generator.model_id}"

    def expand(self, query: str, max_variants: int, *, workspace_id: str = "") -> tuple[TransformedQuery, ...]:
        if max_variants <= 1:
            return (TransformedQuery(text=query, kind="original"),)
        generated = _split_lines(_generate_text(self._generator, query, _MULTI_QUERY_SYSTEM_PROMPT, workspace_id))
        return _multi_variants(query, max_variants, generated)


class DeterministicStepBackQueryTransform:
    """Model-free step-back prompting: pairs the query with a broader concept variant.

    The broader variant is the query's content terms with specifics and function
    words dropped, an ordinary variant text (no embed_text) that feeds semantic
    and lexical search over stored chunks and is never cited. The runtime
    RRF-merges the original and broader rankings. A blank broadening or one that
    collapses back to the original yields the byte-identical original-only path.
    """

    @property
    def cache_id(self) -> str:
        return "stepback-deterministic-v1"

    def expand(self, query: str, max_variants: int, *, workspace_id: str = "") -> tuple[TransformedQuery, ...]:
        del workspace_id
        return _stepback_variants(query, max_variants, _deterministic_stepback(query))


class ModelStepBackQueryTransform:
    """Model-backed step-back prompting: a real generator writes the broader question.

    The generated broader question is an ordinary variant text (no embed_text),
    so it drives semantic and lexical search over stored chunks and is never
    cited; the runtime RRF-merges it with the original. Blank or failed
    generation degrades to the original-only path.
    """

    def __init__(self, generator: Generator) -> None:
        self._generator = generator

    @property
    def cache_id(self) -> str:
        return f"stepback-model:{self._generator.model_id}"

    def expand(self, query: str, max_variants: int, *, workspace_id: str = "") -> tuple[TransformedQuery, ...]:
        broader = (
            _generate_text(self._generator, query, _STEP_BACK_SYSTEM_PROMPT, workspace_id) if max_variants > 1 else ""
        )
        return _stepback_variants(query, max_variants, broader)
