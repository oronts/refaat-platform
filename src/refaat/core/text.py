from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Mapping

TOKEN_PATTERN = re.compile(r"[\w'-]+", re.UNICODE)
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ0-9])")
HORIZONTAL_WHITESPACE_PATTERN = re.compile(r"[\t \f\v]+")
# Commas and semicolons always separate clauses; '.', '!', '?' only when sentence-final
# (before a capital/digit-led follow-on or at the end), so intra-token dots in abbreviations
# ("U.S."), decimals ("3.5"), and version numbers ("3.11") never spawn a clause. The and/or
# alternation is case-insensitive in isolation so the capital/digit lookahead stays case-sensitive.
CLAUSE_SEPARATOR_PATTERN = re.compile(r"[;,]+|[.!?]+(?=\s+[A-ZÀ-ÖØ-Þ0-9]|\s*$)|\s+(?i:and|or)\s+")

# Single source of truth for the co-occurrence graph extractor's stopwords.
# infrastructure/graph.py imports this; the set and its use are behavior-preserving.
STOPWORDS = frozenset(
    {
        "about",
        "after",
        "also",
        "before",
        "between",
        "from",
        "have",
        "into",
        "more",
        "other",
        "that",
        "their",
        "these",
        "they",
        "this",
        "through",
        "using",
        "with",
        "your",
    }
)

# Query-oriented function words dropped by content_terms, used to skip empty and
# content-identical clauses so a decomposition subquery is genuinely distinct.
QUERY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)

# Curated high-frequency function words for the non-English languages the graph
# extractor recognises. Each set is deterministic and dependency-free, chosen to
# drop common connectives, articles, pronouns, and auxiliaries so a tagged
# document's co-occurrence graph is built from content words rather than glue.
_GERMAN_STOPWORDS = frozenset(
    {
        "aber",
        "andere",
        "auch",
        "auf",
        "aus",
        "bei",
        "dann",
        "das",
        "dass",
        "dem",
        "den",
        "der",
        "des",
        "diese",
        "dieser",
        "dieses",
        "durch",
        "ein",
        "eine",
        "einer",
        "eines",
        "haben",
        "ich",
        "ihr",
        "nach",
        "nicht",
        "noch",
        "nur",
        "oder",
        "schon",
        "sehr",
        "sein",
        "sind",
        "über",
        "und",
        "unter",
        "von",
        "war",
        "waren",
        "werden",
        "wenn",
        "wie",
        "wir",
        "zwischen",
    }
)
_FRENCH_STOPWORDS = frozenset(
    {
        "aussi",
        "avec",
        "avoir",
        "aux",
        "cette",
        "comme",
        "dans",
        "des",
        "elle",
        "elles",
        "entre",
        "est",
        "et",
        "étaient",
        "était",
        "être",
        "été",
        "ils",
        "leur",
        "leurs",
        "mais",
        "moins",
        "nous",
        "par",
        "pas",
        "plus",
        "pour",
        "quand",
        "que",
        "qui",
        "quoi",
        "sans",
        "ses",
        "son",
        "sous",
        "sur",
        "toujours",
        "très",
        "une",
        "vous",
    }
)
_SPANISH_STOPWORDS = frozenset(
    {
        "aunque",
        "bajo",
        "como",
        "con",
        "cuando",
        "del",
        "desde",
        "donde",
        "ellos",
        "ellas",
        "entre",
        "era",
        "eran",
        "esta",
        "estas",
        "este",
        "estos",
        "más",
        "muy",
        "nosotros",
        "para",
        "pero",
        "por",
        "porque",
        "que",
        "ser",
        "sobre",
        "son",
        "sus",
        "también",
        "todos",
        "una",
        "unas",
        "unos",
    }
)
_ARABIC_STOPWORDS = frozenset(
    {
        "إلى",
        "أن",
        "إن",
        "التي",
        "الذي",
        "الذين",
        "بعض",
        "بها",
        "على",
        "عند",
        "عندما",
        "كان",
        "كانت",
        "كذلك",
        "لكن",
        "لها",
        "مع",
        "هذا",
        "هذه",
        "هكذا",
        "هناك",
        "هؤلاء",
        "التيار",
        "حيث",
        "خلال",
        "دون",
        "ذلك",
        "أيضا",
        "بين",
        "نحن",
        "كل",
    }
)

# Registry mapping a normalized two-letter code to a stopword set. The "en" key
# maps to the exact existing frozenset objects so an untagged or English corpus
# resolves to the identical set and stays byte-identical.
STOPWORDS_BY_LANGUAGE: dict[str, frozenset[str]] = {
    "en": STOPWORDS,
    "de": _GERMAN_STOPWORDS,
    "fr": _FRENCH_STOPWORDS,
    "es": _SPANISH_STOPWORDS,
    "ar": _ARABIC_STOPWORDS,
}

QUERY_STOPWORDS_BY_LANGUAGE: dict[str, frozenset[str]] = {
    "en": QUERY_STOPWORDS,
    "de": _GERMAN_STOPWORDS,
    "fr": _FRENCH_STOPWORDS,
    "es": _SPANISH_STOPWORDS,
    "ar": _ARABIC_STOPWORDS,
}


def _primary_subtag(language: str | None) -> str:
    """Normalize a language code to its lowercase primary subtag.

    ``en-US`` and ``de_DE`` collapse to ``en`` and ``de``; ``None``, empty, or an
    unknown code falls back to ``en`` so selection never raises and defaults to the
    English behavior.
    """
    if not language:
        return "en"
    primary = re.split(r"[-_]", language.strip(), maxsplit=1)[0].casefold()
    return primary or "en"


def stopwords_for(
    language: str | None,
    registry: Mapping[str, frozenset[str]] | None = None,
) -> frozenset[str]:
    """Return the graph-extractor stopword set for a language code.

    Normalizes the code to its primary subtag. With no ``registry`` the module
    :data:`STOPWORDS_BY_LANGUAGE` is used exactly, so ``None``, ``"en"``, or an
    unrecognized language returns the English :data:`STOPWORDS` object itself and
    the default path is provably the identical set.

    A project may pass its own ``registry`` (from the ``stopwords_by_language``
    setting or :meth:`RuntimeBuilder.with_stopwords`) to override a language or add
    a new one. When given, the primary subtag is resolved in that registry, falling
    back to the registry's own ``"en"`` entry, then the module :data:`STOPWORDS`.
    Never raises. A configured override is a new input, so it cannot shift an
    unconfigured corpus.
    """
    if registry is None:
        return STOPWORDS_BY_LANGUAGE.get(_primary_subtag(language), STOPWORDS)
    subtag = _primary_subtag(language)
    if subtag in registry:
        return registry[subtag]
    if "en" in registry:
        return registry["en"]
    return STOPWORDS


def query_stopwords_for(
    language: str | None,
    registry: Mapping[str, frozenset[str]] | None = None,
) -> frozenset[str]:
    """Return the query-oriented stopword set for a language code.

    Mirrors :func:`stopwords_for`. With no ``registry`` it uses the module
    :data:`QUERY_STOPWORDS_BY_LANGUAGE` exactly and returns the English
    :data:`QUERY_STOPWORDS` object for ``None``, ``"en"``, or an unrecognized
    language. A supplied ``registry`` resolves the primary subtag, falling back to
    its own ``"en"`` entry, then the module :data:`QUERY_STOPWORDS`.
    """
    if registry is None:
        return QUERY_STOPWORDS_BY_LANGUAGE.get(_primary_subtag(language), QUERY_STOPWORDS)
    subtag = _primary_subtag(language)
    if subtag in registry:
        return registry[subtag]
    if "en" in registry:
        return registry["en"]
    return QUERY_STOPWORDS


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).replace("\x00", "").replace("\r\n", "\n")
    lines = [HORIZONTAL_WHITESPACE_PATTERN.sub(" ", line).strip() for line in normalized.split("\n")]
    compacted: list[str] = []
    for line in lines:
        if line or (compacted and compacted[-1]):
            compacted.append(line)
    return "\n".join(compacted).strip()


def normalized_key(value: str) -> str:
    return normalize_text(value).casefold()


def tokens(value: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_PATTERN.finditer(value)]


def name_substrings(query_key: str, max_len: int) -> frozenset[str]:
    """Every contiguous substring of query_key up to max_len, for the entity substring-match branch.

    graph_context matches an entity when its normalized_name is a substring of the normalized query key.
    This enumerates every candidate a term index can equality-match, so it never under-returns a
    non-empty normalized_name graph_context would match: any such name has length at most len(query_key)
    and is therefore produced. The empty string is excluded on purpose (an empty normalized_name matches
    every query, so the caller adds it to the candidate set directly).
    """
    cap = min(max_len, len(query_key))
    if cap <= 0:
        return frozenset()
    return frozenset(
        query_key[start : start + length]
        for length in range(1, cap + 1)
        for start in range(len(query_key) - length + 1)
    )


def estimate_tokens(value: str) -> int:
    return max(1, len(TOKEN_PATTERN.findall(value))) if value.strip() else 0


def truncate_to_tokens(value: str, limit: int) -> str:
    """Truncate text to at most `limit` tokens, measured by the same tokenizer as estimate_tokens.

    The result is a prefix of the original text cut at the end of the limit-th token, so it keeps the
    original spacing and its token count is bounded in this tokenizer's measure even for
    delimiter-separated content, where a whitespace split leaves a single long word unbounded. A single
    unbroken run with no token boundary (for example a long base64 blob or a scriptio-continua CJK span)
    is one token to this tokenizer, so it is kept whole, which is consistent with estimate_tokens.
    """
    if limit <= 0:
        return ""
    end = 0
    for count, match in enumerate(TOKEN_PATTERN.finditer(value), start=1):
        end = match.end()
        if count >= limit:
            break
    return value[:end]


def split_sentences(value: str) -> list[str]:
    normalized = normalize_text(value)
    if not normalized:
        return []
    return [part.strip() for part in SENTENCE_PATTERN.split(normalized) if part.strip()]


def sentence_spans(value: str) -> list[tuple[int, int]]:
    """Return character ``[start, end)`` spans of the sentences in ``value``.

    Unlike :func:`split_sentences`, the returned offsets index into ``value``
    itself, so ``value[start:end]`` is always a contiguous verbatim substring of
    the input. Hierarchical chunking and windowed retrieval rely on this to widen
    a child to neighboring sentences and still quote stored text exactly, never a
    re-joined or synthesized approximation. ``value`` is used as given (it is
    already normalized upstream); an empty input yields no spans.
    """
    if not value:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    for match in SENTENCE_PATTERN.finditer(value):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(value)))
    return spans


def split_clauses(value: str) -> list[str]:
    """Split a query into clauses on punctuation and coordinating conjunctions.

    Returns the non-empty, stripped clause texts in order. A single-clause query
    yields a one-element list, which the query transform treats as the
    byte-identical single-variant fast path.
    """
    normalized = normalize_text(value)
    if not normalized:
        return []
    return [part.strip() for part in CLAUSE_SEPARATOR_PATTERN.split(normalized) if part.strip()]


def overlap_ratio(query_terms: Iterable[str], text: str) -> float:
    """Fraction of a text's tokens that also appear in the query term set.

    Deterministic salience signal for context compression. Returns 0.0 when the
    text has no tokens or the query set is empty, so an empty or off-topic
    sentence scores lowest.
    """
    text_tokens = tokens(text)
    if not text_tokens:
        return 0.0
    query_set = set(query_terms)
    if not query_set:
        return 0.0
    matches = sum(1 for token in text_tokens if token in query_set)
    return matches / len(text_tokens)


def content_terms(value: str) -> list[str]:
    """Deterministic ordered content terms with query stopwords and short function words dropped."""
    result: list[str] = []
    seen: set[str] = set()
    for token in tokens(value):
        if len(token) < 3 or token in QUERY_STOPWORDS:
            continue
        if token not in seen:
            seen.add(token)
            result.append(token)
    return result


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def stable_digest(parts: Iterable[str], *, length: int = 32) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode()).hexdigest()
    return digest[:length]
