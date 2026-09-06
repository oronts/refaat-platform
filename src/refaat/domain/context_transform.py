"""Pre-model context-transform seam: redact sensitive spans from the evidence a
generation model is about to see.

Model providers are a data-egress boundary. This pure, model-independent seam lets a
deployment rewrite the retrieved evidence, graph facts, and session summary carried on a
:class:`GenerationRequest` before the request reaches a real generator, replacing operator
configured spans with a placeholder. The verdict is deterministic and does no I/O; this
module imports only domain value objects and stdlib, never core or infrastructure, so the
domain layer stays leaf. When no transform is installed the generation path is byte-identical
to the untransformed default; this is an opt-in seam.

Honest scope: this covers the GENERATION model's evidence input only. Embeddings (ingest and
query), fine-tuning training data, ingestion-time HyPE, RAPTOR, and vision captioning, and
evaluation-time judging are separate provider egresses this seam does not cover. It is
defense-in-depth against the generation provider, not a complete DLP boundary. See
docs/egress-redaction.md.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from refaat.domain.models import GenerationRequest


@dataclass(frozen=True, slots=True)
class RedactionCount:
    """How many spans one pattern redacted, keyed by an opaque pattern id.

    ``pattern_id`` is a digest of the compiled pattern, never the raw pattern string or the
    matched text, so provenance can be recorded without leaking the secret it protects.
    """

    pattern_id: str
    count: int


@dataclass(frozen=True, slots=True)
class TransformResult:
    """The rewritten request plus redaction provenance handed back from the seam.

    ``request`` is the request to send to the generator. ``redactions`` records the per-pattern
    counts over the surviving spans (counts and pattern ids only). ``images_dropped`` is how many
    images the seam refused: under an active transform, image egress is refused rather than passed,
    because a text redactor cannot vet pixels.
    """

    request: GenerationRequest
    redactions: tuple[RedactionCount, ...]
    images_dropped: int = 0


class ContextTransform(Protocol):
    """Rewrite the evidence a generation request carries before it reaches a model."""

    @property
    def cache_id(self) -> str:
        """A stable identity for this transform's rewrite function.

        Folded into the answer-cache key so an answer produced under one transform (or none) is
        never served after the transform is installed, removed, or changed, because a cache hit
        returns before generation runs.
        """
        ...

    def apply(self, request: GenerationRequest) -> TransformResult: ...


def redact_for_egress(transform: ContextTransform | None, request: GenerationRequest) -> TransformResult:
    """The single shared pre-model egress seam.

    With no transform installed the request passes through untouched (the SAME object), so the
    generation path is byte-identical to the untransformed default. With a transform active the
    request is rewritten, then image egress is refused in the seam itself: any image left on the
    rewritten request is dropped and counted, so every transform inherits the fail-safe and a
    text-only redactor can never leak pixels it did not inspect.
    """
    if transform is None:
        return TransformResult(request=request, redactions=(), images_dropped=0)
    result = transform.apply(request)
    if result.request.images:
        return replace(
            result,
            request=replace(result.request, images=()),
            images_dropped=len(result.request.images),
        )
    return result


__all__ = ["ContextTransform", "RedactionCount", "TransformResult", "redact_for_egress"]
