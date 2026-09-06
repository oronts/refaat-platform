from __future__ import annotations

from refaat.application.services import IngestionService
from refaat.domain.models import (
    ContextPacket,
    GenerationImage,
    GenerationRequest,
)
from refaat.errors import DependencyUnavailableError
from refaat.infrastructure.model_cache import _generation_key

_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc``\x00\x00\x00"
    b"\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
)


class _StubCaptioner:
    provider_id = "stub-captioner-v1"

    def __init__(self, text: str, *, fail: bool = False) -> None:
        self._text = text
        self._fail = fail
        self.calls = 0

    def caption(self, content: bytes, *, media_type: str, title: str) -> str:
        self.calls += 1
        if self._fail:
            raise DependencyUnavailableError("vision provider down")
        return self._text


def test_deterministic_descriptor_is_stable_and_honest() -> None:
    first = IngestionService._deterministic_descriptor(_PNG, "Architecture diagram")
    second = IngestionService._deterministic_descriptor(_PNG, "Architecture diagram")
    assert first == second
    assert first.startswith("image asset ")
    assert "Architecture diagram" in first
    different_bytes = IngestionService._deterministic_descriptor(_PNG + b" ", "Architecture diagram")
    assert different_bytes != first
    no_title = IngestionService._deterministic_descriptor(_PNG, "")
    assert no_title.startswith("image asset ")
    assert "," not in no_title


def _service(captioner: _StubCaptioner | None) -> IngestionService:
    service = IngestionService.__new__(IngestionService)
    service.image_captioner = captioner
    return service


def test_supplied_caption_wins_over_captioner_and_deterministic() -> None:
    captioner = _StubCaptioner("model caption")
    descriptor, provenance = _service(captioner)._image_descriptor(
        _PNG, media_type="image/png", title="T", caption="A red circle on white", alt_text=None
    )
    assert descriptor == "A red circle on white"
    assert provenance == "provided"
    assert captioner.calls == 0


def test_captioner_used_when_no_caption_supplied() -> None:
    captioner = _StubCaptioner("model caption of the scene")
    descriptor, provenance = _service(captioner)._image_descriptor(
        _PNG, media_type="image/png", title="T", caption=None, alt_text=None
    )
    assert descriptor == "model caption of the scene"
    assert provenance == "model:stub-captioner-v1"
    assert captioner.calls == 1


def test_captioner_degrades_to_deterministic_on_failure() -> None:
    captioner = _StubCaptioner("unused", fail=True)
    descriptor, provenance = _service(captioner)._image_descriptor(
        _PNG, media_type="image/png", title="Title", caption=None, alt_text=None
    )
    assert provenance == "deterministic"
    assert descriptor == IngestionService._deterministic_descriptor(_PNG, "Title")


def test_no_captioner_uses_deterministic_descriptor() -> None:
    descriptor, provenance = _service(None)._image_descriptor(
        _PNG, media_type="image/png", title="Title", caption=None, alt_text=None
    )
    assert provenance == "deterministic"
    assert descriptor == IngestionService._deterministic_descriptor(_PNG, "Title")


def _empty_context() -> ContextPacket:
    return ContextPacket(
        query="q",
        hits=(),
        graph_facts=(),
        session_summary=None,
        profile={},
        supplemental_harness={},
        token_count=0,
        routes=(),
    )


def test_generation_key_ignores_absent_images_and_separates_present_ones() -> None:
    base = GenerationRequest(query="q", context=_empty_context(), system_prompt="s", workspace_id="ws")
    with_image = GenerationRequest(
        query="q",
        context=_empty_context(),
        system_prompt="s",
        workspace_id="ws",
        images=(GenerationImage(chunk_id="chunk_a", media_type="image/png", content=_PNG),),
    )
    other_image = GenerationRequest(
        query="q",
        context=_empty_context(),
        system_prompt="s",
        workspace_id="ws",
        images=(GenerationImage(chunk_id="chunk_a", media_type="image/png", content=_PNG + b"x"),),
    )
    identity = "vision-gen:abc"
    text_only_key = _generation_key(identity, "ws", base)
    assert text_only_key == _generation_key(
        identity, "ws", GenerationRequest(query="q", context=_empty_context(), system_prompt="s", workspace_id="ws")
    )
    assert _generation_key(identity, "ws", with_image) != text_only_key
    assert _generation_key(identity, "ws", with_image) != _generation_key(identity, "ws", other_image)
