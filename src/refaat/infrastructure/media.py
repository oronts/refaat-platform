from __future__ import annotations

import io
import tempfile

from refaat.domain.models import ExtractedDocument, JsonObject


class TesseractImageExtractor:
    media_types = frozenset({"image/png", "image/jpeg", "image/webp", "image/tiff"})

    def __init__(self, languages: str = "eng") -> None:
        self.languages = languages

    def extract(self, content: bytes, *, metadata: JsonObject) -> ExtractedDocument:
        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Install Pillow and pytesseract to enable local image OCR") from exc
        image = Image.open(io.BytesIO(content))
        text = pytesseract.image_to_string(image, lang=str(metadata.get("languages", self.languages)))
        return ExtractedDocument(
            text=text,
            metadata={"width": image.width, "height": image.height, "extractor": "tesseract"},
        )


class FasterWhisperAudioExtractor:
    media_types = frozenset(
        {
            "audio/mpeg",
            "audio/mp4",
            "audio/wav",
            "audio/x-wav",
            "audio/webm",
            "video/mp4",
            "video/webm",
        }
    )

    def __init__(self, model: str = "small", device: str = "cpu", compute_type: str = "int8") -> None:
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self._model: object | None = None

    def _load(self) -> object:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError("Install faster-whisper to enable local audio transcription") from exc
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def extract(self, content: bytes, *, metadata: JsonObject) -> ExtractedDocument:
        suffix = str(metadata.get("suffix", ".bin"))
        with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
            handle.write(content)
            handle.flush()
            model = self._load()
            segments, info = model.transcribe(  # type: ignore[attr-defined]
                handle.name,
                language=metadata.get("language"),
                vad_filter=True,
            )
            values = list(segments)
        return ExtractedDocument(
            text=" ".join(segment.text.strip() for segment in values),
            metadata={
                "language": getattr(info, "language", None),
                "duration": getattr(info, "duration", None),
                "extractor": "faster-whisper",
            },
        )
