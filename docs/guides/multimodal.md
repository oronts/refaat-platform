# Documents, images, audio, and multimodal ingestion

Refaat normalizes every supported source into extracted text plus metadata before chunking, embedding, graph extraction, and evidence persistence.

## Built-in document formats

The default extractor registry supports:

| Media type | Extractor |
|---|---|
| Plain text, Markdown, XML, YAML, source code, SQL | text decoder |
| JSON and JSON-LD | structured path flattening |
| CSV | row-to-text extraction |
| HTML/XHTML | visible-text extraction with scripts/styles removed |
| PDF | page-aware text extraction |
| DOCX | paragraphs and tables |

Install `refaat[documents]` for HTML, PDF, and DOCX dependencies.

## Images

The optional image extractor uses Pillow and Tesseract through `refaat[media]`. A production image pipeline should validate file type, dimensions, decompression limits, language packs, OCR quality, and malware scanning before ingestion.

Image recognition beyond OCR is not built into the core. Implement a `DocumentExtractor` or upstream connector that converts approved vision-model output into text and metadata, then register it through `RuntimeBuilder` or the `refaat.extractors` entry-point group.

## Audio

The optional audio path uses `faster-whisper` when `refaat[media]` is installed. Production systems should define language, diarization, retention, consent, and redaction policy outside the generic runtime.

## Metadata safety

Ingestion filters metadata keys commonly associated with credentials before persistence. This is a defensive baseline, not a complete PII or DLP system. Do not place access tokens, passwords, signed URLs, or provider credentials in document metadata.

## Custom media type

```python
from refaat import RuntimeBuilder
from refaat.domain.models import ExtractedDocument, JsonObject


class DiagramExtractor:
    media_types = frozenset({"application/x-acme-diagram"})

    def extract(self, content: bytes, *, metadata: JsonObject) -> ExtractedDocument:
        text = decode_diagram(content)
        return ExtractedDocument(text=text, metadata={"format": "acme-diagram"})


runtime = RuntimeBuilder().with_extractor(DiagramExtractor()).build()
```

The extractor must return non-empty text. Keep network calls, model credentials, retry policy, and file validation explicit rather than hiding them inside an unbounded parser.

## Source identity

For every modality, assign a stable source ID and retain the original in the configured blob store. That allows replacement, deletion, audit, and index rebuild without relying on transient extracted output.
