# Multimodal generation

Refaat can ingest an image and answer over it while keeping the citation contract intact. An image becomes a real stored chunk whose text is a descriptor, so it is embedded, retrieved, and cited exactly like a text chunk. Multimodal generation is off by default: a corpus with no images and a run with no vision model configured behave identically to a text-only deployment.

## The ingestion-to-descriptor-chunk model

When you ingest an image, the bytes are stored in the blob store and one descriptor chunk is written. The chunk carries the descriptor text plus a nullable `image_ref` (the blob key) and a nullable `media_type`. For a text corpus both stay NULL, so nothing changes.

The descriptor is what the runtime embeds and retrieves. A cited image resolves through this stored descriptor chunk, never through the raw image or a caption that was not stored as a chunk.

```python
from refaat import RuntimeBuilder

runtime = RuntimeBuilder().build()
project = runtime.project(workspace="acme", corpus="assets")

project.ingest_image(
    content=image_bytes,
    media_type="image/png",
    source_id="diagram.png",
    title="System diagram",
    caption="A red circle on a white background architecture diagram",
)

answer = project.ask("red circle architecture diagram")
for citation in answer.citations:
    print(citation.chunk_id, citation.quote)
```

Accepted image types are `image/png`, `image/jpeg`, `image/webp`, and `image/gif`. Each image is validated against that allowlist and a size bound (`REFAAT_IMAGE_MAX_BYTES`) before its bytes are stored. The content is never decoded or executed, only stored and referenced.

## The offline-deterministic versus vision-model boundary

The descriptor is honest by construction:

- If you supply a `caption` or `alt_text`, that text is the descriptor.
- Otherwise, if a vision captioner is configured, it writes the caption, and it degrades to the deterministic descriptor on any failure.
- Otherwise the descriptor is deterministic: `image asset <short-digest>, <title>`, derived from the image bytes digest and the provided title.

The deterministic path never fabricates a visual description of pixels it cannot see. It is a stable, reproducible label, so re-ingesting an unchanged image yields an identical descriptor and the write stays idempotent.

A real multimodal model activates only when configured, through a lazily imported adapter that the offline gate never constructs:

| Setting | Default | Effect |
|---|---|---|
| `REFAAT_IMAGE_CAPTIONER` | `none` | `openai_vision` writes the caption at ingest from a real vision model |
| `REFAAT_MULTIMODAL_GENERATION` | `none` | `openai_vision` answers over the actual image at query time |
| `REFAAT_MULTIMODAL_MODEL` | `gpt-4o-mini` | The vision model id |
| `REFAAT_MULTIMODAL_API_KEY` | none | The vision provider key, or the shared `REFAAT_OPENAI_API_KEY` |

## Generation over the image

When a multimodal generator is active, the runtime loads the pixels of the retrieved image-descriptor hits and attaches them to the generation request. The default extractive generator ignores the pixels and grounds on the descriptor text, so the offline answer is deterministic. A real vision generator receives the image content alongside the text evidence and answers over the actual image, and it still returns citations to the real stored image-descriptor chunks. The final answer runs through the same citation resolution as every other query, so every citation resolves against a stored chunk. A vision caption or a raw image that was not stored as a chunk can never appear as a citation.

## Custom adapters

Inject your own captioner or vision generator through the builder when you do not use the OpenAI-compatible defaults:

```python
runtime = RuntimeBuilder().with_image_captioner(my_captioner).with_multimodal_generator(my_vision_generator).build()
```

A captioner implements `caption(content, *, media_type, title)` and a `provider_id`. A vision generator implements the `Generator` port (`generate` and `stream`) and reads `request.images`, each carrying the `chunk_id`, `media_type`, and bytes of one retrieved image-descriptor chunk.

## Surfaces

Every surface exposes image ingestion:

- Python: `project.ingest_image(...)` and `runtime.ingest_image(...)`.
- CLI: `refaat ingest-image WORKSPACE CORPUS PATH --caption ...`.
- HTTP: `POST /v1/corpora/{corpus_id}/documents/image` with the image file plus `source_id`, and optional `title`, `caption`, `alt_text`, and `metadata_json` form fields.
