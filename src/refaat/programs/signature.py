from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

INPUT_ROLE = "input"
OUTPUT_ROLE = "output"


@dataclass(frozen=True, slots=True)
class _FieldMarker:
    role: str
    description: str
    metadata: Mapping[str, Any]


def InputField(*, description: str = "", sensitive: bool = False, tunable: bool = False, **metadata: Any) -> Any:
    return _FieldMarker(
        role=INPUT_ROLE,
        description=description,
        metadata={"sensitive": sensitive, "tunable": tunable, **metadata},
    )


def OutputField(*, description: str = "", grounded: bool = False, required: bool = True, **metadata: Any) -> Any:
    return _FieldMarker(
        role=OUTPUT_ROLE,
        description=description,
        metadata={"grounded": grounded, "required": required, **metadata},
    )


@dataclass(frozen=True, slots=True)
class SignatureField:
    name: str
    annotation: str
    role: str
    description: str
    metadata: Mapping[str, Any]


class Signature:
    """Typed declaration of a program's inputs and outputs.

    Fields are declared with :func:`InputField` and :func:`OutputField`; the
    class collects them into :attr:`fields` and exposes a stable content hash
    used to key artifacts and caches.
    """

    fields: tuple[SignatureField, ...] = ()

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        annotations = getattr(cls, "__annotations__", {})
        resolved: list[SignatureField] = []
        for name, annotation in annotations.items():
            marker = getattr(cls, name, None)
            if not isinstance(marker, _FieldMarker):
                continue
            resolved.append(
                SignatureField(
                    name=name,
                    annotation=str(annotation),
                    role=marker.role,
                    description=marker.description,
                    metadata=dict(marker.metadata),
                )
            )
            delattr(cls, name)
        cls.fields = tuple(resolved)

    @classmethod
    def input_fields(cls) -> tuple[SignatureField, ...]:
        return tuple(field for field in cls.fields if field.role == INPUT_ROLE)

    @classmethod
    def output_fields(cls) -> tuple[SignatureField, ...]:
        return tuple(field for field in cls.fields if field.role == OUTPUT_ROLE)

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Compile the output fields into a closed JSON-Schema object.

        Each output field's annotation is mapped through the shared program-layer
        annotation table; an unsupported annotation raises ``ValidationError``.
        The schema is fed to the same ``validate_object`` body used for tool
        arguments, so structured output has one validation path.
        """
        from refaat.programs.schema import compile_output_schema

        fields = [
            (field.name, field.annotation, bool(field.metadata.get("required", True))) for field in cls.output_fields()
        ]
        return compile_output_schema(fields)

    @classmethod
    def output_schema_digest(cls) -> str:
        canonical = json.dumps(cls.json_schema(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()

    @classmethod
    def signature_hash(cls) -> str:
        payload = [
            {
                "name": field.name,
                "annotation": field.annotation,
                "role": field.role,
                "description": field.description,
                "metadata": dict(field.metadata),
            }
            for field in cls.fields
        ]
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()
