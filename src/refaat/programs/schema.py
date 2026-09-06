from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from refaat.errors import ValidationError

# The single JSON-type table that backs every schema check in the program
# layer. Tool-argument validation here and structured-output validation both
# drive off this one table, so there is one validator body and no dependency on
# the third-party jsonschema package (kept test-only).
_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}

# The one explicit annotation-to-JSON-Schema-property table. A typed output
# Signature is compiled through this table into a JSON-Schema object that is
# then fed to the SAME validate_object body used for tool arguments, so there is
# exactly one validation path and no second schema engine.
_ANNOTATION_PROPERTIES: dict[str, dict[str, Any]] = {
    "str": {"type": "string"},
    "int": {"type": "integer"},
    "float": {"type": "number"},
    "bool": {"type": "boolean"},
    "list[str]": {"type": "array", "items": {"type": "string"}},
}


def _normalize_annotation(annotation: str) -> str:
    # A Signature stores str(annotation). With `from __future__ import
    # annotations` in force (the module convention) this is already "str" or
    # "list[str]"; without it the raw type stringifies as "<class 'str'>". Both
    # forms normalize to the same table key so schema compilation is robust.
    if annotation.startswith("<class '") and annotation.endswith("'>"):
        return annotation[len("<class '") : -len("'>")].rsplit(".", 1)[-1]
    return annotation


def annotation_to_property(annotation: str) -> dict[str, Any]:
    """Map a Signature field annotation string to a JSON-Schema property.

    Raises :class:`ValidationError` for an annotation outside the explicit,
    supported table so an unsupported output type fails loud at compile time
    rather than silently producing an unenforced schema.
    """

    prop = _ANNOTATION_PROPERTIES.get(_normalize_annotation(annotation))
    if prop is None:
        raise ValidationError(f"unsupported output annotation for schema compilation: {annotation!r}")
    return dict(prop)


def compile_output_schema(fields: Sequence[tuple[str, str, bool]]) -> dict[str, Any]:
    """Compile ``(name, annotation, required)`` triples into a JSON-Schema object.

    The result is a closed object schema (``additionalProperties`` false) with
    one property per field and a required list preserving declaration order, so
    two identical output Signatures compile to a byte-identical schema.
    """

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, annotation, is_required in fields:
        properties[name] = annotation_to_property(annotation)
        if is_required:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def _matches(value: Any, json_type: str) -> bool:
    expected = _JSON_TYPES.get(json_type)
    if expected is None:
        return True
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, expected)


def validate_object(value: Any, schema: Mapping[str, Any]) -> tuple[bool, str]:
    """Validate a value against a small, explicit subset of JSON Schema.

    Checks that ``value`` is an object, that every required field is present,
    that each recognized field has the declared JSON type, and, when
    ``additionalProperties`` is false, that no unexpected field appears. Returns
    a ``(ok, reason)`` pair with a stable reason string so traces are
    reproducible.
    """

    if not isinstance(value, dict):
        return False, "value must be an object"
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        properties = {}
    required = schema.get("required", ())
    additional = schema.get("additionalProperties", True)
    for name in required:
        if name not in value:
            return False, f"missing required field: {name}"
    for name, field_value in value.items():
        spec = properties.get(name)
        if spec is None:
            if additional is False:
                return False, f"unexpected field: {name}"
            continue
        expected = spec.get("type") if isinstance(spec, Mapping) else None
        if expected is not None:
            if not isinstance(expected, str) or expected not in _JSON_TYPES:
                return False, f"field {name!r} declares unsupported type {expected!r}"
            if not _matches(field_value, expected):
                return False, f"field {name!r} must be of type {expected}"
    return True, "valid"


def validate_structured(value: Any, schema: Mapping[str, Any]) -> tuple[bool, str]:
    """Validate a structured program output against a compiled output schema.

    Delegates to the single shared :func:`validate_object` body, so structured
    output and tool arguments share one validator and there is no second
    validation engine and no jsonschema import on the runtime path.
    """

    return validate_object(value, schema)
