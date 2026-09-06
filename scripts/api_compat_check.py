from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

# The committed openapi.json is guaranteed to equal the live create_app().openapi()
# by the separate openapi drift gate, so this gate treats it as the authoritative
# current surface and classifies it against the frozen baseline. Both gates are in
# the required set, so a corrupted openapi.json fails the openapi gate independently.

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "docs" / "reference" / "openapi.baseline.json"
CURRENT = ROOT / "docs" / "reference" / "openapi.json"

HTTP_METHODS = frozenset({"get", "put", "post", "delete", "options", "head", "patch", "trace"})
DOC_KEYS = frozenset({"title", "description", "example", "examples", "summary"})

BREAKING = "breaking"
COMPATIBLE = "compatible"


@dataclass(frozen=True, slots=True)
class Change:
    kind: str
    location: str
    detail: str


def _normalize(value: object) -> object:
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in sorted(value.items()) if key not in DOC_KEYS}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def _signature(value: object) -> str:
    return json.dumps(_normalize(value), sort_keys=True, ensure_ascii=False)


def _as_dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _str_set(value: object) -> set[str]:
    return {str(item) for item in value} if isinstance(value, list) else set()


def _ref_name(schema: object) -> str | None:
    if isinstance(schema, dict):
        reference = schema.get("$ref")
        if isinstance(reference, str):
            return reference.rsplit("/", 1)[-1]
    return None


def _operations(spec: dict[str, object]) -> dict[tuple[str, str], dict[str, object]]:
    result: dict[tuple[str, str], dict[str, object]] = {}
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return result
    for path, item in paths.items():
        if not isinstance(item, dict):
            continue
        for method, operation in item.items():
            if method.lower() in HTTP_METHODS and isinstance(operation, dict):
                result[(str(path), method.lower())] = operation
    return result


def _parameters(operation: dict[str, object]) -> dict[tuple[str, str], bool]:
    result: dict[tuple[str, str], bool] = {}
    values = operation.get("parameters")
    if not isinstance(values, list):
        return result
    for parameter in values:
        if isinstance(parameter, dict):
            key = (str(parameter.get("name")), str(parameter.get("in")))
            result[key] = bool(parameter.get("required", False))
    return result


def _parameter_schemas(operation: dict[str, object]) -> dict[tuple[str, str], dict[str, object]]:
    result: dict[tuple[str, str], dict[str, object]] = {}
    values = operation.get("parameters")
    if not isinstance(values, list):
        return result
    for parameter in values:
        if isinstance(parameter, dict):
            key = (str(parameter.get("name")), str(parameter.get("in")))
            result[key] = _as_dict(parameter.get("schema"))
    return result


def _content_targets(container: dict[str, object]) -> dict[str, str]:
    result: dict[str, str] = {}
    content = container.get("content")
    if not isinstance(content, dict):
        return result
    for media_type, media in content.items():
        if not isinstance(media, dict):
            continue
        schema = media.get("schema")
        ref = _ref_name(schema)
        result[str(media_type)] = ref if ref is not None else _signature(schema)
    return result


def _type_signature(schema: dict[str, object]) -> str:
    declared = schema.get("type")
    if declared is not None:
        return json.dumps(declared, sort_keys=True)
    members: list[object] = []
    for combinator in ("anyOf", "oneOf", "allOf"):
        values = schema.get(combinator)
        if isinstance(values, list):
            for member in values:
                if isinstance(member, dict):
                    members.append(member.get("type"))
    return json.dumps(sorted(json.dumps(member, sort_keys=True) for member in members))


def _enum_values(schema: dict[str, object]) -> set[str] | None:
    values = schema.get("enum")
    if not isinstance(values, list):
        return None
    return {json.dumps(value, sort_keys=True) for value in values}


def _classify_content(location: str, before: dict[str, object], after: dict[str, object]) -> list[Change]:
    changes: list[Change] = []
    before_targets = _content_targets(before)
    after_targets = _content_targets(after)
    for media_type in sorted(before_targets.keys() - after_targets.keys()):
        changes.append(Change(BREAKING, f"{location}.content.{media_type}", "media type removed"))
    for media_type in sorted(after_targets.keys() - before_targets.keys()):
        changes.append(Change(COMPATIBLE, f"{location}.content.{media_type}", "media type added"))
    for media_type in sorted(before_targets.keys() & after_targets.keys()):
        if before_targets[media_type] != after_targets[media_type]:
            changes.append(Change(BREAKING, f"{location}.content.{media_type}", "media type schema changed"))
    return changes


def _classify_operation(
    path: str,
    method: str,
    before: dict[str, object],
    after: dict[str, object],
    before_root_security: object,
    after_root_security: object,
) -> list[Change]:
    changes: list[Change] = []
    location = f"paths.{path}.{method}"

    before_parameters = _parameters(before)
    after_parameters = _parameters(after)
    before_parameter_schemas = _parameter_schemas(before)
    after_parameter_schemas = _parameter_schemas(after)
    for key in sorted(before_parameters.keys() - after_parameters.keys()):
        changes.append(Change(BREAKING, f"{location}.parameters.{key[1]}.{key[0]}", "parameter removed"))
    for key in sorted(after_parameters.keys() - before_parameters.keys()):
        required = after_parameters[key]
        kind = BREAKING if required else COMPATIBLE
        detail = "required parameter added" if required else "optional parameter added"
        changes.append(Change(kind, f"{location}.parameters.{key[1]}.{key[0]}", detail))
    for key in sorted(before_parameters.keys() & after_parameters.keys()):
        prefix = f"{location}.parameters.{key[1]}.{key[0]}"
        if not before_parameters[key] and after_parameters[key]:
            changes.append(Change(BREAKING, prefix, "parameter became required"))
        changes.extend(
            _classify_property(prefix, before_parameter_schemas.get(key, {}), after_parameter_schemas.get(key, {}))
        )

    before_body = _as_dict(before.get("requestBody"))
    after_body = _as_dict(after.get("requestBody"))
    if before_body and not after_body:
        changes.append(Change(BREAKING, f"{location}.requestBody", "request body removed"))
    elif after_body and not before_body:
        kind = BREAKING if after_body.get("required", False) else COMPATIBLE
        changes.append(Change(kind, f"{location}.requestBody", "request body added"))
    elif before_body and after_body:
        if not before_body.get("required", False) and after_body.get("required", False):
            changes.append(Change(BREAKING, f"{location}.requestBody", "request body became required"))
        changes.extend(_classify_content(f"{location}.requestBody", before_body, after_body))

    before_responses = _as_dict(before.get("responses"))
    after_responses = _as_dict(after.get("responses"))
    for code in sorted(before_responses.keys() - after_responses.keys()):
        changes.append(Change(BREAKING, f"{location}.responses.{code}", "response removed"))
    for code in sorted(after_responses.keys() - before_responses.keys()):
        changes.append(Change(COMPATIBLE, f"{location}.responses.{code}", "response added"))
    for code in sorted(before_responses.keys() & after_responses.keys()):
        before_response = before_responses[code]
        after_response = after_responses[code]
        if isinstance(before_response, dict) and isinstance(after_response, dict):
            changes.extend(_classify_content(f"{location}.responses.{code}", before_response, after_response))

    before_security = before.get("security") if "security" in before else before_root_security
    after_security = after.get("security") if "security" in after else after_root_security
    if _signature(before_security) != _signature(after_security):
        changes.append(Change(BREAKING, f"{location}.security", "operation security changed"))

    return changes


def _classify_enum(location: str, before: dict[str, object], after: dict[str, object]) -> list[Change]:
    changes: list[Change] = []
    before_enum = _enum_values(before)
    after_enum = _enum_values(after)
    if before_enum is None and after_enum is None:
        return changes
    before_enum = before_enum or set()
    after_enum = after_enum or set()
    for value in sorted(before_enum - after_enum):
        changes.append(Change(BREAKING, f"{location}.enum", f"enum value removed: {value}"))
    for value in sorted(after_enum - before_enum):
        changes.append(Change(COMPATIBLE, f"{location}.enum", f"enum value added: {value}"))
    return changes


def _classify_property(location: str, before: dict[str, object], after: dict[str, object]) -> list[Change]:
    changes: list[Change] = []
    if _ref_name(before) != _ref_name(after):
        # A property whose $ref target changes (or flips inline to a reference) is a shape change a
        # client depends on; the body path already diffs refs, so the property path matches it.
        changes.append(Change(BREAKING, f"{location}.$ref", "schema target changed"))
    elif _type_signature(before) != _type_signature(after):
        changes.append(Change(BREAKING, f"{location}.type", "type changed"))
    changes.extend(_classify_enum(location, before, after))
    return changes


def _classify_schema(name: str, before: dict[str, object], after: dict[str, object]) -> list[Change]:
    changes: list[Change] = []
    location = f"components.schemas.{name}"

    changes.extend(_classify_enum(location, before, after))

    before_properties = _as_dict(before.get("properties"))
    after_properties = _as_dict(after.get("properties"))
    for prop in sorted(before_properties.keys() - after_properties.keys()):
        changes.append(Change(BREAKING, f"{location}.properties.{prop}", "property removed"))
    for prop in sorted(after_properties.keys() - before_properties.keys()):
        changes.append(Change(COMPATIBLE, f"{location}.properties.{prop}", "property added"))

    before_required = _str_set(before.get("required"))
    after_required = _str_set(after.get("required"))
    for prop in sorted(after_required - before_required):
        changes.append(Change(BREAKING, f"{location}.required.{prop}", "property became required"))
    for prop in sorted(before_required - after_required):
        changes.append(Change(COMPATIBLE, f"{location}.required.{prop}", "property became optional"))

    for prop in sorted(before_properties.keys() & after_properties.keys()):
        before_property = before_properties[prop]
        after_property = after_properties[prop]
        if isinstance(before_property, dict) and isinstance(after_property, dict):
            changes.extend(_classify_property(f"{location}.properties.{prop}", before_property, after_property))
    return changes


def _schemas(spec: dict[str, object]) -> dict[str, object]:
    components = spec.get("components")
    if not isinstance(components, dict):
        return {}
    schemas = components.get("schemas")
    return schemas if isinstance(schemas, dict) else {}


def _security_schemes(spec: dict[str, object]) -> dict[str, object]:
    components = spec.get("components")
    if not isinstance(components, dict):
        return {}
    schemes = components.get("securitySchemes")
    return schemes if isinstance(schemes, dict) else {}


def classify(baseline: dict[str, object], current: dict[str, object]) -> list[Change]:
    changes: list[Change] = []

    before_root_security = baseline.get("security")
    after_root_security = current.get("security")

    before_operations = _operations(baseline)
    after_operations = _operations(current)
    for key in sorted(before_operations.keys() - after_operations.keys()):
        changes.append(Change(BREAKING, f"paths.{key[0]}.{key[1]}", "operation removed"))
    for key in sorted(after_operations.keys() - before_operations.keys()):
        changes.append(Change(COMPATIBLE, f"paths.{key[0]}.{key[1]}", "operation added"))
    for key in sorted(before_operations.keys() & after_operations.keys()):
        changes.extend(
            _classify_operation(
                key[0],
                key[1],
                before_operations[key],
                after_operations[key],
                before_root_security,
                after_root_security,
            )
        )

    before_schemas = _schemas(baseline)
    after_schemas = _schemas(current)
    for name in sorted(before_schemas.keys() - after_schemas.keys()):
        changes.append(Change(BREAKING, f"components.schemas.{name}", "schema removed"))
    for name in sorted(after_schemas.keys() - before_schemas.keys()):
        changes.append(Change(COMPATIBLE, f"components.schemas.{name}", "schema added"))
    for name in sorted(before_schemas.keys() & after_schemas.keys()):
        before_schema = before_schemas[name]
        after_schema = after_schemas[name]
        if isinstance(before_schema, dict) and isinstance(after_schema, dict):
            changes.extend(_classify_schema(name, before_schema, after_schema))

    before_schemes = _security_schemes(baseline)
    after_schemes = _security_schemes(current)
    for name in sorted(before_schemes.keys() - after_schemes.keys()):
        changes.append(Change(BREAKING, f"components.securitySchemes.{name}", "security scheme removed"))
    for name in sorted(after_schemes.keys() - before_schemes.keys()):
        changes.append(Change(COMPATIBLE, f"components.securitySchemes.{name}", "security scheme added"))
    for name in sorted(before_schemes.keys() & after_schemes.keys()):
        # Any change to a scheme definition (the auth mechanism, the api-key header name and location,
        # the bearer format) is client-breaking, so a signature change is denied.
        if _signature(before_schemes[name]) != _signature(after_schemes[name]):
            changes.append(Change(BREAKING, f"components.securitySchemes.{name}", "security scheme definition changed"))

    return sorted(changes, key=lambda change: (change.kind, change.location, change.detail))


def _version_tuple(spec: dict[str, object]) -> tuple[int, ...]:
    info = spec.get("info")
    raw = info.get("version") if isinstance(info, dict) else ""
    parts: list[int] = []
    for token in str(raw).split("."):
        digits = "".join(character for character in token if character.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _load(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"OpenAPI document is not an object: {path}")
    return document


def _label(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def run_check(baseline_path: Path, current_path: Path) -> int:
    baseline = _load(baseline_path)
    current = _load(current_path)

    baseline_version = _version_tuple(baseline)
    current_version = _version_tuple(current)
    if baseline_version > current_version:
        print(f"api_compat failed: baseline version {baseline_version} is ahead of current {current_version}")
        return 1

    changes = classify(baseline, current)
    breaking = [change for change in changes if change.kind == BREAKING]
    if not changes and baseline_version != current_version:
        print(f"api_compat failed: identical surface but baseline version {baseline_version} != {current_version}")
        return 1

    summary = {
        "baseline": _label(baseline_path),
        "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "breaking": [[change.location, change.detail] for change in breaking],
        "compatible_count": len(changes) - len(breaking),
        "result": "fail" if breaking else "pass",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if breaking else 0


def run_write(baseline_path: Path, current_path: Path) -> int:
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(current_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Pinned {_label(baseline_path)} to {_label(current_path)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify the OpenAPI surface against a frozen baseline")
    parser.add_argument("--baseline", default=str(BASELINE), help="baseline OpenAPI document")
    parser.add_argument("--current", default=str(CURRENT), help="current OpenAPI document")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="fail closed on any breaking change (default)")
    action.add_argument("--write", action="store_true", help="re-pin the baseline to the current surface")
    args = parser.parse_args(argv)

    baseline_path = Path(args.baseline)
    current_path = Path(args.current)
    try:
        if args.write:
            return run_write(baseline_path, current_path)
        return run_check(baseline_path, current_path)
    except (OSError, ValueError) as error:
        print(f"api_compat failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
