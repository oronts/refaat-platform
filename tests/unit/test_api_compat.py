from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import api_compat_check as compat


def _spec(paths: dict | None = None, schemas: dict | None = None, version: str = "0.1.0") -> dict:
    return {
        "openapi": "3.1.0",
        "info": {"version": version},
        "paths": paths or {},
        "components": {"schemas": schemas or {}},
    }


def _kinds(baseline: dict, current: dict) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {compat.BREAKING: [], compat.COMPATIBLE: []}
    for change in compat.classify(baseline, current):
        result[change.kind].append(change.location)
    return result


def test_identical_surface_has_no_changes() -> None:
    spec = _spec(paths={"/a": {"get": {"responses": {"200": {}}}}})
    assert compat.classify(spec, spec) == []


def test_removed_operation_is_breaking() -> None:
    before = _spec(paths={"/a": {"get": {"responses": {"200": {}}}}})
    after = _spec(paths={})
    kinds = _kinds(before, after)
    assert kinds[compat.BREAKING] == ["paths./a.get"]
    assert kinds[compat.COMPATIBLE] == []


def test_added_operation_is_compatible() -> None:
    before = _spec(paths={})
    after = _spec(paths={"/a": {"get": {"responses": {"200": {}}}}})
    kinds = _kinds(before, after)
    assert kinds[compat.COMPATIBLE] == ["paths./a.get"]
    assert kinds[compat.BREAKING] == []


def test_added_required_parameter_is_breaking() -> None:
    before = _spec(paths={"/a": {"get": {"parameters": [], "responses": {"200": {}}}}})
    after = _spec(
        paths={
            "/a": {"get": {"parameters": [{"name": "q", "in": "query", "required": True}], "responses": {"200": {}}}}
        }
    )
    kinds = _kinds(before, after)
    assert kinds[compat.BREAKING] == ["paths./a.get.parameters.query.q"]


def test_added_optional_parameter_is_compatible() -> None:
    before = _spec(paths={"/a": {"get": {"parameters": [], "responses": {"200": {}}}}})
    after = _spec(
        paths={
            "/a": {"get": {"parameters": [{"name": "q", "in": "query", "required": False}], "responses": {"200": {}}}}
        }
    )
    kinds = _kinds(before, after)
    assert kinds[compat.COMPATIBLE] == ["paths./a.get.parameters.query.q"]
    assert kinds[compat.BREAKING] == []


def test_parameter_becoming_required_is_breaking() -> None:
    before = _spec(
        paths={
            "/a": {"get": {"parameters": [{"name": "q", "in": "query", "required": False}], "responses": {"200": {}}}}
        }
    )
    after = _spec(
        paths={
            "/a": {"get": {"parameters": [{"name": "q", "in": "query", "required": True}], "responses": {"200": {}}}}
        }
    )
    kinds = _kinds(before, after)
    assert kinds[compat.BREAKING] == ["paths./a.get.parameters.query.q"]


def test_removed_response_property_is_breaking() -> None:
    before = _spec(schemas={"Answer": {"properties": {"text": {"type": "string"}, "score": {"type": "number"}}}})
    after = _spec(schemas={"Answer": {"properties": {"text": {"type": "string"}}}})
    kinds = _kinds(before, after)
    assert kinds[compat.BREAKING] == ["components.schemas.Answer.properties.score"]


def test_added_response_property_is_compatible() -> None:
    before = _spec(schemas={"Answer": {"properties": {"text": {"type": "string"}}}})
    after = _spec(schemas={"Answer": {"properties": {"text": {"type": "string"}, "score": {"type": "number"}}}})
    kinds = _kinds(before, after)
    assert kinds[compat.COMPATIBLE] == ["components.schemas.Answer.properties.score"]
    assert kinds[compat.BREAKING] == []


def test_removed_schema_is_breaking() -> None:
    before = _spec(schemas={"Answer": {"properties": {}}})
    after = _spec(schemas={})
    kinds = _kinds(before, after)
    assert kinds[compat.BREAKING] == ["components.schemas.Answer"]


def test_property_added_to_required_is_breaking() -> None:
    before = _spec(schemas={"Answer": {"properties": {"text": {"type": "string"}}, "required": []}})
    after = _spec(schemas={"Answer": {"properties": {"text": {"type": "string"}}, "required": ["text"]}})
    kinds = _kinds(before, after)
    assert kinds[compat.BREAKING] == ["components.schemas.Answer.required.text"]


def test_property_type_change_is_breaking() -> None:
    before = _spec(schemas={"Answer": {"properties": {"score": {"type": "number"}}}})
    after = _spec(schemas={"Answer": {"properties": {"score": {"type": "string"}}}})
    kinds = _kinds(before, after)
    assert kinds[compat.BREAKING] == ["components.schemas.Answer.properties.score.type"]


def test_removed_enum_value_is_breaking_added_is_compatible() -> None:
    before = _spec(schemas={"Mode": {"properties": {"kind": {"type": "string", "enum": ["a", "b"]}}}})
    after = _spec(schemas={"Mode": {"properties": {"kind": {"type": "string", "enum": ["a", "c"]}}}})
    kinds = _kinds(before, after)
    assert kinds[compat.BREAKING] == ["components.schemas.Mode.properties.kind.enum"]
    assert kinds[compat.COMPATIBLE] == ["components.schemas.Mode.properties.kind.enum"]


def test_description_only_change_is_ignored() -> None:
    before = _spec(schemas={"Answer": {"description": "old", "properties": {"text": {"type": "string"}}}})
    after = _spec(schemas={"Answer": {"description": "new", "properties": {"text": {"type": "string"}}}})
    assert compat.classify(before, after) == []


def test_removed_response_media_type_is_breaking() -> None:
    before = _spec(
        paths={
            "/a": {"get": {"responses": {"200": {"content": {"application/json": {"schema": {"$ref": "#/x/Foo"}}}}}}}
        }
    )
    after = _spec(paths={"/a": {"get": {"responses": {"200": {"content": {}}}}}})
    kinds = _kinds(before, after)
    assert kinds[compat.BREAKING] == ["paths./a.get.responses.200.content.application/json"]


def test_changed_ref_target_is_breaking() -> None:
    before = _spec(
        paths={"/a": {"post": {"requestBody": {"content": {"application/json": {"schema": {"$ref": "#/x/Foo"}}}}}}}
    )
    after = _spec(
        paths={"/a": {"post": {"requestBody": {"content": {"application/json": {"schema": {"$ref": "#/x/Bar"}}}}}}}
    )
    kinds = _kinds(before, after)
    assert kinds[compat.BREAKING] == ["paths./a.post.requestBody.content.application/json"]


def test_root_security_removal_is_breaking() -> None:
    before = _spec(paths={"/a": {"get": {"responses": {"200": {}}}}})
    before["security"] = [{"bearerAuth": []}]
    after = _spec(paths={"/a": {"get": {"responses": {"200": {}}}}})
    kinds = _kinds(before, after)
    assert kinds[compat.BREAKING] == ["paths./a.get.security"]


def test_explicit_unauth_operation_ignores_root_security() -> None:
    before = _spec(paths={"/a": {"get": {"security": [], "responses": {"200": {}}}}})
    before["security"] = [{"bearerAuth": []}]
    after = _spec(paths={"/a": {"get": {"security": [], "responses": {"200": {}}}}})
    assert compat.classify(before, after) == []


def test_parameter_type_change_is_breaking() -> None:
    before = _spec(
        paths={
            "/a": {
                "get": {
                    "parameters": [{"name": "q", "in": "query", "required": True, "schema": {"type": "string"}}],
                    "responses": {"200": {}},
                }
            }
        }
    )
    after = _spec(
        paths={
            "/a": {
                "get": {
                    "parameters": [{"name": "q", "in": "query", "required": True, "schema": {"type": "integer"}}],
                    "responses": {"200": {}},
                }
            }
        }
    )
    kinds = _kinds(before, after)
    assert kinds[compat.BREAKING] == ["paths./a.get.parameters.query.q.type"]


def test_parameter_enum_value_removed_is_breaking() -> None:
    before = _spec(
        paths={
            "/a": {
                "get": {
                    "parameters": [{"name": "q", "in": "query", "schema": {"type": "string", "enum": ["a", "b"]}}],
                    "responses": {"200": {}},
                }
            }
        }
    )
    after = _spec(
        paths={
            "/a": {
                "get": {
                    "parameters": [{"name": "q", "in": "query", "schema": {"type": "string", "enum": ["a"]}}],
                    "responses": {"200": {}},
                }
            }
        }
    )
    kinds = _kinds(before, after)
    assert kinds[compat.BREAKING] == ["paths./a.get.parameters.query.q.enum"]


def test_enum_schema_value_removed_is_breaking() -> None:
    before = _spec(schemas={"Mode": {"type": "string", "enum": ["a", "b"]}})
    after = _spec(schemas={"Mode": {"type": "string", "enum": ["a"]}})
    kinds = _kinds(before, after)
    assert kinds[compat.BREAKING] == ["components.schemas.Mode.enum"]


def _write(path: Path, spec: dict) -> str:
    path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def test_main_check_fails_closed_on_breaking_pair(tmp_path: Path) -> None:
    baseline = _write(tmp_path / "baseline.json", _spec(paths={"/a": {"get": {"responses": {"200": {}}}}}))
    current = _write(tmp_path / "current.json", _spec(paths={}))
    assert compat.main(["--check", "--baseline", baseline, "--current", current]) == 1


def test_main_check_passes_on_compatible_pair(tmp_path: Path) -> None:
    baseline = _write(tmp_path / "baseline.json", _spec(paths={"/a": {"get": {"responses": {"200": {}}}}}))
    current = _write(
        tmp_path / "current.json",
        _spec(paths={"/a": {"get": {"responses": {"200": {}}}}, "/b": {"get": {"responses": {"200": {}}}}}),
    )
    assert compat.main(["--check", "--baseline", baseline, "--current", current]) == 0


def test_main_check_passes_when_identical(tmp_path: Path) -> None:
    spec = _spec(paths={"/a": {"get": {"responses": {"200": {}}}}})
    baseline = _write(tmp_path / "baseline.json", spec)
    current = _write(tmp_path / "current.json", spec)
    assert compat.main(["--check", "--baseline", baseline, "--current", current]) == 0


def test_main_fails_closed_on_missing_baseline(tmp_path: Path) -> None:
    current = _write(tmp_path / "current.json", _spec())
    missing = str(tmp_path / "absent.json")
    assert compat.main(["--check", "--baseline", missing, "--current", current]) == 1


def test_main_fails_closed_on_malformed_baseline(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{ not json", encoding="utf-8")
    current = _write(tmp_path / "current.json", _spec())
    assert compat.main(["--check", "--baseline", str(baseline), "--current", str(current)]) == 1


def test_main_fails_closed_when_baseline_version_ahead(tmp_path: Path) -> None:
    baseline = _write(tmp_path / "baseline.json", _spec(version="0.2.0"))
    current = _write(tmp_path / "current.json", _spec(version="0.1.0"))
    assert compat.main(["--check", "--baseline", baseline, "--current", current]) == 1


def test_main_fails_closed_on_identical_surface_with_version_skew(tmp_path: Path) -> None:
    baseline = _write(tmp_path / "baseline.json", _spec(version="0.1.0"))
    current = _write(tmp_path / "current.json", _spec(version="0.2.0"))
    assert compat.main(["--check", "--baseline", baseline, "--current", current]) == 1


def test_property_ref_target_swap_is_breaking() -> None:
    before = _spec(
        schemas={
            "Req": {"type": "object", "properties": {"limits": {"$ref": "#/components/schemas/A"}}},
            "A": {"type": "object"},
            "B": {"type": "object"},
        }
    )
    after = _spec(
        schemas={
            "Req": {"type": "object", "properties": {"limits": {"$ref": "#/components/schemas/B"}}},
            "A": {"type": "object"},
            "B": {"type": "object"},
        }
    )
    changes = compat.classify(before, after)
    assert [(c.kind, c.location) for c in changes] == [
        (compat.BREAKING, "components.schemas.Req.properties.limits.$ref")
    ]


def _security_spec(schemes: dict) -> dict:
    return {"openapi": "3.1.0", "info": {"version": "0.1.0"}, "paths": {}, "components": {"securitySchemes": schemes}}


def test_security_scheme_definition_change_is_breaking() -> None:
    before = _security_spec({"apiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"}})
    after = _security_spec({"apiKeyAuth": {"type": "apiKey", "in": "header", "name": "Authorization"}})
    changes = compat.classify(before, after)
    assert [(c.kind, c.location, c.detail) for c in changes] == [
        (compat.BREAKING, "components.securitySchemes.apiKeyAuth", "security scheme definition changed")
    ]


def test_security_scheme_removed_is_breaking_added_is_compatible() -> None:
    before = _security_spec({"bearerAuth": {"type": "http", "scheme": "bearer"}})
    after = _security_spec({"oauth": {"type": "oauth2"}})
    results = {(c.kind, c.detail) for c in compat.classify(before, after)}
    assert (compat.BREAKING, "security scheme removed") in results
    assert (compat.COMPATIBLE, "security scheme added") in results


def test_committed_baseline_matches_live_surface() -> None:
    assert compat.main(["--check"]) == 0


def test_write_repins_baseline_verbatim(tmp_path: Path) -> None:
    current = tmp_path / "current.json"
    current.write_text(json.dumps(_spec(paths={"/a": {"get": {}}}), indent=2) + "\n", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    assert compat.main(["--write", "--baseline", str(baseline), "--current", str(current)]) == 0
    assert baseline.read_bytes() == current.read_bytes()


@pytest.mark.parametrize("value", ["0.1.0", "1.0.0", "0.2.0rc1"])
def test_version_tuple_parses(value: str) -> None:
    assert compat._version_tuple(_spec(version=value)) == tuple(
        int("".join(c for c in token if c.isdigit()) or "0") for token in value.split(".")
    )
