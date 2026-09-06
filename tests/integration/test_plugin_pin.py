"""refaat plugin pin computes an entry point's content-hash pin without importing its code."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.integration.test_plugin_pin_enforcement import GROUP, SKILL_SRC, _make_dist, _on_path

from refaat.errors import ConfigurationError
from refaat.infrastructure.plugins import PluginRegistry, pin_for_entry_point

pytestmark = pytest.mark.integration


def test_computed_pin_round_trips_and_is_admitted(tmp_path: Path) -> None:
    root = tmp_path / "site"
    code = {"okpkg/__init__.py": "", "okpkg/tool.py": SKILL_SRC}
    expected_hash = _make_dist(root, "ok-pkg", "1.0.0", code, f"[{GROUP}]\ndemo = okpkg.tool:DemoSkill\n")
    with _on_path(root):
        descriptor = pin_for_entry_point(GROUP, "demo")
        assert descriptor.code_hash == expected_hash
        assert descriptor.value == "okpkg.tool:DemoSkill"
        assert descriptor.dist_name == "ok-pkg"
        assert descriptor.dist_version == "1.0.0"
        # The computed pin is exactly what the loader admits.
        registry = PluginRegistry({"demo"}, pins={f"{GROUP}:demo": descriptor})
        assert set(registry.discover(GROUP)) == {"demo"}


def test_pin_refuses_an_unknown_entry_point(tmp_path: Path) -> None:
    root = tmp_path / "site"
    _make_dist(
        root,
        "ok-pkg",
        "1.0.0",
        {"okpkg/__init__.py": "", "okpkg/tool.py": SKILL_SRC},
        f"[{GROUP}]\ndemo = okpkg.tool:DemoSkill\n",
    )
    with _on_path(root), pytest.raises(ConfigurationError, match="no entry point"):
        pin_for_entry_point(GROUP, "absent")


def test_pin_refuses_a_distribution_with_no_record(tmp_path: Path) -> None:
    root = tmp_path / "site"
    _make_dist(
        root,
        "ok-pkg",
        "1.0.0",
        {"okpkg/__init__.py": "", "okpkg/tool.py": SKILL_SRC},
        f"[{GROUP}]\ndemo = okpkg.tool:DemoSkill\n",
        write_record=False,
    )
    with _on_path(root), pytest.raises(ConfigurationError):
        pin_for_entry_point(GROUP, "demo")


def test_cli_rejects_an_unknown_group() -> None:
    from typer.testing import CliRunner

    from refaat.cli import app

    result = CliRunner().invoke(app, ["plugin", "pin", "not.a.group:demo"])
    assert result.exit_code != 0
