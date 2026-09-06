"""The plugin scaffolder writes a conformant package and structural verify catches a missing member."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from refaat.conformance import structural_conformance
from refaat.devkit.scaffold import PortScaffolder
from refaat.errors import ConflictError, ValidationError

pytestmark = pytest.mark.integration


def test_scaffold_writes_an_entry_point_plugin_project(tmp_path: Path) -> None:
    target = PortScaffolder(tmp_path).create("ChunkingStrategy", "acme_chunker")
    assert (target / "pyproject.toml").exists()
    assert (target / "src" / "acme_chunker" / "__init__.py").exists()
    assert (target / "tests" / "test_conformance.py").exists()
    pyproject = (target / "pyproject.toml").read_text()
    assert '[project.entry-points."refaat.chunkers"]' in pyproject
    assert "acme_chunker = " in pyproject
    readme = (target / "README.md").read_text()
    assert "refaat plugin verify acme_chunker:AcmeChunker --port ChunkingStrategy" in readme


def test_scaffolded_stub_is_structurally_conformant(tmp_path: Path) -> None:
    target = PortScaffolder(tmp_path).create("ChunkingStrategy", "acme_chunker")
    src = str(target / "src")
    sys.path.insert(0, src)
    try:
        module = importlib.import_module("acme_chunker")
        assert structural_conformance(module.AcmeChunker, "ChunkingStrategy") == []
    finally:
        sys.path.remove(src)
        sys.modules.pop("acme_chunker", None)


def test_structural_conformance_flags_a_missing_member() -> None:
    class Incomplete:
        strategy_id = "x"
        # no chunk method

    violations = structural_conformance(Incomplete, "ChunkingStrategy")
    assert [v.member for v in violations] == ["chunk"]

    class Complete:
        strategy_id = "x"

        def chunk(self, text, *, metadata, settings):
            return []

    assert structural_conformance(Complete, "ChunkingStrategy") == []


def test_scaffolder_refuses_a_none_tier_or_unknown_port(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="no injection seam"):
        PortScaffolder(tmp_path).create("FileGuard", "x_guard")
    with pytest.raises(ValidationError, match="Unknown port"):
        PortScaffolder(tmp_path).create("NoSuchPort", "x_thing")


def test_scaffolder_refuses_a_duplicate_project(tmp_path: Path) -> None:
    PortScaffolder(tmp_path).create("Connector", "acme_conn")
    with pytest.raises(ConflictError):
        PortScaffolder(tmp_path).create("Connector", "acme_conn")


def test_builder_tier_scaffold_shows_the_injection_snippet(tmp_path: Path) -> None:
    target = PortScaffolder(tmp_path).create("Embedder", "acme_embedder")
    pyproject = (target / "pyproject.toml").read_text()
    assert "entry-points" not in pyproject
    readme = (target / "README.md").read_text()
    assert "with_embedder(AcmeEmbedder())" in readme


def test_skill_annotation_members_are_scaffolded_as_attributes_not_methods(tmp_path: Path) -> None:
    target = PortScaffolder(tmp_path).create("Skill", "acme_skill")
    impl = (target / "src" / "acme_skill" / "__init__.py").read_text()
    # name and description are bare annotations on the Skill Protocol, so they must be data attributes.
    assert "name = None" in impl
    assert "description = None" in impl
    assert "def name(" not in impl
    assert "def run(" in impl
    src = str(target / "src")
    sys.path.insert(0, src)
    try:
        module = importlib.import_module("acme_skill")
        instance = module.AcmeSkill()
        assert not callable(instance.name)
        assert structural_conformance(module.AcmeSkill, "Skill") == []
    finally:
        sys.path.remove(src)
        sys.modules.pop("acme_skill", None)


def test_scaffolder_refuses_a_config_tier_port(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="no plugin injection seam"):
        PortScaffolder(tmp_path).create("BlobStore", "acme_blob")


def test_scaffolder_refuses_a_keyword_name(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="keyword or stdlib"):
        PortScaffolder(tmp_path).create("Connector", "class")
    with pytest.raises(ValidationError, match="keyword or stdlib"):
        PortScaffolder(tmp_path).create("Connector", "typing")


def test_method_stub_uses_the_real_parameter_names(tmp_path: Path) -> None:
    target = PortScaffolder(tmp_path).create("ChunkingStrategy", "acme_chunker")
    impl = (target / "src" / "acme_chunker" / "__init__.py").read_text()
    assert "def chunk(self, text, *, metadata, settings):" in impl
