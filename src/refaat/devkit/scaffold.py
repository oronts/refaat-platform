"""Scaffold an installable plugin package for a typed port.

Generalizes the skill creator to any injectable port: given a port name and a plugin name, it writes a
project with an implementation stub (attributes, properties, and methods taken from the live Protocol,
with the real method parameter names), a conformance test, a README with the allowlist and pin steps,
and, for an entry-point-tier port, the correct entry-point block. The author fills the member bodies
and installs the package.
"""

from __future__ import annotations

import inspect
import keyword
import re
import sys
from pathlib import Path

from refaat.devkit.catalog import PortSpec, get_port
from refaat.errors import ConflictError, ValidationError
from refaat.version import __version__

PLUGIN_NAME = re.compile(r"[a-z][a-z0-9_]{2,63}")


def _valid_plugin_name(name: str) -> bool:
    # A valid import name that is not a keyword or a stdlib module, so the generated package and its
    # entry point and test all import cleanly.
    return bool(
        PLUGIN_NAME.fullmatch(name)
        and not keyword.iskeyword(name)
        and not keyword.issoftkeyword(name)
        and name not in sys.stdlib_module_names
    )


def _class_name(plugin_name: str) -> str:
    return "".join(part.title() for part in plugin_name.split("_"))


def _protocol(port_name: str) -> type | None:
    import refaat.ports as ports

    return getattr(ports, port_name, None)


def _member_kind(port_name: str, member: str) -> str:
    proto = _protocol(port_name)
    if proto is None:
        return "method"
    attr = getattr(proto, member, None)
    if isinstance(attr, property):
        return "property"
    if callable(attr):
        return "method"
    # A bare annotation (for example Skill.name: str) is a data attribute the runtime reads, not a
    # method; emitting it as a method would produce a stub that satisfies presence but breaks at use.
    annotations: dict[str, object] = {}
    for klass in getattr(proto, "__mro__", [proto]):
        annotations.update(getattr(klass, "__annotations__", {}))
    return "attribute" if member in annotations else "method"


def _method_params(port_name: str, member: str) -> str:
    proto = _protocol(port_name)
    func = getattr(proto, member, None) if proto is not None else None
    if func is None or not callable(func):
        return "self, *args, **kwargs"
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return "self, *args, **kwargs"
    parts: list[str] = []
    star_emitted = False
    for param in signature.parameters.values():
        if param.kind is param.KEYWORD_ONLY and not star_emitted:
            parts.append("*")
            star_emitted = True
        if param.kind is param.VAR_POSITIONAL:
            parts.append(f"*{param.name}")
            star_emitted = True
        elif param.kind is param.VAR_KEYWORD:
            parts.append(f"**{param.name}")
        else:
            parts.append(param.name)
    return ", ".join(parts) or "self"


class PortScaffolder:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, port_name: str, plugin_name: str) -> Path:
        spec = get_port(port_name)
        if spec is None:
            raise ValidationError(f"Unknown port {port_name!r}; run 'refaat plugin ports' for the catalog")
        if spec.tier == "none":
            raise ValidationError(
                f"port {port_name!r} has no injection seam yet, so it cannot be scaffolded as a plugin"
            )
        if spec.tier == "config":
            raise ValidationError(
                f"port {port_name!r} is selected by the {spec.seam!r} setting and has no plugin injection seam yet"
            )
        if not _valid_plugin_name(plugin_name):
            raise ValidationError(
                "Plugin names must be lowercase identifiers of 3 to 64 characters, not a keyword or stdlib module name"
            )
        target = self.root / plugin_name
        if target.resolve().parent != self.root or target.exists():
            raise ConflictError(f"Plugin project already exists or escapes the root: {plugin_name}")
        package = target / "src" / plugin_name
        tests = target / "tests"
        package.mkdir(parents=True)
        tests.mkdir(parents=True)
        (target / "pyproject.toml").write_text(self._pyproject(spec, plugin_name), encoding="utf-8")
        (package / "__init__.py").write_text(self._implementation(spec, plugin_name), encoding="utf-8")
        (tests / "test_conformance.py").write_text(self._test(spec, plugin_name), encoding="utf-8")
        (target / "README.md").write_text(self._readme(spec, plugin_name), encoding="utf-8")
        return target

    @staticmethod
    def _pyproject(spec: PortSpec, plugin_name: str) -> str:
        class_name = _class_name(plugin_name)
        entry_points = ""
        if spec.tier == "entry_point":
            entry_points = f'\n[project.entry-points."{spec.seam}"]\n{plugin_name} = "{plugin_name}:{class_name}"\n'
        return (
            "[build-system]\n"
            'requires = ["setuptools>=75"]\n'
            'build-backend = "setuptools.build_meta"\n\n'
            "[project]\n"
            f'name = "refaat-plugin-{plugin_name.replace("_", "-")}"\n'
            'version = "0.1.0"\n'
            f'description = "A Refaat {spec.name} adapter"\n'
            'requires-python = ">=3.12"\n'
            f'dependencies = ["refaat>={__version__},<1"]\n'
            f"{entry_points}\n"
            "[tool.setuptools.packages.find]\n"
            'where = ["src"]\n'
            f'include = ["{plugin_name}*"]\n'
        )

    @staticmethod
    def _implementation(spec: PortSpec, plugin_name: str) -> str:
        class_name = _class_name(plugin_name)
        body = [
            "from __future__ import annotations\n\n",
            f"class {class_name}:\n",
            f'    """A {spec.name} adapter.\n\n    {spec.notes}\n\n'
            f'    Run \'refaat plugin describe {spec.name}\' for the full contract.\n    """\n\n',
        ]
        for member in spec.members:
            kind = _member_kind(spec.name, member)
            if kind == "attribute":
                body.append(f"    {member} = None  # implement {spec.name}.{member}\n\n")
            elif kind == "property":
                body.append("    @property\n")
                body.append(f"    def {member}(self):\n")
                body.append(f'        raise NotImplementedError("implement {spec.name}.{member}")\n\n')
            else:
                params = _method_params(spec.name, member)
                body.append(f"    def {member}({params}):\n")
                body.append(f'        raise NotImplementedError("implement {spec.name}.{member}")\n\n')
        return "".join(body)

    @staticmethod
    def _test(spec: PortSpec, plugin_name: str) -> str:
        class_name = _class_name(plugin_name)
        return (
            "from __future__ import annotations\n\n"
            "from refaat.conformance import structural_conformance\n\n"
            f"from {plugin_name} import {class_name}\n\n\n"
            "def test_structural_conformance() -> None:\n"
            f'    assert structural_conformance({class_name}, "{spec.name}") == []\n'
        )

    @staticmethod
    def _readme(spec: PortSpec, plugin_name: str) -> str:
        class_name = _class_name(plugin_name)
        if spec.tier == "entry_point":
            register = (
                f"This package publishes the `{spec.seam}` entry point. After installing it, allow the plugin, "
                f"then pin it (see the plugin pinning guide):\n\n"
                "```bash\n"
                f"export REFAAT_PLUGIN_ALLOWLIST={plugin_name}\n"
                "```\n"
            )
        else:
            register = (
                f"Inject the adapter in process on the builder:\n\n"
                "```python\n"
                "from refaat.application.runtime import RuntimeBuilder\n\n"
                f"runtime = RuntimeBuilder(settings).{spec.seam}({class_name}()).build()\n"
                "```\n"
            )
        return (
            f"# refaat-plugin-{plugin_name.replace('_', '-')}\n\n"
            f"A Refaat `{spec.name}` adapter.\n\n"
            f"Implement the members in `src/{plugin_name}/__init__.py`, then verify:\n\n"
            "```bash\n"
            f"refaat plugin verify {plugin_name}:{class_name} --port {spec.name}\n"
            "```\n\n"
            f"{register}"
        )
