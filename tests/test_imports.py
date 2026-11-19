"""Static import check — runs without torch, TensorRT, camera SDK or relay.

Many modules import torch / ultralytics / the Hikrobot SDK, so they cannot be
imported on a CI runner. Instead, parse every Python file and verify that each
`from yarnbreak.<mod> import <name>` points at a module that exists and a name
that module defines at top level. This catches broken paths after a refactor
(e.g. a leftover `from relay_hub import ...` from before the package move).
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "yarnbreak"

# Modules that used to live at the repo root and must now be imported from the package.
MOVED = {
    "relay_hub", "modbus_relay", "state_manager", "alarm_logger", "live_view",
    "mvs_sdk", "comb_pipeline", "auth", "inference_worker",
}


def _py_files():
    for p in ROOT.rglob("*.py"):
        if any(part in {"venv", ".venv", "__pycache__", ".git"} for part in p.parts):
            continue
        yield p


def _top_level_names(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
    return names


FILES = sorted(_py_files())


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_imports_resolve(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mod = node.module
            assert mod.split(".")[0] not in MOVED, (
                f"{path.name}:{node.lineno} imports '{mod}' from the old root location; "
                f"use 'yarnbreak.{mod}'")
            if mod == "yarnbreak":
                for a in node.names:
                    assert (PKG / f"{a.name}.py").exists(), f"{path.name}:{node.lineno} no module yarnbreak.{a.name}"
            elif mod.startswith("yarnbreak."):
                target = PKG / (mod.split(".", 1)[1].replace(".", "/") + ".py")
                assert target.exists(), f"{path.name}:{node.lineno} no module {mod}"
                defined = _top_level_names(target)
                for a in node.names:
                    assert a.name == "*" or a.name in defined, (
                        f"{path.name}:{node.lineno} '{a.name}' is not defined in {mod}")
        elif isinstance(node, ast.Import):
            for a in node.names:
                assert a.name.split(".")[0] not in MOVED, (
                    f"{path.name}:{node.lineno} imports '{a.name}' from the old root location")
                if a.name.startswith("yarnbreak."):
                    target = PKG / (a.name.split(".", 1)[1].replace(".", "/") + ".py")
                    assert target.exists(), f"{path.name}:{node.lineno} no module {a.name}"
