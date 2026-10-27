#!/usr/bin/env python3
"""Apply narrow, repeatable fixes to bundled and persisted startup code."""

import argparse
import ast
import importlib.metadata
from pathlib import Path


def replace_once(source: str, old: str, new: str, path: Path) -> str:
    if new in source:
        return source
    if source.count(old) != 1:
        raise RuntimeError(f"Unsupported startup code in {path}: expected {old!r} once")
    return source.replace(old, new, 1)


def save_patch(path: Path, original: str, patched: str) -> None:
    if patched != original:
        ast.parse(patched, filename=str(path))
        path.write_text(patched, encoding="utf-8")
        print(f"[STARTUP] Patched {path}")


def patch_manager(path: Path) -> None:
    original = path.read_text(encoding="utf-8")
    # PIP_USER must remain enabled for installs into the persisted user site,
    # but pip list must also see the image's system-wide packages.
    patched = replace_once(
        original,
        "subprocess.check_output(make_pip_cmd(['list']), universal_newlines=True)",
        "subprocess.check_output(make_pip_cmd(['list']), universal_newlines=True, "
        'env={**os.environ, "PIP_USER": "false"})',
        path,
    )
    save_patch(path, original, patched)


def patch_llm_party(path: Path) -> None:
    original = path.read_text(encoding="utf-8")
    patched = replace_once(
        original,
        "import packaging.tags",
        "import packaging.tags\nfrom packaging.requirements import Requirement",
        path,
    )
    patched = replace_once(
        patched,
        "importlib.metadata.version(package_name)",
        "importlib.metadata.version(Requirement(package_name).name)",
        path,
    )
    # The voice extra is not a distribution. Check its runtime dependencies
    # separately so an incomplete user installation can still be repaired.
    patched = replace_once(
        patched,
        'if not package_is_installed("py-cord[voice]"):',
        'if not all(package_is_installed(name) for name in ("py-cord", "PyNaCl", "davey")):',
        path,
    )
    save_patch(path, original, patched)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manager", action="store_true")
    parser.add_argument("--nodes-dir", type=Path)
    args = parser.parse_args()
    if args.manager:
        distribution = importlib.metadata.distribution("comfyui-manager")
        patch_manager(Path(distribution.locate_file("comfyui_manager/common/manager_util.py")))
    if args.nodes_dir:
        path = args.nodes_dir / "comfyui_LLM_party" / "install.py"
        if path.is_file():
            patch_llm_party(path)


if __name__ == "__main__":
    main()
