#!/usr/bin/env python3
"""Extract the pinned Spark CUDA backend without installing its old kitchen."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import zipfile


WHEEL_SHA256 = "0dc086dfe117ab581d329d1b84b901542aea694671c19f9d18c773c166e05867"
PACKAGE_NAME = "comfyui_spark_cuda"


def disable_registration(source: str) -> str:
    tree = ast.parse(source)
    calls = [node for node in tree.body
             if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
             and isinstance(node.value.func, ast.Name) and node.value.func.id == "_register"]
    if len(calls) != 1:
        raise RuntimeError("Expected exactly one top-level Spark backend registration")
    call = calls[0]
    lines = source.splitlines(keepends=True)
    lines[call.lineno - 1:call.end_lineno] = [
        "# Private Spark backend: never replace the core CUDA registry.\n"
    ]
    result = "".join(lines)
    ast.parse(result)
    return result


def install_backend(wheel: Path, target: Path) -> None:
    if target.name == "comfy_kitchen":
        raise ValueError("The private Spark backend must not replace comfy_kitchen")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if digest != WHEEL_SHA256:
        raise RuntimeError(f"Spark wheel SHA256 mismatch: {digest}")
    # Select fixed members instead of extracting the old comfy_kitchen package.
    with zipfile.ZipFile(wheel) as archive:
        source = archive.read("comfy_kitchen/backends/cuda/__init__.py").decode("utf-8")
        files = {
            "__init__.py": disable_registration(source).encode("utf-8"),
            "_C.abi3.so": archive.read("comfy_kitchen/backends/cuda/_C.abi3.so"),
        }
        for name in ("LICENSE", "NOTICE"):
            members = [m for m in archive.namelist() if m.endswith("/licenses/" + name)]
            if len(members) != 1:
                raise RuntimeError(f"Missing or ambiguous Spark license member: {name}")
            files[name] = archive.read(members[0])
    target.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (target / name).write_bytes(data)
    (target / "manifest.json").write_text(json.dumps({
        "wheel": wheel.name,
        "sha256": digest,
        "core_kitchen_unchanged": True,
        "global_registration_disabled": True,
        "gpu_architecture": "sm120",
        "cuda": "13.0",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Installed private Spark CUDA backend: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--target", type=Path)
    args = parser.parse_args()
    target = args.target
    if target is None:
        target = Path(importlib.metadata.distribution("comfy-kitchen").locate_file(PACKAGE_NAME))
    install_backend(args.wheel, target)


if __name__ == "__main__":
    main()
