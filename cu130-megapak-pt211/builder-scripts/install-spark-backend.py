#!/usr/bin/env python3
"""Install the pinned official Spark-H3 isolated backend without changing core kitchen."""

import argparse
import hashlib
import importlib.metadata
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


ARCHIVE_NAME = "ComfyUI-Spark-H3-0.2.0-linux-x86_64-sm120-cu130.zip"
ARCHIVE_SHA256 = "b51a3ab20c6e677de67bfc3693727ff3789c9d2077680b4ff540d5766593858d"
WHEEL_NAME = "spark_h3_kitchen-0.2.37+spark.h3.sm120.cu130.1-cp312-abi3-linux_x86_64.whl"
WHEEL_MEMBER = "ComfyUI-Spark-H3/wheelhouse/" + WHEEL_NAME


def extract_backend(archive: Path, output_dir: Path) -> Path:
    data = archive.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise RuntimeError(f"Spark archive SHA256 mismatch: {digest}")
    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
        if bundle.namelist().count(WHEEL_MEMBER) != 1:
            raise RuntimeError("Missing or ambiguous pinned Spark wheel")
        wheel = bundle.read(WHEEL_MEMBER)
    # New nodes use the INT8 APIs and CUDA helpers as well as spark_attn.
    # Reject a wheel that could overwrite official core files.
    with zipfile.ZipFile(io.BytesIO(wheel)) as package:
        files = package.namelist()
        if any(name.split("/", 1)[0] not in {
            "spark_h3_kitchen", "spark_h3_kitchen-0.2.37+spark.h3.sm120.cu130.1.dist-info"
        } for name in files):
            raise RuntimeError("Spark wheel must contain only its isolated package and metadata")
        for name in ("spark_h3_kitchen/__init__.py",
                     "spark_h3_kitchen/backends/cuda/__init__.py",
                     "spark_h3_kitchen/backends/cuda/_C.abi3.so"):
            if files.count(name) != 1:
                raise RuntimeError(f"Missing or ambiguous Spark wheel member: {name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / WHEEL_NAME
    path.write_bytes(wheel)
    return path


def install_backend(archive: Path) -> None:
    core_version = importlib.metadata.version("comfy-kitchen")
    with tempfile.TemporaryDirectory() as temporary:
        wheel = extract_backend(archive, Path(temporary))
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)],
            env={**os.environ, "PIP_USER": "false"}, check=True,
        )
    if importlib.metadata.version("comfy-kitchen") != core_version:
        raise RuntimeError("Spark installation changed the official comfy-kitchen version")
    print("Installed official isolated spark-h3-kitchen (SM120 / CUDA 13.0)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    args = parser.parse_args()
    install_backend(args.archive)


if __name__ == "__main__":
    main()
