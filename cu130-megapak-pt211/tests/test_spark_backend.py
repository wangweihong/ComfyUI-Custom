"""Official isolated Spark installation and persisted-node migrations."""

import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


installer = load("spark_installer", ROOT / "builder-scripts/install-spark-backend.py")
startup = load("spark_startup_patches", ROOT / "runner-scripts/patch-startup.py")

NODE = '''def run(controller):
    try:
        from comfy_kitchen.backends.cuda import spark_attn
    except (ImportError, OSError) as error:
        raise RuntimeError("missing Spark extension") from error
    result = spark_attn()
    controller.backend = "comfy-kitchen-spark-global"
    return result
'''


class SparkBackendTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = self.root / installer.ARCHIVE_NAME
        self.output = self.root / "wheelhouse"

    def make_archive(self, extra_files=None, omit=None, member=None):
        files = {
            "spark_h3_kitchen/__init__.py": b"# isolated public API",
            "spark_h3_kitchen/backends/cuda/__init__.py": b"# native wrapper",
            "spark_h3_kitchen/backends/cuda/_C.abi3.so": b"native extension",
            "spark_h3_kitchen/functional.py": b"# INT8 and quantized attention APIs",
            "spark_h3_kitchen-0.2.37+spark.h3.sm120.cu130.1.dist-info/METADATA": b"Name: spark-h3-kitchen",
        }
        files.update(extra_files or {})
        if omit:
            files.pop(omit)
        wheel = io.BytesIO()
        with zipfile.ZipFile(wheel, "w") as archive:
            for name, data in files.items():
                archive.writestr(name, data)
        data = wheel.getvalue()
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr(member or installer.WHEEL_MEMBER, data)
        patcher = patch.object(installer, "ARCHIVE_SHA256", hashlib.sha256(self.archive.read_bytes()).hexdigest())
        patcher.start()
        self.addCleanup(patcher.stop)
        return data

    def test_whole_isolated_wheel_is_preserved(self):
        data = self.make_archive()
        result = installer.extract_backend(self.archive, self.output)
        self.assertEqual(result.name, installer.WHEEL_NAME)
        self.assertEqual(result.read_bytes(), data)
        with zipfile.ZipFile(result) as wheel:
            self.assertIn("spark_h3_kitchen/functional.py", wheel.namelist())

    def test_bad_hash_is_rejected_before_writing(self):
        self.make_archive()
        self.archive.write_bytes(self.archive.read_bytes() + b"corrupted")
        with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
            installer.extract_backend(self.archive, self.output)
        self.assertFalse(self.output.exists())

    def test_core_package_files_are_rejected_before_writing(self):
        self.make_archive(extra_files={"comfy_kitchen/__init__.py": b"must not install"})
        with self.assertRaisesRegex(RuntimeError, "only its isolated package"):
            installer.extract_backend(self.archive, self.output)
        self.assertFalse(self.output.exists())

    def test_incomplete_backend_is_rejected(self):
        self.make_archive(omit="spark_h3_kitchen/backends/cuda/_C.abi3.so")
        with self.assertRaisesRegex(RuntimeError, "Missing or ambiguous Spark wheel member"):
            installer.extract_backend(self.archive, self.output)
        self.assertFalse(self.output.exists())

    def test_different_backend_implementation_is_not_selected(self):
        self.make_archive(member=installer.WHEEL_MEMBER.replace("0.2.37", "0.2.39"))
        with self.assertRaisesRegex(RuntimeError, "Missing or ambiguous pinned Spark wheel"):
            installer.extract_backend(self.archive, self.output)
        self.assertFalse(self.output.exists())

    def test_install_does_not_resolve_dependencies_or_use_user_site(self):
        self.make_archive()
        with patch.object(installer.importlib.metadata, "version", return_value="0.2.40"):
            with patch.object(installer.subprocess, "run") as install:
                installer.install_backend(self.archive)
        command = install.call_args.args[0]
        self.assertEqual(command[:6], [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps"])
        self.assertTrue(command[-1].endswith(installer.WHEEL_NAME))
        self.assertEqual(install.call_args.kwargs["env"]["PIP_USER"], "false")
        self.assertTrue(install.call_args.kwargs["check"])

    def test_legacy_global_and_private_nodes_use_official_package(self):
        for previous in ["comfy_kitchen.backends.cuda", "comfyui_spark_cuda"]:
            with self.subTest(previous=previous):
                path = self.root / "comfyui_backend.py"
                path.write_text(NODE.replace("comfy_kitchen.backends.cuda", previous))
                startup.patch_spark(path)
                before = path.read_bytes(), path.stat().st_mtime_ns
                startup.patch_spark(path)
                self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns))
                cuda = types.ModuleType("spark_h3_kitchen.backends.cuda")
                cuda.spark_attn = lambda: "official isolated result"
                namespace = {}
                exec(path.read_text(), namespace)
                controller = types.SimpleNamespace()
                with patch.dict(sys.modules, {"spark_h3_kitchen.backends.cuda": cuda}):
                    self.assertEqual(namespace["run"](controller), "official isolated result")
                self.assertEqual(controller.backend, "spark-h3-kitchen-isolated")

    def test_new_upstream_namespace_is_left_unchanged(self):
        path = self.root / "comfyui_backend.py"
        path.write_text(NODE.replace("comfy_kitchen.backends.cuda", "spark_h3_kitchen.backends.cuda"))
        before = path.read_bytes(), path.stat().st_mtime_ns
        startup.patch_spark(path)
        self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns))

    def test_formatted_legacy_import_preserves_alias(self):
        path = self.root / "comfyui_backend.py"
        path.write_text("from comfyui_spark_cuda import (\n    spark_attn as attention,\n)\n")
        startup.patch_spark(path)
        self.assertIn("from spark_h3_kitchen.backends.cuda import (", path.read_text())
        self.assertIn("spark_attn as attention", path.read_text())

    def test_unknown_or_duplicate_imports_fail_without_partial_writes(self):
        for source in [NODE.replace("comfy_kitchen.backends.cuda", "unknown_backend"), NODE + NODE]:
            path = self.root / "comfyui_backend.py"
            path.write_text(source)
            with self.assertRaisesRegex(RuntimeError, "expected one known spark_attn import"):
                startup.patch_spark(path)
            self.assertEqual(path.read_text(), source)


if __name__ == "__main__":
    unittest.main()
