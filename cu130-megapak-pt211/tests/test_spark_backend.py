"""Private Spark installation and call-routing regressions."""

import hashlib
import importlib.util
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

WRAPPER = '''events = []
def _register():
    events.append("global registration")
def spark_attn():
    return "private CUDA call"
_register(
)
'''

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
        self.wheel = self.root / "spark.whl"
        self.target = self.root / "comfyui_spark_cuda"

    def make_wheel(self, wrapper=WRAPPER):
        with zipfile.ZipFile(self.wheel, "w") as archive:
            archive.writestr("comfy_kitchen/backends/cuda/__init__.py", wrapper)
            archive.writestr("comfy_kitchen/backends/cuda/_C.abi3.so", b"native extension")
            archive.writestr("comfy_kitchen/int8_decode.py", "must not be extracted")
            archive.writestr("kitchen.dist-info/licenses/LICENSE", "license")
            archive.writestr("kitchen.dist-info/licenses/NOTICE", "notice")
        digest = hashlib.sha256(self.wheel.read_bytes()).hexdigest()
        self.hash_patch = patch.object(installer, "WHEEL_SHA256", digest)
        self.hash_patch.start()
        self.addCleanup(self.hash_patch.stop)

    def test_only_private_members_are_extracted_and_registration_is_disabled(self):
        self.make_wheel()
        installer.install_backend(self.wheel, self.target)
        self.assertEqual({p.name for p in self.target.iterdir()},
                         {"__init__.py", "_C.abi3.so", "LICENSE", "NOTICE", "manifest.json"})
        namespace = {}
        exec((self.target / "__init__.py").read_text(), namespace)
        self.assertEqual(namespace["events"], [])
        self.assertEqual(namespace["spark_attn"](), "private CUDA call")
        self.assertFalse((self.root / "comfy_kitchen").exists())

    def test_bad_hash_is_rejected_before_writing(self):
        self.make_wheel()
        self.wheel.write_bytes(self.wheel.read_bytes() + b"corrupted")
        with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
            installer.install_backend(self.wheel, self.target)
        self.assertFalse(self.target.exists())

    def test_core_package_target_is_rejected(self):
        self.make_wheel()
        target = self.root / "comfy_kitchen"
        with self.assertRaisesRegex(ValueError, "must not replace"):
            installer.install_backend(self.wheel, target)
        self.assertFalse(target.exists())

    def test_unknown_registration_structure_is_rejected_before_writing(self):
        self.make_wheel(WRAPPER + "_register()\n")
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            installer.install_backend(self.wheel, self.target)
        self.assertFalse(self.target.exists())

    def test_node_uses_private_module_and_patch_does_not_rewrite_on_restart(self):
        path = self.root / "comfyui_backend.py"
        path.write_text(NODE)
        startup.patch_spark(path)
        before = path.read_bytes(), path.stat().st_mtime_ns
        startup.patch_spark(path)
        self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns))
        private = types.ModuleType("comfyui_spark_cuda")
        private.spark_attn = lambda: "private result"
        namespace = {}
        exec(path.read_text(), namespace)
        controller = types.SimpleNamespace()
        with patch.dict(sys.modules, {"comfyui_spark_cuda": private}):
            self.assertEqual(namespace["run"](controller), "private result")
        self.assertEqual(controller.backend, "comfy-kitchen-spark-private")


if __name__ == "__main__":
    unittest.main()
