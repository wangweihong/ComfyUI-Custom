"""Regression cases for startup dependency detection and patch migrations."""

import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "runner-scripts" / "patch-startup.py"
spec = importlib.util.spec_from_file_location("startup_patches", SCRIPT)
startup_patches = importlib.util.module_from_spec(spec)
spec.loader.exec_module(startup_patches)

LLM_SOURCE = '''import importlib.metadata
import packaging.tags

def package_is_installed(package_name):
    try:
        importlib.metadata.version(package_name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False

def manage_discord_packages():
    if not package_is_installed("py-cord[voice]"):
        install_package("py-cord[voice]")
'''

MANAGER_SOURCE = '''import os
import subprocess

def query():
    return subprocess.check_output(make_pip_cmd(['list']), universal_newlines=True)
'''


class StartupPatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "module.py"

    def load_llm(self, installed):
        self.path.write_text(LLM_SOURCE)
        startup_patches.patch_llm_party(self.path)
        namespace = {}
        exec(compile(self.path.read_text(), str(self.path), "exec"), namespace)
        installs = []
        namespace["install_package"] = installs.append

        def version(name):
            if name not in installed:
                raise startup_patches.importlib.metadata.PackageNotFoundError(name)
            return "1.0"

        return namespace, installs, version

    def test_installed_voice_extra_does_not_run_pip(self):
        namespace, installs, version = self.load_llm({"py-cord", "PyNaCl", "davey"})
        with patch("importlib.metadata.version", side_effect=version):
            self.assertTrue(namespace["package_is_installed"]("py-cord[voice]"))
            namespace["manage_discord_packages"]()
        self.assertEqual(installs, [])

    def test_missing_voice_dependencies_are_detected(self):
        for missing in ("py-cord", "PyNaCl", "davey"):
            with self.subTest(missing=missing):
                namespace, installs, version = self.load_llm({"py-cord", "PyNaCl", "davey"} - {missing})
                with patch("importlib.metadata.version", side_effect=version):
                    namespace["manage_discord_packages"]()
                self.assertEqual(installs, ["py-cord[voice]"])

    def test_manager_sees_global_packages_without_changing_install_environment(self):
        self.path.write_text(MANAGER_SOURCE)
        startup_patches.patch_manager(self.path)
        namespace = {"make_pip_cmd": lambda args: ["python", "-m", "pip", *args]}
        exec(compile(self.path.read_text(), str(self.path), "exec"), namespace)
        with patch.dict(os.environ, {"PIP_USER": "true"}):
            with patch("subprocess.check_output", return_value="torch 2.11.0") as query:
                namespace["query"]()
            self.assertEqual(query.call_args.kwargs["env"]["PIP_USER"], "false")
            self.assertEqual(os.environ["PIP_USER"], "true")

    def test_patches_are_idempotent_and_do_not_rewrite_files(self):
        for source, apply in ((LLM_SOURCE, startup_patches.patch_llm_party),
                              (MANAGER_SOURCE, startup_patches.patch_manager)):
            with self.subTest(patch=apply.__name__):
                self.path.write_text(source)
                apply(self.path)
                content, modified = self.path.read_bytes(), self.path.stat().st_mtime_ns
                apply(self.path)
                self.assertEqual(self.path.read_bytes(), content)
                self.assertEqual(self.path.stat().st_mtime_ns, modified)

    def test_unknown_upstream_source_is_rejected_without_partial_writes(self):
        source = LLM_SOURCE.replace('if not package_is_installed("py-cord[voice]"):', "if True:")
        self.path.write_text(source)
        with self.assertRaisesRegex(RuntimeError, "Unsupported startup code"):
            startup_patches.patch_llm_party(self.path)
        self.assertEqual(self.path.read_text(), source)


class NodeStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle = self.root / "bundle"
        self.nodes = self.root / "nodes"
        self.bundle.mkdir()
        self.nodes.mkdir()
        (self.bundle / "example_node.py.example").write_text("sample")
        (self.bundle / "bundled-node.py").write_text("bundled")
        entrypoint = SCRIPT.with_name("entrypoint.sh").read_text()
        setup = entrypoint.split("# Copy Custom Nodes from cache to workdir if not exist", 1)[1]
        setup = setup.split("# Run user's pre-start script", 1)[0]
        self.command = "startup_stage() { :; };\n" + setup.replace(
            "/default-comfyui-bundle/ComfyUI/custom_nodes", str(self.bundle)
        ).replace("/root/ComfyUI/custom_nodes", str(self.nodes)).replace("cd /root", "cd /")

    def run_setup(self, **kwargs):
        return subprocess.run(["bash", "-ec", self.command], capture_output=True, text=True, **kwargs)

    def test_old_volume_does_not_restore_removed_nodes(self):
        (self.nodes / "example_node.py.example").write_text("existing")
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.nodes / ".image-nodes-initialized").is_file())
        self.assertFalse((self.nodes / "bundled-node.py").exists())

    def test_failed_copy_is_retried_even_if_sample_was_copied(self):
        tools = self.root / "bin"
        tools.mkdir()
        copier = tools / "cp"
        copier.write_text("#!/bin/sh\nexit 42\n")
        copier.chmod(0o755)
        result = self.run_setup(env={**os.environ, "PATH": f"{tools}:{os.environ['PATH']}"})
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.nodes / ".image-nodes-initialized").exists())
        self.assertTrue((self.nodes / ".image-nodes-copying").exists())
        (self.nodes / "example_node.py.example").write_text("partial copy")
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.nodes / "bundled-node.py").is_file())
        self.assertFalse((self.nodes / ".image-nodes-copying").exists())

    def test_copy_preserves_user_files_and_is_skipped_on_restart(self):
        existing = self.nodes / "bundled-node.py"
        existing.write_text("user modified")
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(existing.read_text(), "user modified")
        (self.bundle / "late-node.py").write_text("late")
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.nodes / "late-node.py").exists())


if __name__ == "__main__":
    unittest.main()
