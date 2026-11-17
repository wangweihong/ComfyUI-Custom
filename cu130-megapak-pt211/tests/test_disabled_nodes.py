"""Exercise the real startup block against new and existing user volumes."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DisabledNodeConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.user = self.root / "user-scripts"
        self.config = self.user / "disabled-nodes.yaml"
        self.bundle = ROOT / "runner-scripts/disabled-nodes.yaml"
        entrypoint = (ROOT / "runner-scripts/entrypoint.sh").read_text()
        block = entrypoint.split("# Configure disabled nodes independently", 1)[1]
        block = block.split('startup_stage "Disabled nodes configuration"', 1)[0]
        self.block = ("# Configure disabled nodes independently" + block).replace(
            "/runner-scripts/disabled-nodes.yaml", str(self.bundle)
        ).replace("/root/user-scripts", str(self.user))

    def run_setup(self, args=None, twice=False):
        env = os.environ.copy()
        env.pop("CLI_ARGS", None)
        if args is not None:
            env["CLI_ARGS"] = args
        command = self.block + (self.block if twice else "") + '\nprintf "ARGS=%s\\n" "$CLI_ARGS"\n'
        result = subprocess.run(["bash", "-euc", command], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.split("ARGS=", 1)[1].strip()

    def test_new_volume_seeds_exactly_three_nodes(self):
        args = self.run_setup()
        self.assertEqual(self.config.read_bytes(), self.bundle.read_bytes())
        node_ids = [line.strip()[2:] for line in self.config.read_text().splitlines()
                    if line.strip().startswith("- ")]
        self.assertEqual(node_ids, ["PixaromaLoadImage", "PixaromaLoadImageMini", "PixaromaPromptReader"])
        self.assertEqual(args, f"--disabled-nodes-config {self.config}")

    def test_existing_config_settings_and_user_script_are_unchanged(self):
        self.user.mkdir()
        self.config.write_text("disabled_nodes: []\n")
        script = self.user / "pre-start.sh"
        script.write_text("# user custom startup\n")
        settings = self.root / "comfy.settings.json"
        settings.write_text('{"custom":true}\n')
        originals = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in [self.config, script, settings]}
        self.run_setup()
        for path, before in originals.items():
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)

    def test_restart_preserves_arguments_and_does_not_duplicate_flag(self):
        args = self.run_setup("--preview-method auto", twice=True)
        self.assertEqual(args, f"--preview-method auto --disabled-nodes-config {self.config}")

    def test_explicit_custom_config_takes_precedence_in_both_forms(self):
        for flag in ["--disabled-nodes-config /custom/list.yaml",
                     "--disabled-nodes-config=/custom/list.yaml"]:
            with self.subTest(flag=flag):
                args = "--fast " + flag
                self.assertEqual(self.run_setup(args, twice=True), args)

    def test_existing_local_pre_start_flag_is_not_duplicated(self):
        args = f"--disabled-nodes-config {self.config}"
        self.assertEqual(self.run_setup(args), args)

    def test_existing_symlink_is_preserved(self):
        self.user.mkdir()
        target = self.root / "custom.yaml"
        target.write_text("disabled_nodes: []\n")
        self.config.symlink_to(target)
        self.run_setup()
        self.assertTrue(self.config.is_symlink())
        self.assertEqual(target.read_text(), "disabled_nodes: []\n")

    def test_dangling_symlink_does_not_write_to_user_target(self):
        self.user.mkdir()
        target = self.root / "not-created.yaml"
        self.config.symlink_to(target)
        self.run_setup()
        self.assertTrue(self.config.is_symlink())
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
