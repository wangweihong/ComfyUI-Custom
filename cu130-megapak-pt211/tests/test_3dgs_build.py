import os
from pathlib import Path
import subprocess
import tempfile
import unittest

REPO=Path(__file__).resolve().parents[2]

class BuildInterfaces(unittest.TestCase):
    def make(self,value):
        return subprocess.run(['make','-n','build-cu130-megapak-pt211','INSTALL_3DGS='+value],cwd=REPO,text=True,capture_output=True)

    def test_enabled_and_default_tags_and_build_arg(self):
        for flag,suffix in [('false',''),('true','-3dgs')]:
            result=self.make(flag)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('COMPONENT_TAG="cu130-megapak-pt211'+suffix+'"',result.stdout)
            self.assertIn('--build-arg INSTALL_3DGS='+flag,result.stdout)

    def test_invalid_flags_fail_before_build(self):
        for value in ['','TRUE','yes','true false','%']:
            result=self.make(value)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('INSTALL_3DGS must be true or false',result.stderr)

    def test_default_is_false(self):
        result=subprocess.run(['make','-n','build-cu130-megapak-pt211'],cwd=REPO,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--build-arg INSTALL_3DGS=false',result.stdout)
        self.assertNotIn('COMPONENT_TAG="cu130-megapak-pt211-3dgs"',result.stdout)

    def test_status_uses_selected_variant(self):
        result=subprocess.run(['make','-n','status','INSTALL_3DGS=true'],cwd=REPO,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('cu130-megapak-pt211:cu130-megapak-pt211-3dgs',result.stdout)

    def test_clean_does_not_remove_other_variant(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);log=root/'removed'
            docker=root/'docker'
            docker.write_text('''#!/bin/sh
if [ "$1" = images ]; then
  echo example.invalid/test:cu130-megapak-pt211-20000101
  echo example.invalid/test:cu130-megapak-pt211-3dgs-20000101
elif [ "$1" = rmi ]; then
  echo "$2" >> "$REMOVED_LOG"
fi
''')
            docker.chmod(0o755)
            for flag,suffix in [('false',''),('true','-3dgs')]:
                log.unlink(missing_ok=True)
                result=subprocess.run(['make','clean-cu130-megapak-pt211','INSTALL_3DGS='+flag,
                    'REGISTRY=example.invalid','IMAGE_NAME=test'],cwd=REPO,capture_output=True,text=True,
                    env=dict(os.environ,PATH=str(root)+os.pathsep+os.environ['PATH'],REMOVED_LOG=str(log)))
                self.assertEqual(result.returncode,0,result.stderr)
                removed=log.read_text().splitlines()
                self.assertIn('example.invalid/test:cu130-megapak-pt211'+suffix+'-20000101',removed)
                self.assertEqual(len(removed),3,removed)


if __name__=="__main__":unittest.main()
