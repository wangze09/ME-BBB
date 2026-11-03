"""Code-only smoke tests: no raw datasets, GPU or model download needed."""
import ast
import pathlib
import re
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
SCRIPTS=ROOT/'scripts'

class TestPublicCode(unittest.TestCase):
    def test_scripts_compile(self):
        files=list(SCRIPTS.rglob('*.py'))
        self.assertEqual(len(files),13)
        for path in files:
            ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
    def test_core_files(self):
        for name in ('README.md','README_zh-CN.md','LICENSE','THIRD_PARTY_NOTICES.md','configs/model_versions.json'):
            self.assertTrue((ROOT/name).is_file(),name)
    def test_no_private_workstation_paths(self):
        for path in SCRIPTS.rglob('*.py'):
            body=path.read_text(encoding='utf-8-sig')
            self.assertNotRegex(body,r'[A-Za-z]:\\|/data/zwang/|经纬BBB|BBB课题')
    def test_data_not_bundled(self):
        self.assertFalse(list((ROOT/'data').rglob('*.csv')))
        self.assertFalse(list((ROOT/'data').rglob('*.npz')))
    def test_models_not_bundled(self):
        self.assertFalse(list(ROOT.rglob('*.safetensors')))
        self.assertFalse(list(ROOT.rglob('*.joblib')))

if __name__=='__main__':
    unittest.main()
