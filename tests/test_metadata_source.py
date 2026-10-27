"""Execute the actual pinned-source off branch with package import forbidden."""
import ast
import builtins
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

class MetadataOffSourceTests(unittest.TestCase):
    def test_disabled_constructor_has_no_graphbudget_import(self):
        root=os.environ.get('GRAPHBUDGET_SGLANG_SOURCE')
        if not root:self.skipTest('requires pinned patched source')
        path=Path(root)/'python/sglang/srt/layers/attention/triton_backend.py'
        module=ast.parse(path.read_text())
        cls=next(x for x in module.body if isinstance(x,ast.ClassDef) and x.name=='TritonAttnBackend')
        init=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='__init__')
        assignment=next(x for x in init.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='graphbudget_metadata_enabled' for t in x.targets))
        guarded=next(x for x in init.body if isinstance(x,ast.If) and isinstance(x.test,ast.Attribute) and x.test.attr=='graphbudget_metadata_enabled')
        imports=[x for x in ast.walk(init) if isinstance(x,ast.ImportFrom) and x.module=='graphbudget.metadata']
        self.assertEqual(len(imports),1)
        self.assertIn(imports[0],list(ast.walk(guarded)))
        def reject(name,*args,**kwargs):
            if name.startswith('graphbudget'):raise AssertionError('off path imported graphbudget')
            return builtins.__import__(name,*args,**kwargs)
        instance=SimpleNamespace()
        code=compile(ast.fix_missing_locations(ast.Module(body=[assignment,guarded],type_ignores=[])),str(path),'exec')
        exec(code,{'self':instance,'get_bool_env_var':lambda name,default:False,
                   '__builtins__':{**vars(builtins),'__import__':reject}})
        self.assertFalse(instance.graphbudget_metadata_enabled)
        replay=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='init_forward_metadata_replay_cuda_graph')
        imports=[x for x in ast.walk(replay) if isinstance(x,ast.ImportFrom) and x.module=='graphbudget.metadata']
        self.assertEqual(len(imports),1)
        guards=[x for x in ast.walk(replay) if isinstance(x,ast.If) and isinstance(x.test,ast.Attribute) and x.test.attr=='graphbudget_metadata_enabled']
        self.assertTrue(any(imports[0] in list(ast.walk(x)) for x in guards))
