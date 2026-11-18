"""Optimization-cache fallback checks using tiny fixtures, never real model files."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from backend import graph_cache
from tools import model_integrity


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.model=self.root/'source.onnx'
        self.model.write_bytes(b'source')
        self.calls=[]
        self.reject_cached=False
        self.reject_serialization=False
        self.patches=[patch.object(graph_cache,'DIRECTORY',self.root/'graphs'),
                      patch.object(model_integrity,'CACHE',self.root/'verified.json'),
                      patch.object(graph_cache,'hardware',return_value=['test-hardware']),
                      patch.object(graph_cache.ort,'InferenceSession',side_effect=self.session)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def session(self,path,options,providers):
        self.calls.append(Path(path))
        if self.reject_cached and Path(path).parent==graph_cache.DIRECTORY:
            raise RuntimeError('cached graph unsupported')
        if options.optimized_model_filepath:
            Path(options.optimized_model_filepath).write_bytes(b'optimized')
            if self.reject_serialization:
                raise RuntimeError('cannot serialize provider graph')
        return SimpleNamespace(source=Path(path))

    def load(self):
        options=SimpleNamespace(optimized_model_filepath='',graph_optimization_level=99)
        return graph_cache.load(self.model,options,['CPUExecutionProvider'])

    def test_unchanged_graph_is_loaded_from_cache(self):
        self.load()
        result=self.load()
        self.assertEqual(result.source.parent,graph_cache.DIRECTORY)
        self.assertNotEqual(result.source,self.model)

    def test_corrupted_generated_graph_is_rebuilt(self):
        self.load()
        target=next(graph_cache.DIRECTORY.glob('*.onnx'))
        target.write_bytes(b'corrupted-cache')
        self.assertEqual(self.load().source,self.model)

    def test_unsupported_cached_graph_falls_back_to_source(self):
        self.load()
        self.reject_cached=True
        self.assertEqual(self.load().source,self.model)

    def test_failed_serialization_does_not_save_partial_graph(self):
        self.reject_serialization=True
        self.assertEqual(self.load().source,self.model)
        self.assertEqual(list(graph_cache.DIRECTORY.glob('*.onnx')),[])

    def test_cache_can_be_disabled(self):
        with patch.dict('os.environ',{'VOICE_ORT_CACHE':'0'}):
            self.assertEqual(self.load().source,self.model)
        self.assertFalse(graph_cache.DIRECTORY.exists())

    def test_rvc_exports_keep_original_graph(self):
        self.model=self.root/'local-rvc/voice/voice.onnx'
        self.model.parent.mkdir(parents=True)
        self.model.write_bytes(b'voice')
        self.assertEqual(self.load().source,self.model)
        self.assertFalse(graph_cache.DIRECTORY.exists())

    def test_unwritable_manifest_does_not_break_inference_session(self):
        with patch.object(Path,'write_text',side_effect=PermissionError('read-only cache')):
            self.assertEqual(self.load().source,self.model)


if __name__=='__main__':
    unittest.main(verbosity=2)
