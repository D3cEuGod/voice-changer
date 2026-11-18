"""Validate cache hits, changed files and forced checks without touching real models."""
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools import model_integrity as integrity


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.file = self.root/'model.onnx'
        self.file.write_bytes(b'valid-model')
        self.sha = hashlib.sha256(self.file.read_bytes()).hexdigest()
        self.cache = patch.object(integrity, 'CACHE', self.root/'cache.json')
        self.cache.start()

    def tearDown(self):
        self.cache.stop()
        self.temp.cleanup()

    def test_unchanged_file_does_not_get_rehashed(self):
        self.assertTrue(integrity.verified(self.file, self.sha))
        with patch.object(hashlib, 'file_digest', side_effect=AssertionError('unexpected rehash')):
            self.assertTrue(integrity.verified(self.file, self.sha))

    def test_same_size_modification_is_rejected(self):
        self.assertTrue(integrity.verified(self.file, self.sha))
        stamp = self.file.stat().st_mtime_ns
        self.file.write_bytes(b'wrong-model')
        os.utime(self.file, ns=(stamp, stamp+10_000_000))
        self.assertFalse(integrity.verified(self.file, self.sha))

    def test_replacement_preserving_size_and_mtime_is_rejected(self):
        self.assertTrue(integrity.verified(self.file, self.sha))
        stamp = self.file.stat().st_mtime_ns
        replacement = self.root/'replacement'
        replacement.write_bytes(b'wrong-model')
        os.utime(replacement, ns=(stamp, stamp))
        replacement.replace(self.file)
        self.assertFalse(integrity.verified(self.file, self.sha))

    def test_invalid_cache_gets_rebuilt(self):
        for data in ('[]','{broken', '{"version":1,"files":[] }'):
            integrity.CACHE.write_text(data)
            self.assertTrue(integrity.verified(self.file, self.sha))

    def test_forced_check_rehashes(self):
        self.assertTrue(integrity.verified(self.file, self.sha))
        with patch.object(hashlib, 'file_digest', wraps=hashlib.file_digest) as digest:
            self.assertTrue(integrity.verified(self.file, self.sha, force=True))
            digest.assert_called_once()

    def test_wrong_expected_hash_is_not_accepted_from_cache(self):
        self.assertTrue(integrity.verified(self.file, self.sha))
        self.assertFalse(integrity.verified(self.file, '0'*64))


if __name__=='__main__':
    unittest.main(verbosity=2)
