"""Installed character registration, API validation and category visibility."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from backend import local_rvc,engine
from backend.app import app
from fastapi.testclient import TestClient


class RegistryTests(unittest.TestCase):
    def test_only_valid_installed_ids_are_exposed(self):
        with tempfile.TemporaryDirectory() as directory:
            registry=Path(directory)/'profiles.json'
            registry.write_text(json.dumps({'version':1,'voices':{
                'tannhauser':{'display_name':'诗歌剧','label':'weights'},
                '../escape':{'label':'bad'},'Bad-ID':{'label':'bad'}}}),encoding='utf-8')
            with patch.object(local_rvc,'REGISTRY',registry):
                self.assertEqual(local_rvc.installed_voices(),{'tannhauser':'诗歌剧'})
                registry.unlink()
                self.assertEqual(local_rvc.installed_voices(),{})

    def test_voice_api_includes_installed_horse(self):
        if 'tannhauser' not in engine.VOICES:
            self.skipTest('The user-provided voice is not installed on this machine')
        client=TestClient(app)
        response=client.get('/api/voices')
        self.assertIn({'id':'tannhauser','name':'诗歌剧'},response.json())
        choices=client.get('/openapi.json').json()['components']['schemas']['VoiceID']['enum']
        self.assertIn('tannhauser',choices)
        self.assertEqual(client.post('/api/convert?voice_id=not_installed',content=b'bad').status_code,422)


if __name__=='__main__':
    unittest.main(verbosity=2)
