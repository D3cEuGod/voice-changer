"""Cache successful hashes of unchanged local model files across launches."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.run/model-verification-v1.json'
_lock = threading.Lock()


def signature(path):
    st = path.stat()
    return [st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_dev, st.st_ino]


def verified(path, expected, *, force=False):
    """Rehash new/replaced/modified files; cache only a successful SHA-256 check.

    This is an integrity cache for locally installed files, not a trust boundary
    against a user who can alter both the models and the cache. Force a full
    check with VOICE_VERIFY_MODELS=1.
    """
    path = Path(path).resolve()
    if not path.is_file():
        return False
    with _lock:
        before = signature(path)
        try:
            data = json.loads(CACHE.read_text(encoding='utf-8'))
            entries = data['files'] if isinstance(data, dict) and data.get('version') == 1 else {}
            if not isinstance(entries, dict):
                entries = {}
        except (OSError, ValueError, KeyError, TypeError):
            entries = {}
        key = str(path)
        if not force and os.environ.get('VOICE_VERIFY_MODELS') != '1':
            if entries.get(key) == {'signature': before, 'sha256': expected}:
                return True
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if signature(path) != before:
            raise RuntimeError(f'Model changed while being verified: {path.name}')
        if actual != expected:
            return False
        entries[key] = {'signature': before, 'sha256': actual}
        part = None
        try:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=CACHE.parent, suffix='.json.tmp', mode='w',
                                             encoding='utf-8', delete=False) as stream:
                part = Path(stream.name)
                json.dump({'version': 1, 'files': entries}, stream)
            part.replace(CACHE)
        except OSError:
            # A read-only cache must not prevent an otherwise verified model loading.
            pass
        finally:
            if part is not None:
                try:
                    part.unlink(missing_ok=True)
                except OSError:
                    pass
        return True
