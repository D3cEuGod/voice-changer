"""Machine-specific, locally generated ORT optimization cache."""
from functools import lru_cache
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import tempfile
import time

import onnxruntime as ort
from tools.model_integrity import signature, verified

DIRECTORY = Path(__file__).resolve().parents[1]/'.run/ort-cache'
LOG = logging.getLogger('uvicorn.error')


@lru_cache(maxsize=1)
def hardware():
    adapters = []
    if os.name == 'nt':
        import winreg
        key = r'SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}'
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as root:
                for i in range(winreg.QueryInfoKey(root)[0]):
                    name = winreg.EnumKey(root,i)
                    if not name.isdigit():
                        continue
                    with winreg.OpenKey(root,name) as adapter:
                        values = []
                        for prop in ('DriverDesc','DriverVersion','MatchingDeviceId'):
                            try:
                                values.append(winreg.QueryValueEx(adapter,prop)[0])
                            except OSError:
                                values.append('')
                        adapters.append(values)
        except OSError:
            pass
    return [platform.machine(),platform.processor(),platform.version(),adapters]


def load(path, options, providers):
    path = Path(path).resolve()
    start = time.perf_counter()
    if os.environ.get('VOICE_ORT_CACHE') == '0' or 'local-rvc' in path.parts:
        # These RVC exports contain DirectML fusions that cannot be reloaded
        # from an optimized ONNX file. Keep their verified original graphs.
        session = ort.InferenceSession(str(path),options,providers=providers)
        LOG.info('Loaded %s in %.2f s',path.parent.name+'/'+path.name,time.perf_counter()-start)
        return session
    identity = [1,str(path),signature(path),ort.__version__,providers,hardware()]
    key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
    cached = DIRECTORY/f'{path.stem}-{key}.onnx'
    manifest = cached.with_suffix('.json')
    try:
        data = json.loads(manifest.read_text())
        if data['identity'] == identity and verified(cached,data['sha256']):
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            session = ort.InferenceSession(str(cached),options,providers=providers)
            LOG.info('Loaded %s from local cache in %.2f s',path.name,time.perf_counter()-start)
            return session
    except (OSError,ValueError,KeyError,TypeError,RuntimeError):
        pass
    except Exception as exc:
        LOG.warning('Rebuilding unreadable optimization cache for %s: %s',path.name,exc)
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    part = None
    try:
        DIRECTORY.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=DIRECTORY,suffix='.onnx',delete=False) as stream:
            part = Path(stream.name)
        options.optimized_model_filepath = str(part)
    except OSError:
        options.optimized_model_filepath = ''
    try:
        try:
            session = ort.InferenceSession(str(path),options,providers=providers)
        except Exception:
            if not options.optimized_model_filepath:
                raise
            options.optimized_model_filepath = ''
            session = ort.InferenceSession(str(path),options,providers=providers)
        if options.optimized_model_filepath and part is not None and part.is_file() and part.stat().st_size:
            try:
                with part.open('rb') as stream:
                    digest = hashlib.file_digest(stream,'sha256').hexdigest()
                part.replace(cached)
                manifest.write_text(json.dumps({'identity':identity,'sha256':digest}),encoding='utf-8')
                verified(cached,digest)
            except OSError:
                LOG.warning('Could not save optimization cache for %s',path.name)
        LOG.info('Prepared %s in %.2f s',path.name,time.perf_counter()-start)
        return session
    finally:
        if part is not None:
            try:
                part.unlink(missing_ok=True)
            except OSError:
                pass
