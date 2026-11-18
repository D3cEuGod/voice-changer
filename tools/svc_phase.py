"""Preserve So-VITS-SVC's overtone phases in new and previously cached exports."""
from pathlib import Path
import tempfile

import numpy as np

# One reproducible uniform phase draw, in cycles. The fundamental starts at zero.
PHASE_CYCLES = (0., .6369617, .2697867, .0409735, .0165276, .8132702, .9127556, .6066358, .7294966)
PHASE_VERSION = "1"


def correct_cached_phase(path):
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(path)
    metadata = {item.key: item.value for item in model.metadata_props}
    if metadata.get("svc_harmonic_phase") == PHASE_VERSION:
        return
    sine = next(node for node in model.graph.node if node.name == "/dec/m_source/l_sin_gen/Sin")
    position = list(model.graph.node).index(sine)
    radians = (2 * np.pi * np.asarray(PHASE_CYCLES, dtype=np.float32))[None, None]
    model.graph.initializer.append(numpy_helper.from_array(radians, "overtone_phase_radians"))
    original = sine.input[0]
    sine.input[0] = "phased_excitation"
    model.graph.node.insert(position, helper.make_node(
        "Add", [original, "overtone_phase_radians"], ["phased_excitation"], name="RestoreOvertonePhase"))
    helper.set_model_props(model, dict(metadata, svc_harmonic_phase=PHASE_VERSION))
    onnx.checker.check_model(model)
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as tmp:
        part = Path(tmp.name)
    try:
        onnx.save(model, part)
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)
