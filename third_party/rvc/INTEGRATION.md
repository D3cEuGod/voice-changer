# RVC

Upstream: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI

Export source revision: `d60cd3d009b7e438f92f8624fa89b710d10c6a39`.

The upstream MIT license is included here. `setup_models.py` checks out that
revision into the ignored build directory. `tools/export_voices.py` uses its
pitch-conditioned RVC v2 ONNX network, exporting the selected speaker into a
one-second, 100-frame graph. `tools/import_rvc.py` accepts supplied tensor-only
checkpoints, validates matching 768-dimensional retrieval indexes, builds the
graph, verifies a one-second output, and records SHA-256 checksums.

The application's streaming adapter uses ContentVec layer 12, RMVPE, eight
inverse-squared-distance retrieval neighbors, and RVC-style unvoiced feature
protection. At full strength it does not mix the caller's waveform back into
an imported character model and does not apply another voice converter after it.
It retains the existing GTCRN/VAD front end and SOLA packet join.

Default imported-model settings are retrieval 0.75 / protection 0.33 for the
character preset, and retrieval at most 0.5 / unvoiced blend at most 0.15 for the
call preset (a smaller blend retains more original linguistic features).
Pitch can follow a configured target register or a fixed semitone shift. The
original weights and indexes remain separate files under `models/local-rvc/`,
with their own source and model terms; they are not added to Git history.
