"""Generate reference data for the C++ numeric core's test.

The C++ in ofx/plugin/stereo_pipeline.cpp is a port of Python that is itself
validated against stock iw3 at max absolute difference 0. This dumps that
Python's answers so tests/cpp/test_pipeline.cpp can check the port against them
rather than against re-derived expectations.

    F:\\_AI_PROJECTS_\\nunif\\venv\\Scripts\\python.exe tools/dump_pipeline_reference.py

Writes tests/cpp/pipeline_reference.bin. The format is deliberately trivial so
the C++ reader is a dozen lines:

    "IW3P" | case count | then per case:
    name length | name | int count | ints | input count | floats |
    output count | floats
"""

import os
import struct
import sys
import types

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUNIF_ROOT = os.environ.get("NUNIF_ROOT", r"F:\_AI_PROJECTS_\nunif")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, NUNIF_ROOT)

import stereo_warp  # noqa: E402
import stereo_warp_onnx  # noqa: E402

OUTPUT = os.path.join(REPO_ROOT, "tests", "cpp", "pipeline_reference.bin")

cases = []


def add(name, ints, inputs, outputs):
    cases.append((name,
                  np.asarray(ints, dtype=np.int32),
                  np.asarray(inputs, dtype=np.float32).ravel(),
                  np.asarray(outputs, dtype=np.float32).ravel()))


# --- the antialiased depth resize -------------------------------------------
# The one ONNX Resize could not reproduce. Python is the reference here.
session = stereo_warp_onnx.StereoWarpSession.__new__(stereo_warp_onnx.StereoWarpSession)
session._weight_cache = {}

for index, (in_w, in_h, out_w, out_h) in enumerate([
        (1920, 800, 938, 392),
        (1920, 1080, 940, 528),
        (384, 216, 128, 72),
        (518, 518, 392, 392),
        (193, 107, 100, 60),
        (100, 60, 193, 107),   # upscale, which takes the scale < 1 branch
]):
    rng = np.random.default_rng(index)
    source = rng.random((1, 1, in_h, in_w), dtype=np.float32)
    resized = session._resize_depth(source, (1, 3, in_h, in_w), None) if False else None
    # call the axis machinery directly: _resize_depth derives its target from a
    # stereo width, and these cases are about the resampling itself
    array = source.astype(np.float64)
    array = session._apply_axis(array, session._axis_weights(in_w, out_w), 3)
    array = session._apply_axis(array, session._axis_weights(in_h, out_h), 2)
    resized = np.clip(array, 0, 1).astype(np.float32)
    add(f"resize_{in_w}x{in_h}_to_{out_w}x{out_h}",
        [in_w, in_h, out_w, out_h], source, resized)


# --- the mappers -------------------------------------------------------------
# Exposed in the plugin as Foreground Scale, so the reference goes through
# iw3's own resolve_mapper_name() rather than through a name chosen here.
from iw3.mapper import resolve_mapper_name  # noqa: E402

ramp = np.linspace(0.0, 1.0, 257, dtype=np.float32)
for type_index, mapper_type in enumerate(["mul", "shift"]):
    for scale in [-3.0, -2.5, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 2.25, 3.0]:
        name = resolve_mapper_name(None, scale, metric_depth=False, mapper_type=mapper_type)
        import torch
        mapped = stereo_warp.get_mapper(name)(torch.from_numpy(ramp)).numpy()
        add(f"mapper_{mapper_type}_{scale}",
            [type_index, int(round(scale * 1000))], ramp, mapped)


# --- the model's input tensor ------------------------------------------------
for index, (width, height, divergence, convergence, border) in enumerate([
        (192, 108, 2.0, 0.5, 0),
        (192, 108, 2.0, 0.5, 1),
        (938, 392, 5.0, 1.0, 1),
        (128, 72, 0.0, 0.0, 1),
        (64, 64, 10.0, -0.5, 1),
]):
    rng = np.random.default_rng(100 + index)
    depth = rng.random((1, 1, height, width), dtype=np.float32)
    base_size = max(height, width)
    tensor = stereo_warp_onnx.StereoWarpSession._make_input_tensor(
        depth, divergence, convergence, base_size, bool(border))
    add(f"input_tensor_{width}x{height}_{divergence}_{convergence}_{border}",
        [width, height, border], depth, tensor)


# --- autoDepthSize -----------------------------------------------------------
# Cross-checked against iw3_ext's own rule, not against a copy of it.
from iw3_ext.depth_file import model_depth_size  # noqa: E402

args = types.SimpleNamespace(resolution=None, limit_resolution=False)
sizes = []
frames = [(1920, 800), (1920, 1080), (3840, 2160), (1280, 720), (640, 360), (392, 392)]
for width, height in frames:
    model_h, model_w = model_depth_size(height, width, args)
    sizes.extend([width, height, model_w, model_h])
add("auto_depth_size", sizes, [], [])


# --- Dubois ------------------------------------------------------------------
rng = np.random.default_rng(7)
pixels = 64
left = rng.random((3, pixels), dtype=np.float32)
right = rng.random((3, pixels), dtype=np.float32)
matrix_left = np.array([[0.4561, 0.500484, 0.176381],
                        [-0.0400822, -0.0378246, -0.0157589],
                        [-0.0152161, -0.0205971, -0.00546856]])
matrix_right = np.array([[-0.0434706, -0.0879388, -0.00155529],
                         [0.378476, 0.73364, -0.0184503],
                         [-0.0721527, -0.112961, 1.2264]])
out = np.clip(matrix_left @ left + matrix_right @ right, 0, 1).astype(np.float32)
add("dubois", [pixels], np.concatenate([left.ravel(), right.ravel()]), out)


# --- write -------------------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "wb") as handle:
    handle.write(b"IW3P")
    handle.write(struct.pack("<i", len(cases)))
    for name, ints, inputs, outputs in cases:
        encoded = name.encode("utf-8")
        handle.write(struct.pack("<i", len(encoded)))
        handle.write(encoded)
        handle.write(struct.pack("<i", ints.size))
        handle.write(ints.tobytes())
        handle.write(struct.pack("<i", inputs.size))
        handle.write(inputs.tobytes())
        handle.write(struct.pack("<i", outputs.size))
        handle.write(outputs.tobytes())

print(f"wrote {OUTPUT}")
print(f"  {len(cases)} cases, {os.path.getsize(OUTPUT) / 1024:.0f} KiB")
for name, *_ in cases:
    print(f"    {name}")
