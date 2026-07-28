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

import torch  # noqa: E402

import stereo_inpaint  # noqa: E402
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


# --- monobw ------------------------------------------------------------------
# The forward warp for the inpaint pipeline. stereo_inpaint.py matches stock iw3
# at difference 0, so this carries that across to the C++ and the CUDA kernels,
# which compile the same header.
#
# MonoBW has no weights, so there is no checkpoint to find and this always runs.
_monobw = stereo_inpaint.MonoBW().eval()


def _monobw_pair(image_hw, depth_hw, seed):
    generator = torch.Generator().manual_seed(seed)
    # Real content, not noise. The warp lands between pixels, so the error it
    # reports is the local gradient times a sub-pixel position difference --
    # and noise has the steepest gradient any image can have, at every pixel.
    # It would report a tolerance the plugin will never actually see. The same
    # reasoning is in tests/test_stereo_warp_onnx.py.
    coarse_image = torch.rand((1, 3, 12, 20), generator=generator)
    image = torch.nn.functional.interpolate(coarse_image, size=image_hw, mode="bilinear",
                                            align_corners=False).clamp(0, 1)
    coarse = torch.rand((1, 1, 8, 8), generator=generator)
    depth = torch.nn.functional.interpolate(coarse, size=depth_hw, mode="bilinear",
                                            align_corners=False)
    # Hard edges, because those are what fold the index map and open the holes.
    # A smooth depth field would exercise neither the cummax nor the mask.
    depth[:, :, depth_hw[0] // 4:depth_hw[0] // 2, depth_hw[1] // 4:depth_hw[1] // 2] = 0.95
    depth[:, :, -depth_hw[0] // 3:, -depth_hw[1] // 3:] = 0.05
    return image, depth.clamp(0, 1)


# the sampling grid, at the depth's own resolution
for index, (depth_hw, image_width, divergence, convergence, border) in enumerate([
        ((108, 192), 384, 2.0, 0.5, False),
        ((108, 192), 384, 5.0, 0.0, False),
        ((392, 938), 1920, 2.0, 0.5, False),
        ((392, 938), 1920, 10.0, 1.0, False),
        ((72, 128), 256, 5.0, 0.5, True),
        ((60, 100), 200, 2.0, -0.5, True),
        ((64, 64), 64, 0.0, 0.5, False),
]):
    _, depth = _monobw_pair((depth_hw[0] * 2, depth_hw[1] * 2), depth_hw, 200 + index)
    with torch.inference_mode():
        border_pix = 0
        if border:
            border_pix = round(divergence * 0.75 * 0.01 * image_width *
                               (depth_hw[1] / image_width))
        grid = _monobw.compute_backward_grid(depth, divergence, convergence,
                                             border_pix=border_pix)
    add(f"monobw_grid_{depth_hw[1]}x{depth_hw[0]}_{divergence}_{convergence}_{int(border)}",
        [depth_hw[1], depth_hw[0], image_width, int(border),
         int(round(divergence * 1000)), int(round(convergence * 1000))],
        depth, grid[:, 0])


# the whole of forward(): warped eye and hole mask, at the frame's resolution
for index, (image_hw, depth_hw, divergence, convergence, border, fix_mask) in enumerate([
        ((216, 384), (108, 192), 2.0, 0.5, False, 1),
        ((216, 384), (108, 192), 5.0, 0.5, False, 0),
        ((216, 384), (108, 192), 5.0, 0.5, False, 2),
        ((216, 384), (108, 192), 8.0, 1.0, True, 1),
        ((160, 288), (80, 144), 3.0, 0.25, False, 1),
        # depth larger than the frame, and an odd size on both
        ((107, 193), (216, 384), 4.0, 0.5, False, 1),
]):
    image, depth = _monobw_pair(image_hw, depth_hw, 300 + index)
    with torch.inference_mode():
        eye, mask = _monobw(image, depth, divergence=divergence, convergence=convergence,
                            preserve_screen_border=border,
                            fix_screen_border_mask=fix_mask, return_mask=True)
    add(f"monobw_forward_{image_hw[1]}x{image_hw[0]}_{depth_hw[1]}x{depth_hw[0]}"
        f"_{divergence}_{convergence}_{int(border)}_{fix_mask}",
        [image_hw[1], image_hw[0], depth_hw[1], depth_hw[0], int(border), fix_mask,
         int(round(divergence * 1000)), int(round(convergence * 1000))],
        np.concatenate([image.numpy().ravel(), depth.numpy().ravel()]),
        np.concatenate([eye.numpy().ravel(), mask.float().numpy().ravel()]))


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
