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


# the mask morphology, on masks a real warp produced rather than drawn by hand
for index, (image_hw, depth_hw, divergence, inner, outer) in enumerate([
        ((216, 384), (108, 192), 5.0, 0, 0),
        ((216, 384), (108, 192), 5.0, 2, 0),
        ((216, 384), (108, 192), 5.0, 0, 3),
        ((216, 384), (108, 192), 5.0, 2, 3),
        ((216, 384), (108, 192), 10.0, 4, 4),
        # base_width scaling: the frame is twice the depth's width, so a count
        # of 1 becomes 2 and one of 3 becomes 6.
        ((160, 288), (80, 144), 3.0, 1, 3),
]):
    image, depth = _monobw_pair(image_hw, depth_hw, 400 + index)
    with torch.inference_mode():
        _, raw = _monobw(image, depth, divergence=divergence, convergence=0.5,
                         preserve_screen_border=False, fix_screen_border_mask=1,
                         return_mask=True)
        processed = stereo_inpaint.MonoBWInpaintImage.preprocess_mask(
            raw, target_size=raw.shape[-2:],
            inner_dilation=inner, outer_dilation=outer, base_width=depth_hw[1])
    add(f"mask_preprocess_{image_hw[1]}x{image_hw[0]}_{depth_hw[1]}_{inner}_{outer}",
        [image_hw[1], image_hw[0], inner, outer, depth_hw[1]],
        raw.float(), processed.float())


# exactly what the inpaint network is handed, for each eye
#
# This is the flip ordering, which is the part of the plugin's monobw path most
# likely to be wrong and least likely to be caught by anything else. iw3 warps
# the right eye in mirrored coordinates and inpaints it in frame coordinates,
# and does the opposite for the left, so between them there are four flips whose
# order matters. Dumping the network's actual inputs pins all of it down without
# needing the network.
for index, (image_hw, depth_hw, divergence, convergence, inner, outer) in enumerate([
        ((216, 384), (108, 192), 2.0, 0.5, 0, 0),
        ((216, 384), (108, 192), 5.0, 0.5, 2, 3),
        ((160, 288), (80, 144), 8.0, 1.0, 1, 1),
]):
    image, depth = _monobw_pair(image_hw, depth_hw, 500 + index)
    with torch.inference_mode():
        results = stereo_inpaint._apply_divergence_monobw(
            _monobw, image, depth, divergence=divergence, convergence=convergence,
            synthetic_view="both", preserve_screen_border=False, fix_screen_border_mask=1)
        left_eye, right_eye, left_mask, right_mask = results
        for side, eye, mask in (("left", left_eye, left_mask), ("right", right_eye, right_mask)):
            # _inpaint_single flips the left eye and not the right, then runs
            # preprocess_mask on whatever orientation that left it in.
            if side == "left":
                eye, mask = eye.flip(-1), mask.flip(-1)
            mask = stereo_inpaint.MonoBWInpaintImage.preprocess_mask(
                mask, target_size=eye.shape[-2:],
                inner_dilation=inner, outer_dilation=outer, base_width=depth_hw[1])
            add(f"monobw_eye_{side}_{image_hw[1]}x{image_hw[0]}_{depth_hw[1]}"
                f"_{divergence}_{convergence}_{inner}_{outer}",
                [image_hw[1], image_hw[0], depth_hw[1], depth_hw[0],
                 1 if side == "right" else 0, inner, outer,
                 int(round(divergence * 1000)), int(round(convergence * 1000))],
                np.concatenate([image.numpy().ravel(), depth.numpy().ravel()]),
                np.concatenate([eye.numpy().ravel(), mask.float().numpy().ravel()]))


# --- mlbw_l2_inpaint ---------------------------------------------------------
# The half of mask_mlbw_l2 that is not the network: the two backward warps, the
# softmax blend, and the mask post-process. The network itself is ONNX, so these
# cases take its three outputs as *inputs* -- which is also exactly what the
# plugin's kernels receive from ORT.
#
# Needs the checkpoint, to get deltas and weights with a real distribution.
# Drawing them by hand would put the taps somewhere a real one never goes.
MASK_MLBW_CHECKPOINT = os.path.join(
    NUNIF_ROOT, "iw3", "pretrained_models", "hub", "checkpoints",
    "iw3_mask_mlbw_l2_d1_20250903.pth")

if os.path.exists(MASK_MLBW_CHECKPOINT):
    _mlbw = stereo_warp.load_mask_mlbw_l2(MASK_MLBW_CHECKPOINT, device="cpu")

    def _mlbw_heads(depth, divergence, convergence, border=False):
        x = stereo_warp._make_input_tensor(
            depth, divergence=divergence, convergence=convergence,
            image_width=max(depth.shape[-2:]), preserve_screen_border=border)
        with torch.inference_mode():
            return _mlbw(x)

    # the weight resize, at the ratios this pipeline actually produces
    #
    # Same buildResampleAxis the depth resize uses, so this is not a new filter
    # -- but it is a new *caller*, and Inpaint Max Width makes downscaling the
    # normal case here where the depth path only ever upscales.
    # The source is a real layer weight, not noise. That is not a detail: a
    # softmax weight map is smooth almost everywhere with sharp ridges where the
    # layers swap over, and noise -- which has the steepest gradient a signal can
    # have at every sample -- reports a tolerance three times what this resampler
    # actually sees. Measured: 6.3e-5 on noise against 2.5e-6 on a real weight
    # map, from identical code. The same trap is recorded in _monobw_pair.
    for index, (in_w, in_h, out_w, out_h) in enumerate([
            (384, 216, 256, 144),      # Inpaint Max Width's ratio, downscaling
            (384, 216, 144, 81),       # a wider ratio still
            (192, 108, 384, 216),      # depth smaller than the frame: upscale
            (192, 108, 192, 108),      # equal, where the resize must be identity
    ]):
        _, depth = _monobw_pair((in_h * 2, in_w * 2), (in_h, in_w), 600 + index)
        _, layer_weight, _ = _mlbw_heads(depth, 5.0, 0.5)
        source = layer_weight[:, 0:1].contiguous()
        with torch.inference_mode():
            resized = torch.nn.functional.interpolate(
                source, size=(out_h, out_w),
                mode="bilinear", align_corners=True, antialias=True)
        add(f"mlbw_weight_resize_{in_w}x{in_h}_to_{out_w}x{out_h}",
            [in_w, in_h, out_w, out_h], source.numpy(), resized.numpy())

    # the two warps and the blend, at the frame's resolution
    for index, (image_hw, depth_hw, divergence, convergence) in enumerate([
            ((216, 384), (216, 384), 2.0, 0.5),     # equal, the Full case
            ((216, 384), (108, 192), 5.0, 0.5),     # depth smaller: weights upscale
            ((108, 192), (216, 384), 5.0, 0.5),     # depth larger: weights downscale
            ((160, 288), (80, 144), 8.0, 1.0),
            ((107, 193), (216, 384), 4.0, -0.5),    # odd on both sides
    ]):
        image, depth = _monobw_pair(image_hw, depth_hw, 700 + index)
        delta, layer_weight, _ = _mlbw_heads(depth, divergence, convergence)
        with torch.inference_mode():
            weights = layer_weight
            if image.shape[-2:] != layer_weight.shape[-2:]:
                weights = torch.nn.functional.interpolate(
                    layer_weight, size=image.shape[-2:], mode="bilinear",
                    align_corners=True, antialias=True)
            scale = torch.tensor(1.0 / (depth_hw[1] // 2 - 1))
            grid = stereo_warp._make_grid(1, depth_hw[1], depth_hw[0], image.device)
            padded = stereo_warp._pad_delta_y(delta)
            eye = torch.zeros_like(image)
            for layer in range(2):
                eye = eye + stereo_warp._backward_warp(
                    image, grid, padded[:, layer * 2:layer * 2 + 2], scale) * weights[:, layer:layer + 1]
            eye = eye.clamp(0, 1)
        add(f"mlbw_warp_{image_hw[1]}x{image_hw[0]}_{depth_hw[1]}x{depth_hw[0]}"
            f"_{divergence}_{convergence}",
            [image_hw[1], image_hw[0], depth_hw[1], depth_hw[0]],
            np.concatenate([image.numpy().ravel(), delta.numpy().ravel(),
                            layer_weight.numpy().ravel()]),
            eye.numpy())

    # the mask post-process, on logits the real head produced
    for index, (image_hw, depth_hw, divergence, inner, outer) in enumerate([
            ((216, 384), (108, 192), 5.0, 0, 0),
            ((216, 384), (108, 192), 5.0, 2, 0),
            ((216, 384), (108, 192), 5.0, 0, 3),
            ((216, 384), (108, 192), 5.0, 2, 3),
            ((216, 384), (216, 384), 10.0, 4, 4),   # no resize, so the threshold alone
            ((160, 288), (80, 144), 3.0, 1, 3),     # base_width scaling, 2x
    ]):
        _, depth = _monobw_pair(image_hw, depth_hw, 800 + index)
        _, _, logits = _mlbw_heads(depth, divergence, 0.5)
        with torch.inference_mode():
            mask = stereo_inpaint._postprocess_hole_mask(
                logits, target_size=image_hw, threshold=stereo_inpaint.MASK_MLBW_THRESHOLD,
                inner_dilation=inner, outer_dilation=outer)
        add(f"mlbw_mask_{image_hw[1]}x{image_hw[0]}_{depth_hw[1]}x{depth_hw[0]}_{inner}_{outer}",
            [image_hw[1], image_hw[0], depth_hw[1], depth_hw[0], inner, outer],
            logits, mask.float())

    # exactly what the inpaint network is handed, for each eye
    #
    # The monobw version of this case caught the flip ordering, which is the
    # part of that path most likely to be wrong and least likely to be found by
    # anything else. This one is worth more, not less: monobw hides its left-eye
    # mirror inside a kernel's write index, whereas here both eyes come back in
    # frame orientation and the mirror has to be an explicit pass.
    for index, (image_hw, depth_hw, divergence, convergence, inner, outer) in enumerate([
            ((216, 384), (216, 384), 2.0, 0.5, 0, 0),
            ((216, 384), (108, 192), 5.0, 0.5, 2, 3),
            ((160, 288), (80, 144), 8.0, 1.0, 1, 1),
    ]):
        image, depth = _monobw_pair(image_hw, depth_hw, 900 + index)
        with torch.inference_mode():
            left_eye, right_eye, left_mask, right_mask = stereo_warp.apply_divergence_mlbw(
                _mlbw, image, depth, divergence=divergence, convergence=convergence,
                synthetic_view="both", preserve_screen_border=False, enable_amp=False)
            for side, eye, mask in (("left", left_eye, left_mask),
                                    ("right", right_eye, right_mask)):
                # _inpaint_single flips the left eye and not the right, then runs
                # preprocess_mask on whatever orientation that left it in.
                if side == "left":
                    eye, mask = eye.flip(-1), mask.flip(-1)
                mask = stereo_inpaint.MLBWInpaintImage.preprocess_mask(
                    mask, target_size=eye.shape[-2:],
                    inner_dilation=inner, outer_dilation=outer)
                add(f"mlbw_eye_{side}_{image_hw[1]}x{image_hw[0]}_{depth_hw[1]}"
                    f"_{divergence}_{convergence}_{inner}_{outer}",
                    [image_hw[1], image_hw[0], depth_hw[1], depth_hw[0],
                     1 if side == "right" else 0, inner, outer,
                     int(round(divergence * 1000)), int(round(convergence * 1000))],
                    np.concatenate([image.numpy().ravel(), depth.numpy().ravel()]),
                    np.concatenate([eye.numpy().ravel(), mask.float().numpy().ravel()]))
else:
    print(f"skipping mlbw cases: {MASK_MLBW_CHECKPOINT} not found")


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
