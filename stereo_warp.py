"""Standalone stereo synthesis from a colour frame and a depth map.

This is iw3's ``row_flow_v2`` warping path extracted into one file. It depends
on PyTorch and nothing else -- no ``nunif`` imports, no ``args`` object, every
setting passed explicitly.

The point of the extraction is that once depth comes from somewhere else (a
depth pass rendered in another tool, a clip on a Resolve timeline), the only
parts of iw3 that still matter are the warp and the model that drives it.

Correspondence with the original, for anyone diffing the two:

    iw3/utils.py            apply_divergence()          -> synthesize_stereo()
    iw3/utils.py            the ``stereo_width`` resize -> _resize_depth()
    iw3/backward_warp.py    apply_divergence_nn_LR()    -> synthesize_stereo()
    iw3/backward_warp.py    apply_divergence_nn_delta() -> _warp_one_view()
    iw3/backward_warp.py    make_input_tensor()         -> _make_input_tensor()
    iw3/backward_warp.py    backward_warp()             -> _backward_warp()
    iw3/mapper.py           get_mapper()                -> get_mapper()
    iw3/models/row_flow_v2  RowFlowV2                   -> RowFlowV2

Arithmetic is kept expression-for-expression identical to the original, even
where it could be simplified, because ``tests/test_stereo_warp.py`` requires a
maximum absolute difference of exactly 0 against stock iw3. Simplifying a
floating-point expression is enough to break that.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "RowFlowV2",
    "load_row_flow_v2",
    "get_mapper",
    "synthesize_stereo",
    "ROW_FLOW_V2_URL",
]

ROW_FLOW_V2_URL = "https://github.com/nagadomi/nunif/releases/download/0.0.0/iw3_row_flow_v2_20240130.pth"

MODEL_NAME = "sbs.row_flow_v2"

# The model is trained with a 28 pixel replication pad, cropped off again after
# inference, so the convolutions never see the frame edge.
OFFSET = 28


# ---------------------------------------------------------------------------
# model


class RowFlowV2(nn.Module):
    """The ``row_flow_v2`` network, inference path only.

    Submodule names match ``iw3/models/row_flow_v2.py`` so the published
    checkpoint loads without any key remapping.

    Input is (B, 3, H, W): disparity, divergence feature, convergence feature.
    Output is (B, 2, H, W): a per-pixel sampling delta, y always zero.

    This is only iw3's ``delta_output=True`` path. The alternative path warps
    the colour image inside the model, which is not what the caller wants here
    -- the delta gets applied to the colour frame at its own, higher,
    resolution.
    """

    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(*[
            nn.ReplicationPad2d((1, 1, 0, 0)),
            nn.Conv2d(3, 16, kernel_size=(1, 3), stride=1, padding=0),
            nn.ReLU(inplace=True),
        ])
        self.non_overlap = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
        self.overlap_residual = nn.Sequential(*[
            nn.ReplicationPad2d((4, 4, 0, 0)),
            nn.Conv2d(16, 16, kernel_size=(1, 9), stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d((4, 4, 0, 0)),
            nn.Conv2d(16, 32, kernel_size=(1, 9), stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d((4, 4, 0, 0)),
            nn.Conv2d(32, 32, kernel_size=(1, 9), stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=0),
        ])
        # Only used by the training-time path that warps inside the model.
        # Registered so the published state dict loads strictly.
        self.register_buffer("delta_scale", torch.tensor(1.0 / 127.0))
        self.pre_pad = nn.ReplicationPad2d((OFFSET,) * 4)

    def forward(self, x):
        x = self.pre_pad(x)
        x = self.feature(x)
        delta = self.non_overlap(x) + self.overlap_residual(x)
        delta = torch.cat([delta, torch.zeros_like(delta)], dim=1)
        return F.pad(delta, [-OFFSET] * 4)


def _remap_state_dict(state_dict):
    """Rewrite nunif's ``nn.Sequential(OrderedDict(...))`` keys to plain indices.

    nunif names the padding layers (``feature.pad0.``), which carry no
    parameters, and numbers the rest as strings. Dropping the named entries and
    keeping the numbers gives exactly this module's key layout.
    """
    remapped = {}
    for key, value in state_dict.items():
        parts = key.split(".")
        if len(parts) >= 2 and parts[0] == "feature":
            # feature.0.weight -> index 1 here, because the pad is a real
            # module in this implementation rather than a named entry.
            remapped[f"feature.{int(parts[1]) + 1}." + ".".join(parts[2:])] = value
        elif len(parts) >= 2 and parts[0] == "overlap_residual":
            # nunif indices 0,2,4,6 are the convs; interleaved pads make them
            # 1,4,7,10 once the pads are counted as modules.
            index = {0: 1, 2: 4, 4: 7, 6: 10}[int(parts[1])]
            remapped[f"overlap_residual.{index}." + ".".join(parts[2:])] = value
        else:
            remapped[key] = value
    return remapped


def load_row_flow_v2(path=None, device="cpu"):
    """Load the published ``row_flow_v2`` checkpoint.

    ``path`` may be a local ``.pth`` file or ``None`` to download the release
    build to the torch hub cache. Returns a ``RowFlowV2`` in eval mode.

    Note that iw3's ``create_stereo_model()`` re-reads its checkpoint on every
    call; cache the result yourself in anything interactive.
    """
    if path is None:
        data = torch.hub.load_state_dict_from_url(ROW_FLOW_V2_URL, weights_only=True, map_location="cpu")
    else:
        data = torch.load(path, map_location="cpu", weights_only=True)

    if "nunif_model" not in data:
        raise ValueError("not a nunif checkpoint")
    if data.get("name") != MODEL_NAME:
        raise ValueError(f"expected {MODEL_NAME}, got {data.get('name')!r}")

    model = RowFlowV2()
    model.load_state_dict(_remap_state_dict(data["state_dict"]), strict=True)
    return model.eval().to(device)


# ---------------------------------------------------------------------------
# depth mappers
#
# Verbatim from iw3/mapper.py. These turn a model's depth output into the
# disparity the warp expects. If the depth is already disparity-like -- which
# is the normal case for depth arriving from outside iw3 -- pass mapper=None,
# which is the same function as iw3's "none".


def _softplus01_legacy(depth, c=6):
    min_v = math.log(1 + math.exp(0 * 12.0 - c)) / (12 - c)
    max_v = math.log(1 + math.exp(1 * 12.0 - c)) / (12 - c)
    v = torch.log(1. + torch.exp(depth * 12.0 - c)) / (12 - c)
    return (v - min_v) / (max_v - min_v)


def _softplus01(x, bias, scale):
    min_v = math.log(1 + math.exp((0 - bias) * scale))
    max_v = math.log(1 + math.exp((1 - bias) * scale))
    v = torch.log(1. + torch.exp((x - bias) * scale))
    return (v - min_v) / (max_v - min_v)


def _inv_softplus01(x, bias, scale):
    min_v = ((torch.zeros(1, dtype=x.dtype, device=x.device) - bias) * scale).expm1().clamp(min=1e-6).log()
    max_v = ((torch.ones(1, dtype=x.dtype, device=x.device) - bias) * scale).expm1().clamp(min=1e-6).log()
    v = ((x - bias) * scale).expm1().clamp(min=1e-6).log()
    return (v - min_v) / (max_v - min_v)


def _distance_to_disparity(x, c):
    c1 = 1.0 + c
    min_v = c / c1
    return ((c / (c1 - x)) - min_v) / (1.0 - min_v)


def _shift_relative_depth(x, min_distance, max_distance=16):
    provisional_max_distance = min_distance + max_distance
    A = 1.0 / provisional_max_distance
    B = (1.0 / min_distance) - (1.0 / provisional_max_distance)
    distance = 1 / (A + B * x)

    new_min_distance = 1.0
    distance = (new_min_distance - min_distance) + distance

    new_x = 1.0 / distance

    min_value = 1.0 / (max_distance + 1)
    value_range = 1.0 - 1.0 / (max_distance + 1)
    new_x = (new_x - min_value) / value_range

    return new_x


_SOFTPLUS_PARAMS = {
    "mul_1": {"bias": 0.343, "scale": 12},
    "mul_2": {"bias": 0.515, "scale": 12},
    "mul_3": {"bias": 0.687, "scale": 12},
}
_INV_SOFTPLUS_PARAMS = {
    "inv_mul_1": {"bias": -0.002102, "scale": 7.8788},
    "inv_mul_2": {"bias": -0.0003, "scale": 6.2626},
    "inv_mul_3": {"bias": -0.0001, "scale": 3.4343},
}
_SHIFT_PARAMS = {
    "shift_30": {"min_distance": 3.0},
    "shift_20": {"min_distance": 2.0},
    "shift_14": {"min_distance": 1.4},
    "shift_08": {"min_distance": 0.8},
    "shift_06": {"min_distance": 0.6},
    "shift_045": {"min_distance": 0.45},
}
_DIV_PARAMS = {
    "div_25": 2.5,
    "div_10": 1,
    "div_6": 0.6,
    "div_4": 0.4,
    "div_2": 0.2,
    "div_1": 0.1,
}


def _resolve_mapper_function(name):
    if name == "pow2":
        return lambda x: x ** 2
    elif name == "none":
        return lambda x: x
    elif name == "softplus":
        return _softplus01_legacy
    elif name == "softplus2":
        return lambda x: _softplus01_legacy(x) ** 2
    elif name in _SOFTPLUS_PARAMS:
        param = _SOFTPLUS_PARAMS[name]
        return lambda x: _softplus01(x, **param)
    elif name in _INV_SOFTPLUS_PARAMS:
        param = _INV_SOFTPLUS_PARAMS[name]
        return lambda x: _inv_softplus01(x, **param)
    elif name in _SHIFT_PARAMS:
        param = _SHIFT_PARAMS[name]
        return lambda x: _shift_relative_depth(x, **param)
    elif name in _DIV_PARAMS:
        param = _DIV_PARAMS[name]
        return lambda x: _distance_to_disparity(x, param)
    else:
        raise NotImplementedError(f"mapper={name}")


def get_mapper(name):
    """Resolve an iw3 mapper name to a callable. ``None`` gives the identity.

    Supports iw3's chaining (``"a:b"``) and interpolation (``"a+b=0.25"``)
    syntax, since ``resolve_mapper_name()`` produces both.
    """
    if name is None:
        return lambda x: x

    names = name.split(":") if ":" in name else [name]
    functions = []
    for one in names:
        if "+" in one:
            one, weight = one.split("=")
            weight = 0.5 if not weight else float(weight)
            assert 0.0 <= weight <= 1.0
            name_a, name_b = one.split("+")
            mapper_a = _resolve_mapper_function(name_a)
            mapper_b = _resolve_mapper_function(name_b)
            functions.append(
                lambda x, a=mapper_a, b=mapper_b, w=weight: a(x) * (1 - w) + b(x) * w)
        else:
            functions.append(_resolve_mapper_function(one))

    def chained(x):
        for function in functions:
            x = function(x)
        return x

    return chained


# ---------------------------------------------------------------------------
# warping


def _autocast(device, enabled):
    """iw3's ``nunif.device.autocast``, minus the backends we do not target.

    CPU autocast is disabled there because it is unexpectedly slow, and that
    choice changes the numbers, so it has to be reproduced rather than tidied.
    """
    device_type = device.split(":")[0] if isinstance(device, str) else device.type
    if device_type == "cpu":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)
    if device_type == "mps":
        return torch.autocast(device_type="mps", dtype=torch.float16, enabled=False)
    return torch.autocast(device_type=device_type, dtype=None, enabled=enabled)


def _make_grid(batch, width, height, device):
    mesh_y, mesh_x = torch.meshgrid(torch.linspace(-1, 1, height, device=device),
                                    torch.linspace(-1, 1, width, device=device), indexing="ij")
    mesh_y = mesh_y.reshape(1, 1, height, width).expand(batch, 1, height, width)
    mesh_x = mesh_x.reshape(1, 1, height, width).expand(batch, 1, height, width)
    return torch.cat((mesh_x, mesh_y), dim=1)


def _make_input_tensor(depth, divergence, convergence, image_width, preserve_screen_border):
    """Build the model's (B, 3, H, W) input: disparity, divergence, convergence.

    iw3 builds this one batch element at a time and stacks. Doing it batched is
    elementwise-identical and is what the ONNX export will need.
    """
    divergence_pix = divergence * 0.5 * 0.01 * image_width
    divergence_value = divergence_pix / 32.0
    convergence_value = (-divergence_pix * convergence) / 32.0

    divergence_feat = torch.full_like(depth, divergence_value)
    convergence_feat = torch.full_like(depth, convergence_value)

    if preserve_screen_border:
        # Force the parallax at the screen border to zero.
        # Left exactly as iw3 writes it: image_width cancels out algebraically,
        # but cancelling it by hand changes the rounding.
        border_pix = round(divergence * 0.75 * 0.01 * image_width * (depth.shape[-1] / image_width))
        if border_pix > 0:
            view_shape = [1] * (depth.ndim - 1) + [-1]
            border_weight_l = torch.linspace(0.0, 1.0, border_pix, dtype=depth.dtype,
                                             device=depth.device).view(view_shape)
            border_weight_r = torch.linspace(1.0, 0.0, border_pix, dtype=depth.dtype,
                                             device=depth.device).view(view_shape)

            divergence_feat[..., :border_pix] *= border_weight_l
            divergence_feat[..., -border_pix:] *= border_weight_r
            convergence_feat[..., :border_pix] *= border_weight_l
            convergence_feat[..., -border_pix:] *= border_weight_r

    return torch.cat([depth, divergence_feat, convergence_feat], dim=1)


def _backward_warp(c, grid, delta, delta_scale):
    """Sample ``c`` through ``grid + delta * delta_scale``.

    The grid is built at the *depth's* resolution and interpolated up to the
    colour image's, which is what lets a small depth map drive a large frame.
    """
    grid = grid + delta * delta_scale
    if c.shape[2] != grid.shape[2] or c.shape[3] != grid.shape[3]:
        grid = F.interpolate(grid, size=c.shape[-2:],
                             mode="bilinear", align_corners=True, antialias=False)
    grid = grid.permute(0, 2, 3, 1)
    z = F.grid_sample(c, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return torch.clamp(z, 0, 1)


def _warp_one_view(model, c, depth, divergence, convergence, steps, shift,
                   preserve_screen_border, enable_amp):
    """One synthetic eye. ``shift`` is -1 for the left eye, +1 for the right.

    The right eye is produced by mirroring, running the same left-eye model,
    and mirroring back.
    """
    if shift > 0:
        c = torch.flip(c, (3,))
        depth = torch.flip(depth, (3,))

    B, _, H, W = depth.shape
    base_size = max(H, W)
    divergence_step = divergence / steps
    grid = _make_grid(B, W, H, c.device)
    # Note the depth's width, not the image's.
    delta_scale = torch.tensor(1.0 / (W // 2 - 1), dtype=c.dtype, device=c.device)

    depth_warp = depth
    delta_steps = []
    for step in range(steps):
        x = _make_input_tensor(depth_warp,
                               divergence=divergence_step,
                               convergence=convergence,
                               image_width=base_size,
                               preserve_screen_border=preserve_screen_border)
        with _autocast(device=depth.device, enabled=enable_amp):
            delta = model(x)

        delta_steps.append(delta)
        if step + 1 < steps:
            depth_warp = _backward_warp(depth_warp, grid, delta, delta_scale)

    z = c
    for delta in delta_steps:
        z = _backward_warp(z, grid, delta, delta_scale)

    if shift > 0:
        z = torch.flip(z, (3,))

    return z


def _resize_depth(depth, image_shape, stereo_width):
    """iw3's ``args.stereo_width`` depth resize.

    This is the mechanism that stops a full-resolution depth map from breaking
    the warp -- without it the model sees detail at a scale it was not trained
    on and the result stripes. The new height uses the *image's* aspect ratio,
    not the depth's, deliberately.
    """
    if stereo_width is None:
        return depth

    H, W = image_shape[2:]
    stereo_width = min(W, stereo_width)
    if depth.shape[3] == stereo_width:
        return depth

    new_w = stereo_width
    new_h = int(H * (stereo_width / W))
    depth = F.interpolate(depth, size=(new_h, new_w),
                          mode="bilinear", align_corners=True, antialias=True)
    return torch.clamp(depth, 0, 1)


def synthesize_stereo(
    image,
    depth,
    model,
    *,
    divergence=2.0,
    convergence=0.5,
    synthetic_view="both",
    stereo_width=None,
    steps=None,
    preserve_screen_border=False,
    enable_amp=True,
    mapper=None,
):
    """Turn one colour frame and one depth map into a left/right pair.

    Args:
        image: float32, RGB, 0..1, CHW or BCHW.
        depth: float32, 0..1, 1HW or B1HW. Larger is nearer. Need not match
            ``image``'s resolution -- the warp is built at the depth's size and
            interpolated up.
        model: a ``RowFlowV2`` from ``load_row_flow_v2()``, on the same device.
        divergence: stereo strength, as a percentage of image width. iw3's
            default is 2.0.
        convergence: the depth value that lands on the screen plane. 0 puts
            everything behind the screen, 1 everything in front.
        synthetic_view: ``"both"`` synthesises both eyes at half strength each;
            ``"left"``/``"right"`` keeps the original as the other eye and
            doubles the divergence.
        stereo_width: resize the depth to this width before warping. Leave at
            ``None`` only when the depth is already at a sane working
            resolution; full-resolution depth needs this set.
        steps: warp iterations, default 1. Higher subdivides the divergence.
        preserve_screen_border: taper parallax to zero at the left and right
            edges.
        enable_amp: autocast the model. No effect on CPU, which iw3 leaves
            disabled because it is slow there.
        mapper: iw3 mapper name to apply to ``depth`` first, or ``None`` to use
            the depth as-is. ``None`` and ``"none"`` are the same function.

    Returns:
        ``(left, right)``, each matching ``image``'s shape and dtype.
    """
    if synthetic_view not in {"both", "left", "right"}:
        raise ValueError(f"synthetic_view={synthetic_view!r}")
    if torch.is_tensor(convergence):
        raise TypeError("convergence must be a float; per-frame convergence is not supported here")

    batched = image.ndim == 4
    if not batched:
        image = image.unsqueeze(0)
        depth = depth.unsqueeze(0)

    steps = 1 if steps is None else steps

    depth = get_mapper(mapper)(depth)
    depth = _resize_depth(depth, image.shape, stereo_width)

    if synthetic_view == "both":
        left = _warp_one_view(model, image, depth, divergence, convergence, steps, -1,
                              preserve_screen_border, enable_amp)
        right = _warp_one_view(model, image, depth, divergence, convergence, steps, 1,
                               preserve_screen_border, enable_amp)
    elif synthetic_view == "right":
        left = image
        right = _warp_one_view(model, image, depth, divergence * 2, convergence, steps, 1,
                               preserve_screen_border, enable_amp)
    else:
        left = _warp_one_view(model, image, depth, divergence * 2, convergence, steps, -1,
                              preserve_screen_border, enable_amp)
        right = image

    if not batched:
        left = left.squeeze(0)
        right = right.squeeze(0)

    return left, right
