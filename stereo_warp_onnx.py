"""The ONNX Runtime implementation of stereo_warp, with the same signature.

Needs onnxruntime and numpy. No PyTorch — that is the whole point: this is the
shape the Resolve plugin's C++ will take, and it is validated against the
PyTorch version in tests/test_stereo_warp_onnx.py.

    from stereo_warp_onnx import StereoWarpSession
    session = StereoWarpSession("models")
    left, right = session.synthesize_stereo(image, depth, divergence=2.0)

Execution providers, measured on an RTX 5080 (see docs/phase2-onnx.md):

    CUDAExecutionProvider   correct, but only with use_tf32=False, which this
                            module sets by default. TF32 costs three orders of
                            magnitude of accuracy to save 0.8 ms.
    CPUExecutionProvider    correct, ~250 ms at 1080p. A fallback, not a plan.
    DmlExecutionProvider    WRONG. It miscomputes row_flow_v2 itself, by whole
                            units, on a graph the CPU and CUDA providers agree
                            on. Refused below rather than silently used.
"""

import math
import os

import numpy as np
import onnxruntime as ort

__all__ = ["StereoWarpSession", "get_mapper"]

# DirectML does not merely lose precision on row_flow_v2, it returns values off
# by ~3.4 absolute on a 0..1 signal, while grid_sample and the grid build are
# both fine. Until that is understood, using it would ship silent corruption.
BROKEN_PROVIDERS = {"DmlExecutionProvider"}


# ---------------------------------------------------------------------------
# mappers, numpy versions of the ones in stereo_warp.py


def _softplus01_legacy(depth, c=6):
    min_v = math.log(1 + math.exp(0 * 12.0 - c)) / (12 - c)
    max_v = math.log(1 + math.exp(1 * 12.0 - c)) / (12 - c)
    v = np.log(1. + np.exp(depth * 12.0 - c)) / (12 - c)
    return (v - min_v) / (max_v - min_v)


def _softplus01(x, bias, scale):
    min_v = math.log(1 + math.exp((0 - bias) * scale))
    max_v = math.log(1 + math.exp((1 - bias) * scale))
    v = np.log(1. + np.exp((x - bias) * scale))
    return (v - min_v) / (max_v - min_v)


def _inv_softplus01(x, bias, scale):
    min_v = np.log(np.clip(np.expm1((0 - bias) * scale), 1e-6, None))
    max_v = np.log(np.clip(np.expm1((1 - bias) * scale), 1e-6, None))
    v = np.log(np.clip(np.expm1((x - bias) * scale), 1e-6, None))
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
    distance = (1.0 - min_distance) + distance
    new_x = 1.0 / distance
    min_value = 1.0 / (max_distance + 1)
    value_range = 1.0 - 1.0 / (max_distance + 1)
    return (new_x - min_value) / value_range


_SOFTPLUS_PARAMS = {"mul_1": (0.343, 12), "mul_2": (0.515, 12), "mul_3": (0.687, 12)}
_INV_SOFTPLUS_PARAMS = {"inv_mul_1": (-0.002102, 7.8788), "inv_mul_2": (-0.0003, 6.2626),
                        "inv_mul_3": (-0.0001, 3.4343)}
_SHIFT_PARAMS = {"shift_30": 3.0, "shift_20": 2.0, "shift_14": 1.4,
                 "shift_08": 0.8, "shift_06": 0.6, "shift_045": 0.45}
_DIV_PARAMS = {"div_25": 2.5, "div_10": 1, "div_6": 0.6, "div_4": 0.4, "div_2": 0.2, "div_1": 0.1}


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
        bias, scale = _SOFTPLUS_PARAMS[name]
        return lambda x: _softplus01(x, bias, scale)
    elif name in _INV_SOFTPLUS_PARAMS:
        bias, scale = _INV_SOFTPLUS_PARAMS[name]
        return lambda x: _inv_softplus01(x, bias, scale)
    elif name in _SHIFT_PARAMS:
        return lambda x: _shift_relative_depth(x, _SHIFT_PARAMS[name])
    elif name in _DIV_PARAMS:
        return lambda x: _distance_to_disparity(x, _DIV_PARAMS[name])
    else:
        raise NotImplementedError(f"mapper={name}")


def get_mapper(name):
    """Resolve an iw3 mapper name to a callable. ``None`` gives the identity."""
    if name is None:
        return lambda x: x
    names = name.split(":") if ":" in name else [name]
    functions = []
    for one in names:
        if "+" in one:
            one, weight = one.split("=")
            weight = 0.5 if not weight else float(weight)
            name_a, name_b = one.split("+")
            a = _resolve_mapper_function(name_a)
            b = _resolve_mapper_function(name_b)
            functions.append(lambda x, a=a, b=b, w=weight: a(x) * (1 - w) + b(x) * w)
        else:
            functions.append(_resolve_mapper_function(one))

    def chained(x):
        for function in functions:
            x = function(x)
        return x

    return chained


# ---------------------------------------------------------------------------


class StereoWarpSession:
    """Holds the ORT sessions. Build one and keep it.

    Creating a session reads and compiles the graph, so doing it per frame
    would dominate the cost -- the same trap as iw3's create_stereo_model(),
    which re-reads its checkpoint on every call.
    """

    def __init__(self, models_dir="models", provider=None, use_tf32=False, allow_broken=False):
        self.models_dir = models_dir
        available = ort.get_available_providers()

        if provider is None:
            for candidate in ("CUDAExecutionProvider", "CPUExecutionProvider"):
                if candidate in available:
                    provider = candidate
                    break
            else:
                provider = available[0]
        if provider in BROKEN_PROVIDERS and not allow_broken:
            raise ValueError(
                f"{provider} miscomputes row_flow_v2 and would silently corrupt output; "
                f"pass allow_broken=True only to reproduce that")
        if provider not in available:
            raise ValueError(f"{provider} not available; have {available}")

        self.provider = provider
        options = ort.SessionOptions()
        options.log_severity_level = 3

        if provider == "CUDAExecutionProvider":
            # TF32 is on by default and takes the model from 4e-6 to 1.5e-3
            # against PyTorch, to save under a millisecond. Not worth it.
            entry = (provider, {"use_tf32": int(bool(use_tf32))})
        else:
            entry = provider

        self._warp = ort.InferenceSession(
            os.path.join(models_dir, "stereo_warp.onnx"), options, providers=[entry])
        # Resampling weights only change when a size changes, which is once per
        # timeline in practice, so they are worth keeping.
        self._weight_cache = {}

    # -- the pieces that stayed outside the graph --------------------------

    @staticmethod
    def _aa_weights(in_size, out_size):
        """PyTorch's antialiased bilinear resampling weights for one axis.

        ONNX Resize will not do this. Exported with antialias=0 the graph
        reproduces torch's *plain* resize to 5e-5, but patching antialias=1 --
        nunif's technique, and the ONNX spec does define the attribute -- lands
        0.23 away from torch on a 0..1 signal, whichever coordinate transform
        mode is used. ORT's antialias is simply a different filter.

        So it is done here instead, which is what the plan anticipated for
        anything that would not export. The reconstruction below matches torch
        to ~1e-5, the residual being float64 here against float32 there.

        Note ``center = scale * (i + 0.5)`` regardless of align_corners: torch's
        antialias path uses that even though its plain align_corners path uses
        ``scale * i``. Using the latter here is wrong by 0.2.
        """
        scale = (in_size - 1) / (out_size - 1) if out_size > 1 else 0.0
        support = scale if scale >= 1.0 else 1.0
        invscale = (1.0 / scale) if scale >= 1.0 else 1.0

        rows = []
        for i in range(out_size):
            center = scale * (i + 0.5)
            xmin = max(int(center - support + 0.5), 0)
            xmax = min(int(center + support + 0.5), in_size)
            index = np.arange(xmin, xmax)
            weight = np.maximum(0.0, 1.0 - np.abs((index - center + 0.5) * invscale))
            total = weight.sum()
            rows.append((xmin, (weight / total if total else weight).astype(np.float64)))
        return rows

    def _axis_weights(self, in_size, out_size):
        key = (in_size, out_size)
        if key not in self._weight_cache:
            self._weight_cache[key] = self._aa_weights(in_size, out_size)
        return self._weight_cache[key]

    @staticmethod
    def _apply_axis(array, rows, axis):
        array = np.moveaxis(array, axis, -1)
        out = np.empty(array.shape[:-1] + (len(rows),), dtype=np.float64)
        for i, (start, weight) in enumerate(rows):
            out[..., i] = (array[..., start:start + len(weight)] * weight).sum(-1)
        return np.moveaxis(out, -1, axis)

    def _resize_depth(self, depth, image_shape, stereo_width):
        if stereo_width is None:
            return depth
        H, W = image_shape[2:]
        stereo_width = min(W, stereo_width)
        if depth.shape[3] == stereo_width:
            return depth
        new_w = stereo_width
        new_h = int(H * (stereo_width / W))

        resized = depth.astype(np.float64)
        resized = self._apply_axis(resized, self._axis_weights(depth.shape[3], new_w), 3)
        resized = self._apply_axis(resized, self._axis_weights(depth.shape[2], new_h), 2)
        return np.ascontiguousarray(np.clip(resized, 0, 1), dtype=np.float32)

    @staticmethod
    def _make_input_tensor(depth, divergence, convergence, image_width, preserve_screen_border):
        divergence_pix = divergence * 0.5 * 0.01 * image_width
        divergence_value = divergence_pix / 32.0
        convergence_value = (-divergence_pix * convergence) / 32.0

        B, _, H, W = depth.shape
        x = np.empty((B, 3, H, W), dtype=np.float32)
        x[:, 0:1] = depth
        x[:, 1] = divergence_value
        x[:, 2] = convergence_value

        if preserve_screen_border:
            # Kept in iw3's own form; see the note in stereo_warp.py.
            border_pix = round(divergence * 0.75 * 0.01 * image_width * (W / image_width))
            if border_pix > 0:
                left = np.linspace(0.0, 1.0, border_pix, dtype=np.float32)
                right = np.linspace(1.0, 0.0, border_pix, dtype=np.float32)
                x[:, 1:3, :, :border_pix] *= left
                x[:, 1:3, :, -border_pix:] *= right
        return x

    # -- the graph ---------------------------------------------------------

    def _warp_both(self, image, x):
        delta_scale = np.array(1.0 / (x.shape[3] // 2 - 1), dtype=np.float32)
        return self._warp.run(["left", "right"],
                              {"image": image, "x": x, "delta_scale": delta_scale})

    def synthesize_stereo(
        self,
        image,
        depth,
        *,
        divergence=2.0,
        convergence=0.5,
        synthetic_view="both",
        stereo_width=None,
        preserve_screen_border=False,
        mapper=None,
    ):
        """Same contract as stereo_warp.synthesize_stereo, on numpy arrays.

        Not supported here, deliberately: ``steps`` above 1, which needs the
        warp fed back into itself and so wants a loop in the graph. iw3 defaults
        to 1 and the Resolve plugin has no reason to change it.
        """
        if synthetic_view not in {"both", "left", "right"}:
            raise ValueError(f"synthetic_view={synthetic_view!r}")

        batched = image.ndim == 4
        if not batched:
            image = image[None]
            depth = depth[None]
        image = np.ascontiguousarray(image, dtype=np.float32)
        depth = np.ascontiguousarray(depth, dtype=np.float32)

        depth = np.asarray(get_mapper(mapper)(depth), dtype=np.float32)
        depth = self._resize_depth(depth, image.shape, stereo_width)

        # "left"/"right" keep the original as the other eye and double the
        # divergence, as in iw3.
        effective_divergence = divergence if synthetic_view == "both" else divergence * 2
        base_size = max(depth.shape[2], depth.shape[3])
        x = self._make_input_tensor(depth, effective_divergence, convergence,
                                    base_size, preserve_screen_border)

        left, right = self._warp_both(image, x)
        if synthetic_view == "right":
            left = image
        elif synthetic_view == "left":
            right = image

        if not batched:
            left, right = left[0], right[0]
        return left, right
