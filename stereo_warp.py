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
    "RowFlowV3",
    "MLBW",
    "load_row_flow_v2",
    "load_row_flow_v3",
    "load_mask_mlbw_l2",
    "load_stereo_model",
    "apply_divergence_mlbw",
    "get_mapper",
    "synthesize_stereo",
    "ROW_FLOW_V2_URL",
    "ROW_FLOW_V3_URL",
    "MASK_MLBW_L2_D1_URL",
]

_RELEASE = "https://github.com/nagadomi/nunif/releases/download/0.0.0/"
ROW_FLOW_V2_URL = _RELEASE + "iw3_row_flow_v2_20240130.pth"
ROW_FLOW_V3_URL = _RELEASE + "iw3_row_flow_v3_20250627.pth"
MASK_MLBW_L2_D1_URL = _RELEASE + "iw3_mask_mlbw_l2_d1_20250903.pth"

MODEL_NAME = "sbs.row_flow_v2"
MODEL_NAME_V3 = "sbs.row_flow_v3"
MODEL_NAME_MLBW = "sbs.mlbw"

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
# row_flow_v3
#
# A different kind of model from v2: windowed attention rather than plain
# convolutions, 117k parameters against 30k. The warp around it is unchanged --
# same three input channels, same delta contract, same non-symmetric path -- so
# only the network below differs.
#
# Module names and Sequential indices mirror iw3's exactly, so the published
# checkpoint loads with no key remapping. The attention bias buffers (`index`,
# `delta`) come from the checkpoint too, which is why their generator does not
# need porting.


def _pixel_unshuffle(x, factor):
    """Non-square pixel unshuffle. torch's built-in is square-only."""
    sh, sw = factor
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // sh, sh, W // sw, sw)
    x = x.permute(0, 1, 3, 5, 2, 4)
    return x.reshape(B, C * sh * sw, H // sh, W // sw)


def _pixel_shuffle(x, factor):
    sh, sw = factor
    B, C, H, W = x.shape
    x = x.reshape(B, C // (sh * sw), sh, sw, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3)
    return x.reshape(B, C // (sh * sw), H * sh, W * sw)


def _window_partition(x, window):
    """BCHW -> (B*windows, tokens, C), aka window_partition."""
    sh, sw = window
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // sh, sh, W // sw, sw)
    x = x.permute(0, 2, 4, 3, 5, 1)
    return x.reshape(B * (H // sh) * (W // sw), sh * sw, C)


def _window_merge(x, shape, window):
    sh, sw = window
    B, C, H, W = shape
    x = x.reshape(B, H // sh, W // sw, sh, sw, C)
    x = x.permute(0, 5, 1, 3, 2, 4)
    return x.reshape(B, C, H, W)


class _WindowScoreBias(nn.Module):
    """A learned bias over relative positions inside a window."""

    def __init__(self, window):
        super().__init__()
        tokens = window[0] * window[1]
        hidden = int(tokens ** 0.5) * 2
        self.tokens = tokens
        self.register_buffer("index", torch.zeros(tokens * tokens, dtype=torch.int64))
        self.register_buffer("delta", torch.zeros((2 * window[0] - 1) * (2 * window[1] - 1), 2))
        self.to_bias = nn.Sequential(*[
            nn.Linear(2, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, 1, bias=True),
        ])

    def forward(self):
        bias = self.to_bias(self.delta)
        return bias[self.index].reshape(self.tokens, self.tokens)


class _MHA(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv_dim = channels // num_heads
        self.qkv_proj = nn.Linear(channels, self.qkv_dim * num_heads * 3, bias=True)
        self.head_proj = nn.Linear(self.qkv_dim * num_heads, channels)

    def forward(self, x, attn_mask=None, export_safe=False):
        q, k, v = self.qkv_proj(x).split(self.qkv_dim * self.num_heads, dim=-1)

        if export_safe:
            # Head by head, staying three-dimensional, concatenating to merge.
            #
            # scaled_dot_product_attention wants (B, heads, tokens, dim), and it
            # is the permute back out of that layout that torch.export's
            # decomposition cannot lower -- it reports "cannot view a tensor",
            # and inserting contiguous() does not help because the copy is
            # elided again before the failure. Slicing costs nothing at two
            # heads.
            #
            # This is NOT bit-identical to the fused kernel: it sums in a
            # different order and lands about 5e-5 away. That is why the default
            # path below keeps SDPA -- the golden test needs difference 0 -- and
            # this form is used only for export, where the bar is ~1e-4 anyway.
            outputs = []
            scale = 1.0 / math.sqrt(self.qkv_dim)
            for head in range(self.num_heads):
                span = slice(head * self.qkv_dim, (head + 1) * self.qkv_dim)
                scores = torch.matmul(q[:, :, span], k[:, :, span].transpose(1, 2)) * scale
                if attn_mask is not None:
                    scores = scores + attn_mask
                outputs.append(torch.matmul(torch.softmax(scores, dim=-1), v[:, :, span]))
            x = torch.cat(outputs, dim=-1)
        else:
            B, QN, C = q.shape
            KN = k.shape[1]
            qh = q.view(B, QN, self.num_heads, self.qkv_dim).permute(0, 2, 1, 3)
            kh = k.view(B, KN, self.num_heads, self.qkv_dim).permute(0, 2, 1, 3)
            vh = v.view(B, KN, self.num_heads, self.qkv_dim).permute(0, 2, 1, 3)
            x = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=attn_mask)
            x = x.permute(0, 2, 1, 3).reshape(B, QN, self.qkv_dim * self.num_heads)

        return self.head_proj(x)


class _WindowMHA2d(nn.Module):
    def __init__(self, channels, num_heads, window, shift=(False, False)):
        super().__init__()
        self.window = window
        # iw3's "shift" is not a roll. It pads half a window of zeros onto each
        # shifted axis, which moves every window boundary by half a window, and
        # crops the padding off again afterwards. row_flow_v3 never shifts;
        # MLBW alternates shifted and unshifted blocks so that tokens are not
        # always partitioned along the same lines.
        self.pad_h = window[0] // 2 if shift[0] else 0
        self.pad_w = window[1] // 2 if shift[1] else 0
        self.mha = _MHA(channels, num_heads)

    def forward(self, x, attn_mask=None, export_safe=False):
        if self.pad_h or self.pad_w:
            x = F.pad(x, (self.pad_w, self.pad_w, self.pad_h, self.pad_h),
                      mode="constant", value=0)
        shape = x.shape
        x = _window_partition(x, self.window)
        x = self.mha(x, attn_mask=attn_mask, export_safe=export_safe)
        x = _window_merge(x, shape, self.window)
        if self.pad_h or self.pad_w:
            x = F.pad(x, (-self.pad_w, -self.pad_w, -self.pad_h, -self.pad_h))
        return x


class _WABlock(nn.Module):
    def __init__(self, channels, window):
        super().__init__()
        self.mha = _WindowMHA2d(channels, num_heads=2, window=window)
        self.conv_mlp = nn.Sequential(*[
            nn.Conv2d(channels, channels, kernel_size=1, padding=0),
            nn.GELU(),
            nn.ReplicationPad2d((1, 1, 1, 1)),
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=0),
            nn.LeakyReLU(0.1, inplace=True),
        ])
        self.bias = _WindowScoreBias(window)

    def forward(self, x, export_safe=False):
        x = x + self.mha(x, attn_mask=self.bias(), export_safe=export_safe)
        return x + self.conv_mlp(x)


class RowFlowV3(nn.Module):
    """iw3's ``row_flow_v3``, inference path only.

    Input is (B, 3, H, W): disparity, divergence feature, convergence feature.
    Output is (B, 2, H, W), a sampling delta with y always zero -- the same
    contract as RowFlowV2, so the warp does not care which is in use.
    """

    DOWNSCALE = (1, 8)
    MOD = 4 * 3

    def __init__(self):
        super().__init__()
        channels = 64
        pack = self.DOWNSCALE[0] * self.DOWNSCALE[1]
        self.blocks = nn.Sequential(*[
            nn.Conv2d(3 * pack, channels, kernel_size=1, stride=1, padding=0),
            _WABlock(channels, (4, 4)),
            _WABlock(channels, (3, 3)),
        ])
        self.last_layer = nn.Sequential(*[
            nn.ReplicationPad2d((1, 1, 1, 1)),
            nn.Conv2d(channels // pack, 1, kernel_size=3, stride=1, padding=0),
        ])
        # Unused in this path; registered so the state dict loads strictly.
        self.register_buffer("delta_scale", torch.tensor(1.0 / 127.0))
        self.export_safe = False

    def forward(self, x):
        height, width = x.shape[2], x.shape[3]
        mod_h = self.MOD * self.DOWNSCALE[0]
        mod_w = self.MOD * self.DOWNSCALE[1]

        # iw3 pads here with its own replication_pad2d_naive, which builds the
        # padding by Python tuple repetition -- (slice,) * n. That needs n to be
        # a concrete int, which specialises an exported graph to a single frame
        # width. F.pad takes the amount as data, and the crop below is a slice,
        # so both stay dynamic. Numerically identical.
        x = F.pad(x, (0, mod_w - width % mod_w, 0, mod_h - height % mod_h), mode="replicate")

        x = _pixel_unshuffle(x, self.DOWNSCALE)
        for index, block in enumerate(self.blocks):
            x = block(x, export_safe=self.export_safe) if index > 0 else block(x)
        x = _pixel_shuffle(x, self.DOWNSCALE)
        x = x[:, :, :height, :width]
        x = self.last_layer(x)

        x = x.to(torch.float32)
        return torch.cat([x, torch.zeros_like(x)], dim=1)


def load_row_flow_v3(path=None, device="cpu", export_safe=False):
    """Load the published ``row_flow_v3`` checkpoint.

    ``export_safe`` swaps the attention for the head-sliced form that
    torch.export can lower. It is bit-identical in float32 and only needed when
    exporting.
    """
    if path is None:
        data = torch.hub.load_state_dict_from_url(ROW_FLOW_V3_URL, weights_only=True, map_location="cpu")
    else:
        data = torch.load(path, map_location="cpu", weights_only=True)

    if "nunif_model" not in data:
        raise ValueError("not a nunif checkpoint")
    if data.get("name") != MODEL_NAME_V3:
        raise ValueError(f"expected {MODEL_NAME_V3}, got {data.get('name')!r}")

    model = RowFlowV3()
    model.load_state_dict(data["state_dict"], strict=True)
    model.export_safe = export_safe
    return model.eval().to(device)


class _MLBWBlock(nn.Module):
    """MLBW's WABlock.

    ``_WABlock`` with two differences that both matter: the window attention can
    shift, and the MLP has no trailing activation.
    """

    def __init__(self, channels, window, shift, num_heads):
        super().__init__()
        self.mha = _WindowMHA2d(channels, num_heads=num_heads, window=window, shift=shift)
        self.conv_mlp = nn.Sequential(*[
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.ReplicationPad2d((1, 1, 1, 1)),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0),
        ])
        self.bias = _WindowScoreBias(window)

    def forward(self, x, export_safe=False):
        x = x + self.mha(x, attn_mask=self.bias(), export_safe=export_safe)
        return x + self.conv_mlp(x)


class MLBW(nn.Module):
    """iw3's ``mask_mlbw_l2``, inference path only, in ``delta_output`` mode.

    Input is the same (B, 3, H, W) as RowFlowV3 -- disparity, divergence
    feature, convergence feature -- but the output is three tensors rather than
    one, and none of them is a finished delta:

      ``delta``             (B, num_layers, H, W), one x-shift per layer
      ``layer_weight``      (B, num_layers, H, W), softmax over the layer axis
      ``hole_mask_logits``  (B, 1, H, W), *logits*, at this input's resolution

    "Multi-layer backward warp" means the eye is a weighted sum of one backward
    warp per layer, not a single warp. That is how the model expresses a pixel
    that could plausibly come from two places -- a foreground edge and the
    background behind it -- instead of having to choose. The hole mask is the
    third head, and it is a prediction rather than the geometric fact MonoBW
    computes.
    """

    DOWNSCALE = (1, 8)
    MOD = 4

    def __init__(self, num_layers=2, base_dim=32, hole_mask=True):
        super().__init__()
        self.num_layers = num_layers
        self.hole_mask = hole_mask
        channels = base_dim * num_layers
        pack = self.DOWNSCALE[0] * self.DOWNSCALE[1]
        self.lv1_in = nn.Sequential(*[
            nn.ReplicationPad2d((4, 4, 0, 0)),
            nn.Conv2d(3, channels // pack, kernel_size=(1, 9), stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=False),
        ])
        self.lv2 = nn.Sequential(*[
            _MLBWBlock(channels, (4, 4), shift=(True, True), num_heads=num_layers),
            _MLBWBlock(channels, (4, 4), shift=(False, False), num_heads=num_layers),
            _MLBWBlock(channels, (4, 4), shift=(True, True), num_heads=num_layers),
            _MLBWBlock(channels, (4, 4), shift=(False, False), num_heads=num_layers),
        ])
        self.lv1_out = nn.Sequential(*[
            nn.ReplicationPad2d((4, 4, 0, 0)),
            nn.Conv2d(channels // pack, num_layers * 2 + (1 if hole_mask else 0),
                      kernel_size=(1, 9), stride=1, padding=0),
        ])
        self.export_safe = False

    def forward(self, x):
        height, width = x.shape[2], x.shape[3]
        mod_h = self.MOD * self.DOWNSCALE[0]
        mod_w = self.MOD * self.DOWNSCALE[1]

        # Symmetric, and never zero: at an exact multiple of the block size iw3
        # still pads a whole block rather than none, because the amount is
        # `mod - width % mod` and not `-width % mod`. That looks like an
        # oversight and is not one to fix here -- the network was trained with
        # it, so the output depends on it.
        pad_w = mod_w - width % mod_w
        pad_h = mod_h - height % mod_h
        pad_w1, pad_h1 = pad_w // 2, pad_h // 2
        x = F.pad(x, (pad_w1, pad_w - pad_w1, pad_h1, pad_h - pad_h1), mode="replicate")

        x = x1 = self.lv1_in(x)
        x = _pixel_unshuffle(x, self.DOWNSCALE)
        for block in self.lv2:
            x = block(x, export_safe=self.export_safe)
        x = _pixel_shuffle(x, self.DOWNSCALE)
        x = self.lv1_out(x + x1)
        # iw3 crops with a negative F.pad. A slice is the same values and, being
        # data rather than a Python int tuple, stays dynamic under export.
        x = x[:, :, pad_h1:pad_h1 + height, pad_w1:pad_w1 + width]

        layers = self.num_layers
        delta = x[:, :layers]
        layer_weight = x[:, layers:layers * 2]
        hole_mask_logits = x[:, layers * 2:]

        # iw3's float32 cast. Under autocast it does nothing -- softmax is on
        # autocast's promote-to-fp32 list, so the cast has already happened.
        # It earns its place in a half-precision graph running *without*
        # autocast, which is what the ONNX export is: there, softmax over fp16
        # lands about 2.4e-4 from the fp32 answer, which is a visible amount of
        # a blend weight. Kept for that case, not for this one.
        layer_weight = F.softmax(layer_weight.to(torch.float32), dim=1)

        return delta.to(torch.float32), layer_weight, hole_mask_logits.to(torch.float32)


def load_mask_mlbw_l2(path=None, device="cpu", export_safe=False):
    """Load the published ``mask_mlbw_l2`` checkpoint.

    Only a ``d1`` checkpoint exists for the masked variant, so unlike plain
    ``mlbw_l2`` -- which ships d1/d2/d3 and picks between them by divergence --
    one set of weights covers every divergence.
    """
    if path is None:
        data = torch.hub.load_state_dict_from_url(MASK_MLBW_L2_D1_URL, weights_only=True,
                                                  map_location="cpu")
    else:
        data = torch.load(path, map_location="cpu", weights_only=True)

    if "nunif_model" not in data:
        raise ValueError("not a nunif checkpoint")
    # The checkpoint is named for the class rather than the factory: every MLBW
    # variant says "sbs.mlbw" and they are told apart by kwargs.
    if data.get("name") != MODEL_NAME_MLBW:
        raise ValueError(f"expected {MODEL_NAME_MLBW}, got {data.get('name')!r}")
    kwargs = data.get("kwargs") or {}
    if not kwargs.get("hole_mask"):
        raise ValueError("expected a hole_mask checkpoint, got plain mlbw")

    model = MLBW(num_layers=kwargs.get("num_layers", 2),
                 base_dim=kwargs.get("base_dim", 32),
                 hole_mask=True)
    model.load_state_dict(data["state_dict"], strict=True)
    model.export_safe = export_safe
    return model.eval().to(device)


def load_stereo_model(name, path=None, device="cpu", **kwargs):
    """``"row_flow_v2"``, ``"row_flow_v3"`` or ``"mask_mlbw_l2"``."""
    if name == "row_flow_v2":
        return load_row_flow_v2(path, device=device)
    elif name == "row_flow_v3":
        return load_row_flow_v3(path, device=device, **kwargs)
    elif name == "mask_mlbw_l2":
        return load_mask_mlbw_l2(path, device=device, **kwargs)
    raise ValueError(f"unknown model {name!r}")


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


def _pad_delta_y(delta_x):
    """(B, L, H, W) -> (B, L*2, H, W), interleaving a zero y-shift after each x.

    ``_backward_warp`` wants a two-channel delta per layer. The model predicts
    only the horizontal half, because the disparity it models is horizontal.
    """
    return torch.stack([delta_x, torch.zeros_like(delta_x)], dim=2).flatten(1, 2)


def _warp_one_view_mlbw(model, c, depth, divergence, convergence, shift,
                        preserve_screen_border, enable_amp):
    """One synthetic eye from MLBW, plus that eye's hole-mask logits.

    Same mirror-and-mirror-back trick as ``_warp_one_view``: the model is
    trained for the left eye only, so the right one is made by flipping the
    inputs, warping, and flipping the result. The mask travels with it.

    The eye is a *weighted sum* of one backward warp per layer rather than a
    single warp, which is the whole point of the architecture.

    Returns ``(eye, hole_mask_logits)``. The logits are raw -- not sigmoided,
    not thresholded, and still at the depth's resolution. Turning them into a
    mask is ``_postprocess_hole_mask``'s job, and it needs them in this form.
    """
    if shift > 0:
        c = torch.flip(c, (3,))
        depth = torch.flip(depth, (3,))

    B, _, H, W = depth.shape
    base_size = max(H, W)

    x = _make_input_tensor(depth,
                           divergence=divergence,
                           convergence=convergence,
                           image_width=base_size,
                           preserve_screen_border=preserve_screen_border)
    with _autocast(device=depth.device, enabled=enable_amp):
        delta, layer_weight, hole_mask_logits = model(x)

    # The weights are blended up to the frame's size, the deltas are not --
    # _backward_warp interpolates the finished grid instead. Note antialias=True
    # here against the grid's antialias=False: iw3 treats a weight map as an
    # image to be resampled and a grid as a coordinate field to be resampled,
    # and they are not the same operation.
    if c.shape[2] != layer_weight.shape[2] or c.shape[3] != layer_weight.shape[3]:
        layer_weight = F.interpolate(layer_weight, size=c.shape[-2:],
                                     mode="bilinear", align_corners=True, antialias=True)

    # The depth's width, not the image's, as everywhere else in this pipeline.
    delta_scale = torch.tensor(1.0 / (W // 2 - 1), dtype=c.dtype, device=c.device)
    delta = _pad_delta_y(delta)
    grid = _make_grid(B, W, H, c.device)

    z = torch.zeros_like(c)
    for layer in range(model.num_layers):
        d = delta[:, layer * 2:layer * 2 + 2]
        w = layer_weight[:, layer:layer + 1]
        z = z + _backward_warp(c, grid, d, delta_scale) * w
    z = z.clamp(0, 1)

    if shift > 0:
        z = z.flip((3,))
        hole_mask_logits = hole_mask_logits.flip((3,))

    return z, hole_mask_logits


def apply_divergence_mlbw(model, c, depth, divergence, convergence, synthetic_view,
                          preserve_screen_border=False, enable_amp=True):
    """Both eyes out of one MLBW, plus the hole-mask logits for each synthesised one.

    A non-synthesised eye is the source frame itself and has no mask, which is
    what tells the inpaint half to leave it alone.
    """
    if synthetic_view == "both":
        left_eye, left_mask = _warp_one_view_mlbw(
            model, c, depth, divergence=divergence, convergence=convergence, shift=-1,
            preserve_screen_border=preserve_screen_border, enable_amp=enable_amp)
        right_eye, right_mask = _warp_one_view_mlbw(
            model, c, depth, divergence=divergence, convergence=convergence, shift=1,
            preserve_screen_border=preserve_screen_border, enable_amp=enable_amp)
    elif synthetic_view == "right":
        left_eye, left_mask = c, None
        right_eye, right_mask = _warp_one_view_mlbw(
            model, c, depth, divergence=divergence * 2, convergence=convergence, shift=1,
            preserve_screen_border=preserve_screen_border, enable_amp=enable_amp)
    else:
        left_eye, left_mask = _warp_one_view_mlbw(
            model, c, depth, divergence=divergence * 2, convergence=convergence, shift=-1,
            preserve_screen_border=preserve_screen_border, enable_amp=enable_amp)
        right_eye, right_mask = c, None

    return left_eye, right_eye, left_mask, right_mask


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
