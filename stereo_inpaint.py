"""Standalone stereo synthesis by forward warp and inpainting.

This is iw3's ``monobw_inpaint`` method extracted into one file, the same way
``stereo_warp.py`` extracts ``row_flow_v2``/``row_flow_v3``. It depends on
PyTorch and on ``stereo_warp`` for four shape helpers -- no ``nunif`` imports,
no ``args`` object, every setting passed explicitly.

It is a *different pipeline* from ``stereo_warp``, not a different model::

    depth -> MonoBW warp -+-> warped eye --+
                          +-> hole mask ---+-> mask morphology -> LightInpaintV1 -> composite

``MonoBW`` forward-warps the frame and reports which pixels got stretched over
a hole; ``LightInpaintV1`` fills the holes. The first has no learned weights at
all, the second has 2.26M of them and runs at the colour frame's full
resolution, which is where the cost is. See ``docs/monobw-inpaint.md``.

Correspondence with the original, for anyone diffing the two:

    iw3/utils.py                     apply_divergence()        -> synthesize_stereo_inpaint()
    iw3/backward_warp.py             apply_divergence_monobw() -> _apply_divergence_monobw()
    iw3/monobw_inpaint.py            MonoBWInpaintImage        -+
    iw3/base_inpaint.py              BaseImageInpaint          -+-> MonoBWInpaintImage
    iw3/models/monobw.py             MonoBW                    -> MonoBW
    iw3/models/light_inpaint_v1.py   LightInpaintV1            -> LightInpaintV1
    iw3/models/light_inpaint_v1.py   GMLPBlock, GLUConvMLP     -> _GMLPBlock, _GLUConvMLP
    iw3/dilation.py                  dilate(), erode(), ...    -> _dilate(), _erode(), ...
    nunif/modules/attention.py       WindowGMLP2d, GMLP        -> _WindowGMLP2d, _GMLP
    nunif/modules/norm.py            FastLayerNorm             -> _FastLayerNorm
    nunif/modules/gaussian_filter.py GaussianFilter2d, ...     -> _GaussianFilter2d, ...

Arithmetic is kept expression-for-expression identical to the original, even
where it could be simplified, because ``tests/test_stereo_inpaint.py`` requires
a maximum absolute difference of exactly 0 against stock iw3.

The one deliberate departure is padding. iw3 pads with
``replication_pad2d_naive()``, which builds the padding by Python tuple
repetition -- ``(slice,) * n`` -- and that needs ``n`` to be a concrete int,
which bakes one frame size into an exported graph. ``F.pad(mode="replicate")``
takes the amount as data. Numerically identical, and it is the same change
``row_flow_v3`` needed; ``docs/row-flow-v3.md`` records why.

The video model is deliberately not ported. ``light_video_inpaint_v1`` is
driven through a frame queue with pre- and post-padding, and Resolve renders
frames out of order, with gaps and repeats -- see ``docs/phase0-findings.md``.
``light_inpaint_v1`` is per-frame and is the right target.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# The row_flow_v3 work already needed non-square pixel shuffle and window
# partitioning, and both are exercised by its ONNX export. Importing rather
# than copying keeps one definition of each.
from stereo_warp import (
    _autocast,
    _pixel_shuffle,
    _pixel_unshuffle,
    _window_merge,
    _window_partition,
    get_mapper,
)

__all__ = [
    "MonoBW",
    "LightInpaintV1",
    "MonoBWInpaintImage",
    "load_light_inpaint_v1",
    "load_monobw_inpaint",
    "synthesize_stereo_inpaint",
    "LIGHT_INPAINT_V1_URL",
]

_RELEASE = "https://github.com/nagadomi/nunif/releases/download/0.0.0/"
LIGHT_INPAINT_V1_URL = _RELEASE + "iw3_light_inpaint_v1_20250919.pth"

MODEL_NAME = "inpaint.light_inpaint_v1"


# ---------------------------------------------------------------------------
# gaussian filters
#
# From nunif/modules/gaussian_filter.py. The kernels are computed rather than
# loaded -- they are non-persistent buffers in the original too -- so the
# formula has to match, not just the shape.


def _gaussian_kernel1d(kernel_size, sigma=None):
    if kernel_size == 1:
        return torch.ones((kernel_size,))

    # The formula is torchvision's.
    if sigma is None:
        sigma = kernel_size * 0.15 + 0.35
    half = (kernel_size - 1) * 0.5
    x = torch.linspace(-half, half, steps=kernel_size)
    kernel = torch.exp(-0.5 * (x / sigma).pow(2))
    return kernel / kernel.sum()


def _gaussian_kernel2d(kernel_size, sigma=None):
    """``kernel_size`` is (height, width); either may be 1."""
    kernel_y = _gaussian_kernel1d(kernel_size[0], sigma=sigma)
    kernel_x = _gaussian_kernel1d(kernel_size[1], sigma=sigma)
    return torch.mm(kernel_y[:, None], kernel_x[None, :])


def _depthwise_weight(in_channels, kernel):
    """A 2D kernel as a depthwise conv weight, (C, 1, kh, kw)."""
    kernel = kernel.reshape(1, 1, *kernel.shape)
    return kernel.expand(in_channels, 1, *kernel.shape[2:]).contiguous()


class _GaussianFilter2d(nn.Module):
    """nunif's ``GaussianFilter2d``. ``padding`` is (left, right, top, bottom)."""

    def __init__(self, in_channels, kernel_size, padding):
        super().__init__()
        with torch.no_grad():
            kernel = _depthwise_weight(in_channels, _gaussian_kernel2d(kernel_size))
        self.register_buffer("kernel", kernel, persistent=False)
        self.padding = padding

    def forward(self, x):
        x = F.pad(x, self.padding, mode="replicate")
        return F.conv2d(x, weight=self.kernel, bias=None, groups=self.kernel.shape[0])


class _SeparableGaussianFilter2d(nn.Module):
    """nunif's ``SeparableGaussianFilter2d``. One pad, then a 1xK and a Kx1 pass."""

    def __init__(self, in_channels, kernel_size, padding):
        super().__init__()
        with torch.no_grad():
            kernel_h = _depthwise_weight(in_channels, _gaussian_kernel2d([1, kernel_size]))
            kernel_v = _depthwise_weight(in_channels, _gaussian_kernel2d([kernel_size, 1]))
        self.register_buffer("kernel_h", kernel_h, persistent=False)
        self.register_buffer("kernel_v", kernel_v, persistent=False)
        self.padding = (padding,) * 4

    def forward(self, x):
        x = F.pad(x, self.padding, mode="replicate")
        x = F.conv2d(x, weight=self.kernel_h, bias=None, groups=self.kernel_h.shape[0])
        x = F.conv2d(x, weight=self.kernel_v, bias=None, groups=self.kernel_v.shape[0])
        return x


# ---------------------------------------------------------------------------
# mask morphology
#
# Verbatim from iw3/dilation.py, minus the parts only the depth-estimation side
# uses. ``dilate_inner``/``dilate_outer`` are one-sided: they grow the mask
# towards or away from the direction the warp pulled from, and both assume the
# right-eye (unflipped) handedness the inpaint network is trained for.


def _dilate(mask, kernel_size=3):
    return F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)


def _erode(mask, kernel_size=3):
    return -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)


def _closing(mask, kernel_size=3, n_iter=2):
    mask = mask.float()
    for _ in range(n_iter):
        mask = _dilate(mask, kernel_size=kernel_size)
    for _ in range(n_iter):
        mask = _erode(mask, kernel_size=kernel_size)
    return mask


def _mask_closing(mask, kernel_size=3, n_iter=2):
    mask = mask_org = mask.float()
    mask = _closing(mask, kernel_size=kernel_size, n_iter=n_iter)
    # Put back the isolated pixels that closing erased.
    return (mask + mask_org).clamp(0, 1)


def _dilate_outer(mask, n_iter, base_width=None):
    if n_iter <= 0:
        return mask

    mask_dtype = mask.dtype
    mask = mask.bool()
    if base_width is not None:
        n_iter = max(round(mask.shape[-1] / base_width * n_iter), 1)
    for _ in range(n_iter):
        mask = mask | F.pad(mask, (1, 0, 0, 0))[:, :, :, :-1]
    return mask.to(mask_dtype)


def _dilate_inner(mask, n_iter, base_width=None):
    if n_iter <= 0:
        return mask

    mask_dtype = mask.dtype
    mask = mask.bool()
    if base_width is not None:
        n_iter = max(round(mask.shape[-1] / base_width * n_iter), 1)
    for _ in range(n_iter):
        mask = mask | F.pad(mask, (0, 1, 0, 0))[:, :, :, 1:]
    return mask.to(mask_dtype)


# ---------------------------------------------------------------------------
# MonoBW
#
# A forward warp with no learned weights at all. iw3's own header calls it "a
# non-ML, heuristic-based method that generates stereo images from depth with
# results very similar to iw3's row_flow_v3, but faster", and benchmarks it at
# 1800 FPS at FHD.
#
# What it does that a backward warp cannot: it knows where the holes are. The
# destination index map is made monotone with a cummax, so the mapping can be
# inverted; wherever that inverse had to stretch a pixel across a gap, the
# stretch mask marks it, and the inpaint model gets told to fill it.


class MonoBW(nn.Module):
    """iw3's ``sbs.monobw``, inference path only.

    No parameters and no checkpoint -- the only buffer is the smoothing
    kernel, which is computed in ``__init__`` and not persistent.
    """

    def __init__(self, smooth_kernel=9):
        super().__init__()
        if smooth_kernel > 0:
            pad = (smooth_kernel - 1) // 2
            self.smooth_filter = _GaussianFilter2d(
                in_channels=1,
                kernel_size=(1, smooth_kernel),
                padding=(pad, pad, 0, 0),
            )
        else:
            self.smooth_filter = None

    def smoothing(self, dest_index_fix, dest_index):
        """Blur the index map, but only where the monotonisation moved it.

        Mutates ``dest_index_fix`` in place, as the original does.
        """
        if self.smooth_filter is None:
            return dest_index_fix

        dest_index_mask = dest_index != dest_index_fix
        dest_index_mask = F.max_pool2d(dest_index_mask.float(), kernel_size=(1, 5),
                                       stride=1, padding=(0, 2)) > 0
        dest_index_fix[dest_index_mask] = self.smooth_filter(dest_index_fix)[dest_index_mask]
        return dest_index_fix

    @staticmethod
    def interpolate_1d(dest_index, src_index):
        """Invert a monotone 1D mapping, one row per batch element.

        ``searchsorted`` finds where each source index lands in the destination
        map, and the two neighbours are interpolated between. This is the step
        that turns a forward warp into something ``grid_sample`` can apply.
        """
        dest_index = dest_index.contiguous()
        src_index = src_index.contiguous()
        BH, W = dest_index.shape

        idx = torch.searchsorted(dest_index, src_index, right=False)
        idx0 = (idx - 1).clamp(0, W - 1)
        idx1 = idx.clamp(0, W - 1)

        d0 = torch.gather(dest_index, 1, idx0)
        d1 = torch.gather(dest_index, 1, idx1)
        s0 = torch.gather(src_index, 1, idx0)
        s1 = torch.gather(src_index, 1, idx1)

        denom = d1 - d0
        t = (src_index - d0) / (denom + 1e-5)
        return s0 + t * (s1 - s0)

    def compute_backward_grid(self, depth, divergence, convergence, border_pix=0):
        B, _, H, W = depth.shape
        dtype = depth.dtype
        device = depth.device

        if isinstance(convergence, float):
            convergence = torch.tensor(convergence, dtype=dtype, device=device).expand(B, 1).view(B, 1, 1, 1)
        else:
            if convergence.ndim != 4:
                convergence = convergence.view(B, 1, 1, 1)

        src_index = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, 1, H, W)

        base_size = max(H, W)
        delta_scale = base_size / W
        shift_size_px = divergence * (0.01 * delta_scale * (W - 1) * 0.5)
        index_shift = (depth - convergence) * shift_size_px

        if border_pix > 0:
            view_shape = [1] * (index_shift.ndim - 1) + [-1]
            border_weight_l = torch.linspace(0.0, 1.0, border_pix, dtype=dtype, device=device).view(view_shape)
            border_weight_r = torch.linspace(1.0, 0.0, border_pix, dtype=dtype, device=device).view(view_shape)
            index_shift[..., :border_pix] *= border_weight_l
            index_shift[..., -border_pix:] *= border_weight_r

        # Monotonisation: a cummax makes the mapping invertible, which is what
        # stops the warp folding over itself at a depth discontinuity.
        dest_index = src_index + index_shift
        dest_index_fix = torch.cummax(dest_index, dim=-1)[0]

        dest_index = self.smoothing(dest_index_fix, dest_index)

        src_index_flat = src_index.reshape(B * H, W)
        dest_index_flat = dest_index.view(B * H, W)
        index_back = self.interpolate_1d(dest_index_flat, src_index_flat)
        index_back = index_back.reshape(B, 1, H, W)

        grid_x = (index_back / (W - 1)) * 2.0 - 1.0
        mesh_y = torch.linspace(-1, 1, H, device=depth.device, dtype=dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        return torch.cat([grid_x, mesh_y], dim=1)

    @staticmethod
    def compute_stretch_mask(grid, threshold=0.5):
        """Mark pixels the inverse mapping had to stretch, i.e. the holes.

        A step of less than half a pixel between neighbouring grid entries
        means the same source pixel is being read twice or more.
        """
        W = grid.shape[-1]
        grid_x = grid[:, 0:1]
        diff_x = grid_x[..., 1:] - grid_x[..., :-1]
        threshold = (2.0 / (W - 1)) * threshold
        is_stretched = diff_x < threshold
        mask = torch.zeros_like(grid_x, dtype=torch.bool, device=grid.device)
        mask[..., :-1] |= is_stretched
        mask[..., 1:] |= is_stretched
        return mask

    def warp(self, x, grid):
        # iw3 guards this with `grid.shape[-2] != x.shape[-2:]`, an int against
        # a torch.Size, which is always true -- so the interpolate always runs,
        # even when the grid is already the right size. Kept unconditional
        # rather than "fixed", because the golden test compares against that.
        grid = F.interpolate(grid, size=x.shape[-2:], mode="bilinear", align_corners=True)
        grid = grid.permute(0, 2, 3, 1).to(x.dtype)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def forward(self, rgb, depth, divergence, convergence,
                preserve_screen_border=False,
                fix_screen_border_mask=1,  # 0: no fix, 1: the uninpaintable side, 2: both sides
                return_mask=False):
        assert fix_screen_border_mask in {0, 1, 2}
        if preserve_screen_border:
            image_width = rgb.shape[-1]
            depth_width = depth.shape[-1]
            border_pix = round(divergence * 0.75 * 0.01 * image_width * (depth_width / image_width))
        else:
            border_pix = 0

        # The coordinate arithmetic is done in float32 whatever the caller set,
        # because half precision loses index_back to rounding.
        with torch.autocast(device_type=depth.device.type, enabled=False):
            grid_bchw = self.compute_backward_grid(depth, divergence, convergence, border_pix=border_pix)
            # Same always-true guard as in warp(); see the note there.
            grid_bchw = F.interpolate(grid_bchw, size=rgb.shape[-2:], mode="bilinear", align_corners=True)
            warped_rgb = self.warp(rgb, grid_bchw)

            if return_mask:
                # Note this is the full-resolution grid, so the mask comes out
                # at the colour frame's size, not the depth's.
                mask = self.compute_stretch_mask(grid_bchw)
                if not preserve_screen_border and fix_screen_border_mask > 0:
                    # The screen border stretches for a reason that is not a
                    # hole, and inpainting it makes it worse.
                    image_width = rgb.shape[-1]
                    depth_width = depth.shape[-1]
                    border_pix = round(divergence * 0.01 * image_width * (depth_width / image_width)) + 1
                    mask[..., :border_pix] = False
                    if fix_screen_border_mask == 2:
                        mask[..., -border_pix:] = False
            else:
                mask = None

        if return_mask:
            return warped_rgb, mask
        return warped_rgb


# ---------------------------------------------------------------------------
# LightInpaintV1
#
# 2.26M parameters of gated MLP over shifted windows, running on the colour
# frame at full resolution. Module names and Sequential indices mirror iw3's
# exactly, so the published checkpoint loads with no key remapping.


class _FastLayerNorm(nn.LayerNorm):
    """nunif's ``FastLayerNorm``, the idea taken from timm.

    ``nn.LayerNorm`` is on autocast's fp32 list, so it upcasts. This one stays
    in the autocast dtype instead. That changes the numbers, so it has to be
    reproduced rather than replaced with a plain LayerNorm.
    """

    def forward(self, x):
        if torch.is_autocast_enabled(x.device.type):
            dtype = torch.get_autocast_dtype(x.device.type)
            x = x.to(dtype)
            weight = self.weight.to(dtype) if self.weight is not None else None
            bias = self.bias.to(dtype) if self.bias is not None else None
            with torch.amp.autocast(device_type=x.device.type, enabled=False):
                return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
        return super().forward(x)


class _GMLP(nn.Module):
    """A gated MLP block: project up, split, gate one half by a spatial mix."""

    def __init__(self, embed_dim, seq_len, mlp_ratio=1):
        super().__init__()
        self.proj_in = nn.Linear(embed_dim, int(embed_dim * mlp_ratio * 2))
        # Mixing across tokens is a 1x1 conv over the sequence dimension, which
        # is why the window size is baked into the weight shape.
        self.proj_spatial = nn.Conv1d(seq_len, seq_len, kernel_size=1, stride=1, bias=True)
        self.proj_out = nn.Linear(int(embed_dim * mlp_ratio * 2) // 2, embed_dim)

    def forward(self, x, norm1, norm2):
        shortcut = x
        x = norm1(x)
        x = self.proj_in(x)
        x = F.gelu(x)

        u, v = x.chunk(2, dim=-1)
        v = norm2(v)
        v = self.proj_spatial(v)
        x = u * v

        x = self.proj_out(x)
        return x + shortcut


class _WindowGMLP2d(nn.Module):
    """``_GMLP`` over square windows of a BCHW map, optionally shifted.

    The shift is a half-window zero pad on all four sides, which offsets every
    window boundary and is cropped off again afterwards -- the Swin trick, so
    tokens are not permanently cut off from their neighbours.
    """

    def __init__(self, in_channels, window_size, mlp_ratio=2, shift=False):
        super().__init__()
        self.window_size = (window_size, window_size)
        self.shift = shift
        self.pad_h = window_size // 2
        self.pad_w = window_size // 2
        self.gmlp = _GMLP(in_channels, seq_len=window_size * window_size, mlp_ratio=mlp_ratio)

    def forward(self, x, norm1, norm2):
        if self.shift:
            x = F.pad(x, (self.pad_w, self.pad_w, self.pad_h, self.pad_h), mode="constant", value=0)

        shape = x.shape
        x = _window_partition(x, self.window_size)
        x = self.gmlp(x, norm1, norm2)
        x = _window_merge(x, shape, self.window_size)

        if self.shift:
            x = F.pad(x, (-self.pad_w, -self.pad_w, -self.pad_h, -self.pad_h))
        return x


class _GLUConvMLP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, mlp_ratio=2):
        super().__init__()
        mid = int(out_channels * mlp_ratio)
        self.padding = ((kernel_size - 1) // 2,) * 4
        self.w1 = nn.Conv2d(in_channels, mid, kernel_size=1, stride=1, padding=0)
        self.w2 = nn.Conv2d(mid // 2, out_channels, kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        x = self.w1(x)
        x = F.glu(x, dim=1)
        x = F.pad(x, self.padding, mode="replicate")
        return self.w2(x)


class _GMLPBlock(nn.Module):
    def __init__(self, in_channels, window_size, mlp_ratio=2, shift=False, kernel_size=3):
        super().__init__()
        self.gmlp = _WindowGMLP2d(in_channels, window_size=window_size, shift=shift, mlp_ratio=mlp_ratio)
        self.norm1 = _FastLayerNorm(in_channels, bias=False)
        self.norm2 = _FastLayerNorm(in_channels * mlp_ratio, bias=False)
        self.glu_conv = _GLUConvMLP(in_channels, in_channels, mlp_ratio=1, kernel_size=kernel_size)

    def forward(self, x):
        x = x + self.gmlp(x, self.norm1, self.norm2)
        return x + self.glu_conv(x)


class LightInpaintV1(nn.Module):
    """iw3's ``inpaint.light_inpaint_v1``, inference path only.

    Input is the warped eye (B, 3, H, W) in 0..1 and a hole mask (B, 1, H, W);
    output is the eye with the holes filled and everything else passed through
    untouched, because the last step composites by the mask.

    The network is trained for one handedness only -- it expects the holes to
    open towards the same side every time -- which is why ``_inpaint_single()``
    flips the left eye, inpaints, and flips back. ``row_flow`` uses the same
    trick.
    """

    # An I2IBaseModel offset in iw3: forward() crops this many pixels from each
    # side, so 256x448 in gives 224x416 out. infer() below is the model's own
    # method, not I2IBaseModel's, and skips the crop.
    I2I_OFFSET = 16

    DOWNSCALING_FACTOR = 4
    MOD = 16

    def __init__(self):
        super().__init__()
        pack = self.DOWNSCALING_FACTOR ** 2
        C = 96
        C2 = C * 2
        # What a fully masked token becomes: the network's idea of "nothing
        # here yet", substituted for the packed pixels before the first block.
        self.mask_bias = nn.Parameter(torch.zeros(1, C, 1, 1))
        self.patch = nn.Sequential(
            nn.Conv2d(3 * pack, C, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc1 = _GMLPBlock(C, window_size=16, mlp_ratio=2, shift=True)
        self.down = nn.Conv2d(C, C2, kernel_size=2, stride=2, padding=0)
        self.enc2 = nn.Sequential(
            _GMLPBlock(C2, window_size=8, mlp_ratio=2, shift=False),
            _GMLPBlock(C2, window_size=8, mlp_ratio=2, shift=True),
            _GMLPBlock(C2, window_size=8, mlp_ratio=2, shift=False),
            _GMLPBlock(C2, window_size=8, mlp_ratio=2, shift=True),
        )
        self.up = nn.Conv2d(C2, C * 4, kernel_size=1, stride=1, padding=0)
        self.dec1 = _GMLPBlock(C, window_size=16, mlp_ratio=2, shift=False)
        self.to_image = nn.Sequential(
            nn.ReplicationPad2d((1,) * 4),
            nn.Conv2d(C, 3 * pack, kernel_size=3, stride=1, padding=0),
        )
        self.mask_blur = _SeparableGaussianFilter2d(1, kernel_size=15, padding=15 // 2)

    def preprocess(self, x, mask, closing=False, inner_dilation=0, outer_dilation=0, base_width=None):
        """Blank the holes and feather the mask.

        The feathered mask is what the composite blends by, so the seam is not
        a hard edge; the unfeathered one is what tells the network which
        tokens carry no information.
        """
        if closing:
            mask = _mask_closing(mask)
        else:
            mask = mask.float()

        mask = _dilate_inner(mask, n_iter=inner_dilation, base_width=base_width)
        mask = _dilate_outer(mask, n_iter=outer_dilation, base_width=base_width)

        x = x * (1 - mask)
        mask = torch.clamp(self.mask_blur(mask) + mask, 0, 1)
        return x, mask

    def infer(self, x, mask, closing=False, inner_dilation=0, outer_dilation=0, base_width=None):
        """The model's own entry point. Note it skips the i2i offset crop."""
        x, mask = self.preprocess(x, mask, closing=closing,
                                  inner_dilation=inner_dilation, outer_dilation=outer_dilation,
                                  base_width=base_width)
        return self.forward(x, mask, skip_i2i_offset=True)

    def _forward(self, x, mask):
        x = _pixel_unshuffle(x, (self.DOWNSCALING_FACTOR,) * 2)
        x = self.patch(x)

        # A packed token counts as a hole only if every pixel in it is one.
        mask = _pixel_unshuffle(mask, (self.DOWNSCALING_FACTOR,) * 2).amax(dim=1, keepdim=True) > 0.99
        x = torch.where(mask, self.mask_bias.to(x.dtype), x)

        x1 = self.enc1(x)
        x2 = self.down(x1)
        x2 = self.enc2(x2)
        x2 = self.up(x2)
        x2 = _pixel_shuffle(x2, (2, 2))
        x = self.dec1(x1 + x2)
        x = self.to_image(x)
        return _pixel_shuffle(x, (self.DOWNSCALING_FACTOR,) * 2)

    def forward(self, x, mask, skip_i2i_offset=False):
        src = x

        x = (x - 0.5) / 0.5

        input_height, input_width = x.shape[2:]
        # Pad up to a multiple of 64, always by at least one pixel: an exact
        # multiple still gets a whole extra block. That is iw3's arithmetic and
        # the network was trained with it.
        pad1 = (self.MOD * self.DOWNSCALING_FACTOR) - input_width % (self.MOD * self.DOWNSCALING_FACTOR)
        pad2 = (self.MOD * self.DOWNSCALING_FACTOR) - input_height % (self.MOD * self.DOWNSCALING_FACTOR)
        padding = (0, pad1, 0, pad2)
        x = F.pad(x, padding, mode="replicate")
        mask = F.pad(mask, padding, mode="replicate")

        x = self._forward(x, mask)
        x = F.pad(x, (0, -pad1, 0, -pad2))
        mask = F.pad(mask, (0, -pad1, 0, -pad2))

        if not skip_i2i_offset:
            src = F.pad(src.to(x.dtype), (-self.I2I_OFFSET,) * 4)
            mask = F.pad(mask, (-self.I2I_OFFSET,) * 4)
            x = F.pad(x, (-self.I2I_OFFSET,) * 4)

        mask = mask.expand_as(src)
        src = src * (1 - mask) + x * mask

        # iw3 guards this with `if not self.training`; this is an inference-only
        # port, so it always applies.
        return src.clamp(0, 1)


def load_light_inpaint_v1(path=None, device="cpu"):
    """Load the published ``light_inpaint_v1`` checkpoint.

    ``path`` may be a local ``.pth`` file or ``None`` to download the release
    build to the torch hub cache. Returns a ``LightInpaintV1`` in eval mode.
    """
    if path is None:
        data = torch.hub.load_state_dict_from_url(LIGHT_INPAINT_V1_URL, weights_only=True, map_location="cpu")
    else:
        data = torch.load(path, map_location="cpu", weights_only=True)

    if "nunif_model" not in data:
        raise ValueError("not a nunif checkpoint")
    if data.get("name") != MODEL_NAME:
        raise ValueError(f"expected {MODEL_NAME}, got {data.get('name')!r}")

    model = LightInpaintV1()
    model.load_state_dict(data["state_dict"], strict=True)
    return model.eval().to(device)


# ---------------------------------------------------------------------------
# the pipeline


def _apply_divergence_monobw(model, c, depth, divergence, convergence, synthetic_view,
                             preserve_screen_border, fix_screen_border_mask):
    """Both eyes out of one MonoBW, plus the hole mask for each synthesised one.

    The right eye is produced by mirroring, warping, and mirroring back, the
    same as ``stereo_warp``'s ``_warp_one_view``. The mask travels with it.
    """
    if synthetic_view == "both":
        left_eye = model(c, depth, divergence=divergence, convergence=convergence,
                         preserve_screen_border=preserve_screen_border,
                         fix_screen_border_mask=fix_screen_border_mask,
                         return_mask=True)
        right_eye = model(c.flip(dims=[-1]), depth.flip(dims=[-1]),
                          divergence=divergence, convergence=convergence,
                          preserve_screen_border=preserve_screen_border,
                          fix_screen_border_mask=fix_screen_border_mask,
                          return_mask=True)
    elif synthetic_view == "right":
        left_eye = c
        right_eye = model(c.flip(dims=[-1]), depth.flip(dims=[-1]),
                          divergence=divergence * 2, convergence=convergence,
                          preserve_screen_border=preserve_screen_border,
                          fix_screen_border_mask=fix_screen_border_mask,
                          return_mask=True)
    else:
        left_eye = model(c, depth, divergence=divergence * 2, convergence=convergence,
                         preserve_screen_border=preserve_screen_border,
                         fix_screen_border_mask=fix_screen_border_mask,
                         return_mask=True)
        right_eye = c

    left_mask = right_mask = None
    if isinstance(left_eye, tuple):
        left_eye, left_mask = left_eye
    if isinstance(right_eye, tuple):
        right_eye, right_mask = right_eye
    if synthetic_view in {"both", "right"}:
        right_eye = right_eye.flip(dims=[-1])
        right_mask = right_mask.flip(dims=[-1])

    return left_eye, right_eye, left_mask, right_mask


class MonoBWInpaintImage(nn.Module):
    """iw3's ``MonoBWInpaintImage``, with ``BaseImageInpaint`` folded in.

    Holds the two halves of the pipeline: a ``MonoBW`` with no weights and a
    ``LightInpaintV1`` with 2.26M of them.
    """

    def __init__(self, model, device="cpu"):
        super().__init__()
        self.model = model
        self.monobw = MonoBW().eval().to(device)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.eval()

    @staticmethod
    def _resize(x, max_width):
        """iw3's ``inpaint_max_width``: cap the resolution the inpaint runs at.

        Note ``new_h`` is derived from the rounded-up ``max_width`` against the
        *original* width, which is iw3's arithmetic, not a simplification of it.
        """
        if max_width is not None and x.shape[-1] > max_width:
            if max_width % 2 != 0:
                max_width += 1
            new_w = max_width
            new_h = int((max_width / x.shape[-1]) * x.shape[-2])
            if new_h % 2 != 0:
                new_h += 1
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", antialias=True, align_corners=False)
        return x

    def apply_warp(self, x, depth, divergence, convergence, synthetic_view, preserve_screen_border=False):
        return _apply_divergence_monobw(
            self.monobw, x, depth,
            divergence=divergence,
            convergence=convergence,
            synthetic_view=synthetic_view,
            preserve_screen_border=preserve_screen_border,
            fix_screen_border_mask=1,  # fix the uninpaintable side
        )

    @staticmethod
    def preprocess_mask(mask, target_size, inner_dilation=0, outer_dilation=0, base_width=None):
        if mask.shape[-2:] != target_size:
            mask = F.interpolate(mask, size=target_size, mode="nearest")
        mask = mask > 0
        mask = _mask_closing(mask)
        mask = _dilate_outer(mask, n_iter=outer_dilation, base_width=base_width)
        mask = _dilate_inner(mask, n_iter=inner_dilation, base_width=base_width)
        return mask

    def _inpaint_single(self, eye, mask, is_left, inner_dilation=0, outer_dilation=0, base_width=None):
        # The network is trained for one handedness, so the left eye goes
        # through mirrored and comes back.
        if is_left:
            eye, mask = eye.flip(-1), mask.flip(-1)

        mask = self.preprocess_mask(
            mask,
            target_size=eye.shape[-2:],
            inner_dilation=inner_dilation,
            outer_dilation=outer_dilation,
            base_width=base_width,
        )
        eye = self.model.infer(eye, mask)

        if is_left:
            eye = eye.flip(-1)
        return eye

    def _inpaint(self, left_eye, right_eye, left_mask, right_mask, synthetic_view,
                 inner_dilation=0, outer_dilation=0, base_width=None):
        if synthetic_view in {"both", "left"}:
            left_eye = self._inpaint_single(
                left_eye, left_mask, is_left=True,
                inner_dilation=inner_dilation, outer_dilation=outer_dilation, base_width=base_width)
        if synthetic_view in {"both", "right"}:
            right_eye = self._inpaint_single(
                right_eye, right_mask, is_left=False,
                inner_dilation=inner_dilation, outer_dilation=outer_dilation, base_width=base_width)
        return left_eye, right_eye

    def forward(self, x, depth, divergence, convergence, synthetic_view="both",
                inner_dilation=0, outer_dilation=0, preserve_screen_border=False):
        left_eye, right_eye, left_mask, right_mask = self.apply_warp(
            x, depth,
            divergence=divergence,
            convergence=convergence,
            synthetic_view=synthetic_view,
            preserve_screen_border=preserve_screen_border,
        )
        return self._inpaint(
            left_eye, right_eye, left_mask, right_mask, synthetic_view,
            inner_dilation=inner_dilation,
            outer_dilation=outer_dilation,
            # The dilation counts are quoted against the depth's width, so the
            # same setting means the same thing at any output resolution.
            base_width=depth.shape[-1],
        )

    def infer(self, x, depth, divergence, convergence, synthetic_view="both", max_width=None,
              inner_dilation=0, outer_dilation=0, preserve_screen_border=False, enable_amp=True):
        """One frame. ``x`` and ``depth`` are batched, and iw3 only ever passes B=1."""
        with _autocast(self.device, enabled=enable_amp):
            x = self._resize(x, max_width)
            return self(
                x, depth,
                divergence=divergence,
                convergence=convergence,
                synthetic_view=synthetic_view,
                inner_dilation=inner_dilation,
                outer_dilation=outer_dilation,
                preserve_screen_border=preserve_screen_border,
            )


def load_monobw_inpaint(path=None, device="cpu"):
    """The whole pipeline: a ``MonoBW`` plus a loaded ``LightInpaintV1``."""
    return MonoBWInpaintImage(load_light_inpaint_v1(path, device=device), device=device)


def synthesize_stereo_inpaint(
    image,
    depth,
    model,
    *,
    divergence=2.0,
    convergence=0.5,
    synthetic_view="both",
    max_width=None,
    inner_dilation=0,
    outer_dilation=0,
    preserve_screen_border=False,
    enable_amp=True,
    mapper=None,
):
    """Turn one colour frame and one depth map into a left/right pair.

    The same contract as ``stereo_warp.synthesize_stereo()``, with the settings
    that only apply to this pipeline swapped in: no ``stereo_width`` and no
    ``steps`` (iw3 does not resize depth for the inpaint methods, and the warp
    is single-shot), plus ``max_width`` and the two dilations.

    Args:
        image: float32, RGB, 0..1, CHW or BCHW.
        depth: float32, 0..1, 1HW or B1HW. Larger is nearer. Need not match
            ``image``'s resolution -- the warp is built at the depth's size and
            interpolated up.
        model: a ``MonoBWInpaintImage`` from ``load_monobw_inpaint()``, on the
            same device.
        divergence: stereo strength, as a percentage of image width. iw3's
            default is 2.0.
        convergence: the depth value that lands on the screen plane. 0 puts
            everything behind the screen, 1 everything in front.
        synthetic_view: ``"both"`` synthesises both eyes at half strength each;
            ``"left"``/``"right"`` keeps the original as the other eye and
            doubles the divergence.
        max_width: cap the width the inpaint runs at. The output is that size,
            not the input's -- this resizes the picture, it does not tile it.
        inner_dilation: grow the hole mask towards the occluder.
        outer_dilation: grow it the other way. Both are counted against the
            depth's width, so they scale with output resolution.
        preserve_screen_border: taper parallax to zero at the left and right
            edges. Also disables the screen-border mask fix, since with the
            parallax tapered there is no border stretch to suppress.
        enable_amp: autocast the inpaint model. No effect on CPU, which iw3
            leaves disabled because it is slow there. The warp itself is always
            float32.
        mapper: iw3 mapper name to apply to ``depth`` first, or ``None`` to use
            the depth as-is. ``None`` and ``"none"`` are the same function.

    Returns:
        ``(left, right)``, each matching ``image``'s shape and dtype -- unless
        ``max_width`` resized it.
    """
    if synthetic_view not in {"both", "left", "right"}:
        raise ValueError(f"synthetic_view={synthetic_view!r}")
    if torch.is_tensor(convergence):
        raise TypeError("convergence must be a float; per-frame convergence is not supported here")

    batched = image.ndim == 4
    if not batched:
        image = image.unsqueeze(0)
        depth = depth.unsqueeze(0)

    depth = get_mapper(mapper)(depth)

    # iw3 runs the inpaint methods one frame at a time rather than batching
    # them, and batched inference is not bit-identical to a loop -- different
    # sizes pick different kernels. Reproduced, not tidied.
    left_eyes = []
    right_eyes = []
    for i in range(depth.shape[0]):
        left_eye, right_eye = model.infer(
            image[i:i + 1], depth[i:i + 1],
            divergence=divergence,
            convergence=float(convergence),
            synthetic_view=synthetic_view,
            max_width=max_width,
            inner_dilation=inner_dilation,
            outer_dilation=outer_dilation,
            preserve_screen_border=preserve_screen_border,
            enable_amp=enable_amp,
        )
        left_eyes.append(left_eye)
        right_eyes.append(right_eye)

    if len(left_eyes) == 1:
        left, right = left_eyes[0], right_eyes[0]
    else:
        left = torch.cat(left_eyes, dim=0)
        right = torch.cat(right_eyes, dim=0)

    if not batched:
        left = left.squeeze(0)
        right = right.squeeze(0)

    return left, right
