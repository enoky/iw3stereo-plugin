"""Stage 0 probe: does shifted-window attention survive ONNX export at dynamic shapes?

Instrumentation, not product. No fidelity to any checkpoint -- random weights,
right *shape*.

The question is narrow. ``mask_mlbw_l2``'s WABlock is ``row_flow_v3``'s block
plus ``shift``, which pads half a window on each side, partitions, and crops
back. ``row_flow_v3``'s blocks never shift, so that padding-and-cropping around
a window partition had never been through the exporter here, and
``docs/monobw-inpaint.md`` predicted it would be the problem.

It is not. See ``docs/mlbw-inpaint-plan.md`` for the answer and the numbers.

Runs four variants, because "does it export" is really two questions crossed:
shift on or off, and SDPA against the head-sliced attention ``row_flow_v3``
needed. Export each, then run at six sizes and compare against PyTorch.

    F:/_AI_PROJECTS_/nunif/venv/Scripts/python.exe tools/probe_shifted_window.py

Set PYTHONIOENCODING=utf-8 first, or the exporter's own progress ticks kill it
on a cp1252 console before it reports anything.
"""
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stereo_warp import (  # noqa: E402
    _MHA, _WindowScoreBias, _window_partition, _window_merge,
    _pixel_shuffle, _pixel_unshuffle,
)

OPSET = 20
WINDOW = (4, 4)
CHANNELS = 64          # base_dim 32 * num_layers 2
DOWNSCALE = (1, 8)
MOD = 4


class ShiftedWindowMHA2d(nn.Module):
    """WindowMHA2d with nunif's ``shift``, which is padding rather than a roll."""

    def __init__(self, channels, num_heads, window, shift):
        super().__init__()
        self.window = window
        self.shift = shift
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


class WABlock(nn.Module):
    def __init__(self, channels, window, shift):
        super().__init__()
        self.mha = ShiftedWindowMHA2d(channels, num_heads=2, window=window, shift=shift)
        self.conv_mlp = nn.Sequential(*[
            nn.Conv2d(channels, channels, kernel_size=1, padding=0),
            nn.GELU(),
            nn.ReplicationPad2d((1, 1, 1, 1)),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
        ])
        self.bias = _WindowScoreBias(window)

    def forward(self, x, export_safe=False):
        x = x + self.mha(x, attn_mask=self.bias(), export_safe=export_safe)
        return x + self.conv_mlp(x)


class Probe(nn.Module):
    """The shape of MLBW's body: pad, unshuffle, four blocks, shuffle, crop.

    Same alternation of shifted and unshifted blocks the real one has, and the
    same symmetric padding, because the interaction between "pad the input by
    pad//2 each side" and "pad the windows by 2 each side" is exactly what could
    go wrong.
    """

    def __init__(self, shift, export_safe):
        super().__init__()
        self.export_safe = export_safe
        pack = DOWNSCALE[0] * DOWNSCALE[1]
        self.lv1_in = nn.Sequential(*[
            nn.ReplicationPad2d((4, 4, 0, 0)),
            nn.Conv2d(3, CHANNELS // pack, kernel_size=(1, 9), padding=0),
            nn.LeakyReLU(0.2, inplace=False),
        ])
        on = (True, True) if shift else (False, False)
        self.lv2 = nn.ModuleList([
            WABlock(CHANNELS, WINDOW, shift=on),
            WABlock(CHANNELS, WINDOW, shift=(False, False)),
            WABlock(CHANNELS, WINDOW, shift=on),
            WABlock(CHANNELS, WINDOW, shift=(False, False)),
        ])
        self.lv1_out = nn.Sequential(*[
            nn.ReplicationPad2d((4, 4, 0, 0)),
            nn.Conv2d(CHANNELS // pack, 5, kernel_size=(1, 9), padding=0),
        ])

    def forward(self, x):
        height, width = x.shape[2], x.shape[3]
        mod_h = MOD * DOWNSCALE[0]
        mod_w = MOD * DOWNSCALE[1]
        pad_w = mod_w - width % mod_w
        pad_h = mod_h - height % mod_h
        # Symmetric, the way MLBW._calc_pad does it outside training.
        pad_w1, pad_w2 = pad_w // 2, pad_w - pad_w // 2
        pad_h1, pad_h2 = pad_h // 2, pad_h - pad_h // 2
        x = F.pad(x, (pad_w1, pad_w2, pad_h1, pad_h2), mode="replicate")

        x = x1 = self.lv1_in(x)
        x = _pixel_unshuffle(x, DOWNSCALE)
        for block in self.lv2:
            x = block(x, export_safe=self.export_safe)
        x = _pixel_shuffle(x, DOWNSCALE)
        x = self.lv1_out(x + x1)
        return x[:, :, pad_h1:pad_h1 + height, pad_w1:pad_w1 + width]


def export(model, path):
    x = torch.rand(2, 3, 108, 192)
    batch = torch.export.Dim("batch", min=1, max=64)
    height = torch.export.Dim("height", min=16, max=8192)
    width = torch.export.Dim("width", min=16, max=8192)
    torch.onnx.export(
        model, (x,), path,
        input_names=["x"], output_names=["y"],
        dynamic_shapes={"x": {0: batch, 2: height, 3: width}},
        opset_version=OPSET, dynamo=True, external_data=False)


SIZES = [(108, 192), (392, 938), (392, 940), (384, 960), (100, 200), (528, 940)]


def check(model, path):
    import onnxruntime as ort
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    worst = 0.0
    for height, width in SIZES:
        generator = torch.Generator().manual_seed(height * 10000 + width)
        x = torch.rand(1, 3, height, width, generator=generator)
        with torch.inference_mode():
            expected = model(x).numpy()
        got = session.run(None, {"x": x.numpy()})[0]
        if got.shape != expected.shape:
            print(f"    {height}x{width}: SHAPE {got.shape} != {expected.shape}")
            worst = float("inf")
            continue
        diff = float(np.abs(got - expected).max())
        worst = max(worst, diff)
        print(f"    {height}x{width}: {diff:.3e}")
    return worst


def main():
    root = tempfile.mkdtemp(prefix="probe_shifted_window_")
    results = {}
    for shift in (False, True):
        for export_safe in (False, True):
            name = f"shift={shift} export_safe={export_safe}"
            print(f"\n=== {name}")
            torch.manual_seed(72)
            model = Probe(shift=shift, export_safe=export_safe).eval()
            path = os.path.join(root, f"probe_shift{int(shift)}_safe{int(export_safe)}.onnx")
            try:
                export(model, path)
            except Exception as error:
                print(f"    EXPORT FAILED: {type(error).__name__}: {str(error)[:400]}")
                results[name] = "export failed"
                continue
            try:
                results[name] = check(model, path)
            except Exception as error:
                print(f"    RUN FAILED: {type(error).__name__}: {str(error)[:400]}")
                results[name] = "run failed"

    print("\n=== summary")
    for name, value in results.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
