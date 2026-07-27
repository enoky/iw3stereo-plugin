# Phase 2 — ONNX

Deliverables: `models/row_flow_v2.onnx`, `models/stereo_warp.onnx`,
`stereo_warp_onnx.py`, and the timing table below.

Measured on an RTX 5080, Windows 11, onnxruntime-gpu 1.28.0 and
onnxruntime-directml 1.24.4, opset 17.

## What went into the graph

`stereo_warp.onnx` is the whole warp in one call: the network run twice (once
mirrored for the right eye), the sampling grid, the grid's interpolation up to
image resolution, `grid_sample`, and the clamp. One `session.run` per frame
produces both eyes.

Three things stayed outside, each for its own reason:

| Outside the graph | Why |
| --- | --- |
| the depth mapper | twenty variants, each would need its own graph; it is elementwise and costs nothing |
| `preserve_screen_border` | its ramp width is `round()`ed from divergence at runtime — control flow, not arithmetic |
| the `stereo_width` resize | **ONNX Resize cannot reproduce it.** See below. |

Because `preserve_screen_border` has to be applied to the feature channels, the
graph takes the model's input tensor `x` (disparity, divergence, convergence)
rather than building it from depth internally. That costs one upload of two
cheap channels and buys back the whole feature.

`grid_sample` **does** export at opset 17 and runs correctly on the CUDA and CPU
providers, so the `cv2.remap` fallback the plan allowed for was not needed.

## ONNX Resize will not do PyTorch's antialiased downscale

iw3 resizes depth with `F.interpolate(..., align_corners=True, antialias=True)`,
and that resize is load-bearing: it is what stops full-resolution depth from
striping the warp.

- Exported with `antialias=0`, the graph matches torch's **plain** resize to
  **5.5e-5**. The export itself is fine.
- Patched to `antialias=1` — nunif's own `patch_resize_antialias` technique, and
  a real opset 18 attribute that survives into the file — the output sits
  **0.23** away from torch on a 0..1 signal. With `align_corners=False` it
  improves to 0.054, still useless. ORT's antialias is simply a different filter.

So it is reimplemented in `stereo_warp_onnx.py` as a separable triangle filter
with cached per-axis weights, matching torch to **~1e-5**. The one non-obvious
detail, which costs 0.2 if you get it wrong:

```python
center = scale * (i + 0.5)   # even when align_corners=True
```

torch's *plain* align_corners path uses `scale * i`, but its **antialias** path
uses `scale * (i + 0.5)` regardless. The weights only change when a size
changes — once per timeline in practice — so caching them makes this free.

## Execution providers

Checked against the PyTorch reference at five shapes (matched, small depth,
square depth, 1080p, odd dimensions), via `tools/check_ort.py`.

| Provider | Network alone | Whole pipeline | Verdict |
| --- | --- | --- | --- |
| CUDA, `use_tf32=0` | **3.96e-6** | 3.03e-4 | **correct — what to ship** |
| CPU | 2.38e-6 | 3.03e-4 | correct, far too slow |
| CUDA, default (TF32 on) | 1.50e-3 | 2.03e-3 | precision loss, see below |
| TensorRT | 6.26e-3 | 6.50e-3 | precision loss, unconstrained |
| **DirectML** | **3.38** | **0.996** | **wrong — refused in code** |

Three findings worth keeping:

**TF32 is on by default and costs three orders of magnitude.** The CUDA EP takes
the network from 3.96e-6 to 1.50e-3 to save about 0.8 ms. `StereoWarpSession`
passes `use_tf32=0`.

**DirectML miscomputes `row_flow_v2` itself**, by 3.38 absolute on a 0..1
signal, in a clean isolated venv. It is not precision and it is not the warp:
the grid build and `grid_sample` were checked separately on DML and are both
fine to 1.2e-7. `stereo_warp_onnx.py` raises rather than use it, because a
silently corrupt result is worse than no result. This is a **shipping
constraint**: the GPU path is NVIDIA-only unless someone finds what DML is doing
to the convolution stack.

**The pipeline residual is not the provider.** CUDA and CPU agree on it to the
digit, so it is the exported graph differing from PyTorch — the grid
interpolation and `grid_sample` kernels — not an execution problem. It scales
with how far the grid is upsampled and with how adversarial the depth is: 3.0e-4
on the reference set, and 7.3e-4 when the same case was measured at true 1080p
on pure noise depth. Against the Phase 1 PyTorch implementation on realistic
content the whole pipeline agrees to **2e-4**, which is what
`tests/test_stereo_warp_onnx.py` enforces.

## Timing

Wall time per frame, both eyes, one `session.run`, mean of 30 after warm-up.
Run-to-run variation is roughly ±15%.

| Configuration | CUDA (tf32 off) | TensorRT | DirectML | CPU |
| --- | --- | --- | --- | --- |
| 1080p, depth 960x540 | **13 ms** | 10 ms | 14 ms* | 258 ms |
| 1080p, depth 1920x1080 | 42 ms | 20 ms | 31 ms* | 610 ms |
| 4K, depth 1920x1080 | 59 ms | 36 ms | 56 ms* | 936 ms |
| 4K, depth 3840x2160 | 161 ms | 85 ms | 100 ms* | 2353 ms |

\* DirectML timings are for a provider that returns wrong pixels. Listed only to
show it would not have been worth it anyway.

**The number Phase 3 was gated on: ~13 ms for a 1080p frame** with depth at
960x540, which is the configuration `stereo_width` exists to produce. That is
about 75 fps of warp — comfortably interactive, and it leaves real headroom
under Resolve's own per-frame budget.

Reading the table:

- **Depth resolution dominates, not image resolution.** 1080p goes from 13 ms to
  42 ms purely by handing it full-resolution depth instead of half. Setting
  `stereo_width` is a 3x performance decision as well as a quality one.
- **4K is not interactive but is renderable** at ~17 fps with half-resolution
  depth. Frame caching matters there; at 1080p it is close to optional.
- **TensorRT is meaningfully faster** but currently ~6.6e-3 off. It builds its
  own reduced-precision plans; constraining it to fp32 is worth trying if 4K
  interactivity becomes a goal, and is not worth it otherwise.

## Not implemented

`steps > 1` needs the warp fed back into itself, which wants a loop in the
graph. iw3 defaults to 1 and the plugin has no reason to change it. The PyTorch
implementation in `stereo_warp.py` supports it and is tested at steps 1-3.
