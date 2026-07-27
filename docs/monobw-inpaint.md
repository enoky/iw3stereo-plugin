# monobw_inpaint + light_inpaint_v1

Investigation only. Nothing is implemented yet; this is the map that makes the
implementation session efficient, and the numbers that decide whether it is
worth having.

## What it is, and why it is not a model swap

`row_flow_v2` and `row_flow_v3` are interchangeable because the pipeline around
them is identical. `monobw_inpaint` is a **different pipeline**:

```
depth ─> MonoBW warp ─┬─> warped eye ─┐
                      └─> hole mask ──┴─> mask morphology ─> LightInpaintV1 ─> composite
```

Forward-warp-and-fill rather than backward-warp. Two stages, two very different
kinds of work, and a mask travelling between them.

## The two halves are nothing alike

**MonoBW has no learned weights.** Its own header says so: *"a non-ML,
heuristic-based method that generates stereo images from depth with results very
similar to iw3's row_flow_v3, but faster"*. It is pure tensor arithmetic, so
nothing to export — but everything to port. Its notable operations:

| Operation | Note for a CUDA port |
| --- | --- |
| `torch.cummax` along width | an inclusive scan, one per row |
| `torch.searchsorted` + `gather` | binary search per element, for the inverse mapping |
| Gaussian smoothing of the index map | separable, kernel is a fixed buffer |
| `dest_index_fix[mask] = ...` | boolean masked assignment; trivial in a kernel |
| `compute_stretch_mask` | neighbour difference against a threshold |

It is also cheap: iw3 benchmarks it at 1800 FPS at FHD.

**LightInpaintV1 is 2.26M parameters** — 75x `row_flow_v2`, 19x `row_flow_v3` —
and runs on the **colour frame at full resolution**, not on downsampled depth.
That is what makes it expensive.

## The number that decides it

Measured on an RTX 5080, PyTorch, the inpaint model alone:

| | per eye | both eyes |
| --- | --- | --- |
| 1920x1036 | 24.2 ms | **48.4 ms** |
| 1080p | 26.1 ms | 52.1 ms |
| 4K | 116.2 ms | 232.4 ms |

Against the ~5 ms the current warp costs end to end, HD is roughly **10x
slower** — about 15 fps rather than 200 — and 4K becomes render-only. And that
is the inpaint model on its own, before the warp, the mask work, or any
ONNX and packing overhead.

The trade is real: inpainting fixes occlusion artifacts that backward warping
cannot, because it has something to put in the holes rather than smearing an
edge. Whether that is worth 10x is a judgement to make on real footage, not on
a benchmark.

## The export blocker, already located

Same class of problem as `row_flow_v3`, and expected to need the same kind of
hunt:

```
dynamo export: OK
  256x448: max abs diff 1.281e-06        <- the traced size
  392x938: FAILED  Expand node ...       <- every other size
  260x452: FAILED  Expand node ...
```

It exports and is accurate where it was traced, and fails everywhere else on an
`Expand` node with a baked dimension. `row_flow_v3` needed three separate fixes
to clear this class of problem — the attention head merge, the padding, and a
batch-1 specialisation. How many this needs is unknown; one is located, and
`WindowGMLP2d`'s shifted window partitioning is the obvious suspect for the
rest.

## The port inventory

For the standalone PyTorch stage, with no nunif imports and a diff-0 bar:

| From | Needs |
| --- | --- |
| `iw3/models/monobw.py` | `MonoBW`, ~150 lines with its Gaussian filter |
| `iw3/dilation.py` | `dilate`, `erode`, `closing`, `mask_closing`, `dilate_inner`, `dilate_outer` |
| `iw3/models/light_inpaint_v1.py` | `LightInpaintV1`, `GLUConvMLP`, `GMLPBlock` |
| `nunif/modules/attention.py` | `WindowGMLP2d`, `GMLP` |
| `nunif/modules/norm.py` | `FastLayerNorm` (LayerNorm plus an autocast dtype rule that matters for diff 0) |
| `nunif/modules/gaussian_filter.py` | `GaussianFilter2d`, `SeparableGaussianFilter2d`, the kernel builders |
| `iw3/monobw_inpaint.py`, `iw3/base_inpaint.py` | `MonoBWInpaintImage`, `preprocess_mask`, `_inpaint_single`, `_inpaint` |

Roughly 530 lines, all of which must be exact.

Already available from the `row_flow_v3` work: the non-square pixel
shuffle/unshuffle and the window partition and merge helpers, in
`stereo_warp.py`.

## Two things to get right

**Only the image inpaint model fits Resolve.** There is also
`light_video_inpaint_v1`, which `MonoBWInpaintVideo` drives through a frame
queue with `pre_padding` and `post_padding`. Resolve renders frames out of
order, with gaps and repeats — measured in `docs/phase0-findings.md` — so a
temporal queue is a poor fit. `light_inpaint_v1` is per-frame and is the right
target.

**The offset convention.** `LightInpaintV1` has `i2i_offset = 16`, so its
`forward()` crops 16 pixels from each side; 256x448 in gives 224x416 out. The
plugin has to pad before and account for that. Note `LightInpaintV1.infer()` is
the model's own method, not the `I2IBaseModel` one, and calls `forward()` with
`skip_i2i_offset=True`.

Also worth knowing: `_inpaint_single()` flips the left eye horizontally, inpaints,
and flips back — the network is trained for one handedness only, the same trick
`row_flow` uses.

## Not decided

Whether to build it at all past the standalone stage. The staged plan was:
PyTorch first with a golden test at difference 0, judge the quality against the
10x cost on real footage, then decide about CUDA and ONNX.
