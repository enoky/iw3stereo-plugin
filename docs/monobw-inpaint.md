# monobw_inpaint + light_inpaint_v1

The standalone PyTorch stage is **done**: `stereo_inpaint.py`, 22/22 at max
absolute difference 0 against stock iw3. Nothing past that is built — no CUDA
kernels, no ONNX, no plugin parameter. The decision that gate was waiting on is
below, and the cost turned out to be about half what this document first
predicted.

## What it is, and why it is not a model swap

`row_flow_v2` and `row_flow_v3` are interchangeable because the pipeline around
them is identical. `monobw_inpaint` is a **different pipeline**:

```
depth ─> MonoBW warp ─┬─> warped eye ─┐
                      └─> hole mask ──┴─> mask morphology ─> LightInpaintV1 ─> composite
```

Forward-warp-and-fill rather than backward-warp. Two stages, two very different
kinds of work, and a mask travelling between them. That is why it lives in its
own file rather than joining `stereo_warp.py`'s Model parameter.

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

It is also cheap: measured at **1.7 - 3.5 ms** for both eyes at HD, mask
included, which is the same order as the whole `row_flow_v3` path.

**LightInpaintV1 is 2.26M parameters** — 75x `row_flow_v2`, 19x `row_flow_v3` —
and runs on the **colour frame at full resolution**, not on downsampled depth.
That is what makes it expensive.

## The number that decides it

Measured on an RTX 5080, PyTorch, both eyes, end to end — warp, mask
morphology, and inpaint — against `row_flow_v3` on the same input:

| | monobw_inpaint | row_flow_v3 | ratio |
| --- | --- | --- | --- |
| 1920x1036, depth 392x938 | **25.9 - 26.2 ms** | 3.7 - 4.2 ms | **~6.7x** |
| 4K, depth 518x910 | ~124 ms | 6.7 ms | ~19x |

So HD is about **38 fps** rather than 250, and 4K stays render-only. Of the
26 ms, 23.6 is the inpaint network and the rest is the warp and the mask work.

An earlier draft of this document put HD at 48.4 ms and called it 10x. That
table was measured **without autocast**, and the model is almost exactly twice
as fast in half precision — which is what iw3 runs by default:

| inpaint model, one eye | fp16 | fp32 |
| --- | --- | --- |
| 1920x1036 | 11.8 ms | 22.6 ms |
| 4K | 57.5 ms | 105.3 ms |

The trade is still real, just cheaper than feared: inpainting fixes occlusion
artifacts that backward warping cannot, because it has something to put in the
holes rather than smearing an edge. Whether that is worth 6.7x is a judgement to
make on real footage, not on a benchmark — and now there is something to make it
with.

## What got ported

`stereo_inpaint.py`, 908 lines — the ~530 of code this document first estimated,
plus the comments explaining which expressions cannot be tidied — no nunif
imports:

| From | What |
| --- | --- |
| `iw3/models/monobw.py` | `MonoBW`, whole |
| `iw3/dilation.py` | `_dilate`, `_erode`, `_closing`, `_mask_closing`, `_dilate_inner`, `_dilate_outer` |
| `iw3/models/light_inpaint_v1.py` | `LightInpaintV1`, `_GLUConvMLP`, `_GMLPBlock` |
| `nunif/modules/attention.py` | `_WindowGMLP2d`, `_GMLP` |
| `nunif/modules/norm.py` | `_FastLayerNorm` |
| `nunif/modules/gaussian_filter.py` | `_GaussianFilter2d`, `_SeparableGaussianFilter2d`, the kernel builders |
| `iw3/monobw_inpaint.py`, `iw3/base_inpaint.py` | `MonoBWInpaintImage` — the two classes folded into one |
| `iw3/backward_warp.py` | `_apply_divergence_monobw` |
| `iw3/utils.py` | `synthesize_stereo_inpaint`, the `apply_divergence` equivalent |

The non-square pixel shuffle/unshuffle, the window partition and merge, the
autocast rule and `get_mapper` all come from `stereo_warp.py` unchanged — they
were already written and already exercised by the `row_flow_v3` export.

Deliberately not ported: the video model, `basic_module_init`/`icnr_init` (an
inference port loads weights rather than initialising them), and
`shift_mask_token` (`LightInpaintV1` never enables it, and the checkpoint
confirms it — there are no `shift_mask_bias` keys).

One deliberate departure, and it is the same one `row_flow_v3` needed:
`replication_pad2d_naive` builds padding by Python tuple repetition, which bakes
a frame size into an exported graph, so `F.pad(mode="replicate")` is used
instead. Numerically identical; `docs/row-flow-v3.md` records why it matters.

Things that look like bugs and were kept anyway, because the golden test
compares against them:

- `MonoBW.warp()` guards its grid resize with `grid.shape[-2] != x.shape[-2:]`,
  an int against a `torch.Size`, which is always true. The grid is therefore
  interpolated twice, the second time to the size it already is.
- `LightInpaintV1` pads up to a multiple of 64 but never by zero — an exact
  multiple still gets a whole extra 64 pixels. The network was trained that way.
- `_resize` derives the new height from the rounded-up `max_width` against the
  *original* width.

## The golden test

`tests/test_stereo_inpaint.py`, 22 cases at difference 0, against
`create_stereo_model("monobw_inpaint", ...)` driven through `apply_divergence`
— iw3's real path, not a hand-assembled one. Every axis the pipeline has:
divergence, convergence, synthetic view, mapper, both mask dilations,
`inpaint_max_width`, mismatched and odd depth sizes, batching, amp on and off,
and CPU.

Diff 0 on the first run is not on its own evidence of anything, so the harness
was checked by breaking the implementation on purpose. It catches all of these:

| Mutation | Diff |
| --- | --- |
| `MonoBW` index smoothing off | 0.18 |
| pad only to the next multiple of 64 | 0.17 |
| window shift off | 0.035 |
| plain `LayerNorm` instead of `FastLayerNorm` | 4.9e-4 |

The last one is why `_FastLayerNorm` is in the port at all: `nn.LayerNorm` is on
autocast's fp32 list and upcasts, nunif's does not, and that is worth 5e-4.

Two mutations changed nothing, both for understandable reasons rather than
missing coverage: dropping the second grid interpolate in `warp()` (it really is
a no-op, it is kept only for faithfulness), and moving the packed-token mask
threshold from 0.99 to 0.5 (the blur radius against the 4x4 pack size leaves no
token in that band).

## The export blocker, still there

Untouched by this stage, and the same class of problem as `row_flow_v3`:

```
dynamo export: OK
  256x448: max abs diff 1.281e-06        <- the traced size
  392x938: FAILED  Expand node ...       <- every other size
  260x452: FAILED  Expand node ...
```

It exports and is accurate where it was traced, and fails everywhere else on an
`Expand` node with a baked dimension. `row_flow_v3` needed three separate fixes
to clear this class of problem — the attention head merge, the padding, and a
batch-1 specialisation. One of those three, the padding, is already in this port
because it was written that way from the start. How many more this needs is
unknown; `_WindowGMLP2d`'s shifted window partitioning is the obvious suspect.

## Two things to get right

**Only the image inpaint model fits Resolve.** There is also
`light_video_inpaint_v1`, which `MonoBWInpaintVideo` drives through a frame
queue with `pre_padding` and `post_padding`. Resolve renders frames out of
order, with gaps and repeats — measured in `docs/phase0-findings.md` — so a
temporal queue is a poor fit. `light_inpaint_v1` is per-frame and is the right
target.

**The offset convention.** `LightInpaintV1` has `i2i_offset = 16`, so its
`forward()` crops 16 pixels from each side; 256x448 in gives 224x416 out.
`LightInpaintV1.infer()` is the model's own method, not the `I2IBaseModel` one,
and calls `forward()` with `skip_i2i_offset=True` — which is the path the
pipeline uses, so nothing is cropped in practice. A tiled implementation would
need the other one.

Also worth knowing: `_inpaint_single()` flips the left eye horizontally,
inpaints, and flips back — the network is trained for one handedness only, the
same trick `row_flow` uses.

## Not decided

Whether to build it past the standalone stage. The staged plan was: PyTorch
first with a golden test at difference 0, judge the quality against the cost on
real footage, then decide about CUDA and ONNX. The first step is done and the
cost is 6.7x, not 10x. The judgement is the next step, and it needs footage
rather than another benchmark.
