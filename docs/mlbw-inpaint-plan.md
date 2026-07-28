# mlbw_l2_inpaint — implementation plan

Investigation and plan. Nothing is implemented. This is the map that makes the
implementation session efficient, written the same way `docs/monobw-inpaint.md`
was and with the benefit of having since built that one end to end.

The headline: **this should be materially cheaper than `monobw_inpaint` was**,
because the expensive half is already built and the warp half is a better fit
for ONNX than MonoBW's was. The one thing that could turn that around is a
single unknown, and it should be probed before anything else is written.

## What it is

A third stereo pipeline, alongside the backward warp (`row_flow_*`) and the
forward-warp-and-fill (`monobw_inpaint`):

```
depth ─> mask_mlbw_l2 ─┬─> delta ×2      ─┐
                       ├─> layer weights ─┴─> 2 backward warps, weighted sum ─> eye
                       └─> hole mask logits ─> morphology ─> LightInpaintV1 ─> composite
```

Two things distinguish it from what exists.

**The warp is multi-layer.** The network emits *two* sampling deltas and a
softmax weight per layer, and the eye is the weighted sum of two backward warps
of the same frame. `row_flow` does one warp; this does two and blends them. That
is what "MLBW" is — multi-layer backward warp — and it is how the model
represents a pixel that could plausibly come from two places.

**The hole mask is predicted, not derived.** MonoBW computes its mask
geometrically: where the inverted index map had to stretch, there is a hole.
`mask_mlbw_l2` emits mask *logits* as a third output and the holes are wherever
`sigmoid(logits) > 0.15`. That is a learned judgement rather than an arithmetic
one, with consequences noted under *Risks*.

Everything after the mask — morphology, the inpaint network, the composite — is
identical to `monobw_inpaint` and is already built.

## The facts this plan rests on

Read out of nunif rather than assumed:

| | |
| --- | --- |
| model | `sbs.mask_mlbw_l2` = `MLBW(num_layers=2, base_dim=32, hole_mask=True)` |
| checkpoint | `iw3_mask_mlbw_l2_d1_20250903.pth`, present locally |
| parameters | **0.233M** — twice `row_flow_v3`, a fraction of the inpaint model's 2.26M |
| input | 3 channels, and it is **`make_input_tensor` verbatim** — the same disparity/divergence/convergence tensor `row_flow` takes |
| output | 5 channels at depth resolution: 2 deltas, 2 layer weights, 1 mask logit |
| driver | `iw3/mlbw_inpaint.py`, `apply_divergence_nn_delta_weight` in `iw3/backward_warp.py` |
| mask rule | `postprocess_hole_mask`, threshold `MASK_MLBW_THRESHOLD = 0.15` |

**There is only one mask checkpoint.** Plain `mlbw_l2` ships d1/d2/d3 variants
and `load_mlbw_model` picks one by divergence, which would have meant the model
changing under the user as they dragged a slider. `mask_mlbw_l2` has only `d1`,
so one graph covers every divergence. That removes what would otherwise have
been the nastiest piece of plumbing in this whole plan.

## What already exists

This is most of the job, and it is the reason to do this next rather than
something else.

| Piece | Where | State |
| --- | --- | --- |
| `LightInpaintV1`, `LightVideoInpaintV1` | `stereo_inpaint.py` | ported, diff 0 |
| both inpaint graphs, fp16 | `models/light_*inpaint_v1.onnx` | exported, in the bundle |
| fp16 cast kernels either side of ORT | `monobw_gpu.cu` | done |
| `dilate_inner` / `dilate_outer` / `closing` | `monobw_math.h`, `monobw_gpu.cu` | done |
| the twelve-frame window, cache, `getFramesNeeded` | `iw3stereo.cpp`, `monobw_gpu.cu` | done |
| Inpaint Max Width, reduce-and-composite | `monobw_gpu.cu` | done |
| `make_input_tensor` | `buildInputTensor`, CPU and CUDA | done, and **the input is identical** |
| `backward_warp` | inside `stereo_warp.onnx` | done, same delta contract |
| `WindowMHA2d`, `WindowScoreBias`, window partition/merge | `stereo_warp.py`'s `RowFlowV3` | ported |
| the export-safe attention rewrite | `stereo_warp.py`, `export_safe=True` | done, and will be needed again |

## What is new

| | Notes |
| --- | --- |
| `MLBW` model | shifted-window `WABlock`, symmetric mod padding, softmax layer weights, three outputs |
| the multi-layer warp | two `backward_warp` calls weighted by a softmax and summed |
| `postprocess_hole_mask` | different from monobw's mask path in three ways, below |
| the left-eye mirror | monobw folds it into a kernel's write index; here the eyes arrive in frame orientation and need an explicit flip |

`postprocess_hole_mask` is *not* `preprocess_mask` with different constants. It:

1. runs `closing(n_iter=1)` on the **logits**, before any sigmoid,
2. interpolates them to the frame's size (bilinear, `align_corners=True`),
3. takes `sigmoid > 0.15`,
4. dilates **inner then outer** — the opposite order to `monobw`'s
   `preprocess_mask`, which does outer then inner.

Step 4 does not commute in general, and step 1 operating on logits rather than
on a binary mask means the existing `mask_closing` is the wrong function. Both
are the kind of detail that a diff-0 test catches and a reading does not.

## The gating unknown: shifted-window attention through ONNX

`WABlock` here is `row_flow_v3`'s block with one addition — `shift`, which
`WindowMHA2d` implements by padding half a window on each side, partitioning,
and cropping back. `row_flow_v3`'s blocks never shift.

`docs/monobw-inpaint.md` predicted, before any of that work started, that
"`WindowGMLP2d`'s shifted window partitioning is the obvious suspect" for export
trouble. That prediction was never tested, because `light_inpaint_v1` turned out
to export cleanly for an unrelated reason. It is still untested, and it is
squarely on this model's critical path.

**Probe it first, before porting anything.** Build `MLBW` badly if necessary —
random weights, no fidelity — and try to export it at dynamic shapes. The answer
takes an hour and decides the shape of everything after it:

- exports and runs at several sizes → the rest is work, not risk
- fails → the fix is likely `row_flow_v3`'s attention rewrite applied again,
  which is already written and understood
- fails in some new way → that is worth knowing before a diff-0 port is sunk
  into it

This is the same discipline that paid off with the temporal-access probe: the
question that could kill the approach gets asked with the cheapest possible
experiment, not discovered three days in.

## Staging

Each stage ends somewhere committable, and each has a bar that is a number.

**0 — probe the export.** Above. Half a day at most.

**1 — `MLBW` standalone in PyTorch, diff 0.** Into `stereo_warp.py` beside
`RowFlowV3`, not `stereo_inpaint.py`: it is a warp network and it shares
`RowFlowV3`'s attention machinery. Bar: max absolute difference **0** against
`create_stereo_model("mlbw_l2_inpaint", ...)` driven through `apply_divergence`,
across the axes `tests/test_stereo_inpaint.py` already covers.

Watch for: the symmetric mod padding (`pad//2` each side, and it always pads at
least one whole block even at an exact multiple), `layer_weight.to(float32)`
before the softmax, and the fact that `delta` is one channel per layer until
`pad_delta_y` widens it.

**2 — the ONNX graph.** Boundary by the rule the other two exports already
follow: fixed arithmetic in, anything whose *amount* is a runtime parameter out.

- **In**: the network, `pad_delta_y`, the grid, both backward warps, the softmax
  weighting and sum, both eyes by mirroring, and the mask logits passed straight
  out.
- **Out**: the input tensor (its `preserve_screen_border` ramp is `round()`ed at
  runtime), the mapper, the depth resize, and all mask morphology.

Signature: `(image, x, delta_scale) -> (left, right, left_mask, right_mask)` —
`stereo_warp.onnx`'s exact inputs with two more outputs. `OrtRuntime` already
binds outputs by the names the graph declares, so a four-output graph needs no
runtime change.

Export in **fp16**, and give the session `enable_mem_pattern = false`. Both are
already plumbed per-model. The warp model is small enough that neither is
forced, but matching the inpaint graphs costs nothing and iw3 runs everything
under autocast anyway.

Bar: within **2e-5** of the PyTorch, six sizes including 1036x1920, plus a test
that the mask output actually varies with the input — a graph that dropped the
third head would pass every tolerance otherwise.

**3 — the mask post-process in CUDA.** Four kernels, three of which exist in
some form: `closing` on logits (reuse `maskDilateAt`/`maskErodeAt`), a bilinear
upscale from depth to frame resolution, `sigmoid > 0.15`, and the two
directional dilations (the existing single-window `maskFinishAt`, with the
arguments swapped for the reversed order).

Bar: exact agreement on the thresholded mask against Python reference data from
`tools/dump_pipeline_reference.py`, as the monobw morphology already is. A mask
is a threshold on a value and cannot be checked with a tolerance.

**4 — the plugin.** A fifth Model option, and a sixth if the temporal variant is
wanted — `MLBWInpaintVideo` exists upstream and the twelve-frame machinery here
is model-agnostic, so it should be nearly free once the image one works.

The one genuinely new piece of plumbing: the warp graph returns both eyes in
frame orientation, whereas `MonoBwGpu::prepareEye` hands the inpaint network a
mirrored left eye by writing to the opposite column. Here the mirror has to be
an explicit pass — `flipKernel` already exists — on the left eye and its mask
before the inpaint, and on the result after. Getting this wrong produces a
picture rather than an error, so it deserves the same treatment the monobw
version got: reference data recording *exactly what the network is handed* for
each eye, checked against the kernels.

## Risks, in the order they are likely to bite

**The shifted-window export.** Stage 0 exists for this. If it needs the
attention rewrite, note that `row_flow_v3`'s version is not bit-identical — it
lands about 5e-5 away — so the PyTorch model keeps an `export_safe` flag and the
diff-0 test uses the default path. Same arrangement here.

**The predicted mask has no independent check.** MonoBW's mask is arithmetic, so
"is this mask right" has an answer that does not involve the network. This one
is whatever the model says, so the *only* guard is diff 0 against iw3. That
raises the value of stage 1's bar and lowers the value of eyeballing it.

**Two warps, not one.** The graph does `backward_warp` twice per eye and sums.
Cost should still be small — 0.233M parameters and the warp itself is cheap —
but it is not free, and the honest comparison is against `row_flow_v3`'s 4-5 ms
rather than against the inpaint's 18 ms. Measure it in stage 2 rather than
assuming it disappears next to the inpaint.

**The mask is at depth resolution.** MonoBW's mask is computed at frame
resolution; this one is a network output at the depth's size and is upscaled.
The dilation counts are quoted against `base_width`, which
`postprocess_hole_mask` takes from the *logits'* width — so the same setting
means something different here than in the monobw path. Do not share the
parameter without checking that.

## What this plan assumes, and how it could be wrong

It assumes the inpaint half is genuinely reusable as-is. That is well founded —
`MLBWInpaintImage` and `MonoBWInpaintImage` both derive from `BaseImageInpaint`
and differ only in `apply_warp` and `preprocess_mask` — but it has not been
tested by actually swapping one for the other.

It assumes the plugin's temporal window is model-agnostic. It is written against
`MonoBwGpu` in one place (`buildVideoWindow` calls `prepareEye`), so a small
amount of indirection will be needed there.

And it assumes stage 0 comes back green. If it does not, and the attention
rewrite does not fix it, the fallback is the one MonoBW ended up taking: keep
the model out of ONNX and write it as CUDA. That would be a much larger job
here than it was there — MonoBW had no weights, and this has 0.233M of them —
so a red probe is a genuine decision point rather than a detour.

## Lessons from the monobw work worth carrying over

Written down because each of these cost real time to learn:

- **Any inpaint session needs `enable_mem_pattern = false`.** Without it the
  twelve-frame graph takes 15.7 GiB and sixteen seconds instead of 9.0 and a
  quarter of a second. It is not a tuning knob.
- **Benchmark like for like.** PyTorch under autocast is fp16; an ONNX graph may
  not be. Comparing them silently produced a cost figure that was wrong by a
  factor of two and stood in the docs for days.
- **Reference images must be real content, not noise.** Noise has the steepest
  gradient an image can have at every pixel, so a warp lands a tolerance that the
  plugin will never see — 4.7e-5 on noise against 2.4e-6 on real content, from
  identical code.
- **Nothing tests the OFX glue.** Three bugs in the monobw work reached the user,
  and all three were in the plumbing rather than the arithmetic: a window that
  could not clamp at the start of a clip, a buffer left holding the wrong frame,
  a session option that was measured and then not carried across. Every one of
  them produced *a picture* rather than an error. If this pipeline adds a fourth,
  the answer is a test that drives frames in sequence through the entry points,
  not another point fix.
