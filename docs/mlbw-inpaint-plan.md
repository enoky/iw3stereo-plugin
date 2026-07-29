# mlbw_l2_inpaint — plan, and what building it changed

**Built.** All five stages are done and both Model options ship. What follows is
the original plan with each stage's outcome folded in where it landed, because
three of the five stages changed on contact with a measurement and the reasons
are worth more than a tidy plan would have been.

The prediction was that this would be **materially cheaper than
`monobw_inpaint`**, because the expensive half already existed and the warp half
looked like a better fit for ONNX. Both halves of that were right, and the
cheapness came from somewhere slightly different than expected: not from the warp
exporting cleanly — it does not, and had to move to CUDA — but from almost
everything it needed in CUDA already being there.

What actually changed, in order:

| Stage | Plan said | Turned out |
| --- | --- | --- |
| 0 probe | shifted-window attention is the risk | not a variable at all; `export_safe` is |
| 1 PyTorch | diff 0 | diff 0, and two plan claims were wrong |
| 2 ONNX | the whole warp, in fp16 | the network only, in fp32 |
| 3 CUDA | bigger, because stage 2 handed it the warp | smaller, because the filter already existed |
| 4 plugin | a fifth option, maybe a sixth | both, and one cache bug caught in the writing |

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

Steps 1–3 each change the answer, and step 1 operating on logits rather than on
a binary mask means the existing `mask_closing` is the wrong function — the kind
of detail a diff-0 test catches and a reading does not.

Step 4 turns out not to matter, which is worth recording because the first draft
of this plan asserted the opposite. They are two morphological dilations by
opposite unit structuring elements, and dilation commutes; measured over 3200
cases including ones saturating against the frame edge, swapping the order
changes nothing. Keep iw3's order, but do not spend a test on it.

## The gating unknown: shifted-window attention through ONNX — ANSWERED, green

`WABlock` here is `row_flow_v3`'s block with one addition — `shift`, which
`WindowMHA2d` implements by padding half a window on each side, partitioning,
and cropping back. `row_flow_v3`'s blocks never shift.

`docs/monobw-inpaint.md` predicted, before any of that work started, that
"`WindowGMLP2d`'s shifted window partitioning is the obvious suspect" for export
trouble. That prediction was never tested, because `light_inpaint_v1` turned out
to export cleanly for an unrelated reason. So it was tested here first, before
porting anything: `tools/probe_shifted_window.py` builds MLBW's body at the
right shape with random weights and exports it at dynamic shapes.

**Shift is not the problem, and it is not even a variable.** Four runs, `shift`
crossed with `export_safe`:

| | SDPA | head-sliced (`export_safe`) |
| --- | --- | --- |
| no shift | export fails | **6.0e-7** |
| shift | export fails | **6.0e-7** |

Six sizes each — 108x192, 392x938, 392x940, 384x960, 100x200, 528x940 — every
one within 6e-7 of PyTorch, and the shifted and unshifted columns agree to the
last digit. The prediction was wrong: shifted-window partitioning exports
cleanly.

What does decide it is the same thing that decided `row_flow_v3`: SDPA's head
permute, which `torch.export`'s decomposition cannot lower. Without the rewrite
both variants die identically in `Run decompositions`, with or without shift.
That rewrite already exists in `stereo_warp.py` as `export_safe`, so the fix was
written before the problem was found.

Two things follow. The risk register below loses its top entry. And stage 1's
model needs the same `export_safe` flag `RowFlowV3` carries, for the same
reason — the rewrite is ~5e-5 away from the fused kernel, so diff 0 needs the
SDPA path and export needs the other one.

This is the discipline that paid off with the temporal-access probe: ask the
question that could kill the approach with the cheapest possible experiment. It
cost an hour, and the answer was the opposite of the documented expectation.

## Staging

Each stage ends somewhere committable, and each has a bar that is a number.

**0 — probe the export.** **Done**, green. Above.

**1 — `MLBW` standalone in PyTorch, diff 0.** **Done.** 22 cases at max absolute
difference **0** against `create_stereo_model("mlbw_l2_inpaint", ...)` driven
through `apply_divergence`, sharing the axis sweep with the monobw suite.

Nine mutations were run against the finished port to check the suite has teeth;
seven were caught, by 1 to 54 failing cases each. The two survivors were both
the test suite telling the truth: the dilation order genuinely does not matter
(above), and iw3's float32 cast before the softmax is a no-op under autocast,
because autocast already promotes softmax to fp32. That cast is not dead code —
it is what keeps stage 2's fp16 graph honest, where the same softmax lands
2.4e-4 away — but nothing at stage 1 can observe it, and a test pretending
otherwise would have been a test of nothing.

Original plan for this stage, kept for the record: into `stereo_warp.py` beside
`RowFlowV3`, not `stereo_inpaint.py`: it is a warp network and it shares
`RowFlowV3`'s attention machinery. Bar: max absolute difference **0** against
`create_stereo_model("mlbw_l2_inpaint", ...)` driven through `apply_divergence`,
across the axes `tests/test_stereo_inpaint.py` already covers.

Watch for: the symmetric mod padding (`pad//2` each side, and it always pads at
least one whole block even at an exact multiple), `layer_weight.to(float32)`
before the softmax, and the fact that `delta` is one channel per layer until
`pad_delta_y` widens it.

**2 — the ONNX graph.** **Done**, but not the graph this plan asked for. The
boundary moved, for a measured reason.

The plan put the whole warp in one graph, reasoning that — unlike MonoBW's,
which needs `cummax` and `searchsorted` — every operation in it has an ONNX
equivalent. Every operation does. One of them does not have the *same*
equivalent.

iw3 resizes the layer weights with
`F.interpolate(..., align_corners=True, antialias=True)`. **PyTorch's
antialiased kernel silently ignores `align_corners`** and applies a half-pixel
coordinate transform regardless. Upscaling 4 → 8 it returns 1.0, 1.143 … 3.714
where an align_corners resize ends at 4.0. The export emits `antialias=1`
faithfully and ONNX Runtime honours `align_corners` faithfully, and the two
disagree by **3.6e-2** on the finished eye. No `Resize` configuration
reproduces PyTorch here, in either direction, because the disagreement is in
the coordinate transform rather than in the filter.

It could be built by hand from gathers, but the antialias filter's support
scales with the resize ratio, so a static graph needs a fixed tap count and
blends wrongly once the ratio exceeds it — silently, producing a picture rather
than an error. That is this project's documented failure mode, so it was not
built. The resize, both warps and the blend go to CUDA, where the tap count is
just a loop bound.

This is not a retreat to a worse position. It is the split `monobw_inpaint`
already uses — network in the graph, geometry in the kernel — and it makes
stage 2 smaller and stage 3 larger rather than adding work overall.

So: `mlbw_net.onnx`, signature `x -> (delta, layer_weight, mask_logits)`.

- **In**: the network, and nothing else.
- **Out**: the input tensor, the mapper, the grid, both backward warps, the
  weight resize, the blend, the mirroring, and all mask morphology.

**float32, not fp16.** Also measured rather than assumed. The deltas this
emits become grid coordinates, and at 1920 wide the grid's spacing is 1.04e-3
against fp16's resolution of 9.77e-4 near the ±1 edges — the same order, so
neighbouring columns stop being distinguishable and the warp reads the wrong
pixel. End to end the fp16 warp lands **1.0** from the fp32 one. The inpaint
graphs survive fp16 because they carry pixels, which have 8 bits of meaning;
this one carries coordinates, which do not. At 0.233M parameters there is
nothing to gain anyway.

Measured against PyTorch across six sizes to 1036x1920: **2.3e-5** on the
deltas, 9.5e-6 on the weights, 4.8e-5 on the logits, and **1.3e-6** on a
finished eye — the eye is tighter than the heads that made it, because a 2e-5
delta is a sub-pixel shift. Four mutations of the export were run and all four
were caught, one of them by the export refusing to run at all.

**One finding that outlives this stage.** Through the fill, the comparison
changes in kind: the mask is `sigmoid(logits) > 0.15`, and a threshold has no
tolerance. At HD exactly **one pixel in 1,989,120** sat close enough to the
boundary to land on the other side of it, and because a flipped mask pixel gets
*filled*, that one pixel is a 1.1e-2 difference on its own, with about 17
neighbours dragged past 2e-4 by the fill's feather. Everything else agrees to
1e-6, and the mean over the frame is 8.8e-8.

A max-abs bar is therefore the wrong instrument end to end, and the suite uses
the mean and a count of materially-different pixels instead. **The same thing
will happen between the plugin's CUDA mask and iw3's**, for the same reason. It
is inherent to any two implementations that differ at all, and it is not a
defect in either — worth knowing before it is discovered as a bug in stage 4.

**3 — the geometry and the mask post-process in CUDA.** **Done.** It looked
bigger than planned, because stage 2 handed it the warp, and then turned out
smaller, because most of what stage 2 sent here already existed.

The antialiased resize is the reason the warp left the graph — and this repo had
already hit that wall once, for the depth resize, and answered it the same way.
`buildResampleAxis` was already written, already public "because the CUDA path
uploads exactly these weights rather than deriving its own", and already covered
by reference data in both directions. The layer weights are a new *caller* of
that filter, not a new filter. Its two kernels moved into `resample_gpu.cuh` so
there is one definition rather than two copies.

Worth recording as a process note: stage 2 spent its time rediscovering
something `stereo_warp_onnx.py` documents in a docstring. Reading the existing
resize before building the full-warp graph would have been the cheaper order.

Results, all against the Python that matches iw3 at diff 0:

| | |
| --- | --- |
| CPU core (`test_pipeline`) | 76 checks, warp within **2.4e-6**, mask **exact** |
| CUDA (`test_mlbw_gpu`) | 23 checks, warp within **2.9e-6**, mask **exact** |
| mutations of the kernels | **7 of 7 caught** |

The mask being exact is a stronger claim here than for monobw, whose mask input
is already binary. This one thresholds a *resized continuous* value, so any
disagreement in the upscale's last bit near 0.15 would surface as a wrong pixel.

**The mirroring resolved more simply than expected.** Working it through: the
left eye's model sees the frame and `_inpaint_single` flips the result; the right
eye's model sees the mirror and `apply_divergence` flips it back before
`_inpaint_single` leaves it alone. Both therefore hand the network *the mirror of
their own warp coordinates* — so the warp writes mirrored for both eyes and the
logits are flipped for both eyes, and the only per-eye difference left is which
image the warp samples. That was worth confirming against the real Python rather
than trusting the derivation, which is what the `mlbw_eye_*` cases do.

One detail the reference data earns its keep on: the logits must be flipped
*before* the post-process, not after. The closing and the resize are symmetric,
but the dilation window is not.

Original plan for this stage follows.

Two groups of kernels.

*The mask post-process*, as originally scoped. Four steps, three of which exist
in some form: `closing` on logits (reuse `maskDilateAt`/`maskErodeAt`), a
bilinear upscale from depth to frame resolution, `sigmoid > 0.15`, and the two
directional dilations (the existing single-window `maskFinishAt` — and since
the order provably does not matter, either order will do).

*The geometry*, newly arrived:

- the **antialiased layer-weight resize**, which is why any of this is here.
  Reproduce ATen's `upsample_bilinear2d_aa` exactly: `scale = (in-1)/(out-1)`,
  `support = max(scale, 1)`, `center = scale*(i+0.5)`, taps from
  `xmin = max(int(center - support + 0.5), 0)` to
  `min(int(center + support + 0.5), in)`, triangle weights
  `max(0, 1 - |(j + xmin - center + 0.5) * invscale|)` normalised by their sum,
  with `invscale = 1/scale` when downscaling and 1 when up. Separable, so one
  pass per axis. **Note it uses the align_corners scale with a half-pixel
  centre** — that hybrid is the whole reason it is not an ONNX Resize.
- two **backward warps** per eye: grid plus `delta * delta_scale`, the grid
  bilinearly resized to the frame (ordinary bilinear, `align_corners=True`,
  *no* antialias), then a border-clamped bilinear sample. `bilinearSampleBorder`
  already exists.
- the **softmax blend**, a two-term weighted sum, and the mirroring for the
  right eye.

Bar: exact agreement on the thresholded mask against Python reference data from
`tools/dump_pipeline_reference.py`, as the monobw morphology already is — a mask
is a threshold and cannot be checked with a tolerance — and float32 epsilon on
the warp, which is how the existing numeric core is judged. Reference data must
record the **antialias tap weights** for at least one upscale and one downscale
ratio; getting that filter subtly wrong is the single most likely way this stage
produces a plausible picture that is not iw3's.

**4 — the plugin.** **Done**, both options: `mlbw_l2_inpaint` and
`mlbw_l2_inpaint_video`. The twelve-frame machinery was model-agnostic as
predicted, so the temporal variant cost one window-build function and a branch.

Three pieces are now shared rather than copied — the antialiased resample
(`resample_gpu.cuh`), the fp16 and reduce-and-composite boundary either side of
an inpaint graph (`inpaint_boundary.cuh`), and both fill graphs themselves.
monobw's 30 checks passed across every extraction.

One bug this stage introduced and caught before it shipped: `settingsFingerprint`
did not include the method, so switching between the two *video* pipelines with
everything else untouched would have served the other model's cached window and
looked like the switch had not taken.

Most of the work went into the test rather than the plugin, deliberately.
`test_inpaint_ort` now drives both mlbw paths with real ORT on the real bundle,
real kernels and real device pointers, and the assertions are aimed at the three
bugs that actually reached the user last time: a flat or black frame fails, two
eyes agreeing on more than 98% of samples fails (a no-op mirror looks exactly
like a working plugin producing no 3D), and the temporal case traces each cached
frame back to the slot it was stored in.

Original plan for this stage follows.

A fifth Model option, and a sixth if the temporal variant is
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

~~**The shifted-window export.**~~ Retired by stage 0. What remains of it is not
a risk but a constraint on stage 1: the attention rewrite is not bit-identical —
about 5e-5 from the fused kernel — so the model carries an `export_safe` flag,
diff 0 is measured on the default path, and export uses the other one. Exactly
`RowFlowV3`'s arrangement.

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
