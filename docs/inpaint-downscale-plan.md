# Improving the reduced-resolution inpaint path

An assessment of the "High-Resolution Inpainting Patch for iw3" against this
plugin, and a plan for the parts that apply.

The patch targets stock iw3's `--inpaint-max-width`. This plugin's equivalent is
**Inpaint Max Width**, and the two are not the same code — the plugin already
diverges from iw3 here, deliberately and in the direction the patch is going.
That changes which features are worth taking.

## Feature by feature

**1 — High-resolution warping. Already implemented; nothing to do.**

The patch's headline fix is that iw3 resizes the *frame* before warping
(`_resize(x, max_width)` in `BaseImageInpaint.infer`), so the whole picture is
synthesised at the reduced size and the output is soft everywhere. The plugin has
never done that. `MonoBwGpu`/`MlbwGpu` warp and build the hole mask at full
frame resolution; `prepareInpaintInput` reduces *only* what the fill graph is
handed, and `compositeUpscaledKernel` puts the result back into the
full-resolution eye. `ofx/plugin/README.txt` already promises this in as many
words: "The warp and the hole detection stay at full resolution; only the network
runs small."

Note the Python port in `stereo_inpaint.py` *does* follow iw3 and resize first.
That is correct and must stay: it is what holds the diff-0 golden tests. The
divergence is the plugin's alone.

**2 — Alpha-premultiplied downsampling. Worth doing.**

`downscaleEyeKernel` is a plain box average over the source footprint. It has no
idea which of the pixels it is averaging are inside a hole, so invalid content
bleeds into the valid pixels next to every hole edge, and the network is then
handed that as if it were real.

The patch's fix is standard and correct: weight the average by validity and
renormalise.

    valid   = 1 - mask
    low     = downscale(eye * valid) / max(downscale(valid), eps)

One correction to the patch's framing, which matters for expectations: it
describes the invalid pixels as "black", producing "dark halos". That is iw3's
symptom. Here the holes contain *smeared* content — monobw stretches an edge
across them and the backward warps clamp at the border — so the artefact is not a
dark halo but edge content bleeding inwards. Same cause, same fix, less dramatic
before-and-after.

**3a — Bicubic upsampling. Worth doing.**

`compositeUpscaledKernel` upsamples the fill bilinearly. Bicubic is better on
thin structures, which is exactly where a hole is narrow enough for the
difference to show.

**3b — Exact compositing. Already half-true; the other half is a real change.**

The patch's stated problem is double-blending: the network's output has already
been composited against its input internally, so blending it *again* on the way
back up softens everything. The plugin does not have that bug in the region that
matters — the composite short-circuits to the untouched full-resolution eye
wherever the feathered mask is zero:

    if (m <= 0) { destination = eye; return; }

So outside the feather we are already exact. What the patch does better is
*inside* the feather band, where we linearly blend the original against the
upscaled fill and therefore lose full-resolution detail across that band. Their
residual form keeps it:

    out = fill_up + (eye - eye_low_up) * (1 - blurred_mask)

This one is worth taking, but it is the most invasive of the three and it changes
pixels in a band around every hole. It should be measured, not assumed.

## How much does this actually buy?

All three scale with the downscale ratio, so the answer depends on how hard
Inpaint Max Width is being pushed:

| Frame | Max Width | Ratio | Box footprint | Expected benefit |
| --- | --- | --- | --- | --- |
| 1920 | Full | 1.0 | 1x1 | **none** — none of this code runs |
| 1920 | 1280 (default) | 1.5 | ~2x2 | modest |
| 1920 | 720 | 2.7 | ~3x3 | clear |
| 3840 | 720 | 5.3 | ~5x5 | large |

At the shipped default on an HD timeline this is a refinement. For someone on a
small card running 720 on 4K — which is exactly who reaches for the setting — it
is the difference the patch describes.

## What is not protected

Worth stating before touching any of it: **nothing tests this path's output.**
The Python golden tests cover `max_width` at diff 0, but they test the *port*,
which follows iw3 and resizes first — they never exercise the plugin's
downscale/composite kernels at all. `test_inpaint_ort` runs a reduced-resolution
case but only asserts that a buffer came back and is not flat.

So the reduced-resolution path is simultaneously the least-tested thing in the
plugin and the thing being changed. Reference data comes first.

## Plan

**0 — reference data and a test, before any change.** Extend
`tools/dump_pipeline_reference.py` with `inpaint_downscale_*` cases: a
full-resolution eye and mask in, the reduced eye and the recomposited frame out,
at 1.5x, 2.7x and 5.3x. Add a handler to `test_mlbw_gpu.cu` (or a new
`test_inpaint_boundary.cu`) so the kernels are checked against Python.

Bar: the existing bilinear/box behaviour reproduced within 1e-5, *first* — so the
harness is proven against the code as it stands before the code moves. Then each
change below flips its own reference and the diff is visible.

**1 — premultiplied downscale.** `downscaleEyeKernel` gains the mask as an
argument, accumulates `sum(eye * valid)` and `sum(valid)`, and divides. Where a
footprint is entirely masked the weight is zero and it falls back to the plain
average — that pixel is inside a hole and about to be replaced anyway, but it
must be finite, not a NaN.

Cost: one extra plane read per output pixel. Negligible.

**2 — bicubic upsample in the composite.** Replace the four-tap bilinear in
`compositeUpscaledKernel` with a sixteen-tap Catmull-Rom, matching PyTorch's
`mode="bicubic"` coefficient (a = -0.75) so the reference data can be generated
from `F.interpolate`. Clamp the taps at the edges as the bilinear does.

Cost: 16 reads instead of 4, only for pixels where `m > 0`. The short-circuit
already skips everything outside a hole, so this is confined to hole
neighbourhoods.

**3 — residual composite.** The largest behavioural change and the one to do
last, on its own commit, so it can be reverted independently if it does not look
better. Needs `eye_low_up` — the *downscaled* eye upsampled back — which the
kernel does not currently have. Either upsample `_eyeHalf` in the same kernel (a
second sixteen-tap read) or precompute it into a scratch plane.

Bar for this one is not a tolerance but a comparison: render the same frame both
ways at 720-on-4K and look at the band around a hole. If it is not visibly
better, do not keep it — it costs a buffer and a second resample.

**4 — the user-facing note.** `ofx/plugin/README.txt` already explains that only
the network runs small. If 1-3 land it should say the downscale is
hole-aware, because that is the part a user would otherwise attribute to the
model.

## Recommendation

Do 0, 1 and 2. They are small, isolated, and strictly better at every ratio.

Treat 3 as a proposal to be judged on a rendered frame. It is the only one that
changes pixels people are currently happy with, and the plugin does not have the
double-blending bug that motivates it upstream — only the softer transition band,
which is a smaller claim.

Skip feature 4 of the patch entirely (scene-boundary temporal buffering): it is
about iw3's CLI scene-cut detection feeding its frame queue, and the plugin's
twelve-frame window is addressed by frame number from Resolve, so the bug cannot
occur here.
