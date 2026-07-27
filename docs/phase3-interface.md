# Phase 3 — plugin interface

**Status: agreed. Decisions recorded at the bottom.**

The plan asks for the interface to be written down and agreed before the plugin
is built, so this is that document. Everything below follows from a measurement
in `docs/phase0-findings.md` or `docs/phase2-onnx.md`; where it follows from a
guess instead, it is called out under *Open questions*.

## What it is

One OFX plugin, **`com.nunif.iw3.stereo`**, labelled **"iw3 Stereo"** in the
group **"iw3"**.

Its real form is a **Fusion node**: two image inputs, one image output. That is
where Resolve instantiates an OFX effect in the general context and gives it a
second clip. Dropped on a timeline clip in Edit or Color it gets the filter
context and one input — see *Open questions* for what, if anything, it should do
there.

It does the warping half of iw3. It does not estimate depth; depth arrives as a
clip.

## Inputs

| Input | Fusion colour | Required | Contents |
| --- | --- | --- | --- |
| `Source` | yellow | yes | the colour frame, RGBA float |
| `Depth` | green | no | disparity, larger = nearer, read from the red channel |
| `EffectMask` | blue | no | added by Fusion itself; not read by the plugin |

With `Depth` unconnected the plugin passes `Source` through unchanged and says
so in the node's status. It must not output black: "nothing happened" is a
readable failure, a black frame is not.

Resolve delivers both clips at **composition resolution**, float32 RGBA, exactly
packed, always the full frame. The depth clip's own resolution is not visible to
the plugin — a 960x384 file arrives as 1920x800.

## Parameters

### Stereo

| Parameter | Type | Range | Default | Meaning |
| --- | --- | --- | --- | --- |
| **Divergence** | double | 0 – 10, slider 0 – 5 | **2.0** | Strength of the effect, as a percentage of image width. iw3 calls 0–2 reasonable and its GUI offers 1.0–5.0. |
| **Convergence** | double | −1 – 2, slider 0 – 1 | **0.5** | The depth value that lands on the screen plane. 0 puts everything behind the screen, 1 everything in front. |
| **Preserve Screen Border** | bool | — | **off** | Tapers parallax to zero at the left and right edges. |

Both eyes are always synthesised, each carrying half the parallax. iw3's
`--synthetic-view` can instead keep one eye pristine and put the full divergence
on the other; that is **not exposed** — see the decisions below.

### Depth

| Parameter | Type | Range | Default | Meaning |
| --- | --- | --- | --- | --- |
| **Depth Is Inverted** | bool | — | **off** | External depth tools disagree about whether white is near or far. |
| **Foreground Scale** | double | −3 – 3 | **0** | iw3's `--foreground-scale`. 0 is off. This is exposed instead of `--mapper` because iw3's own help says "directly using this option is not recommended, use --foreground-scale instead". |
| **Mapper Type** | choice | Multiply / Shift | **Multiply** | Which family Foreground Scale selects from. Only meaningful when Foreground Scale is non-zero. Metric-depth mappers are not offered — depth arriving as a video clip is relative, not metric. |
| **Stereo Width** | int | 0 – 8192 | **0 = Auto** | Width the depth is resized to before warping. See below — this is not optional. |
| **Depth Range** | choice | As delivered / Undo video expansion | **As delivered** | See below. |

### Output

| Parameter | Type | Range | Default | Meaning |
| --- | --- | --- | --- | --- |
| **Output** | choice | Anaglyph / Left eye / Right eye / Half SBS / Depth (debug) | **Anaglyph** | What the single output image contains. |

## Behaviour that is not a parameter

### Stereo Width is mandatory, and Auto is the right default

Resolve hands the plugin depth at composition resolution unconditionally. That
is precisely the case iw3's `stereo_width` exists to prevent: the warp takes its
sampling grid and delta scale from the depth's own width, and the networks are
trained around what a depth model emits. Measured in `iw3_ext`, on a 1920x1036
frame, horizontal-noise score for the same depth content was 0.36 at 518 lines,
3.09 at 700, and **7.32 at full frame**.

So **Auto** does not mean "leave it alone". It means the sizing rule already
written for this in `iw3_ext/depth_file.py`:

> bring the depth down to the shape a depth model would have produced for a
> frame this size — lower bound 392, rounded to a multiple of 14, aspect
> preserved — and only when it is larger than that.

For a 1920x800 comp that gives roughly **938x392**. An explicit Stereo Width
overrides it, matching iw3's GUI choices of 1920 / 1280 / 640.

The resampling must be the antialiased separable filter from Phase 2, not a
naive resize, and not ONNX Resize — which cannot reproduce it.

### Depth Range

Resolve applies a **limited-to-full range expansion** to the depth clip: values
arrive spanning roughly −0.01 to 1.005 rather than 0 to 1. The transform is
affine and invertible, so it acts like a change in divergence and convergence
rather than distorting geometry.

The plugin always clamps to 0–1 before use, because the mappers and the model
assume that range. The **Depth Range** parameter additionally offers to undo the
expansion, `v_original = (v * 219 + 16) / 255`.

**Measured afterwards: the expansion is correct and must be left alone.** iw3
applies the same one when reading its own depth videos, so Resolve's handling is
what makes the plugin agree with it. The documented recommendation is therefore
to leave Data Levels at whatever Resolve chose, and to leave this parameter on
"As delivered". Its other setting is for a file whose range tag is wrong, and
will make a correctly tagged one worse. See the correction in
`docs/phase0-findings.md`.

### Rendering

- One ONNX Runtime session for the whole process, built on first render, held
  behind a `call_once`. Bring-up costs 60–240 ms for the CUDA provider plus
  40–55 ms for the session; that is a one-off, not per frame.
- Loaded **by absolute path from inside the bundle**, never linked. Resolve
  ships its own `onnxruntime.dll` (1.13, CPU-only) in a directory Windows
  searches first.
- `use_tf32=0` on the CUDA provider.
- Resolve renders on one thread, full frame, frames roughly in order but with
  gaps and repeats. Any cache is keyed by frame number. The session still gets a
  mutex — the single thread is an observation, not a guarantee.
- If the CUDA provider will not start, fall back to CPU, and **say so in the
  node's status**: CPU is ~20x slower and a user must not discover that by
  wondering why playback died.

### Failure behaviour

| Situation | What the user sees |
| --- | --- |
| Depth not connected | Source passed through, status says why |
| Model file missing from the bundle | Source passed through, status names the missing file |
| CUDA unavailable | Works on CPU, status says it is on CPU and slow |
| Inference fails | Source passed through, status shows the ORT error |

Never a black frame, never a silent wrong result.

## Deliberately not exposed

- **`steps`** — iw3 defaults to 1, and the ONNX graph does not implement more
  (it needs the warp fed back into itself). `stereo_warp.py` supports it and is
  tested at 1–3, so it can be added if a reason appears.
- **The mapper by name** — twenty variants, and iw3 recommends Foreground Scale.
- **Method selection** — `row_flow_v2` only, per the plan. `row_flow_v3` and the
  `mlbw` family are candidates for later; the inpaint methods are a different
  shape of problem.
- **DirectML / provider choice** — DirectML miscomputes the model, so there is
  nothing to choose between.

## Decisions

Settled 2026-07-27.

1. **Half SBS only.** Full SBS would need the plugin to declare an output region
   twice the input width, which `multiRes=0` suggests Resolve will not honour.
   Not shipped, not tested further.
2. **Output defaults to Anaglyph**, so the node's effect is visible the moment
   it is added.
3. **Depth Range defaults to As delivered** — since confirmed correct by
   measurement. iw3 reads its own depth videos with the same limited-to-full
   expansion Resolve applies, so "As delivered" is what matches it. The Undo
   option is for a mis-tagged file only.
4. **No Edit/Color page variant.** Fusion is where a depth pass is composited.
   The filter context is still described, so the effect appears rather than
   failing oddly, but it passes the source through.
5. **Dubois anaglyph**, not a plain channel swap.
6. **Weights may be redistributed** — nagadomi confirmed. The exported `.onnx`
   ships with the plugin.
7. **Synthetic View removed** after first use, as overlapping with Output.
   The two are not quite equivalent and the difference is worth recording in
   case it is wanted back: Output picks which rendered image you see, while
   Synthetic View changed *how it was rendered*. With it set to Left or Right,
   one eye stayed the untouched original and the other carried the full
   divergence — sharper in one eye, more distortion in the other. Both-eyes,
   which is what remains, splits the parallax evenly. For anaglyph and SBS,
   which is what this plugin outputs, even splitting is the better choice
   anyway; the pristine-eye behaviour matters mainly when one eye is delivered
   as-is downstream.

### On the Dubois matrices

Red/cyan Dubois, applied to the gamma-encoded values Resolve delivers:

```
out.r =  0.4561   L.r + 0.500484 L.g + 0.176381   L.b
        -0.0434706 R.r - 0.0879388 R.g - 0.00155529 R.b
out.g = -0.0400822 L.r - 0.0378246 L.g - 0.0157589  L.b
        +0.378476  R.r + 0.73364   R.g - 0.0184503  R.b
out.b = -0.0152161 L.r - 0.0205971 L.g - 0.00546856 L.b
        -0.0721527 R.r - 0.112961  R.g + 1.2264     R.b
```

Each row sums to 1.0 across all six coefficients, so a grey stereo pair comes
out the same grey. That is the check worth keeping in a test — a transcription
error in any single coefficient breaks it.
