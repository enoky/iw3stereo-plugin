# Phase 0 — de-risking Resolve

Status: **complete. The answer is architecture A, the OFX C++ plugin.**

## The decision

**Build architecture A: an OFX plugin in C++ with ONNX Runtime, as a Fusion
node.** Every question this phase existed to answer came back favourable, and
the two that could have killed it — a second input clip, and a GPU execution
provider inside Resolve's process — both work.

The evidence, in the order it matters:

1. **Resolve ships the entire OpenFX SDK**, with GPU render suites and
   multi-frame-access samples. Nothing to vendor. (0a)
2. **A plugin we built loads**, and it is Studio. (0d)
3. **The second input clip exists** on the Fusion page. (0c)
4. **ONNX Runtime brings up a CUDA provider inside Resolve** and runs the real
   warp there, at 23.5 ms for a 1920x800 frame with full-resolution depth. (0e)
5. **The licence position is clean** — dropping depth estimation drops every
   non-commercial encumbrance. (0a)

Three constraints came with it. None is a blocker; all three belong in the
plugin's design and its documentation:

- **Fusion page only** for two-input operation. Edit and Color get the filter
  context and one input. (0c)
- **NVIDIA only** for the GPU path, because DirectML miscomputes the model.
  (Phase 2)
- **`stereo_width` is mandatory**, because Resolve always delivers depth at
  composition resolution, which is the exact case it exists to prevent. (0f)

Architecture B is also now *less* attractive than the plan assumed: it is built
entirely on the Python scripting API, and `fusionscript.dll` access-violates on
import under Python 3.12.

---

The evidence for each of the above follows.

Test machine: Windows 11 Pro 22631, RTX 5080, DaVinci Resolve 21.0.1.11,
CUDA toolkits 11.8–13.2, Visual Studio Community 2026 + Build Tools 2022/2019,
CMake 4.1.1, Ninja. nunif venv: Python 3.12.10, torch 2.12.0+cu130.

## 0a — edition, SDK, toolchain, licences

### Edition

Resolve 21.0.1.11 is installed at `C:\Program Files\Blackmagic Design\DaVinci Resolve`.
The Add/Remove Programs entry reads "DaVinci Resolve", not "DaVinci Resolve Studio" —
but that entry is written by a shared installer and is **not** a reliable edition
signal. Two things point at Studio:

- the user states Studio 21 is licensed;
- `C:\Program Files\Common Files\OFX\Plugins` already contains third-party OFX
  bundles (`Topaz Video.ofx.bundle`, `NewBlueFX`), which free Resolve refuses to
  load.

Treated as **confirmed pending 0d**, which tests it directly: if our own plugin
appears, it is Studio.

### Resolve ships the whole OpenFX SDK

This is the single biggest de-risking result, and it is better than the plan
assumed. `C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\Developer\OpenFX`
(README dated 12 May 2026) contains:

| Item | Notes |
| --- | --- |
| `OpenFX-1.4/include` | the OFX C headers, incl. `ofxGPURender.h` |
| `Support/` | the OFX C++ Support wrapper, sources and headers |
| `GainPlugin` | sample with CUDA, OpenCL and Metal render paths |
| `TemporalBlurPlugin` | **multiple frame access** |
| `RandomFrameAccessPlugin` | **random frame access** |
| `DissolveTransitionPlugin` | transition context |

Nothing needs to be vendored, and the GPU render suites (`setSupportsCudaRender`,
`setSupportsCudaStream`) are available, which is what keeps buffers on the GPU
instead of round-tripping per frame.

The neighbouring `Developer/` folders also confirm the other surfaces exist
locally: `Scripting`, `Workflow Integrations`, `Fusion Fuse`, `DaVinciCTL`,
`CodecPlugin`.

### Toolchain

Two snags, both solved, both worth not rediscovering:

1. **The sample projects do not build as shipped.** They target
   `PlatformToolset v120` (VS 2013) and import `CUDA 8.0.props`. Replaced with a
   CMake build (`ofx/CMakeLists.txt`) against the shipped headers.
2. **`Support/Library/ofxsHWNDInteract.cpp` cannot compile at all.** It is
   written against `ofxHWNDInteract.h`, a C header Resolve did not ship — only
   the C++ wrapper half is present. It is excluded from the build; nothing here
   needs HWND interacts.

Also note CMake 4.1.1 has no "Visual Studio 18 2026" generator, so the build
uses the VS 2022 generator and the 2022 Build Tools (MSVC 19.44). That works.

Result: `ofx/build/bundles/iw3probe.ofx.bundle/Contents/Win64/iw3probe.ofx`,
exporting `OfxGetNumberOfPlugins` and `OfxGetPlugin` — a well-formed OFX binary.

### Installing needs elevation

`C:\Program Files\Common Files\OFX\Plugins` is not writable without
administrator rights (confirmed: `UnauthorizedAccessException`). Neither
`OFX_PLUGIN_PATH` nor a literal `Common Files\OFX` string appears anywhere in
Resolve's 401 binaries in ASCII or UTF-16, so the env-var escape hatch is
unproven — assume the standard directory and an elevated install step.
`scripts/install-ofx.ps1` does the copy and refuses to run unelevated.

### Licences

- **nunif itself is MIT** (`F:\_AI_PROJECTS_\nunif\LICENSE`, © 2019-2023 nagadomi).
- **`iw3_row_flow_v2_20240130.pth` carries no separate licence.** It is
  nagadomi's own weight, served from the nunif GitHub releases, and `NOTICE`
  does not list it among the third-party assets. The `cc-by-nc-4.0` restrictions
  in `iw3/README.md` attach to third-party **depth** models (Depth-Anything V2,
  Video-Depth-Anything, and metric variants) — none of which this project needs,
  because depth arrives from outside.
- So the warp half is redistributable under MIT on its face. Still worth
  confirming with nagadomi in writing before shipping converted weights, since
  "no separate licence stated" is an inference, not a grant.

This is a real advantage of the architecture: dropping depth estimation drops
every non-commercial encumbrance in the project.

## 0b — a custom OFX plugin builds

Done. `ofx/probe/iw3probe.cpp` registers two effects in one binary:

- **iw3 Probe (Single)** — one Source clip, inverted. Establishes that a plugin
  we built loads and renders.
- **iw3 Probe (Dual)** — Source + Depth clips, rendered side by side, with the
  Depth clip marked optional so the effect still renders when nothing is wired
  to it. If depth never arrives the right half is flat magenta.

It carries the parameter set the real plugin will need (divergence, convergence,
synthetic view, depth-inverted), so Phase 3 does not discover late that Resolve
renders some param type badly.

Everything it observes is appended to `%LOCALAPPDATA%\iw3probe\probe.log`:
host name and version, supported contexts, GPU render flags, which context each
effect is described in, clip connection changes, and for every `render()` call
the frame time, render window, render scale, thread id, parameter values, and
each image's bounds, bit depth, components and row stride.

That log is how 0c, 0e and 0f get answered.

## 0d — the plugins load  **ANSWERED: yes, and it is Studio**

Resolve 21.0.1 loaded `iw3probe.ofx.bundle` on startup and described both
effects. **Studio confirmed** — the application's own status bar reads "DaVinci
Resolve Studio 21", and free Resolve refuses third-party OFX in any case.

What the host reports about itself:

```
host: name='DaVinciResolve' label='DaVinci Resolve' version=21.0.1 api=1.4
host: multiRes=0 tiles=0 temporalClipAccess=1 multipleClipDepths=0
      multipleClipPARs=0 overlays=1
host: openCL=0 cuda=1 cudaStream=1 metal=0
host: contexts=[filter, general, transition, generator]
```

Three of those matter:

- **`general` is in the supported context list.** That is the OFX context in
  which extra input clips are legal — the filter context permits exactly one.
  So a second clip is not ruled out at the host level, which was the main worry
  behind 0c.
- **`cuda=1 cudaStream=1`** — the GPU render suites are live, so the plugin can
  keep buffers on the GPU and hand ORT a device pointer rather than round-trip
  through host memory every frame.
- **`temporalClipAccess=1`** — neighbouring frames are reachable, which matters
  if a temporally-stable variant is ever wanted.

`multiRes=0` and `tiles=0` mean Resolve always asks for the full frame at one
resolution. That is good news for an ML plugin: no tiled render windows to
stitch, and the render window equals the frame.

### Aside: the Python scripting API cannot be driven from Python 3.12

`fusionscript.dll` **crashes with an access violation (0xC0000005) while being
imported** as an extension module by Python 3.12.10 — before any API call. It
declares no `pythonXX.dll` import at all, and adding the Resolve directory via
`os.add_dll_directory` plus `PATH` does not help, so this looks like an ABI
mismatch inside the module's own init rather than a missing dependency.

Not blocking for architecture A, which never touches the scripting API, but it
is a real constraint on architecture B and on any Workflow Integration panel,
both of which are *built* on that API. Options if B is ever chosen: find the
Python version BMD actually built against, or drive scripts from Resolve's own
Fusion Console, which uses Resolve's embedded interpreter.

## 0c — the second input clip  **ANSWERED: yes, on the Fusion page only**

The context Resolve instantiates the effect in depends on where it is used, and
the two answers are different:

| Where | Context | Depth clip |
| --- | --- | --- |
| Edit page (and Color page) | `filter` | **no** — filter permits exactly one input |
| Fusion page | `general` | **yes** |

From the log:

```
describeInContext: context=filter  withDepth=0      <- dropped on a timeline clip
describeInContext: context=general withDepth=1      <- added as a Fusion node
createInstance: dual=1 src=00000219F4CB26C0 depth=00000219F4CB2CB0
changedClip: 'Depth' connected=1
```

As a Fusion node the effect gets three inputs: the yellow source, a **green
`Depth` input**, and a blue `EffectMask` that Fusion adds by itself. Clip
connection changes are reported to the plugin, so it can tell whether depth is
present and say so in the UI.

`general` does appear in the host's advertised context list, but advertising it
is not the same as using it: dropping the effect on a timeline clip still gets
the filter context. So the shape of the product is decided by this result:

- **Fusion page — the real plugin.** Two inputs, wired as a node graph. This is
  where a creator would work anyway if they are compositing a depth pass.
- **Edit / Color page — single input.** Depth has to be smuggled in: packed in
  the source's alpha, or as a stacked/side-by-side source the plugin splits.
  One binary can serve both, since `describeInContext` already branches on
  context; the filter build would carry a "depth source" parameter.

### What render() actually receives

Measured over 443 render calls on the Edit page plus the Fusion node:

- **Always the full frame.** `window=(0,0)-(1920,1080)` at `scale=1.0`, matching
  `multiRes=0 tiles=0`. No tiled render windows to stitch — unusually pleasant
  for an ML effect.
- **float32 RGBA, `rowBytes=30720`** = 1920 x 4 channels x 4 bytes, exactly
  packed with no row padding. It converts to a tensor without repacking.
- **Single render thread** (`thread=5040` on the Edit page, `34276` in Fusion;
  one thread each). One guarded ORT session should not contend, though the
  plugin should not *depend* on that.
- **Frames arrive in order but with gaps and repeats** during scrub and
  playback (845, 848, 851, ... then 876 four times). Any cache must be keyed by
  frame number rather than assuming sequential access.

## 0d — the plugins load  **ANSWERED: yes, and it is Studio**

## 0e — ONNX Runtime with a GPU EP  **HALF ANSWERED**

Phase 2 measured this outside Resolve. Full detail in `docs/phase2-onnx.md`;
the two results that bear on the architecture:

- **A 1080p frame warps in ~13 ms**, both eyes, on the CUDA execution provider
  with depth at 960x540. Interactive with headroom.
- **DirectML returns wrong pixels.** It miscomputes `row_flow_v2` by 3.38
  absolute on a 0..1 signal, in a clean isolated environment, while the grid
  build and `grid_sample` are fine on it. That kills the fallback the plan
  assumed was available, and makes the GPU path **NVIDIA-only**. CPU works
  correctly but takes 258 ms per 1080p frame — a render path, not a preview one.

### Inside Resolve's process: it works

`ofx/probe/iw3ort.cpp` runs the real warp through ORT inside Resolve. From the
log:

```
---- ONNX Runtime bring-up (OK) ----
    loading  ...\iw3ort.ofx.bundle\Contents\Win64\ort\onnxruntime.dll
    actually loaded: ...\iw3ort.ofx.bundle\Contents\Win64\ort\onnxruntime.dll
    runtime version: 1.28.0
    available providers: Tensorrt CUDA CPU
    CUDA provider attach: OK in 59.5 ms
    CreateSession OK in 41.1 ms
    provider in use: CUDA
    inference 23.5 ms on CUDA (1920x800)
```

**A CUDA execution provider comes up cleanly inside a host that already holds
its own CUDA context.** Bring-up is 60-240 ms once per process, session creation
another 41-55 ms. The classic failure this item exists to catch does not happen.

Note `actually loaded` — that line exists because of the next finding, and it
confirms the mitigation works.

### Resolve ships its own ONNX Runtime, and it would have been loaded instead

`C:\Program Files\Blackmagic Design\DaVinci Resolve\onnxruntime.dll` is
**version 1.13, from October 2022, CPU-only** — it exports `OrtGetApiBase` and
`OrtSessionOptionsAppendExecutionProvider_CPU` and nothing else.

It sits in the application directory, which Windows searches early when
resolving a DLL by name. A plugin with `onnxruntime.dll` in its import table
would bind to *that* copy: no CUDA provider, and an API surface fifteen versions
behind the one it was compiled against.

The mitigation, in `ofx/probe/ort_runtime.h`: **no import library and no
link-time dependency at all.** The runtime is loaded from an absolute path
inside the plugin's own bundle with `LOAD_WITH_ALTERED_SEARCH_PATH` (so its
sibling provider DLLs resolve from the same folder) and reached through
`OrtGetApiBase`, which is what the C API is designed for. `dumpbin /DEPENDENTS`
on the built plugin lists no ONNX Runtime at all, and the log confirms the
bundle's copy is the one in use.

This applies to any plugin shipping ORT into Resolve, and it would have been an
extremely confusing failure to debug after the fact.

## 0f — colour management of a depth clip  **ANSWERED: it is rescaled**

Measured by logging raw pixel statistics in the plugin and decoding the same
frame of the same file directly.

Source: `F:\_3D_Test_\iw3_ext_demo\dc_depth.mp4`, HEVC `yuv420p10le`, 960x384,
`color_range=1` (limited). Frame 717, R channel, every 4th pixel:

| | min | max | mean |
| --- | --- | --- | --- |
| **what the plugin received** | **0.21021** | **1.00360** | **0.67265** |
| file, Y expanded 16-235 -> 0-1 | 0.20091 | 1.00457 | 0.67095 | 
| file, decoded to full-range RGB | 0.19216 | 0.99608 | 0.66501 |
| file, raw Y / 255 | 0.23529 | 0.92549 | 0.63897 |

**Resolve applies a limited-to-full range expansion**, matching the middle row
and not the others. The giveaway is that the values *exceed 1.0*: no untouched
depth map does that, and only the range expansion produces it.

Two things follow, and the second is the more consequential:

**The transform is affine, not a gamma curve.** `v_full = (v*255 - 16) / 219`.
That means the depth's structure survives — it is rescaled, not bent — so the
error shows up as an effective change in divergence and convergence rather than
as distorted geometry. It is also exactly invertible. The fix is either to set
the depth clip's **Data Levels to Full** in Resolve's clip attributes, or to let
the plugin undo it; the former is better, because the plugin cannot reliably
tell whether an expansion happened.

**Resolve resizes the depth clip to the composition resolution.** The file is
960x384. The plugin receives it at **1920x800**, the comp size, as
`bounds=(0,0)-(1920,800)`. So the plugin never sees depth at its native
resolution and cannot infer it.

That matters more than it first appears. iw3's `stereo_width` exists precisely
because full-resolution depth breaks the warp — it is what stops the striping.
Resolve hands the plugin full-comp-resolution depth *unconditionally*, which is
the exact failure case. So **`stereo_width` is not optional in this plugin**: it
has to downsample the depth it is given, every time, and the resampling has to
be the antialiased filter from Phase 2 rather than a naive one.

## What Phase 3 inherits

Carried forward into the plugin's design, all of it measured rather than assumed:

- Render is called with the **full frame**, at scale 1.0, float32 RGBA exactly
  packed, on a **single thread**, with frames in rough order but with gaps and
  repeats. Cache by frame number; one guarded ORT session suffices.
- **Load ONNX Runtime by absolute path from inside the bundle.** Never link it.
- **Set `use_tf32=0`** on the CUDA provider.
- **Always downsample the depth** to a `stereo_width`, with the antialiased
  filter from Phase 2.
- **Tell users to set the depth clip's Data Levels to Full**, and say why.
- Bring-up costs ~100-300 ms once; budget it on first render, not per frame.
- The probe's per-pixel `getPixelAddress` packing is the wrong approach for the
  real plugin — it costs more than the inference does. Use row pointers, or the
  OFX GPU render suites so the buffers never leave the GPU.
