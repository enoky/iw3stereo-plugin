# The GPU render path

The plugin asks Resolve for CUDA render, and then a frame never leaves the GPU:
Resolve's source buffer, six kernels, ONNX Runtime, and Resolve's destination
buffer are all device memory.

## How Resolve hands over

Declaring `setSupportsCudaRender(true)` and `setSupportsCudaStream(true)` changes
what `render()` receives:

- `OFX::Image::getPixelData()` returns a **device** pointer, not a host one.
- `RenderArguments::pCudaStream` is the stream to queue work on.
- `RenderArguments::isEnabledCudaRender` says whether this particular call is
  actually on the GPU.

BMD's own `GainPlugin` sample establishes that contract; `getRowBytes()` remains
a byte stride and the kernels index in floats.

## Measured

1920x800 frame, depth resized to 938x392, RTX 5080, from the plugin's own log:

| | inference | total |
| --- | --- | --- |
| first frame after load | 221.35 ms | 447.92 ms |
| steady state | **4.95 - 5.38 ms** | **5.10 - 6.33 ms** |

The gap between inference and total is 0.2-0.9 ms. That gap *is* the result:
packing, depth preparation, the resize and the tensor build together now cost
under a millisecond, where the CPU path's per-pixel packing cost more than the
inference did.

Two honest caveats on those numbers:

- **The total is not the whole frame.** ORT's `Run` synchronises, so everything
  through inference is inside the measured window, but compose and unpack are
  still queued on Resolve's stream when it returns. Forcing a synchronise to
  measure them would slow the path being measured.
- **It is not a clean comparison against the 23.5 ms the Phase 0e probe
  reported.** That probe ran with depth at full frame resolution; this runs at
  938x392. Phase 2 measured depth resolution alone as worth roughly 3x, so both
  changes contribute and the split between them is not separately measured.

### The first frame

It used to cost **~450 ms**. Two things caused that, and both are fixed:

**`cudnn_conv_algo_search` defaults to EXHAUSTIVE**, which benchmarks every
convolution algorithm the first time it meets a shape. It is now set to
HEURISTIC, which picks from cuDNN's own table. On a model this small — 30k
parameters, mostly 1x9 row convolutions — there is little for an exhaustive
search to find that a heuristic misses.

**Bring-up happened on the render thread.** The runtime was created lazily on
the first render, so the first visible frame paid for provider start-up, session
creation *and* the first inference. It now starts on a background thread from
the effect's constructor, and does a throwaway 128x128 inference there, so all
of it overlaps with the user wiring the node up. No extra synchronisation was
needed: the function-local static already blocks a render that arrives early.

## Why the arithmetic is not written twice

The obvious way to write this is to transcribe the mappers, the Dubois matrix
and the resample into `.cu`. That would put the numeric core in a file no test
covers — and the whole value of this project is that its output matches iw3.

Instead:

- `plugin/numeric_math.h` holds the pure functions, compiled `__host__
  __device__`. The kernels and the CPU path run the same lines.
- The resample weights come from `buildResampleAxis()` on the **host** and are
  uploaded. The GPU does not rederive them.

So the 36 checks in `tests/cpp/test_pipeline.cpp`, which can only run on the
CPU, cover the arithmetic the GPU executes. What they do not cover is the
kernels' indexing, which only running it in Resolve tests.

## ONNX Runtime with device tensors

`CreateMemoryInfo("Cuda", ...)` plus `CreateIoBinding` / `BindInput` /
`BindOutputToDevice` / `RunWithBinding`. Outputs come back as device pointers
from `GetBoundOutputValues`, owned until the next call, and feed straight into
the compose kernel.

`delta_scale` is deliberately left on the host: it is four bytes, and ORT wants
it where the shape arithmetic happens.

Without this, ORT would copy 25 MB up and 50 MB back per 1080p frame even though
both ends already live on the GPU.

## Left on the table

**Sharing Resolve's stream.** ORT runs on its own stream, so the plugin calls
`cudaStreamSynchronize` on Resolve's stream before inference. The CUDA provider
accepts `user_compute_stream`, which would remove that — but the session is
built once, while the host's stream is only known per render call. Worth
revisiting if the sync ever shows up in a profile; at 0.2-0.9 ms of non-inference
time it currently does not.

## Fallbacks

`renderCuda()` returns false without touching the output, and the CPU path runs,
when: the host renders without CUDA, no CUDA device is present, the ORT session
is not device-capable, a device allocation fails, the stream synchronise fails,
or a kernel launch fails. The build itself also works with no CUDA compiler at
all — CMake warns and omits the `.cu`.
