# resolve-iw3

A DaVinci Resolve stereo-conversion plugin built on iw3's warp and inpaint
models, with depth supplied from outside rather than estimated.

iw3 (part of [nunif](https://github.com/nagadomi/nunif)) converts 2D video to
stereo 3D in two halves: depth estimation, then stereo synthesis. Once depth
comes from somewhere else — a depth pass rendered in another tool, a clip on a
timeline — only the second half still matters. This repo extracts that half and
builds it into a tool creators already use.

## Layout

| Path | What |
| --- | --- |
| `stereo_warp.py` | Phase 1. Standalone PyTorch stereo synthesis, one file, no nunif imports. |
| `stereo_warp_onnx.py` | Phase 2. The same thing on ONNX Runtime and numpy, no PyTorch. |
| `export_onnx.py` | Builds `models/*.onnx` and the reference data. |
| `tools/check_ort.py` | Validates and times any execution provider; needs only ORT and numpy. |
| `tests/` | Golden test against stock iw3 at diff 0, and ONNX against PyTorch. |
| `ofx/` | CMake build for OFX plugins, against the OpenFX SDK Resolve ships. |
| `ofx/plugin/` | **The plugin.** `iw3stereo.cpp` is the OFX glue; `stereo_pipeline.cpp` is the numeric core. |
| `ofx/common/` | ONNX Runtime loader (dynamic, never linked) and the log. |
| `ofx/probe/` | Phase 0 probe plugins — instrumentation, not product. |
| `tests/cpp/` | Checks the C++ numeric core against the Python implementation. |
| `docs/phase3-interface.md` | The agreed plugin interface. |
| `scripts/install-ofx.ps1` | Copies built bundles to Resolve's OFX directory (needs elevation). |
| `scripts/resolve_probe.py` | Queries a running Resolve via the scripting API. |
| `docs/phase0-findings.md` | What Resolve can and cannot do, with evidence. |
| `docs/phase2-onnx.md` | What exported, what did not, and the timing table. |

## Status

| Phase | State |
| --- | --- |
| 0 — de-risk Resolve | **done** — the answer is an OFX C++ plugin, as a Fusion node |
| 1 — standalone PyTorch | **done**, 20/20 at max abs diff 0 vs stock iw3 |
| 2 — ONNX | **done**, 12/12 vs PyTorch within 2e-4; 1080p in ~13 ms on CUDA |
| 3 — the plugin | **built**, 36/36 on the numeric core; awaiting verification in Resolve |

**iw3 Stereo** is a Fusion node with Source and Depth inputs, producing an
anaglyph, either eye, or half SBS. Interface in `docs/phase3-interface.md`.

Correctness carries across three hops, each measured rather than assumed:
`stereo_warp.py` matches stock iw3 at **diff 0**; the ONNX implementation
matches that within **2e-4**; and the C++ numeric core matches the Python at
**float32 epsilon**.

Three constraints worth knowing before reading further, each with evidence in
`docs/`:

- **Fusion page only** for two-input operation — Edit and Color give an OFX
  effect the filter context, which permits one input.
- **NVIDIA only** for the GPU path — DirectML miscomputes `row_flow_v2` by whole
  units and is refused in code.
- **Never link ONNX Runtime** — Resolve ships its own `onnxruntime.dll` (1.13,
  CPU-only) in the application directory, and a normal import would bind to it.

## Running the tests

The tests compare against stock iw3, so they need nunif and its venv:

```bash
cd F:/_AI_PROJECTS_/resolve-iw3 && F:/_AI_PROJECTS_/nunif/venv/Scripts/python.exe -m unittest discover -s tests -v
```

Set `NUNIF_ROOT` if nunif is not at `F:\_AI_PROJECTS_\nunif`.

## Building the OFX plugins

```bash
cd F:/_AI_PROJECTS_/resolve-iw3/ofx && cmake -S . -B build -G "Visual Studio 17 2022" -A x64 && cmake --build build --config Release
```

Then install from an **elevated** PowerShell and restart Resolve:

```bash
powershell -ExecutionPolicy Bypass -File scripts\install-ofx.ps1
```

## Testing the C++ numeric core

The C++ is a port of Python that matches iw3 exactly, so the port is checked
against that Python rather than against re-derived expectations. Generate the
reference data, then run the test:

```bash
cd F:/_AI_PROJECTS_/resolve-iw3 && F:/_AI_PROJECTS_/nunif/venv/Scripts/python.exe tools/dump_pipeline_reference.py && ./ofx/build/tests/test_pipeline.exe tests/cpp/pipeline_reference.bin
```

## Licence

nunif is MIT. See `docs/phase0-findings.md` for the position on the model
weights.
