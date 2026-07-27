"""Validate and time stereo_warp.onnx on whatever execution providers are here.

Needs only onnxruntime and numpy, so it runs in a throwaway venv holding a
single ORT build. That isolation is the point: onnxruntime-gpu and
onnxruntime-directml both install a package called `onnxruntime` and overwrite
each other's DLLs, and a mixed install produces silently wrong numbers.

    python tools/check_ort.py --models-dir models
"""

import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort

CASES = ["match", "small_depth", "square_depth", "hd", "odd"]
INPUTS = ["image", "x", "delta_scale"]


USE_TF32 = 0


def make_session(path, provider):
    options = ort.SessionOptions()
    options.log_severity_level = 3
    if provider == "CUDAExecutionProvider":
        # TF32 is on by default and costs three orders of magnitude of accuracy
        # for well under a millisecond, so the shipped configuration turns it
        # off. --tf32 puts it back to show the difference.
        entry = (provider, {"use_tf32": USE_TF32})
    else:
        entry = provider
    return ort.InferenceSession(path, options, providers=[entry])


def check_provider(path, reference, provider):
    result = {"provider": provider}
    try:
        session = make_session(path, provider)
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        return result

    # Did ORT actually place the graph on this provider, or quietly fall back?
    result["providers_in_use"] = session.get_providers()

    worst = 0.0
    per_case = {}
    for case in CASES:
        feed = {name: reference[f"{case}.{name}"] for name in INPUTS}
        try:
            left, right = session.run(["left", "right"], feed)
        except Exception as error:
            per_case[case] = f"{type(error).__name__}: {error}"
            worst = float("inf")
            continue
        diff = max(float(np.abs(left - reference[f"{case}.left"]).max()),
                   float(np.abs(right - reference[f"{case}.right"]).max()))
        per_case[case] = diff
        worst = max(worst, diff)
    result["max_abs_diff"] = worst
    result["per_case"] = per_case
    return result


def time_provider(path, provider, shapes, repeats=30):
    """Wall time per frame. Warms up first; the first call builds kernels."""
    timings = {}
    try:
        session = make_session(path, provider)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}

    for label, (image_shape, depth_shape) in shapes.items():
        rng = np.random.default_rng(0)
        x = np.empty((depth_shape[0], 3) + depth_shape[2:], dtype=np.float32)
        x[:, 0] = rng.random(depth_shape, dtype=np.float32)[:, 0]
        x[:, 1] = 0.12
        x[:, 2] = -0.06
        feed = {
            "image": rng.random(image_shape, dtype=np.float32),
            "x": x,
            # a 0-d array, not a numpy scalar: ORT rejects np.float32(...)
            "delta_scale": np.array(1.0 / (depth_shape[3] // 2 - 1), dtype=np.float32),
        }
        try:
            for _ in range(3):
                session.run(["left", "right"], feed)
            start = time.perf_counter()
            for _ in range(repeats):
                session.run(["left", "right"], feed)
            elapsed = (time.perf_counter() - start) / repeats
            timings[label] = round(elapsed * 1000, 2)
        except Exception as error:
            timings[label] = f"{type(error).__name__}: {str(error)[:120]}"
    return timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--json", help="also write the results here")
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--tf32", action="store_true", help="let the CUDA EP use TF32")
    args = parser.parse_args()

    global USE_TF32
    USE_TF32 = 1 if args.tf32 else 0

    path = os.path.join(args.models_dir, "stereo_warp.onnx")
    reference = np.load(os.path.join(args.models_dir, "reference.npz"))

    report = {
        "onnxruntime": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "checks": [],
    }
    print(f"onnxruntime {ort.__version__}")
    print(f"available: {report['available_providers']}")

    model_path = os.path.join(args.models_dir, "row_flow_v2.onnx")
    print("\nnetwork alone (row_flow_v2.onnx)")
    for provider in report["available_providers"]:
        try:
            session = make_session(model_path, provider)
            worst = 0.0
            for case in CASES:
                delta = session.run(["delta"], {"x": reference[f"{case}.x"]})[0]
                worst = max(worst, float(np.abs(delta - reference[f"{case}.delta"]).max()))
            print(f"  {provider:28s} max abs diff = {worst:.3e}")
            report.setdefault("model_only", {})[provider] = worst
        except Exception as error:
            print(f"  {provider:28s} FAILED {type(error).__name__}: {str(error)[:100]}")

    print("\nwhole pipeline (stereo_warp.onnx)")
    for provider in report["available_providers"]:
        check = check_provider(path, reference, provider)
        report["checks"].append(check)
        if "error" in check:
            print(f"  {provider:28s} FAILED  {check['error'][:120]}")
            continue
        verdict = "OK" if check["max_abs_diff"] < 1e-3 else "WRONG"
        print(f"  {provider:28s} {verdict:5s} max abs diff = {check['max_abs_diff']:.3e}"
              f"  (in use: {','.join(p.replace('ExecutionProvider', '') for p in check['providers_in_use'])})")
        for case, diff in check["per_case"].items():
            shown = f"{diff:.3e}" if isinstance(diff, float) else diff[:80]
            print(f"      {case:14s} {shown}")

    if not args.skip_timing:
        shapes = {
            "1080p (depth 960x540)": ((1, 3, 1080, 1920), (1, 1, 540, 960)),
            "1080p (depth 1920x1080)": ((1, 3, 1080, 1920), (1, 1, 1080, 1920)),
            "4K (depth 1920x1080)": ((1, 3, 2160, 3840), (1, 1, 1080, 1920)),
            "4K (depth 3840x2160)": ((1, 3, 2160, 3840), (1, 1, 2160, 3840)),
        }
        print("\nper-frame wall time, ms (both eyes, one session.run)")
        report["timings"] = {}
        for provider in report["available_providers"]:
            timings = time_provider(path, provider, shapes)
            report["timings"][provider] = timings
            print(f"  {provider}")
            for label, value in timings.items():
                print(f"      {label:26s} {value}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
