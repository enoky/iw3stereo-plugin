"""Export row_flow_v2 and the full warp pipeline to ONNX.

    F:\\_AI_PROJECTS_\\nunif\\venv\\Scripts\\python.exe export_onnx.py

Writes to models/:

    row_flow_v2.onnx    the network alone: (B,3,h,w) -> delta (B,2,h,w)
    stereo_warp.onnx    the whole warp: image + depth -> left + right
    reference.npz       inputs and PyTorch outputs, so any environment can
                        check an execution provider without needing torch

What is in the graph and what is not
------------------------------------
In:  the model, the divergence/convergence feature channels, the sampling grid,
     the grid interpolation up to image resolution, grid_sample, both eyes.
Out: the depth mapper, the stereo_width resize, and preserve_screen_border.

The mapper stays outside because it is cheap and has twenty variants that
would each need their own graph. preserve_screen_border stays outside because
its ramp width is round()ed from divergence at runtime, which is real control
flow rather than arithmetic; the graph therefore takes the model's input tensor
`x` directly rather than building it from depth.

The stereo_width resize stays outside for a harder reason: **ONNX Resize cannot
reproduce it.** Exported with antialias=0 the graph matches torch's plain
resize to 5e-5, but patching antialias=1 -- nunif's own technique, and a real
opset 18 attribute -- lands 0.23 away from torch on a 0..1 signal under every
coordinate transform mode tried. ORT's antialias is a different filter. It is
reimplemented in stereo_warp_onnx.py instead, matching torch to ~1e-5.

The caller computes three scalars, all trivial:

    base_size        = max(depth_h, depth_w)
    divergence_pix   = divergence * 0.5 * 0.01 * base_size
    divergence_value = divergence_pix / 32          -> x channel 1
    convergence_value = -divergence_pix * convergence / 32   -> x channel 2
    delta_scale      = 1 / (depth_w // 2 - 1)
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

import stereo_warp

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_CHECKPOINT = os.path.join(
    r"F:\_AI_PROJECTS_\nunif", "iw3", "pretrained_models", "hub", "checkpoints",
    "iw3_row_flow_v2_20240130.pth")

OPSET = 17


class StereoWarpGraph(torch.nn.Module):
    """The whole warp as one graph, both eyes, one call per frame.

    The right eye reuses the same left-eye network by mirroring the input and
    mirroring the result back, exactly as iw3 does.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, x, delta_scale):
        # x is the model's own input -- disparity, divergence feature,
        # convergence feature -- built by the caller rather than here. Building
        # it outside costs one extra upload of two cheap channels and buys
        # preserve_screen_border, whose border ramp is round()ed from divergence
        # at runtime and so cannot live in a static graph.
        delta_left = self.model(x)
        delta_right = self.model(torch.flip(x, (3,)))

        grid = self._grid(x)
        left = self._warp(image, grid, delta_left, delta_scale)
        right = torch.flip(
            self._warp(torch.flip(image, (3,)), grid, delta_right, delta_scale), (3,))
        return left, right

    @staticmethod
    def _grid(depth):  # only the shape of `depth` is used
        # torch.linspace bakes its length in at trace time, so the mesh is built
        # from arange over the dynamic shape instead. The operation order below
        # is linspace's own -- start + i * step, not i / (n - 1) * 2 - 1 -- which
        # matters: the naive order costs about 7e-4 of end-to-end accuracy once
        # grid_sample amplifies it across a 1920-pixel width.
        b = depth.shape[0]
        h = depth.shape[2]
        w = depth.shape[3]
        xs = -1 + torch.arange(w, dtype=depth.dtype, device=depth.device) * (2 / (w - 1))
        ys = -1 + torch.arange(h, dtype=depth.dtype, device=depth.device) * (2 / (h - 1))
        mesh_x = xs.reshape(1, 1, 1, w).expand(b, 1, h, w)
        mesh_y = ys.reshape(1, 1, h, 1).expand(b, 1, h, w)
        return torch.cat([mesh_x, mesh_y], dim=1)

    @staticmethod
    def _warp(image, grid, delta, delta_scale):
        grid = grid + delta * delta_scale
        # Unconditional. A traced `if shape != shape` bakes one branch into the
        # graph and silently produces depth-sized output forever after; that is
        # exactly the bug this line exists to avoid. When the sizes already
        # match, bilinear with align_corners=True is the identity.
        grid = F.interpolate(grid, size=image.shape[-2:], mode="bilinear",
                             align_corners=True, antialias=False)
        grid = grid.permute(0, 2, 3, 1)
        z = F.grid_sample(image, grid, mode="bilinear", padding_mode="border",
                          align_corners=True)
        return torch.clamp(z, 0, 1)


def export_model_only(model, path):
    x = torch.rand(1, 3, 216, 384)
    torch.onnx.export(
        model, (x,), path,
        input_names=["x"], output_names=["delta"],
        dynamic_axes={"x": {0: "batch", 2: "height", 3: "width"},
                      "delta": {0: "batch", 2: "height", 3: "width"}},
        opset_version=OPSET, dynamo=False)
    return path


def export_pipeline(model, path):
    graph = StereoWarpGraph(model).eval()
    # Exported with the image and the depth at *different* sizes, so the
    # interpolate is exercised at trace time rather than folded away.
    image = torch.rand(1, 3, 216, 384)
    x = torch.rand(1, 3, 108, 192)
    delta_scale = torch.tensor(1.0 / (192 // 2 - 1))
    torch.onnx.export(
        graph, (image, x, delta_scale), path,
        input_names=["image", "x", "delta_scale"],
        output_names=["left", "right"],
        dynamic_axes={"image": {0: "batch", 2: "image_height", 3: "image_width"},
                      "x": {0: "batch", 2: "depth_height", 3: "depth_width"},
                      "left": {0: "batch", 2: "image_height", 3: "image_width"},
                      "right": {0: "batch", 2: "image_height", 3: "image_width"}},
        opset_version=OPSET, dynamo=False)
    return path


def write_reference(model, path):
    """Reference IO at several shapes, so a provider can be checked without torch."""
    graph = StereoWarpGraph(model).eval()
    arrays = {}
    cases = [
        ("match", (1, 3, 216, 384), (1, 1, 216, 384)),
        ("small_depth", (1, 3, 216, 384), (1, 1, 108, 192)),
        ("square_depth", (1, 3, 216, 384), (1, 1, 518, 518)),
        # Half-height rather than true 1080p: it exercises the same path -- a
        # grid upsampled 2x, which is where the largest ORT-vs-torch difference
        # shows up -- without a 100 MiB reference file. Real 1080p is measured
        # for speed in tools/check_ort.py, where no reference is needed.
        ("hd", (1, 3, 540, 960), (1, 1, 270, 480)),
        ("odd", (1, 3, 217, 385), (1, 1, 107, 193)),
    ]
    for name, image_shape, depth_shape in cases:
        generator = torch.Generator().manual_seed(len(name))
        image = torch.rand(image_shape, generator=generator)
        depth = torch.rand(depth_shape, generator=generator)
        delta_scale = torch.tensor(1.0 / (depth_shape[3] // 2 - 1))
        x = torch.cat([depth,
                       torch.full_like(depth, 0.12),
                       torch.full_like(depth, -0.06)], dim=1)
        with torch.inference_mode():
            left, right = graph(image, x, delta_scale)
            delta = model(x)
        arrays[f"{name}.image"] = image.numpy()
        arrays[f"{name}.x"] = x.numpy()
        arrays[f"{name}.delta_scale"] = delta_scale.numpy()
        arrays[f"{name}.left"] = left.numpy()
        arrays[f"{name}.right"] = right.numpy()
        # The network on its own, so a bad provider can be localised to the
        # model rather than the warp around it.
        arrays[f"{name}.delta"] = delta.numpy()

    np.savez(path, **arrays)
    return [name for name, _, _ in cases]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=MODEL_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model = stereo_warp.load_row_flow_v2(args.checkpoint, device="cpu")

    model_path = export_model_only(model, os.path.join(args.output_dir, "row_flow_v2.onnx"))
    print(f"wrote {model_path} ({os.path.getsize(model_path) / 1024:.0f} KiB)")

    pipeline_path = export_pipeline(model, os.path.join(args.output_dir, "stereo_warp.onnx"))
    print(f"wrote {pipeline_path} ({os.path.getsize(pipeline_path) / 1024:.0f} KiB)")

    reference_path = os.path.join(args.output_dir, "reference.npz")
    cases = write_reference(model, reference_path)
    print(f"wrote {reference_path} with cases: {', '.join(cases)}")


if __name__ == "__main__":
    main()
