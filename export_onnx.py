"""Export row_flow_v2 and the full warp pipeline to ONNX.

    F:\\_AI_PROJECTS_\\nunif\\venv\\Scripts\\python.exe export_onnx.py

Writes to models/:

    row_flow_v2.onnx        the network alone: (B,3,h,w) -> delta (B,2,h,w)
    stereo_warp.onnx        the whole warp: image + depth -> left + right
    stereo_warp_v3.onnx     the same, with row_flow_v3
    light_inpaint_v1.onnx   the inpaint half of monobw_inpaint: eye + hole
                            mask -> filled eye
    light_video_inpaint_v1.onnx
                            the same, twelve frames at a time, with a temporal
                            axis so the fills do not flicker
    reference.npz           inputs and PyTorch outputs, so any environment can
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

What is in the inpaint graph, and what is not
---------------------------------------------
Same rule, applied again: fixed arithmetic goes in, anything whose *amount* is
a runtime parameter stays out.

In:  the mask blur, the blanking, the pad to a multiple of 64, the network, the
     crop back, and the composite by the mask.
Out: mask_closing and the inner/outer dilations, whose iteration counts are
     plugin parameters, and the horizontal flip that gives the left eye the
     handedness the network was trained for.

The warp half is not here at all, and cannot be: `MonoBW` is built on
`torch.cummax` and `torch.searchsorted`, and **neither has an ONNX operator**.
Everything else it needs does. See docs/monobw-inpaint.md.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

import stereo_inpaint
import stereo_warp

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
CHECKPOINT_DIR = os.path.join(
    os.environ.get("NUNIF_ROOT", r"F:\_AI_PROJECTS_\nunif"),
    "iw3", "pretrained_models", "hub", "checkpoints")
CHECKPOINTS = {
    "row_flow_v2": os.path.join(CHECKPOINT_DIR, "iw3_row_flow_v2_20240130.pth"),
    "row_flow_v3": os.path.join(CHECKPOINT_DIR, "iw3_row_flow_v3_20250627.pth"),
    "light_inpaint_v1": os.path.join(CHECKPOINT_DIR, "iw3_light_inpaint_v1_20250919.pth"),
    "light_video_inpaint_v1": os.path.join(CHECKPOINT_DIR,
                                           "iw3_light_video_inpaint_v1_20250919.pth"),
}
DEFAULT_CHECKPOINT = CHECKPOINTS["row_flow_v2"]

OPSET = 17
# row_flow_v3 needs the dynamo exporter, and that needs opset 18.
OPSET_V3 = 18
OPSET_INPAINT = 18


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


def export_pipeline_v3(model, path):
    """row_flow_v3, which needs a different exporter.

    The TorchScript exporter produces a graph that is accurate at the traced
    size and fails at every other one: the window-partition and padding reshapes
    bake in as constants. The dynamo exporter handles genuinely dynamic shapes,
    but only once two things in the model are restructured -- the attention head
    merge and the padding -- both of which stereo_warp.RowFlowV3 already does.
    See its comments for why each was necessary.

    Verified at 392x938, 392x940, 384x960, 528x940, 108x192 and 100x200, all
    within 1.2e-4 of stock iw3.
    """
    graph = StereoWarpGraph(model).eval()
    # Batch 2 in the example, deliberately. torch.export specialises any
    # dimension whose example value is 1, so tracing with a single frame bakes
    # batch=1 into the graph however the Dim is declared.
    image = torch.rand(2, 3, 216, 384)
    x = torch.rand(2, 3, 108, 192)
    delta_scale = torch.tensor(1.0 / (192 // 2 - 1))

    batch = torch.export.Dim("batch", min=1, max=64)
    image_height = torch.export.Dim("image_height", min=16, max=8192)
    image_width = torch.export.Dim("image_width", min=16, max=8192)
    depth_height = torch.export.Dim("depth_height", min=16, max=8192)
    depth_width = torch.export.Dim("depth_width", min=16, max=8192)

    torch.onnx.export(
        graph, (image, x, delta_scale), path,
        input_names=["image", "x", "delta_scale"],
        output_names=["left", "right"],
        dynamic_shapes={"image": {0: batch, 2: image_height, 3: image_width},
                        "x": {0: batch, 2: depth_height, 3: depth_width},
                        "delta_scale": {}},
        opset_version=OPSET_V3, dynamo=True, external_data=False)
    return path


class LightInpaintGraph(torch.nn.Module):
    """The inpaint half of ``monobw_inpaint`` as one graph.

    Takes the warped eye and its hole mask, returns the eye with the holes
    filled and every other pixel passed through untouched -- the composite is
    inside the graph, so the caller does not have to blend.

    ``infer()`` is the model's own method rather than ``I2IBaseModel``'s, and
    calls ``forward(skip_i2i_offset=True)``, so nothing is cropped: the output
    is the same size as the input.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, eye, mask):
        return self.model.infer(eye, mask)


def export_light_inpaint_v1(model, path):
    """``light_inpaint_v1``, the one part of this pipeline that can be a graph.

    The investigation recorded this as blocked -- accurate at the traced size,
    failing everywhere else on an ``Expand`` node with a baked dimension. That
    was iw3's ``replication_pad2d_naive``, which builds padding by Python tuple
    repetition and so needs a concrete width. ``stereo_inpaint.LightInpaintV1``
    uses ``F.pad(mode="replicate")`` instead, and the blocker goes with it: the
    change was already in the port before the export was ever attempted.
    Verified by putting the old padding back, which reproduces the failure at
    exactly the sizes first reported.

    Unlike ``row_flow_v3`` this needs neither the attention rewrite -- gMLP
    mixes tokens with a Conv1d and has no head permute to lower -- nor the
    batch-2 example, which was checked and is not required here. The example is
    batch 2 anyway, to keep the two exports the same shape.
    """
    graph = LightInpaintGraph(model).eval()
    generator = torch.Generator().manual_seed(0)
    eye = torch.rand((2, 3, 256, 448), generator=generator)
    mask = torch.zeros((2, 1, 256, 448))
    mask[:, :, 64:128, 112:150] = 1.0

    batch = torch.export.Dim("batch", min=1, max=64)
    height = torch.export.Dim("height", min=64, max=8192)
    width = torch.export.Dim("width", min=64, max=8192)

    torch.onnx.export(
        graph, (eye, mask), path,
        input_names=["eye", "mask"], output_names=["y"],
        dynamic_shapes={"eye": {0: batch, 2: height, 3: width},
                        "mask": {0: batch, 2: height, 3: width}},
        opset_version=OPSET_INPAINT, dynamo=True, external_data=False)
    return path


class LightVideoInpaintGraph(torch.nn.Module):
    """The temporal inpaint model as one graph: twelve eyes in, twelve out.

    Same boundary as the image graph -- blank, blur, network, composite -- and
    the same reason for it: the mask morphology's counts are plugin parameters.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, eyes, masks):
        return self.model.infer(eyes, masks)


def export_light_video_inpaint_v1(model, path):
    """``light_video_inpaint_v1``, twelve frames at a fixed batch.

    Easier than the image model's export rather than harder, and for a reason
    worth stating: the batch is *not* dynamic. ``enc2.1`` and ``enc2.3`` mix
    along the frame axis with a Conv1d whose weights are (12, 12, 1), so twelve
    is baked into the checkpoint and there is nothing to keep dynamic. Only
    height and width are declared, and the export succeeded first time at every
    size tried.

    The graph therefore has a fixed first dimension. A caller with fewer than
    twelve frames pads by repeating the first and last, which is what
    ``LightVideoInpaintV1.infer`` does and what a window running off the end of
    a clip needs anyway.
    """
    graph = LightVideoInpaintGraph(model).eval()
    generator = torch.Generator().manual_seed(0)
    coarse = torch.rand((stereo_inpaint.SEQ_LEN, 3, 12, 20), generator=generator)
    eyes = F.interpolate(coarse, size=(256, 448), mode="bilinear",
                         align_corners=False).clamp(0, 1)
    masks = torch.zeros((stereo_inpaint.SEQ_LEN, 1, 256, 448))
    masks[:, :, 64:128, 112:150] = 1.0

    height = torch.export.Dim("height", min=64, max=8192)
    width = torch.export.Dim("width", min=64, max=8192)

    torch.onnx.export(
        graph, (eyes, masks), path,
        input_names=["eyes", "masks"], output_names=["y"],
        dynamic_shapes={"eyes": {2: height, 3: width},
                        "masks": {2: height, 3: width}},
        opset_version=OPSET_INPAINT, dynamo=True, external_data=False)
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
    parser.add_argument("--skip-v3", action="store_true")
    parser.add_argument("--skip-inpaint", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model = stereo_warp.load_row_flow_v2(args.checkpoint, device="cpu")

    model_path = export_model_only(model, os.path.join(args.output_dir, "row_flow_v2.onnx"))
    print(f"wrote {model_path} ({os.path.getsize(model_path) / 1024:.0f} KiB)")

    pipeline_path = export_pipeline(model, os.path.join(args.output_dir, "stereo_warp.onnx"))
    print(f"wrote {pipeline_path} ({os.path.getsize(pipeline_path) / 1024:.0f} KiB)")

    if not args.skip_v3 and os.path.exists(CHECKPOINTS["row_flow_v3"]):
        # export_safe swaps in the head-sliced attention, which torch.export can
        # lower. It is about 5e-5 from the fused kernel, well inside the
        # tolerance the ONNX path is judged at.
        v3 = stereo_warp.load_row_flow_v3(CHECKPOINTS["row_flow_v3"], device="cpu", export_safe=True)
        v3_path = export_pipeline_v3(v3, os.path.join(args.output_dir, "stereo_warp_v3.onnx"))
        print(f"wrote {v3_path} ({os.path.getsize(v3_path) / 1024:.0f} KiB)")

    if not args.skip_inpaint and os.path.exists(CHECKPOINTS["light_inpaint_v1"]):
        inpaint = stereo_inpaint.load_light_inpaint_v1(CHECKPOINTS["light_inpaint_v1"], device="cpu")
        inpaint_path = export_light_inpaint_v1(
            inpaint, os.path.join(args.output_dir, "light_inpaint_v1.onnx"))
        print(f"wrote {inpaint_path} ({os.path.getsize(inpaint_path) / 1024:.0f} KiB)")

    if not args.skip_inpaint and os.path.exists(CHECKPOINTS["light_video_inpaint_v1"]):
        video = stereo_inpaint.load_light_video_inpaint_v1(
            CHECKPOINTS["light_video_inpaint_v1"], device="cpu")
        video_path = export_light_video_inpaint_v1(
            video, os.path.join(args.output_dir, "light_video_inpaint_v1.onnx"))
        print(f"wrote {video_path} ({os.path.getsize(video_path) / 1024:.0f} KiB)")

    reference_path = os.path.join(args.output_dir, "reference.npz")
    cases = write_reference(model, reference_path)
    print(f"wrote {reference_path} with cases: {', '.join(cases)}")


if __name__ == "__main__":
    main()
