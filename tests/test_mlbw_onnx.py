"""The exported mask_mlbw_l2 network against the PyTorch one.

The bar is a float tolerance, not 0: different kernels, and the graph runs the
head-sliced attention while the reference runs the fused one. The comparison is
against ``stereo_warp.MLBW`` rather than against re-derived expectations, so
this suite inherits that model's diff-0 guarantee against iw3.

Only the network is a graph. The plan had the warp in here too -- and unlike
MonoBW's warp, every operation in it does have an ONNX equivalent -- but iw3
resizes the layer weights with PyTorch's antialiased kernel, which ignores
``align_corners`` and applies a half-pixel transform instead, and no ONNX
``Resize`` reproduces that. ``export_mlbw_net``'s docstring has the numbers.
The resize, the two backward warps and the blend go to CUDA, which is the same
split ``monobw_inpaint`` already uses.

So the second class below is the one that matters: the real pipeline with the
ONNX network substituted for the PyTorch one, which is how the plugin will be
put together.

    F:\\_AI_PROJECTS_\\nunif\\venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import numpy as np
import onnxruntime as ort
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import stereo_inpaint  # noqa: E402
import stereo_warp  # noqa: E402

MODELS_DIR = os.path.join(REPO_ROOT, "models")
GRAPH = os.path.join(MODELS_DIR, "mlbw_net.onnx")
_CHECKPOINT_DIR = os.path.join(
    os.environ.get("NUNIF_ROOT", r"F:\_AI_PROJECTS_\nunif"),
    "iw3", "pretrained_models", "hub", "checkpoints")
CHECKPOINT = os.path.join(_CHECKPOINT_DIR, "iw3_mask_mlbw_l2_d1_20250903.pth")
INPAINT_CHECKPOINT = os.path.join(_CHECKPOINT_DIR, "iw3_light_inpaint_v1_20250919.pth")

# The graph is float32 -- see export_mlbw_net for why fp16 is not an option for
# a model whose output becomes grid coordinates -- so this is not fp16 round-off.
# It is the head-sliced attention the exporter needs, which sums in a different
# order and lands about 5e-5 from the fused kernel.
TOLERANCE = 2e-4

# Logits are unbounded, so they get their own bar. Looser in absolute terms and
# much tighter relative to the values involved.
MASK_TOLERANCE = 2e-3

# A delta difference of 2e-5 is a sub-pixel shift, so the finished eye is
# actually *tighter* than the heads that produced it -- 1.3e-6 measured.
PIPELINE_TOLERANCE = 2e-4

# Through the fill it is different in kind, not degree, and this is the number
# that matters for the plugin. The mask is `sigmoid(logits) > 0.15`, and a
# threshold has no tolerance: a logit sitting within 5e-5 of the boundary lands
# on either side depending on which kernel computed it, and a mask pixel that
# flips gets *filled*, which is an O(1) change to that pixel.
#
# Measured at HD: exactly one pixel per eye in 1,989,120 crossed the threshold,
# and that single pixel accounts for the whole 1.1e-2 maximum. Around it, the
# fill's 15-tap feather spreads the change to about 17 neighbours above 2e-4.
# Everything else agrees to 1e-6.
#
# So a max-abs assertion is the wrong instrument here: loose enough to pass, it
# would catch nothing. Two statistics that do mean something instead -- the mean,
# which any systematic error moves and a handful of pixels cannot, and the count
# of materially different pixels, which isolated flips cannot inflate.
#
# The same thing will happen between the plugin's CUDA mask and iw3's, for the
# same reason, and it is not a defect in either.
MEAN_TOLERANCE = 1e-6    # measured 8.8e-08
FLIP_MAGNITUDE = 1e-3    # what "materially different" means
FLIP_FRACTION = 1e-5     # measured 5.0e-07, i.e. one pixel

SIZES = [(216, 384), (392, 938), (392, 940), (384, 960), (100, 200), (1036, 1920)]


def make_pair(height, width, depth_height=None, depth_width=None, batch=1, seed=0):
    """Real content, not noise.

    Noise has the steepest gradient an image can have at every pixel, so a warp
    over it reports a tolerance the plugin will never see. The depth carries
    hard-edged blocks because those are what open the holes this model predicts.
    """
    generator = torch.Generator().manual_seed(seed)
    depth_height = depth_height or height
    depth_width = depth_width or width

    coarse = torch.rand((batch, 3, 12, 20), generator=generator)
    image = torch.nn.functional.interpolate(coarse, size=(height, width), mode="bilinear",
                                            align_corners=False).clamp(0, 1)
    coarse_depth = torch.rand((batch, 1, 8, 8), generator=generator)
    depth = torch.nn.functional.interpolate(coarse_depth, size=(depth_height, depth_width),
                                            mode="bilinear", align_corners=False)
    depth[:, :, depth_height // 4:depth_height // 2, depth_width // 4:depth_width // 2] = 0.95
    depth[:, :, -depth_height // 3:, -depth_width // 3:] = 0.05
    return image, depth.clamp(0, 1)


def make_input(depth, divergence=2.0, convergence=0.5, preserve_screen_border=False):
    return stereo_warp._make_input_tensor(
        depth, divergence=divergence, convergence=convergence,
        image_width=max(depth.shape[-2:]),
        preserve_screen_border=preserve_screen_border)


class _GraphModel(torch.nn.Module):
    """The ONNX network wearing ``MLBW``'s interface.

    ``apply_divergence_mlbw`` only ever calls ``model(x)`` and reads
    ``num_layers``, so this is enough to run the real warp against the graph.
    An ``nn.Module`` rather than a plain object so it can be dropped into
    ``MLBWInpaintImage`` in place of the real one.
    """

    num_layers = 2

    def __init__(self, session):
        super().__init__()
        self.session = session

    def forward(self, x):
        outputs = self.session.run(["delta", "layer_weight", "mask_logits"],
                                   {"x": x.numpy().astype(np.float32)})
        return tuple(torch.from_numpy(array) for array in outputs)


class MLBWNetOnnxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(GRAPH):
            raise unittest.SkipTest("mlbw_net.onnx not exported; run export_onnx.py")
        if not os.path.exists(CHECKPOINT):
            raise unittest.SkipTest(f"checkpoint not found: {CHECKPOINT}")
        cls.session = ort.InferenceSession(GRAPH, providers=["CPUExecutionProvider"])
        cls.model = stereo_warp.load_mask_mlbw_l2(CHECKPOINT, device="cpu")
        cls.graph_model = _GraphModel(cls.session)

    def assert_close(self, depth, label="", **settings):
        x = make_input(depth, **settings)
        got = self.graph_model(x)
        with torch.inference_mode():
            want = self.model(x)

        for name, expected, actual in zip(("delta", "layer_weight", "mask_logits"), want, got):
            self.assertEqual(expected.shape, actual.shape, f"{name} shape {label}")
            diff = float((expected - actual).abs().max())
            bar = MASK_TOLERANCE if name == "mask_logits" else TOLERANCE
            self.assertLess(diff, bar, f"{name} max abs diff {diff:.3e} {label}")

    # -- shapes ------------------------------------------------------------

    def test_shapes(self):
        for height, width in SIZES:
            with self.subTest(size=(height, width)):
                _, depth = make_pair(height, width, seed=height)
                self.assert_close(depth, label=f"{height}x{width}")

    def test_exact_multiple_of_the_pad_modulus(self):
        """A width that is already a multiple still gets a whole block of pad."""
        for height, width in ((128, 192), (64, 256)):
            with self.subTest(size=(height, width)):
                _, depth = make_pair(height, width, seed=width)
                self.assert_close(depth, label=f"{height}x{width}")

    def test_odd_dimensions(self):
        for height, width in ((97, 161), (101, 103), (63, 65)):
            with self.subTest(size=(height, width)):
                _, depth = make_pair(height, width, seed=height)
                self.assert_close(depth, label=f"{height}x{width}")

    def test_batched(self):
        _, depth = make_pair(216, 384, batch=3)
        self.assert_close(depth, label="batch 3")

    # -- settings ----------------------------------------------------------

    def test_divergence(self):
        _, depth = make_pair(216, 384)
        for divergence in (0.0, 1.0, 2.0, 5.0, 10.0):
            with self.subTest(divergence=divergence):
                self.assert_close(depth, divergence=divergence)

    def test_convergence(self):
        _, depth = make_pair(216, 384)
        for convergence in (-0.5, 0.0, 0.5, 1.0, 2.0):
            with self.subTest(convergence=convergence):
                self.assert_close(depth, convergence=convergence)

    def test_preserve_screen_border(self):
        _, depth = make_pair(216, 384)
        for divergence in (1.0, 2.0, 5.0):
            with self.subTest(divergence=divergence):
                self.assert_close(depth, divergence=divergence, preserve_screen_border=True)

    # -- the heads, which a tolerance alone would not pin down -------------

    def test_the_three_heads_are_distinct(self):
        """A graph that wired two outputs to the same tensor would still be close.

        Each head is checked against PyTorch above, so a *wrong* head is caught.
        What this rules out is the shape-compatible mistake: delta and
        layer_weight are both (B, 2, H, W) and swapping or duplicating them
        survives every shape assertion.
        """
        _, depth = make_pair(216, 384)
        delta, layer_weight, mask_logits = self.graph_model(make_input(depth, divergence=5.0))

        self.assertEqual(delta.shape[1], 2)
        self.assertEqual(layer_weight.shape[1], 2)
        self.assertEqual(mask_logits.shape[1], 1)
        self.assertGreater(float((delta - layer_weight).abs().max()), 1e-2,
                           "delta and layer_weight are the same tensor")
        # layer_weight is a softmax over the layer axis and nothing else is.
        self.assertLess(float((layer_weight.sum(dim=1) - 1.0).abs().max()), 1e-5,
                        "layer_weight is not a softmax; the heads are crossed")

    def test_the_mask_follows_the_depth(self):
        _, depth = make_pair(216, 384)
        _, other = make_pair(216, 384, seed=99)
        first = self.graph_model(make_input(depth, divergence=5.0))[2]
        second = self.graph_model(make_input(other, divergence=5.0))[2]

        self.assertGreater(float(first.std()), 1e-3, "mask logits are constant")
        self.assertGreater(float((first - second).abs().max()), 1e-2,
                           "mask did not change when the depth did")


class MLBWPipelineWithGraphTest(unittest.TestCase):
    """The composition that matters: the real pipeline, ONNX network inside.

    This is what the plugin is -- CUDA either side of one ORT call -- so it is
    the arrangement worth testing, not the network alone. The reference is the
    all-PyTorch pipeline, which the golden suite holds at diff 0 against iw3.
    """

    @classmethod
    def setUpClass(cls):
        for path in (GRAPH, CHECKPOINT, INPAINT_CHECKPOINT):
            if not os.path.exists(path):
                raise unittest.SkipTest(f"not available: {path}")
        cls.session = ort.InferenceSession(GRAPH, providers=["CPUExecutionProvider"])
        cls.model = stereo_warp.load_mask_mlbw_l2(CHECKPOINT, device="cpu")
        cls.graph_model = _GraphModel(cls.session)

    def assert_warp_close(self, image, depth, label="", **settings):
        # Pulled out once. Popping them inside each call would leave the second
        # call running on the defaults, which is a way to pass this test that
        # has nothing to do with the graph being right.
        settings.setdefault("divergence", 2.0)
        settings.setdefault("convergence", 0.5)
        with torch.inference_mode():
            want = stereo_warp.apply_divergence_mlbw(
                self.model, image, depth, synthetic_view="both", enable_amp=False, **settings)
            got = stereo_warp.apply_divergence_mlbw(
                self.graph_model, image, depth, synthetic_view="both", enable_amp=False,
                **settings)

        for name, expected, actual in zip(("left", "right", "left_mask", "right_mask"),
                                          want, got):
            self.assertEqual(expected.shape, actual.shape, f"{name} shape {label}")
            diff = float((expected - actual).abs().max())
            bar = MASK_TOLERANCE if name.endswith("mask") else PIPELINE_TOLERANCE
            self.assertLess(diff, bar, f"{name} max abs diff {diff:.3e} {label}")

    def test_both_eyes(self):
        image, depth = make_pair(216, 384)
        self.assert_warp_close(image, depth, label="matched")

    def test_depth_smaller_than_image(self):
        """The weight resize upscales here. Exercised through the real function."""
        image, depth = make_pair(432, 768, depth_height=108, depth_width=192)
        self.assert_warp_close(image, depth, label="depth 108x192 -> image 432x768")

    def test_depth_larger_than_image(self):
        """And downscales here -- which is what Inpaint Max Width produces.

        The image is capped and the depth is not, so this is the normal case
        once that setting is off Full, not an exotic one.
        """
        image, depth = make_pair(108, 192, depth_height=432, depth_width=768)
        self.assert_warp_close(image, depth, label="depth 432x768 -> image 108x192")

    def test_hd(self):
        image, depth = make_pair(1036, 1920, seed=7)
        self.assert_warp_close(image, depth, label="1036x1920")

    def test_settings(self):
        image, depth = make_pair(216, 384)
        for case in (dict(divergence=5.0), dict(convergence=1.0),
                     dict(divergence=5.0, preserve_screen_border=True)):
            with self.subTest(**case):
                self.assert_warp_close(image, depth, **case)

    def test_through_the_whole_inpaint_pipeline(self):
        """Warp, mask morphology and fill, with the graph in the warp's place.

        Judged by how many pixels disagree rather than by how much the worst one
        does, because the mask threshold makes the worst pixel uninformative.
        See FLIP_FRACTION for the measurement behind that.
        """
        inpaint = stereo_inpaint.load_light_inpaint_v1(INPAINT_CHECKPOINT, device="cpu")
        reference = stereo_inpaint.MLBWInpaintImage(inpaint, self.model, device="cpu")
        substituted = stereo_inpaint.MLBWInpaintImage(inpaint, self.model, device="cpu")
        substituted.mask_mlbw = self.graph_model

        for height, width in ((216, 384), (1036, 1920)):
            with self.subTest(size=(height, width)):
                image, depth = make_pair(height, width, seed=height)
                with torch.inference_mode():
                    want = stereo_inpaint.synthesize_stereo_inpaint(
                        image, depth, reference, divergence=5.0, convergence=0.5,
                        enable_amp=False)
                    got = stereo_inpaint.synthesize_stereo_inpaint(
                        image, depth, substituted, divergence=5.0, convergence=0.5,
                        enable_amp=False)

                for eye, expected, actual in zip(("left", "right"), want, got):
                    self.assertEqual(expected.shape, actual.shape, f"{eye} shape")
                    difference = (expected - actual).abs().max(dim=1).values

                    mean = float(difference.mean())
                    self.assertLess(mean, MEAN_TOLERANCE,
                                    f"{eye} mean abs diff {mean:.3e}; that is systematic, "
                                    f"not a few pixels crossing the threshold")

                    material = int((difference > FLIP_MAGNITUDE).sum())
                    fraction = material / difference.numel()
                    self.assertLess(
                        fraction, FLIP_FRACTION,
                        f"{eye}: {material} of {difference.numel()} pixels differ by more "
                        f"than {FLIP_MAGNITUDE:.0e}; that is more than threshold jitter")


if __name__ == "__main__":
    unittest.main()
