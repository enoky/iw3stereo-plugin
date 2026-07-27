"""Golden test: stereo_warp must match stock iw3 at max absolute difference 0.

Anything above 0 means a setting diverged, not that floating point drifted --
both sides run the same torch ops on the same device, so equality is the
correct bar rather than an aspiration.

Run with nunif's venv, which already has torch:

    F:\\_AI_PROJECTS_\\nunif\\venv\\Scripts\\python.exe -m unittest discover -s tests -v

Set NUNIF_ROOT if nunif lives somewhere other than F:\\_AI_PROJECTS_\\nunif.
"""

import os
import sys
import unittest

import torch

NUNIF_ROOT = os.environ.get("NUNIF_ROOT", r"F:\_AI_PROJECTS_\nunif")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, NUNIF_ROOT)

import stereo_warp  # noqa: E402

from iw3 import models as _iw3_models  # noqa: E402,F401  (registers sbs.row_flow_v2)
from iw3.stereo_model_factory import create_stereo_model  # noqa: E402
from iw3.utils import apply_divergence  # noqa: E402

CHECKPOINTS = {
    "row_flow_v2": os.path.join(NUNIF_ROOT, "iw3", "pretrained_models", "hub", "checkpoints",
                                "iw3_row_flow_v2_20240130.pth"),
    "row_flow_v3": os.path.join(NUNIF_ROOT, "iw3", "pretrained_models", "hub", "checkpoints",
                                "iw3_row_flow_v3_20250627.pth"),
}
CHECKPOINT = CHECKPOINTS["row_flow_v2"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_ID = 0 if DEVICE == "cuda" else -1


class Args:
    """The subset of iw3's args that apply_divergence() reads for row_flow_v2."""

    def __init__(self, **overrides):
        self.method = "row_flow_v2"
        self.model_name = "row_flow_v2"
        self.divergence = 2.0
        self.convergence = 0.5
        self.synthetic_view = "both"
        self.mapper = "none"
        self.stereo_width = None
        self.warp_steps = None
        self.preserve_screen_border = False
        self.disable_amp = False
        self.state = {"convergence_model": None}
        for key, value in overrides.items():
            setattr(self, key, value)


def make_pair(height, width, depth_height=None, depth_width=None, batch=1, seed=0, device="cpu"):
    """A deterministic colour/depth pair with enough structure to be a real test.

    Pure noise would warp fine but exercise none of the smooth-gradient and
    hard-edge cases the model was trained for, so the depth is built from a
    low-resolution field upsampled, plus a couple of hard-edged blocks.
    """
    generator = torch.Generator().manual_seed(seed)
    depth_height = depth_height or height
    depth_width = depth_width or width

    image = torch.rand((batch, 3, height, width), generator=generator)
    ramp = torch.linspace(0, 1, width).reshape(1, 1, 1, width)
    image = (image * 0.4 + ramp * 0.6).clamp(0, 1)

    coarse = torch.rand((batch, 1, 8, 8), generator=generator)
    depth = torch.nn.functional.interpolate(
        coarse, size=(depth_height, depth_width), mode="bilinear", align_corners=False)
    depth[:, :, depth_height // 4:depth_height // 2, depth_width // 4:depth_width // 2] = 0.95
    depth[:, :, -depth_height // 3:, -depth_width // 3:] = 0.05
    depth = depth.clamp(0, 1)

    return image.to(device), depth.to(device)


class GoldenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(CHECKPOINT):
            raise unittest.SkipTest(f"checkpoint not downloaded: {CHECKPOINT}")
        cls.reference_model = create_stereo_model("row_flow_v2", divergence=2.0, device_id=DEVICE_ID)
        cls.model = stereo_warp.load_row_flow_v2(CHECKPOINT, device=DEVICE)

    def assert_identical(self, image, depth, **settings):
        args = Args(**settings)

        with torch.inference_mode():
            reference = apply_divergence(depth.clone(), image.clone(), args, self.reference_model)
            actual = stereo_warp.synthesize_stereo(
                image.clone(), depth.clone(), self.model,
                divergence=args.divergence,
                convergence=args.convergence,
                synthetic_view=args.synthetic_view,
                stereo_width=args.stereo_width,
                steps=args.warp_steps,
                preserve_screen_border=args.preserve_screen_border,
                enable_amp=not args.disable_amp,
                mapper=None if args.mapper == "none" else args.mapper,
            )

        for eye, expected, got in zip(("left", "right"), reference, actual):
            self.assertEqual(expected.shape, got.shape, f"{eye} shape, settings={settings}")
            diff = (expected.float() - got.float()).abs().max().item()
            self.assertEqual(diff, 0.0, f"{eye} max abs diff {diff} != 0, settings={settings}")

    # -- one axis at a time ------------------------------------------------

    def test_divergence(self):
        image, depth = make_pair(96, 160, device=DEVICE)
        for divergence in (0.0, 1.0, 2.0, 5.0, 10.0):
            with self.subTest(divergence=divergence):
                self.assert_identical(image, depth, divergence=divergence)

    def test_convergence(self):
        image, depth = make_pair(96, 160, device=DEVICE)
        for convergence in (-0.5, 0.0, 0.5, 1.0, 2.0):
            with self.subTest(convergence=convergence):
                self.assert_identical(image, depth, convergence=convergence)

    def test_synthetic_view(self):
        image, depth = make_pair(96, 160, device=DEVICE)
        for view in ("both", "left", "right"):
            with self.subTest(synthetic_view=view):
                self.assert_identical(image, depth, synthetic_view=view)

    def test_warp_steps(self):
        image, depth = make_pair(96, 160, device=DEVICE)
        for steps in (None, 1, 2, 3):
            with self.subTest(warp_steps=steps):
                self.assert_identical(image, depth, warp_steps=steps)

    def test_preserve_screen_border(self):
        image, depth = make_pair(96, 160, device=DEVICE)
        for divergence in (1.0, 2.0, 5.0):
            with self.subTest(divergence=divergence):
                self.assert_identical(image, depth, divergence=divergence,
                                      preserve_screen_border=True)

    def test_mapper(self):
        image, depth = make_pair(96, 160, device=DEVICE)
        for mapper in ("none", "pow2", "softplus", "mul_2", "inv_mul_1",
                       "div_6", "shift_20", "mul_1+mul_2=0.25", "pow2:mul_1"):
            with self.subTest(mapper=mapper):
                args = Args(mapper=mapper)
                with torch.inference_mode():
                    reference = apply_divergence(depth.clone(), image.clone(), args, self.reference_model)
                    actual = stereo_warp.synthesize_stereo(
                        image.clone(), depth.clone(), self.model, mapper=mapper,
                        divergence=args.divergence, convergence=args.convergence)
                for eye, expected, got in zip(("left", "right"), reference, actual):
                    diff = (expected.float() - got.float()).abs().max().item()
                    self.assertEqual(diff, 0.0, f"{eye} diff {diff} != 0, mapper={mapper}")

    # -- the diff-0 result above is only meaningful if the warp did work ---

    def test_warp_actually_does_something(self):
        """Guard against a vacuous pass.

        If both implementations were broken into returning the input unchanged,
        every equality assertion above would still hold.
        """
        image, depth = make_pair(96, 160, device=DEVICE)
        with torch.inference_mode():
            left, right = stereo_warp.synthesize_stereo(
                image, depth, self.model, divergence=5.0, convergence=0.5)

        self.assertGreater((left - right).abs().max().item(), 0.05, "eyes are identical")
        self.assertGreater((left - image).abs().max().item(), 0.05, "left eye is the source")
        self.assertGreater((right - image).abs().max().item(), 0.05, "right eye is the source")

        # A left-eye view samples from further right, so the parallax should
        # have a consistent sign rather than being symmetric noise.
        with torch.inference_mode():
            left_only, right_passthrough = stereo_warp.synthesize_stereo(
                image, depth, self.model, divergence=5.0, convergence=0.5,
                synthetic_view="left")
        self.assertEqual((right_passthrough - image).abs().max().item(), 0.0,
                         "synthetic_view='left' must pass the right eye through untouched")
        self.assertGreater((left_only - image).abs().max().item(), 0.05)

    # -- stereo_width, the setting that stops full-res depth striping -------

    def test_stereo_width_unset(self):
        image, depth = make_pair(216, 384, device=DEVICE)
        self.assert_identical(image, depth, stereo_width=None)

    def test_stereo_width_set(self):
        image, depth = make_pair(216, 384, device=DEVICE)
        for stereo_width in (64, 128, 256, 384, 512):
            with self.subTest(stereo_width=stereo_width):
                # 512 exceeds the image width and must clamp to it.
                self.assert_identical(image, depth, stereo_width=stereo_width)

    def test_stereo_width_with_mismatched_depth(self):
        image, depth = make_pair(216, 384, depth_height=518, depth_width=518, device=DEVICE)
        for stereo_width in (None, 128, 384):
            with self.subTest(stereo_width=stereo_width):
                self.assert_identical(image, depth, stereo_width=stereo_width)

    # -- shapes ------------------------------------------------------------

    def test_depth_smaller_than_image(self):
        image, depth = make_pair(216, 384, depth_height=108, depth_width=192, device=DEVICE)
        self.assert_identical(image, depth)

    def test_depth_larger_than_image(self):
        image, depth = make_pair(108, 192, depth_height=216, depth_width=384, device=DEVICE)
        self.assert_identical(image, depth)

    def test_depth_different_aspect_ratio(self):
        image, depth = make_pair(216, 384, depth_height=518, depth_width=518, device=DEVICE)
        self.assert_identical(image, depth)

    def test_odd_dimensions(self):
        for height, width in ((97, 161), (99, 199), (101, 103), (63, 65)):
            with self.subTest(size=(height, width)):
                image, depth = make_pair(height, width, device=DEVICE)
                self.assert_identical(image, depth)

    def test_odd_depth_dimensions(self):
        image, depth = make_pair(216, 384, depth_height=107, depth_width=193, device=DEVICE)
        self.assert_identical(image, depth)

    def test_batched(self):
        image, depth = make_pair(96, 160, batch=3, device=DEVICE)
        self.assert_identical(image, depth)

    def test_unbatched_chw(self):
        image, depth = make_pair(96, 160, device=DEVICE)
        image, depth = image.squeeze(0), depth.squeeze(0)
        self.assert_identical(image, depth)
        self.assertEqual(image.ndim, 3)

    # -- amp ---------------------------------------------------------------

    def test_amp_disabled(self):
        image, depth = make_pair(96, 160, device=DEVICE)
        self.assert_identical(image, depth, disable_amp=True)

    def test_cpu(self):
        image, depth = make_pair(96, 160, device="cpu")
        model = stereo_warp.load_row_flow_v2(CHECKPOINT, device="cpu")
        reference_model = create_stereo_model("row_flow_v2", divergence=2.0, device_id=-1)
        args = Args()
        with torch.inference_mode():
            reference = apply_divergence(depth.clone(), image.clone(), args, reference_model)
            actual = stereo_warp.synthesize_stereo(image.clone(), depth.clone(), model)
        for eye, expected, got in zip(("left", "right"), reference, actual):
            diff = (expected.float() - got.float()).abs().max().item()
            self.assertEqual(diff, 0.0, f"cpu {eye} diff {diff} != 0")

    # -- combinations ------------------------------------------------------

    def test_combinations(self):
        image, depth = make_pair(216, 384, depth_height=518, depth_width=518, device=DEVICE)
        cases = [
            dict(divergence=1.0, convergence=0.0, synthetic_view="left", stereo_width=256),
            dict(divergence=5.0, convergence=1.0, synthetic_view="right", stereo_width=128,
                 preserve_screen_border=True),
            dict(divergence=2.0, convergence=0.5, synthetic_view="both", stereo_width=384,
                 warp_steps=2),
            dict(divergence=8.0, convergence=0.25, synthetic_view="both", stereo_width=None,
                 preserve_screen_border=True, warp_steps=3),
        ]
        for case in cases:
            with self.subTest(**case):
                self.assert_identical(image, depth, **case)


class GoldenV3Test(GoldenTest):
    """The same matrix again against row_flow_v3.

    It is a windowed-attention transformer rather than a convolution stack, but
    the warp around it is unchanged -- same three input channels, same delta
    contract, same non-symmetric path -- so every case above should hold at
    difference 0 for it too.
    """

    MODEL = "row_flow_v3"

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(CHECKPOINTS["row_flow_v3"]):
            raise unittest.SkipTest("row_flow_v3 checkpoint not downloaded")
        cls.reference_model = create_stereo_model("row_flow_v3", divergence=2.0, device_id=DEVICE_ID)
        cls.model = stereo_warp.load_row_flow_v3(CHECKPOINTS["row_flow_v3"], device=DEVICE)

    def assert_identical(self, image, depth, **settings):
        settings.setdefault("method", "row_flow_v3")
        super().assert_identical(image, depth, **settings)

    def test_cpu(self):
        image, depth = make_pair(96, 160, device="cpu")
        model = stereo_warp.load_row_flow_v3(CHECKPOINTS["row_flow_v3"], device="cpu")
        reference_model = create_stereo_model("row_flow_v3", divergence=2.0, device_id=-1)
        args = Args(method="row_flow_v3")
        with torch.inference_mode():
            reference = apply_divergence(depth.clone(), image.clone(), args, reference_model)
            actual = stereo_warp.synthesize_stereo(image.clone(), depth.clone(), model)
        for eye, expected, got in zip(("left", "right"), reference, actual):
            diff = (expected.float() - got.float()).abs().max().item()
            self.assertEqual(diff, 0.0, f"cpu {eye} diff {diff} != 0")

    def test_mapper(self):
        image, depth = make_pair(96, 160, device=DEVICE)
        for mapper in ("none", "pow2", "mul_2", "div_6"):
            with self.subTest(mapper=mapper):
                args = Args(mapper=mapper, method="row_flow_v3")
                with torch.inference_mode():
                    reference = apply_divergence(depth.clone(), image.clone(), args, self.reference_model)
                    actual = stereo_warp.synthesize_stereo(
                        image.clone(), depth.clone(), self.model, mapper=mapper,
                        divergence=args.divergence, convergence=args.convergence)
                for eye, expected, got in zip(("left", "right"), reference, actual):
                    diff = (expected.float() - got.float()).abs().max().item()
                    self.assertEqual(diff, 0.0, f"{eye} diff {diff} != 0, mapper={mapper}")

    def test_is_really_v3(self):
        """Guard against the suite silently running v2 on both sides.

        setUpClass loads v3 for the reference and the standalone alike, so a
        wiring mistake that used v2 for both would still pass every case above.
        The two models must disagree.
        """
        image, depth = make_pair(96, 160, device=DEVICE)
        v2 = stereo_warp.load_row_flow_v2(CHECKPOINTS["row_flow_v2"], device=DEVICE)
        with torch.inference_mode():
            from_v3 = stereo_warp.synthesize_stereo(image, depth, self.model, divergence=5.0)
            from_v2 = stereo_warp.synthesize_stereo(image, depth, v2, divergence=5.0)
        difference = (from_v3[0] - from_v2[0]).abs().max().item()
        self.assertGreater(difference, 1e-3,
                           f"row_flow_v3 and row_flow_v2 agree to {difference}; is this really v3?")

    def test_export_safe_attention_is_close(self):
        """The export-only attention is a different reduction order, not identical."""
        image, depth = make_pair(96, 160, device=DEVICE)
        safe = stereo_warp.load_row_flow_v3(CHECKPOINTS["row_flow_v3"], device=DEVICE, export_safe=True)
        with torch.inference_mode():
            expected = stereo_warp.synthesize_stereo(image, depth, self.model)
            actual = stereo_warp.synthesize_stereo(image, depth, safe)
        for eye, want, got in zip(("left", "right"), expected, actual):
            diff = (want - got).abs().max().item()
            self.assertLess(diff, 1e-3, f"{eye} export-safe diff {diff}")


if __name__ == "__main__":
    unittest.main()
