"""Golden test: stereo_inpaint must match stock iw3 at max absolute difference 0.

Same bar and same reasoning as ``test_stereo_warp.py`` -- both sides run the
same torch ops on the same device, so anything above 0 means a setting
diverged, not that floating point drifted.

The reference is iw3's real ``monobw_inpaint`` path: ``create_stereo_model()``
builds the same ``MonoBWInpaint`` the CLI uses, and ``apply_divergence()``
drives it, mapper and per-frame loop included.

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

import stereo_inpaint  # noqa: E402

from iw3 import models as _iw3_models  # noqa: E402,F401  (registers sbs.monobw)
from iw3.stereo_model_factory import create_stereo_model  # noqa: E402
from iw3.utils import apply_divergence  # noqa: E402

CHECKPOINT = os.path.join(NUNIF_ROOT, "iw3", "pretrained_models", "hub", "checkpoints",
                          "iw3_light_inpaint_v1_20250919.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_ID = 0 if DEVICE == "cuda" else -1


class Args:
    """The subset of iw3's args that apply_divergence() reads for monobw_inpaint."""

    def __init__(self, **overrides):
        self.method = "monobw_inpaint"
        self.divergence = 2.0
        self.convergence = 0.5
        self.synthetic_view = "both"
        self.mapper = "none"
        self.preserve_screen_border = False
        self.mask_inner_dilation = 0
        self.mask_outer_dilation = 0
        self.inpaint_max_width = None
        self.disable_amp = False
        self.state = {"convergence_model": None}
        for key, value in overrides.items():
            setattr(self, key, value)


def make_pair(height, width, depth_height=None, depth_width=None, batch=1, seed=0, device="cpu"):
    """A deterministic colour/depth pair with enough structure to be a real test.

    The depth carries hard-edged blocks on purpose: those are what open the
    occlusion holes, and a pipeline whose whole point is filling holes tests
    nothing on a smooth depth field.
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


class GoldenInpaintTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(CHECKPOINT):
            raise unittest.SkipTest(f"checkpoint not downloaded: {CHECKPOINT}")
        cls.reference_model = create_stereo_model("monobw_inpaint", divergence=2.0, device_id=DEVICE_ID)
        cls.reference_model.set_mode("image")
        cls.model = stereo_inpaint.load_monobw_inpaint(CHECKPOINT, device=DEVICE)

    def assert_identical(self, image, depth, **settings):
        args = Args(**settings)

        with torch.inference_mode():
            reference = apply_divergence(depth.clone(), image.clone(), args, self.reference_model)
            actual = stereo_inpaint.synthesize_stereo_inpaint(
                image.clone(), depth.clone(), self.model,
                divergence=args.divergence,
                convergence=args.convergence,
                synthetic_view=args.synthetic_view,
                max_width=args.inpaint_max_width,
                inner_dilation=args.mask_inner_dilation,
                outer_dilation=args.mask_outer_dilation,
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
                self.assert_identical(image, depth, mapper=mapper)

    # -- the mask morphology, which only this pipeline has -----------------

    def test_mask_dilation(self):
        image, depth = make_pair(96, 160, device=DEVICE)
        for inner, outer in ((0, 0), (1, 0), (0, 1), (2, 3), (4, 4)):
            with self.subTest(inner=inner, outer=outer):
                self.assert_identical(image, depth, divergence=5.0,
                                      mask_inner_dilation=inner, mask_outer_dilation=outer)

    def test_mask_dilation_scales_with_resolution(self):
        """base_width is the depth's, so the same setting must survive a rescale."""
        image, depth = make_pair(216, 384, depth_height=108, depth_width=192, device=DEVICE)
        for inner, outer in ((1, 1), (3, 2)):
            with self.subTest(inner=inner, outer=outer):
                self.assert_identical(image, depth, divergence=5.0,
                                      mask_inner_dilation=inner, mask_outer_dilation=outer)

    # -- inpaint_max_width, which resizes the picture rather than tiling ---

    def test_max_width(self):
        image, depth = make_pair(216, 384, device=DEVICE)
        for max_width in (None, 128, 192, 383, 384, 512):
            with self.subTest(max_width=max_width):
                # 383 is odd and must round up; 512 exceeds the width and must
                # leave the frame alone.
                self.assert_identical(image, depth, inpaint_max_width=max_width)

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

    def test_exact_multiple_of_the_pad_modulus(self):
        """64x64 in still gets a whole extra 64 pixels of padding. Check it."""
        image, depth = make_pair(128, 192, device=DEVICE)
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
        model = stereo_inpaint.load_monobw_inpaint(CHECKPOINT, device="cpu")
        reference_model = create_stereo_model("monobw_inpaint", divergence=2.0, device_id=-1)
        reference_model.set_mode("image")
        args = Args()
        with torch.inference_mode():
            reference = apply_divergence(depth.clone(), image.clone(), args, reference_model)
            actual = stereo_inpaint.synthesize_stereo_inpaint(image.clone(), depth.clone(), model)
        for eye, expected, got in zip(("left", "right"), reference, actual):
            diff = (expected.float() - got.float()).abs().max().item()
            self.assertEqual(diff, 0.0, f"cpu {eye} diff {diff} != 0")

    # -- combinations ------------------------------------------------------

    def test_combinations(self):
        image, depth = make_pair(216, 384, depth_height=518, depth_width=518, device=DEVICE)
        cases = [
            dict(divergence=1.0, convergence=0.0, synthetic_view="left", inpaint_max_width=256),
            dict(divergence=5.0, convergence=1.0, synthetic_view="right",
                 preserve_screen_border=True, mask_inner_dilation=2),
            dict(divergence=2.0, convergence=0.5, synthetic_view="both",
                 mask_outer_dilation=3, mapper="div_6"),
            dict(divergence=8.0, convergence=0.25, synthetic_view="both",
                 preserve_screen_border=True, inpaint_max_width=192,
                 mask_inner_dilation=1, mask_outer_dilation=1),
        ]
        for case in cases:
            with self.subTest(**case):
                self.assert_identical(image, depth, **case)

    # -- the diff-0 result above is only meaningful if the pipeline did work

    def test_pipeline_actually_does_something(self):
        """Guard against a vacuous pass.

        If both implementations were broken into returning the input unchanged,
        every equality assertion above would still hold.
        """
        image, depth = make_pair(96, 160, device=DEVICE)
        with torch.inference_mode():
            left, right = stereo_inpaint.synthesize_stereo_inpaint(
                image, depth, self.model, divergence=5.0, convergence=0.5)

        self.assertGreater((left - right).abs().max().item(), 0.05, "eyes are identical")
        self.assertGreater((left - image).abs().max().item(), 0.05, "left eye is the source")
        self.assertGreater((right - image).abs().max().item(), 0.05, "right eye is the source")

        with torch.inference_mode():
            left_only, right_passthrough = stereo_inpaint.synthesize_stereo_inpaint(
                image, depth, self.model, divergence=5.0, convergence=0.5,
                synthetic_view="left")
        self.assertEqual((right_passthrough - image).abs().max().item(), 0.0,
                         "synthetic_view='left' must pass the right eye through untouched")
        self.assertGreater((left_only - image).abs().max().item(), 0.05)

    def test_inpaint_changes_the_warp(self):
        """The inpaint half must contribute, not just pass the warp through.

        Every case above loads the same checkpoint on both sides, so a wiring
        mistake that skipped the network entirely -- an all-zero mask, say --
        would agree with iw3 only if iw3 skipped it too. It does not: compare
        against the bare MonoBW warp and the two must disagree.
        """
        image, depth = make_pair(96, 160, device=DEVICE)
        with torch.inference_mode():
            inpainted, _ = stereo_inpaint.synthesize_stereo_inpaint(
                image, depth, self.model, divergence=10.0, convergence=0.5,
                synthetic_view="left")
            warped, mask = self.model.monobw(
                image, depth, divergence=20.0, convergence=0.5, return_mask=True)

        self.assertGreater(mask.float().mean().item(), 0.0, "no holes to fill; the test proves nothing")
        difference = (inpainted - warped).abs().max().item()
        self.assertGreater(difference, 1e-3,
                           f"inpaint changed the warp by only {difference}; did it run?")

    def test_holes_are_where_the_depth_edges_are(self):
        """The mask must track the occluders, not be uniform or empty."""
        image, depth = make_pair(96, 160, device=DEVICE)
        with torch.inference_mode():
            _, mask = self.model.monobw(image, depth, divergence=10.0, convergence=0.5,
                                        return_mask=True)
        coverage = mask.float().mean().item()
        self.assertGreater(coverage, 0.001, "mask is empty")
        self.assertLess(coverage, 0.5, "mask covers half the frame; that is not a hole mask")


class GoldenVideoInpaintTest(unittest.TestCase):
    """light_video_inpaint_v1, the temporal model, at difference 0.

    This tests the network rather than a pipeline around it. iw3 drives it
    through a stateful frame queue that assumes frames arrive consecutively,
    which Resolve does not do, so the queue is deliberately not ported -- the
    plugin will hand the model the twelve frames it wants directly. What has to
    be right is the model and its padding rule.
    """

    CHECKPOINT = os.path.join(NUNIF_ROOT, "iw3", "pretrained_models", "hub", "checkpoints",
                              "iw3_light_video_inpaint_v1_20250919.pth")

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(cls.CHECKPOINT):
            raise unittest.SkipTest(f"checkpoint not downloaded: {cls.CHECKPOINT}")
        from nunif.models import load_model
        cls.model = stereo_inpaint.load_light_video_inpaint_v1(cls.CHECKPOINT, device=DEVICE)
        cls.reference, _ = load_model(cls.CHECKPOINT, device_ids=[DEVICE_ID], weights_only=True)
        cls.reference = cls.reference.eval()

    @staticmethod
    def make_sequence(frames, height, width, seed=0):
        """A short sequence with moving content and a stationary hole.

        The content has to move, or the temporal axis has nothing to carry and
        a model that ignored it entirely would pass.
        """
        generator = torch.Generator().manual_seed(seed)
        coarse = torch.rand((frames, 3, 12, 20), generator=generator)
        sequence = torch.nn.functional.interpolate(
            coarse, size=(height, width), mode="bilinear", align_corners=False).clamp(0, 1)
        mask = torch.zeros((frames, 1, height, width))
        mask[:, :, :, width // 4:width // 4 + 3] = 1.0
        mask[:, :, height // 3:, width // 2:width // 2 + 5] = 1.0
        return sequence.to(DEVICE), mask.to(DEVICE)

    def assert_identical(self, frames, height, width):
        x, mask = self.make_sequence(frames, height, width, seed=frames * 1000 + height)
        with torch.inference_mode(), torch.autocast(DEVICE, enabled=(DEVICE == "cuda")):
            expected = self.reference.infer(x.clone(), mask.clone())
            actual = self.model.infer(x.clone(), mask.clone())
        self.assertEqual(expected.shape, actual.shape, f"{frames}f {height}x{width}")
        diff = (expected.float() - actual.float()).abs().max().item()
        self.assertEqual(diff, 0.0, f"{frames}f {height}x{width}: max abs diff {diff}")

    def test_full_window(self):
        for height, width in ((128, 192), (216, 384), (107, 193)):
            with self.subTest(size=(height, width)):
                self.assert_identical(stereo_inpaint.SEQ_LEN, height, width)

    def test_short_sequences_are_padded(self):
        """Fewer than twelve frames is padded by repeating the first and last.

        That is also the rule for a window running off the start or end of a
        clip, which the temporal probe showed does happen.
        """
        for frames in (1, 2, 6, 11):
            with self.subTest(frames=frames):
                self.assert_identical(frames, 128, 192)

    def test_more_than_a_window_is_refused(self):
        """Thirteen frames is not a bigger window, and iw3 asserts on it too."""
        x, mask = self.make_sequence(13, 128, 192)
        with self.assertRaises(AssertionError):
            with torch.inference_mode():
                self.model.infer(x, mask)

    def test_the_temporal_axis_is_actually_used(self):
        """Changing a *neighbouring* frame must change this frame's fill.

        This is the property the image model does not have and the whole reason
        for porting a second one. If the window were being processed frame by
        frame, the middle frame's output would not move at all.
        """
        x, mask = self.make_sequence(12, 128, 192, seed=11)
        altered = x.clone()
        # Leave frame 6 alone; scramble everything else.
        generator = torch.Generator().manual_seed(99)
        noise = torch.rand(altered.shape, generator=generator).to(DEVICE)
        keep = altered[6:7].clone()
        altered = noise
        altered[6:7] = keep

        with torch.inference_mode(), torch.autocast(DEVICE, enabled=(DEVICE == "cuda")):
            before = self.model.infer(x, mask)
            after = self.model.infer(altered, mask)

        self.assertEqual((x[6] - altered[6]).abs().max().item(), 0.0,
                         "the frame under test must be unchanged")
        moved = (before[6] - after[6]).abs().max().item()
        self.assertGreater(moved, 1e-3,
                           f"frame 6 moved by only {moved} when its neighbours changed; "
                           "is the temporal axis wired up?")

    def test_is_really_the_video_model(self):
        """Guard against the suite loading the image model on both sides."""
        image = stereo_inpaint.load_light_inpaint_v1(CHECKPOINT, device=DEVICE)
        x, mask = self.make_sequence(12, 128, 192, seed=5)
        with torch.inference_mode(), torch.autocast(DEVICE, enabled=(DEVICE == "cuda")):
            from_video = self.model.infer(x, mask)
            from_image = image.infer(x, mask)
        difference = (from_video - from_image).abs().max().item()
        self.assertGreater(difference, 1e-3,
                           f"the two inpaint models agree to {difference}; is this really the "
                           "video one?")


if __name__ == "__main__":
    unittest.main()
