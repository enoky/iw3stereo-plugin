"""The ONNX implementation against the PyTorch one from Phase 1.

The bar here is a float tolerance, not 0: different kernels. The comparison is
against stereo_warp.synthesize_stereo rather than against re-derived
expectations, so this suite inherits Phase 1's diff-0 guarantee against iw3.

    F:\\_AI_PROJECTS_\\nunif\\venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import stereo_warp  # noqa: E402
import stereo_warp_onnx  # noqa: E402

MODELS_DIR = os.path.join(REPO_ROOT, "models")
_CHECKPOINT_DIR = os.path.join(
    os.environ.get("NUNIF_ROOT", r"F:\_AI_PROJECTS_\nunif"),
    "iw3", "pretrained_models", "hub", "checkpoints")
CHECKPOINT = os.path.join(_CHECKPOINT_DIR, "iw3_row_flow_v2_20240130.pth")
CHECKPOINT_V3 = os.path.join(_CHECKPOINT_DIR, "iw3_row_flow_v3_20250627.pth")

# Real content, not noise. Noise maximises every interpolation difference and
# would report a tolerance the plugin will never actually see.
TOLERANCE = 2e-4


def make_pair(height, width, depth_height=None, depth_width=None, batch=1, seed=0):
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
    depth = depth.clamp(0, 1)
    return image, depth


class OnnxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(os.path.join(MODELS_DIR, "stereo_warp.onnx")):
            raise unittest.SkipTest("models not exported; run export_onnx.py")
        if not os.path.exists(CHECKPOINT):
            raise unittest.SkipTest(f"checkpoint not found: {CHECKPOINT}")
        cls.session = stereo_warp_onnx.StereoWarpSession(MODELS_DIR)
        cls.model = stereo_warp.load_row_flow_v2(CHECKPOINT, device="cpu")

    def assert_close(self, image, depth, **settings):
        with torch.inference_mode():
            expected = stereo_warp.synthesize_stereo(
                image, depth, self.model, enable_amp=False, **settings)
        actual = self.session.synthesize_stereo(
            image.numpy(), depth.numpy(), **settings)

        for eye, want, got in zip(("left", "right"), expected, actual):
            want = want.numpy()
            self.assertEqual(want.shape, got.shape, f"{eye} shape, {settings}")
            diff = np.abs(want - got).max()
            self.assertLess(diff, TOLERANCE, f"{eye} max abs diff {diff:.3e}, {settings}")

    def test_provider_is_not_the_broken_one(self):
        self.assertNotIn(self.session.provider, stereo_warp_onnx.BROKEN_PROVIDERS)

    def test_defaults(self):
        image, depth = make_pair(216, 384)
        self.assert_close(image, depth)

    def test_divergence(self):
        image, depth = make_pair(216, 384)
        for divergence in (0.0, 1.0, 2.0, 5.0, 10.0):
            with self.subTest(divergence=divergence):
                self.assert_close(image, depth, divergence=divergence)

    def test_convergence(self):
        image, depth = make_pair(216, 384)
        for convergence in (-0.5, 0.0, 0.5, 1.0, 2.0):
            with self.subTest(convergence=convergence):
                self.assert_close(image, depth, convergence=convergence)

    def test_synthetic_view(self):
        image, depth = make_pair(216, 384)
        for view in ("both", "left", "right"):
            with self.subTest(synthetic_view=view):
                self.assert_close(image, depth, synthetic_view=view)

    def test_preserve_screen_border(self):
        image, depth = make_pair(216, 384)
        for divergence in (1.0, 2.0, 5.0):
            with self.subTest(divergence=divergence):
                self.assert_close(image, depth, divergence=divergence,
                                  preserve_screen_border=True)

    def test_mapper(self):
        image, depth = make_pair(216, 384)
        for mapper in ("pow2", "softplus", "softplus2", "mul_2", "inv_mul_1",
                       "div_6", "shift_20", "mul_1+mul_2=0.25", "pow2:mul_1"):
            with self.subTest(mapper=mapper):
                self.assert_close(image, depth, mapper=mapper)

    def test_stereo_width(self):
        image, depth = make_pair(216, 384, depth_height=518, depth_width=518)
        for stereo_width in (None, 128, 256, 384, 512):
            with self.subTest(stereo_width=stereo_width):
                self.assert_close(image, depth, stereo_width=stereo_width)

    def test_shapes(self):
        cases = [
            (216, 384, 108, 192),
            (108, 192, 216, 384),
            (216, 384, 518, 518),
            (217, 385, 107, 193),
            (1080, 1920, 540, 960),
        ]
        for height, width, depth_height, depth_width in cases:
            with self.subTest(size=(height, width, depth_height, depth_width)):
                image, depth = make_pair(height, width, depth_height, depth_width)
                self.assert_close(image, depth)

    def test_batched(self):
        image, depth = make_pair(216, 384, batch=3)
        self.assert_close(image, depth)

    def test_unbatched_chw(self):
        image, depth = make_pair(216, 384)
        self.assert_close(image.squeeze(0), depth.squeeze(0))

    def test_combinations(self):
        image, depth = make_pair(216, 384, depth_height=518, depth_width=518)
        cases = [
            dict(divergence=1.0, convergence=0.0, synthetic_view="left", stereo_width=256),
            dict(divergence=5.0, convergence=1.0, synthetic_view="right", stereo_width=128,
                 preserve_screen_border=True),
            dict(divergence=2.0, convergence=0.5, stereo_width=384, mapper="mul_2"),
        ]
        for case in cases:
            with self.subTest(**case):
                self.assert_close(image, depth, **case)


class OnnxV3Test(OnnxTest):
    """The same comparisons against row_flow_v3.

    The tolerance is looser than v2's for a known reason rather than a vague
    one: the exported graph uses the head-sliced attention, which sums in a
    different order from the fused kernel and sits about 5e-5 away before the
    warp amplifies it.
    """

    TOLERANCE = 6e-4

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(os.path.join(MODELS_DIR, "stereo_warp_v3.onnx")):
            raise unittest.SkipTest("row_flow_v3 graph not exported")
        if not os.path.exists(CHECKPOINT_V3):
            raise unittest.SkipTest("row_flow_v3 checkpoint not downloaded")
        cls.session = stereo_warp_onnx.StereoWarpSession(MODELS_DIR, model="row_flow_v3")
        cls.model = stereo_warp.load_row_flow_v3(CHECKPOINT_V3, device="cpu")

    def assert_close(self, image, depth, **settings):
        with torch.inference_mode():
            expected = stereo_warp.synthesize_stereo(
                image, depth, self.model, enable_amp=False, **settings)
        actual = self.session.synthesize_stereo(image.numpy(), depth.numpy(), **settings)
        for eye, want, got in zip(("left", "right"), expected, actual):
            want = want.numpy()
            self.assertEqual(want.shape, got.shape, f"{eye} shape, {settings}")
            diff = np.abs(want - got).max()
            self.assertLess(diff, self.TOLERANCE, f"{eye} max abs diff {diff:.3e}, {settings}")

    def test_graph_is_really_v3(self):
        """Guard: the v2 and v3 graphs must not agree."""
        image, depth = make_pair(216, 384)
        v2 = stereo_warp_onnx.StereoWarpSession(MODELS_DIR, model="row_flow_v2")
        left_v3, _ = self.session.synthesize_stereo(image.numpy(), depth.numpy(), divergence=5.0)
        left_v2, _ = v2.synthesize_stereo(image.numpy(), depth.numpy(), divergence=5.0)
        self.assertGreater(np.abs(left_v3 - left_v2).max(), 1e-3,
                           "the v3 graph agrees with the v2 graph; is it really v3?")


if __name__ == "__main__":
    unittest.main()
