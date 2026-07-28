"""The exported light_inpaint_v1 graph against the PyTorch one.

The bar here is a float tolerance, not 0: different kernels. The comparison is
against stereo_inpaint rather than against re-derived expectations, so this
suite inherits the diff-0 guarantee that suite holds against stock iw3.

Only the inpaint half is a graph. `MonoBW` is built on `torch.cummax` and
`torch.searchsorted`, neither of which has an ONNX operator, so the warp stays
in PyTorch here and will be CUDA in the plugin. The second test below is the
composition that matters: the real pipeline with the ONNX graph substituted for
the network, which is exactly how the plugin will be put together.

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

MODELS_DIR = os.path.join(REPO_ROOT, "models")
GRAPH = os.path.join(MODELS_DIR, "light_inpaint_v1.onnx")
CHECKPOINT = os.path.join(
    os.environ.get("NUNIF_ROOT", r"F:\_AI_PROJECTS_\nunif"),
    "iw3", "pretrained_models", "hub", "checkpoints", "iw3_light_inpaint_v1_20250919.pth")

# Measured 1.0e-6 to 3.6e-6 across every size tried, up to full HD.
TOLERANCE = 2e-5


def make_pair(height, width, batch=1, seed=0):
    """An eye and a hole mask. Real content, not noise.

    The mask is vertical bands, which is the shape a real occlusion mask takes
    -- holes open beside depth edges and run down them.
    """
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.rand((batch, 3, 12, 20), generator=generator)
    eye = torch.nn.functional.interpolate(coarse, size=(height, width), mode="bilinear",
                                          align_corners=False).clamp(0, 1)
    mask = torch.zeros((batch, 1, height, width))
    mask[:, :, :, width // 4:width // 4 + 3] = 1.0
    mask[:, :, height // 3:, width // 2:width // 2 + 5] = 1.0
    return eye, mask


class InpaintOnnxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(GRAPH):
            raise unittest.SkipTest("light_inpaint_v1.onnx not exported; run export_onnx.py")
        if not os.path.exists(CHECKPOINT):
            raise unittest.SkipTest(f"checkpoint not found: {CHECKPOINT}")
        cls.session = ort.InferenceSession(GRAPH, providers=["CPUExecutionProvider"])
        cls.model = stereo_inpaint.load_light_inpaint_v1(CHECKPOINT, device="cpu")

    def run_graph(self, eye, mask):
        return self.session.run(["y"], {"eye": eye.numpy(), "mask": mask.numpy()})[0]

    def assert_close(self, eye, mask, label=""):
        got = self.run_graph(eye, mask)
        with torch.inference_mode():
            want = self.model.infer(eye, mask).numpy()
        self.assertEqual(want.shape, got.shape, f"shape {label}")
        diff = np.abs(want - got).max()
        self.assertLess(diff, TOLERANCE, f"max abs diff {diff:.3e} {label}")
        return diff

    # -- the graph on its own ----------------------------------------------

    def test_shapes(self):
        for height, width in ((256, 448), (392, 938), (260, 452), (100, 200), (216, 384)):
            with self.subTest(size=(height, width)):
                eye, mask = make_pair(height, width, seed=height)
                self.assert_close(eye, mask, f"{height}x{width}")

    def test_size_that_is_an_exact_multiple_of_64(self):
        """The pad is always at least one whole block; check the graph agrees."""
        eye, mask = make_pair(128, 192)
        self.assert_close(eye, mask, "128x192")

    def test_hd(self):
        eye, mask = make_pair(1036, 1920, seed=7)
        self.assert_close(eye, mask, "1036x1920")

    def test_batched(self):
        eye, mask = make_pair(256, 448, batch=3)
        self.assert_close(eye, mask, "batch 3")

    def test_empty_mask_passes_the_eye_through(self):
        """Nothing to fill means nothing changes -- the composite is by the mask."""
        eye, _ = make_pair(256, 448)
        mask = torch.zeros((1, 1, 256, 448))
        got = self.run_graph(eye, mask)
        self.assertLess(np.abs(eye.numpy() - got).max(), 1e-6,
                        "an empty mask must leave the eye untouched")

    def test_full_mask_replaces_everything(self):
        """The opposite end: the network's own output, no source left."""
        eye, _ = make_pair(256, 448)
        mask = torch.ones((1, 1, 256, 448))
        got = self.run_graph(eye, mask)
        self.assertGreater(np.abs(eye.numpy() - got).max(), 0.05,
                           "a full mask must not return the source")
        self.assert_close(eye, mask, "full mask")

    # -- the composition the plugin will use -------------------------------

    def test_pipeline_with_the_graph_substituted(self):
        """The real pipeline, ONNX in place of the network.

        This is the arrangement the plugin will have: the warp and the mask
        morphology outside the graph, the network and its composite inside. If
        the boundary were drawn in the wrong place -- a preprocessing step left
        on the wrong side of it -- this is what would catch it.

        Run without autocast on both sides. The graph is fp32 throughout and
        the torch pipeline is fp16 under autocast, and that gap is far wider
        than anything this test is trying to measure.
        """
        pipeline = stereo_inpaint.MonoBWInpaintImage(self.model, device="cpu")

        generator = torch.Generator().manual_seed(3)
        image = torch.rand((1, 3, 216, 384), generator=generator)
        coarse = torch.rand((1, 1, 8, 8), generator=generator)
        depth = torch.nn.functional.interpolate(coarse, size=(108, 192), mode="bilinear",
                                                align_corners=False)
        depth[:, :, 27:54, 48:96] = 0.95
        depth = depth.clamp(0, 1)

        settings = dict(divergence=5.0, convergence=0.5, enable_amp=False)
        with torch.inference_mode():
            expected = stereo_inpaint.synthesize_stereo_inpaint(image, depth, pipeline, **settings)

        session = self.session

        class OnnxNetwork(torch.nn.Module):
            """Stands in for LightInpaintV1, calling the graph instead."""

            def infer(self, eye, mask):
                y = session.run(["y"], {"eye": eye.contiguous().numpy(),
                                        "mask": mask.contiguous().numpy()})[0]
                return torch.from_numpy(y)

        hybrid = stereo_inpaint.MonoBWInpaintImage(OnnxNetwork(), device="cpu")
        with torch.inference_mode():
            actual = stereo_inpaint.synthesize_stereo_inpaint(image, depth, hybrid, **settings)

        for eye_name, want, got in zip(("left", "right"), expected, actual):
            self.assertEqual(want.shape, got.shape, f"{eye_name} shape")
            diff = (want - got).abs().max().item()
            self.assertLess(diff, TOLERANCE, f"{eye_name} max abs diff {diff:.3e}")

    def test_the_graph_is_not_a_passthrough(self):
        """Guard against a vacuous pass.

        Every tolerance above would hold just as well if the graph returned its
        input. It must not: with a mask, the result has to move.
        """
        eye, mask = make_pair(256, 448)
        got = self.run_graph(eye, mask)
        difference = np.abs(eye.numpy() - got).max()
        self.assertGreater(difference, 0.01,
                           f"graph changed the eye by only {difference}; is it running?")

        # And it must stay local. Not to the binary mask -- preprocess blurs it
        # with a 15-tap gaussian and composites by the blurred version, so the
        # fill feathers seven pixels past the hole on purpose, which is what
        # stops a hard seam. Outside that support the blurred mask is zero and
        # the composite is exactly the source.
        support = torch.nn.functional.max_pool2d(mask, kernel_size=17, stride=1, padding=8)
        outside = np.broadcast_to(support.numpy() == 0, eye.shape)
        untouched = np.abs(eye.numpy() - got)[outside].max()
        self.assertLess(untouched, 1e-5,
                        f"the fill reached {untouched} beyond the mask blur's support")


if __name__ == "__main__":
    unittest.main()
