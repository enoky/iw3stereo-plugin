# row_flow_v3

Supported alongside `row_flow_v2`, selected by the plugin's **Model** parameter.

The warp needed no changes: same three input channels, same delta contract,
same non-symmetric path. Only the network differs — 117k parameters of windowed
attention rather than 30k of row convolutions.

Getting it into ONNX was the whole of the work, and it took two model-level
changes that are worth recording, because neither is obvious and both look like
gratuitous rewrites until you know why.

## The TorchScript exporter cannot do it

Exporting `row_flow_v3` the way `row_flow_v2` is exported produces a graph that
is accurate at the traced size — 7.2e-05 — and **fails at every other size**:

```
392x938: max abs diff 7.248e-05
392x940: FAIL : Non-zero status code returned while running Reshape node
384x960: FAIL : Non-zero status code returned while running Reshape node
```

The window-partition and padding reshapes bake in as constants. A plugin has to
handle whatever composition resolution the user has, so this is unusable.

## Change one: the attention head merge

The dynamo exporter handles genuinely dynamic shapes, but `torch.export`'s
decomposition fails on it:

```
ValueError: Cannot view a tensor with shape [..., 16, 2, 32]
            and strides (1024, 32, 512, 1) as a tensor with shape [..., 16, 64]
```

That is `sliced_sdp`'s last line, `x.permute(0, 2, 1, 3).reshape(...)`, merging
attention heads. `scaled_dot_product_attention` requires `(B, heads, tokens,
dim)`, and it is the permute back out of that layout that cannot be lowered.

**Inserting `.contiguous()` does not fix it.** That was tried, and verified to
be reached — the error is byte-identical, because decomposition elides the copy
again before it fails.

What works is not producing the layout in the first place: run each head
separately, staying three-dimensional, and **concatenate** to merge. At two
heads the slicing costs nothing.

This is *not* bit-identical to the fused kernel — it sums in a different order
and lands about **5e-5** away. So `stereo_warp.RowFlowV3` keeps
`scaled_dot_product_attention` by default, where the golden test demands
difference 0, and uses the head-sliced form only under `export_safe=True`.

## Change two: the padding

With the attention fixed the export succeeded but still failed at sizes other
than the traced one. The cause is in iw3's `replication_pad2d_naive`:

```python
pad_r = (x[:, :, :, -1:],) * right
```

Python tuple repetition needs `right` to be a concrete integer, so `torch.export`
specialises it and bakes one frame width into the graph. `F.pad(mode="replicate")`
takes the amount as data, and the matching crop becomes a slice. Numerically
identical, and the graph stays dynamic.

## Change three, found by a failing test

The first working export still rejected batched input: `Got: 3 Expected: 1`.
`torch.export` specialises any dimension whose example value is 1, however the
`Dim` is declared. Exporting with a batch-2 example fixes it.

## Where it ends up

| | |
| --- | --- |
| standalone PyTorch vs stock iw3 | **0**, at four sizes including odd ones |
| export-safe attention vs stock iw3 | ~5e-5 |
| exported ONNX vs stock iw3 | 4.0e-05 to 1.1e-04, six sizes |
| `tests/test_stereo_warp.py` | the full v2 matrix again, at difference 0 |
| `tests/test_stereo_warp_onnx.py` | the full v2 matrix again, tolerance 6e-4 |

Both suites carry a guard that the v2 and v3 paths actually **disagree** with
each other, because they load v3 on both sides of their comparison and would
otherwise pass just as happily if something had wired v2 into both.

`stereo_warp_v3.onnx` is 858 KiB against `stereo_warp.onnx`'s 124 KiB. Both ship
in the bundle and each gets its own session, built once during the background
start-up.

## Not done

**`row_flow_v3_sym`.** It sets `symmetric=True`, which iw3 routes through
`apply_divergence_nn_symmetric` — one delta warped in both directions rather
than two runs of a mirrored network. That is a different pipeline, not a
different checkpoint.

**Speed.** The plugin logs which model each frame used, but v3 has not been
timed against v2 in Resolve. Expect it to be slower; by how much is unmeasured.
