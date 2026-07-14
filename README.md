# TropNN

TropNN is the minimal research implementation of Pairwise Comparison Lookup
Table networks. It isolates the operator, its training rule, low-level backends,
and controlled probes from the LutFlow language-model stack.

## Operator

For `T` tables and `C` comparisons per table:

```text
m[t,c] = x[a[t,c]] - x[b[t,c]] - theta[t,c]
r[t,c] = 1[m[t,c] > 0]
code[t] = sum_c r[t,c] * 2**c
y = sum_t payload[t, code[t], :]
```

`a` and `b` are fixed anchor indices in the standard layer. `theta` may be fixed
or learned. Each table selects one of `2**C` payload rows. The output is an
additive reduction across tables.

Full comparison compares all coordinate pairs and is the clean geometric
construction. Practical PC-LUT uses `C << input_dim`, multiple sparse tables,
and usually a residual connection.

## Why it is trainable

A visited payload row receives the exact downstream gradient. Thresholds and
upstream coordinates affect a discrete route, so TropNN uses a finite-difference
STE based on the payload difference between the current and neighboring chamber.
Min-margin STE updates the nearest comparison boundary; full STE considers each
bit flip.

This distinction is important:

- Forward and inference use hard comparison plus lookup.
- Payload gradients are exact sparse gradients.
- Threshold and input route gradients are surrogate gradients.
- Fixed-threshold probes remove route-gradient questions and isolate
  representation and decoding.

## API

```python
from tropnn import PairwiseLUT

layer = PairwiseLUT(
    input_dim=256,
    output_dim=256,
    tables=64,
    comparisons=6,
    backend="torch",
)
y = layer(x)
```

Use Torch as the semantic reference. Optimized paths must preserve route,
payload decode, output reduction, and STE semantics.

## Backends

| Backend | Purpose |
|---|---|
| `torch` | Readable reference and correctness oracle |
| `tilelang` | Fused GPU route, lookup, reduction, and backward |
| `triton` | Alternative fused GPU and payload experiments |
| `zig` | CPU packed-payload and anchor-layout implementation |

Packed payloads are cached by parameter version. Training repacks after an
optimizer update; evaluation can retain a packed representation. Low-bit forward
storage does not imply that the optimizer master parameter is low-bit.

## Current evidence

### Low-bit payloads

Binary payloads retain useful behavior in EMNIST and short LUTSSM runs. The
retained kernels use byte-major decode and grouped exact min-margin backward.
See [`../../report/binary_lut_kernel_optimization.md`](../../report/binary_lut_kernel_optimization.md).

This does not make all parameters binary. Anchors are integer indices, routes
are bits, thresholds may be continuous, and optimizer state may be floating
point.

### Depth and refinement

Plain PC-LUT often improves weakly with recursive depth compared with MLPs.
Generic premixing, shared generators, route-conditioned anchors, compare-swap,
and payload-factorization variants have not supplied a robust general fix.

Do not call an additive ensemble a depth sweep. If every route reads the same
input and its scalar outputs are summed, `L x T` is exactly one wider additive
route bank.

### Relation selection

A direct PC-LUT scorer learns a nontrivial binary bilinear relation but remains
far behind dense controls. A fixed `T16/C5` scalar route reaches `0.0679`
Top-16 recall and `0.0078` Top-1 in the controlled relation-energy probe.

Sixteen independent banks improve MSE to `0.2285` Top-16 and `0.0977` Top-1
with 8,192 parameters. A 529-parameter dense MLP reaches `0.2942` and `0.1172`.

The decisive diagnostic freezes all 256 route codes, expands them into 1,280
bits, and trains a dense decoder. Top-16 rises to `0.6597` and Top-1 to `0.4531`
without changing any comparison. The fixed quotient retains substantial
relation information; additive independent-table decoding is the primary
bottleneck on this probe.

The dense decoder is diagnostic, not a fast replacement. The active problem is
cross-table interaction with sparse, bounded-order, or lookup-native execution.

## Controlled probes

From the repository root:

```bash
PYTHONPATH=python/src/tropnn \
python/src/tropnn/.venv/bin/python -m tropnn.tools.bilinear_retrieval_probe --help

PYTHONPATH=python/src/tropnn \
python/src/tropnn/.venv/bin/python -m tropnn.tools.fixed_route_relation_energy_probe --help

PYTHONPATH=python/src/tropnn \
python/src/tropnn/.venv/bin/python -m tropnn.tools.additive_relation_energy_sweep --help

PYTHONPATH=python/src/tropnn \
python/src/tropnn/.venv/bin/python -m tropnn.tools.fixed_l16_route_decoder_probe --help
```

Relevant launchers:

- `scripts/run_tropnn_bilinear_retrieval_4gpu.sh`
- `scripts/run_tropnn_fixed_route_relation_energy_5gpu.sh`
- `scripts/run_tropnn_additive_relation_energy_depth_5gpu.sh`
- `scripts/run_tropnn_fixed_l16_route_decoder_2gpu.sh`

## Benchmarking

A comparison with `nn.Linear` must match batch, sequence length, dimensions,
dtype, output shape, cache state, and forward/backward mode. Route time omits
wide payload reads. Forward-only time omits finite-difference backward, which
can dominate training.

## Development

```bash
cd python/src/tropnn
/home/ubuntu/.nix-profile/bin/uv sync --extra dev
PYTHONPATH=. .venv/bin/pytest tests -q
```

Read [`../../report/README.md`](../../report/README.md) for evidence and
[`../../../doc/PAIRWISE_LUT_CURRENT_STATE.md`](../../../doc/PAIRWISE_LUT_CURRENT_STATE.md)
for the repository-level interpretation.
