# tropnn

`tropnn` is a small demonstration package for PC-LUT networks: pairwise comparison, lookup table read, and accumulation. The public PC-LUT operator intentionally avoids GEMM in its forward path.

## Core operator

For an input row `x`, each table owns `L` fixed coordinate pairs `(a, b)` and trainable thresholds `theta`:

```text
bit_l = 1[x[a_l] - x[b_l] - theta_l > 0]
index = sum_l bit_l * 2^l
payload = LUT[table, index]
y = sum_table payload
```

Training uses a finite-difference straight-through estimator. The selected LUT row receives ordinary gradient. The threshold and input-side gradient use the neighboring row across the closest active comparison boundary.

## Public API

```python
import torch
from tropnn import AbsDiffLUT, PairwiseLUT, PairwiseWalshLUT

x = torch.randn(8, 256)
lut = PairwiseLUT(256, 512, tables=64, comparisons=6)
print(lut(x).shape)

relation = AbsDiffLUT(256, 1, tables=16, comparisons=4)
print(relation(x, x).shape)

walsh = PairwiseWalshLUT(256, 512, tables=64, comparisons=6, walsh_order=2)
print(walsh(x).shape)

affine = PairwiseWalshLUT(
    256,
    512,
    tables=64,
    comparisons=6,
    walsh_order=2,
    slope_order=2,
    lut_dtype="int2",
)
print(affine(x).shape)
```

`PairwiseWalshLUT(..., slope_order=2, lut_dtype="int2")` is the structured
low-bit baseline.  It replaces a free row payload with low-degree Boolean
generators over the comparison signs:

```text
sign_i = 2 * bit_i - 1
basis = [1, sign_i, sign_i * sign_j]

update =
  basis @ shared_bias_generators
  + sum_i low_degree_coeff_i(bits) * margin_i * shared_slope_generator_i
```

The generated bias rows, generated slope coefficients, and slope generators are
materialized through the same low-bit LUT dtype path as `PairwiseLUT`.

## Backends

- `backend="torch"`: reference implementation.
- `backend="tilelang"`: CUDA fused pairwise compare/LUT path.
- `backend="zig"`: CPU inference path for `PairwiseLUT`.

## EMNIST demo

```bash
uv run tropnn-emnist \
  --root data/emnist \
  --download \
  --split digits \
  --epochs 10 \
  --batch-size 512 \
  --tables 64 \
  --comparisons 6 \
  --device cuda
```


## Reproducible experiment categories

The library core exports only PC-LUT layers. Experiment scripts may include dense baselines, including `nn.Linear`, to reproduce report comparisons.

- `tropnn-emnist --family linear|pairwise|pairwise_walsh`; use `--family pairwise_walsh --slope-order 2 --lut-dtype int2` for the Walsh affine baseline
- `tropnn-scaling-benchmark --families paper,untied_paper,linear,pairwise,tied_pairwise,pairwise_walsh,tied_pairwise_walsh,pairwise_walsh_affine,tied_pairwise_walsh_affine`
- `tropnn-recovery-lut-structures --variants free,walsh1,walsh2,coarse`
- `tropnn-dense-projection-fit --variants dense_exact,linear,pairwise`

For EMNIST payload-width/depth sweeps, `scripts/run_tropnn_payload_width_4gpu.sh`
exposes the baseline as `VARIANTS="walsh_affine"` with default
`WALSH_LUT_DTYPE=int2`, `WALSH_ORDER=2`, and `WALSH_SLOPE_ORDER=2`.
