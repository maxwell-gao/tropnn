# tropnn

`tropnn` is a minimal routed neural layer library with two public layers:

- `TropLinear`: multi-head tropical routing with compact selected codes and a shared output projection.
- `PairwiseLinear`: classic pairwise-comparator LUT baseline.

Historical chamber-affine and payload-ablation implementations live in the experiment branch history. `main` keeps only the selected tropical form and the pairwise comparison path.

## Core Idea

`TropLinear` projects the input to a code space, routes each tropical head through a Power-diagram style argmax, reads one compact code per head, and decodes the summed code with a shared output map:

$$
i_h(x)=\arg\max_{k\le K}\langle w_{hk}, Px\rangle+b_{hk},
$$

$$
z(x)=Px+\frac{1}{\sqrt H}\sum_h c_{h,i_h(x)},\qquad y=Wz+\beta.
$$

Training uses a local min-face surrogate against the runner-up cell:

$$
\tilde c_h(x)
=
c_{h,i_h(x)}
+ U(s_{h,i_h}(x)-s_{h,j_h}(x))
(c_{h,j_h(x)}-c_{h,i_h(x)}),
\qquad
U(m)=\frac{0.5}{1+|m|}.
$$

Inference remains fully hard.

## Package Layout

- `layers/tropical.py`: `TropLinear`
- `layers/pairwise.py`: `PairwiseLinear`
- `layers/fan.py`: `TropFanLinear`
- `layers/base.py`: shared routed-layer shell
- `backend.py` and `backends/triton_scores.py`: tropical score backend dispatch
- `backends/tilelang_route.py`: optional fused TileLang tropical route/code backend
- `backends/tropical_zig.py`: optional Zig CPU tropical route/code inference backend
- `backends/fan_zig.py`: optional Zig CPU tropical fan route/value inference backend
- `backends/pairwise_tilelang.py`: optional fused TileLang pairwise route/LUT backend
- `backends/pairwise_zig.py`: optional Zig CPU pairwise inference backend
- `examples/emnist.py`: EMNIST training example for `tropical` and `pairwise`
- `tools/benchmark.py`: backend benchmark for `TropLinear`, `PairwiseLinear`, and `TropFanLinear`
- `tools/profile.py`: forward profiler for `tropical` and `pairwise`
- `tools/ncu_memory_case.py` and `tools/ncu_memory_sweep.py`: Nsight Compute DRAM profiling helpers

## Quick Start

```python
import torch
from tropnn import PairwiseLinear, TropFanLinear, TropLinear

x = torch.randn(8, 256)

tropical = TropLinear(256, 512, heads=32, cells=4, code_dim=32)
print(tropical(x).shape)

tropical_tilelang = TropLinear(256, 512, heads=32, cells=4, code_dim=32, backend="tilelang")
print(tropical_tilelang(x).shape)

tropical_zig = TropLinear(256, 512, heads=32, cells=4, code_dim=32, backend="zig", cpu_param_dtype="f16").eval()
print(tropical_zig(x.cpu()).shape)

pairwise = PairwiseLinear(256, 512, tables=16, comparisons=6)
print(pairwise(x).shape)

pairwise_tilelang = PairwiseLinear(256, 512, tables=16, comparisons=6, backend="tilelang").cuda()
print(pairwise_tilelang(x.cuda()).shape)

pairwise_zig = PairwiseLinear(256, 512, tables=16, comparisons=6, backend="zig", cpu_lut_dtype="f16").eval()
print(pairwise_zig(x.cpu()).shape)

fan_zig = TropFanLinear(256, 512, heads=32, cells=4, code_dim=32, backend="zig", fan_value_mode="basis", cpu_param_dtype="f16").eval()
print(fan_zig(x.cpu()).shape)
```

## EMNIST Example

```bash
uv run python -m tropnn.examples.emnist \
  --root /path/to/emnist \
  --family tropical \
  --split digits \
  --epochs 10 \
  --batch-size 512 \
  --lr 1e-3 \
  --hidden-dim 128 \
  --depth 2 \
  --heads 256 \
  --cells 16 \
  --code-dim 64 \
  --max-train 20000 \
  --max-test 4000 \
  --device cuda \
  --seed 0
```

Pairwise uses the same script with `--family pairwise --pairwise-tables 136 --comparisons 6`.

`backend="tilelang"` is an explicit CUDA backend for `TropLinear` and `PairwiseLinear`. `TropLinear` fuses routing and selected-code accumulation for both eval and train-time min-face backward. `PairwiseLinear` fuses comparator routing, output-block LUT row reads, vectorized LUT-gradient scatter, and block-reduced LUT/STE backward; its TileLang path currently supports the `fast_sigmoid_odd` LUT surrogate. If TileLang compilation fails, ensure a CUDA toolkit compatible with the GPU is first on `PATH` and export `CC=/usr/bin/gcc CXX=/usr/bin/g++`.

`backend="zig"` is an explicit CPU inference backend for `TropLinear`, `TropFanLinear`, and `PairwiseLinear`. It uses the bundled `kernels/src` Zig kernels inside this package and compiles them with the pinned `ziglang==0.16.0` package. `TropLinear` fuses route selection and selected-code accumulation, `TropFanLinear` fuses route selection and generated-value accumulation, and both then use the existing torch output projection. `PairwiseLinear` runs no-cache compare/LUT forward. Install it with:

```bash
uv sync --extra cpu
```

Use `cpu_param_dtype="f32"` / `cpu_lut_dtype="f32"` for exact f32 parameter reads, or `"f16"` to reduce route/code/LUT bandwidth during inference. The Zig backend is inference-only; call `.eval()` before use. If you need to override the compiler path, set `TROPNN_ZIG`, for example `TROPNN_ZIG="/path/to/zig"`.

## Profiling

```bash
uv run python -m tropnn.tools.profile \
  --device cuda \
  --batch-size 512 \
  --tropical-hidden 128 \
  --tropical-heads 256 \
  --tropical-cells 16 \
  --tropical-code-dim 64 \
  --pairwise-hidden 128 \
  --pairwise-tables 136 \
  --pairwise-comparisons 6 \
  --tropical-backend tilelang \
  --pairwise-backend tilelang
```

CPU pairwise benchmark:

```bash
uv run python -m tropnn.tools.benchmark \
  --device cpu \
  --batch-size 1 \
  --steps 1000 \
  --tropical-zig-dtype f16 \
  --pairwise-zig-dtype f16
```

## Bridge Scaling: One Capacity Unit Across Kingdoms

`tools/bridge_scaling.py` is an architecture-agnostic scaling diagnostic. It addresses a structural limitation of `SuperpositionScaling`-style `log L vs log m` fits: model dim `m` is not a comparable capacity unit across architectures. A dense linear-algebra layer with width `m` and a routed comparator-LUT layer with the same `m` can have wildly different *effective* representation capacities, so their `\beta` values are not on the same axis.

The bridge tool replaces `m` with an effective capacity `K_eff` computed from whichever representation views the model exposes:

```text
vector  view: r_i = model(eye_i) in R^m              -> linear-algebra K
chamber view: sigma_i = winner_h(eye_i) in [C]^H     -> combinatorial K
```

For each view it computes:

* `K_metric(alpha)` = `(tr M)^2 / ||M||_F^2` for `M = exp(-alpha D)`, where `D` is the pairwise distinguishability matrix (`1 - cos^2` for vectors, `1 - hamming_match_fraction` for chambers).
* `K_pack(tau)` = greedy anti-clique under `D >= tau`.
* `chamber_entropy` and `chamber_distinct` for routed families.

`fit_bridge_exponents_per_family` then fits `log L = a - beta * log K_eff` per family and pooled across families, producing a `bridge_exponents` block in the benchmark summary.

### Result on the Standard Recovery Toy

Setup: `n_features=256`, `model_dims=8,16,32,64,128`, `alpha=1.0`, `steps=3000`, `batch=1024`, three seeds, six families.

Legacy fit (`log L vs log m`) gives the well-known kingdom split:

```text
paper                     beta = 1.66    R^2 = 0.99
tied_tropical_lowrank      beta = 1.93    R^2 = 0.97
untied_paper              beta = 0.32    R^2 = 0.84
linear                    beta = 0.21    R^2 = 0.57
tied_tropfan_zero_dense    beta = 0.20    R^2 = 0.90
tied_tropical_zero_dense   beta = 0.18    R^2 = 0.89
```

Pooled bridge fit (`log L vs log K_eff` across all 90 runs):

```text
K_eff = bridge_K_vec_a4         beta_pooled = 0.55    R^2 = 0.41
K_eff = bridge_K_vec_a16        beta_pooled = 0.48    R^2 = 0.28
K_eff = bridge_K_chamb_a4       beta_pooled = 0.59    R^2 = 0.43
K_eff = bridge_K_chamb_a16      beta_pooled = 0.62    R^2 = 0.48
K_eff = bridge_chamber_distinct beta_pooled = 1.09    R^2 = 0.33
```

The pooled exponent under `chamber_distinct` lands at `~1.0`, which matches the SuperpositionScaling prediction `L \propto 1 / K_eff`. The legacy split between `paper` (1.66) and `tied_tropical_zero_dense` (0.18) collapses to a much narrower band (0.5-1.1) under any bridge capacity unit. The remaining cross-family variance shows up as dispersion in `R^2` rather than as different exponents, which is the diagnostic signal we want.

### What the Bridge Diagnoses

The clearest picture comes from `K / m` per architecture:

```text
K_vec_a16 / m            m=8   m=16   m=32   m=64  m=128
paper                  11.87  16.00   8.00   3.98   1.99
tied_tropical_lowrank   0.50   4.05   8.00   4.00   2.00
tied_tropfan_zero_dense 1.69   1.72   1.57   1.32   1.13
tied_tropical_zero_dense 0.18  0.13   0.13   0.13   0.50

K_chamb_a16 / m          m=8   m=16   m=32   m=64  m=128
tied_tropical_lowrank   0.26   0.97   8.00   4.00   2.00
tied_tropfan_zero_dense 1.91   1.75   1.57   1.32   1.14
tied_tropical_zero_dense 0.20  0.13   0.09   0.06   0.03
```

The structural failure of `tied_tropical_zero_dense` is now explicit: its chamber capacity per unit `m` *decreases* as `m` grows. The architecture does not use the width it nominally has. This is exactly why its legacy `beta` is small, and the bridge tool surfaces it as a single number per `(family, m)` pair instead of as a fit slope.

Note that `paper`, `tied_tropical_lowrank`, `linear`, and `untied_paper` saturate `K_metric` at `n_features = 256` for `m >= 32`. To resolve scaling cleanly in the linear-algebra kingdom, run with `n_features` at least 4x the largest `m`.

### Usage

```bash
uv run python -m tropnn.tools.scaling_benchmark \
  --families paper,tied_tropical_lowrank,tied_tropical_zero_dense,tied_tropfan_zero_dense \
  --n-features 256 \
  --model-dims 8,16,32,64,128 \
  --alphas 1.0 \
  --batch-size 1024 \
  --steps 3000 \
  --heads 32 \
  --cells 4 \
  --route-terms 4 \
  --fan-value-mode basis \
  --fan-basis-rank 16 \
  --seeds 0,1,2 \
  --device cuda
```

The summary JSON contains a `bridge_exponents` array with `scope in {per_family, pooled}` and `capacity` keys for each `K` choice, alongside the legacy `exponents` block.

### Pairwise: A Non-Inner-Product Architecture Under the Bridge

`PairwiseLinear` is the cleanest architecture in this package that does no inner products in its forward path. It hashes the input via pairwise comparators, looks up an LUT row per table, and sums. Its capacity is driven by `(tables T, comparisons L)`, not by the model dim `m`. The bridge tool catches this directly.

**Experiment 1 (sweep model_dim, fixed `T=32, L=6`):**

```text
family                     m=8   m=16   m=32   m=64   m=128    legacy_beta
paper                      0.0023 0.0010 0.0003 9e-5  3e-5     1.66
tied_tropical_lowrank      0.0020 0.0011 0.0003 6e-5  1e-5     1.93
tied_tropical_zero_dense   0.0046 0.0039 0.0036 0.0032 0.0027  0.18
tied_pairwise              0.0033 0.0029 0.0027 0.0025 0.0024  0.12

K_chamb_a16 per family
tied_tropical_lowrank      2.05  15.6   256    256    256
tied_tropical_zero_dense   1.61  2.14   2.94   3.64   3.61
tied_pairwise              2.54  2.54   2.54   2.54   2.54   <-- constant in m
```

`tied_pairwise`'s `K_chamb_a16` is exactly constant across `m`. The legacy `beta = 0.12` reflects only the slow loss decrease from the LUT row width growing; it does not reflect any change in routing capacity. By the legacy diagnostic, pairwise looks like a non-scaling architecture. By the bridge diagnostic, the right diagnosis is "pairwise's `m` is the wrong axis to vary".

**Experiment 2 (sweep `T`, fixed `m=64, L=6`):**

```text
T=4    T=8    T=16   T=32   T=64   T=128
loss             0.0047 0.0044 0.0036 0.0025 0.0015 0.0010
distinct         24     46     80     135    199    244
K_chamb_a16      1.21   1.46   1.89   2.54   3.21   3.68
overlap*m        53.0   43.5   30.6   15.9   11.2   7.7

beta vs T                  : 0.47  R^2=0.90
beta vs K_chamb_a16        : 1.37  R^2=0.90
beta vs chamber_distinct   : 0.65  R^2=0.83
```

When pairwise is allowed to scale along its native capacity axis `T`, it has a clear scaling law: `loss ~ T^{-0.47}`, `R^2 = 0.90`. Translated to bridge `K_chamb_a16`, the exponent is `1.37`. This sits in the same neighborhood as `paper` (1.66) and `tied_tropical_lowrank` (1.93), even though pairwise has no learned dense projection and no inner-product recovery operator anywhere in its forward path.

**Pooled bridge fit across both experiments** (paper + lowrank + zero_dense varying `m`, pairwise varying `T`, all on the same `log L vs log K_chamb_a16` axis):

```text
beta_pooled = 0.83   R^2 = 0.84   pts = 48
```

The four families lie on roughly the same line, with similar slope and small intercept differences. The bridge tool turns the legacy "pairwise has beta = 0.12 vs paper has beta = 1.66" mismatch into a unified picture: pairwise scales like the others when the capacity unit accounts for combinatorial chamber count instead of vector dim.

This is the practical demonstration of why the bridge tool exists. The legacy `m`-axis privileges architectures whose capacity equals their model width. For routing-based architectures like `PairwiseLinear`, the legacy axis hides the actual scaling behavior; the bridge axis exposes it.

## Scope

This package is deliberately minimal. It does not include the old chamber-affine payload, tropical payload ablations, sparse comparator variants, or experiment orchestration.
