# tropnn

`tropnn` is a minimal routed neural layer library with several public routed layers:

- `TropLinear`: multi-head tropical routing with compact selected codes and a shared output projection.
- `PairwiseLinear`: classic pairwise-comparator LUT baseline.
- `TropFanLinear`: tropical normal-fan routing with geometry-generated values.
- `TropDictLinear`: experimental shared-dictionary payload with sparse per-cell coefficients.

Historical chamber-affine and payload-ablation implementations live in the experiment branch history. `main` keeps the selected tropical form, pairwise comparison path, fan-style controls, and the current dictionary-payload experiment.

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
- `layers/dictlinear.py`: `TropDictLinear`
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

from tropnn import TropDictLinear

dict_layer = TropDictLinear(
    256,
    512,
    heads=32,
    cells=4,
    route_source="sketch",
    dict_size=1024,
    dict_sparsity=4,
    use_route_residual=True,
)
print(dict_layer(x).shape)
```

## Dictionary Payload Experiment

`TropDictLinear` implements the "shared ETF dictionary + sparse cell coefficients" idea for a zero-dense or near-zero-dense payload:

```text
z           = route_project(x)
score[h,k] = route_score(z, h, k)
winner_h    = argmax_k score[h,k]
code[h,k]  = sum_l coeff[h,k,l] * basis[support[h,k,l]]
y           = bias + code_scale * sum_h code[h,winner_h]
```

It supports two route front-ends:

- `route_source="anchors"`: sparse coordinate routing, closest to the original zero-dense read story.
- `route_source="sketch"`: fixed CountSketch input projection followed by full site scoring, which gives stronger feature discrimination without introducing a learned dense input matmul.

The layer exposes `dictionary_loss()` as an ETF-style mean off-diagonal squared-overlap penalty on the shared basis. The scaling benchmark includes `tropical_dict` and `tied_tropical_dict` families, plus `--dict-route-source`, `--dict-sparsity`, `--dict-ortho-weight`, and `--dict-route-residual`.

### Result So Far

The implementation works mechanically and is covered by tests, but the research hypothesis did not reach the target scaling exponent. On the `n_features=256`, `alpha=1.0`, `model_dim=8..128`, `steps=3000`, `seeds=0,1,2` recovery benchmark:

```text
paper                         beta ~= 1.66
tied_tropical_lowrank          beta ~= 1.93
tied_tropical_zero_dense       beta ~= 0.18
tied_tropical_dict anchors     beta ~= 0.24
tied_tropical_dict sketch      beta ~= 0.51
tied_tropical_dict sketch+res  beta ~= 0.52
```

The failure mode is informative. Sketch routing plus dictionary codes brings `overlap * model_dim` close to paper scale (`~1.0-1.5`), so the frame geometry is not the main issue. The missing piece is recovery strength: sparse dictionary combinations do not reproduce the learned feature-specific low-rank basis that `tied_tropical_lowrank` gets from its dense projection. In short, ETF dictionary regularization fixes "features should not overlap too much", but it does not by itself learn "features should reconstruct with the right self-gain and frequency-weighted recovery".

Current conclusion:

```text
Shared ETF dictionary + sparse coefficients is a useful control,
but it is not enough to replace the learned dense low-rank body.
```

Future variants should target stronger feature-specific recovery rather than stronger dictionary orthogonality alone: learned sparse support, top-k sparse decoder rows, shared low-rank dictionary plus selected residuals, or another generated payload that preserves low-rank/tied recovery geometry without reintroducing a full dense matmul.

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

## Scope

This package is deliberately minimal. It does not include the old chamber-affine payload, tropical payload ablations, sparse comparator variants, or experiment orchestration.
