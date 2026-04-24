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
- `layers/base.py`: shared routed-layer shell
- `backend.py` and `backends/triton_scores.py`: tropical score backend dispatch
- `backends/tilelang_route.py`: optional fused TileLang inference route/code backend
- `examples/emnist.py`: EMNIST training example for `tropical` and `pairwise`
- `tools/benchmark.py`: backend benchmark for `TropLinear`
- `tools/profile.py`: forward profiler for `tropical` and `pairwise`

## Quick Start

```python
import torch
from tropnn import PairwiseLinear, TropLinear

x = torch.randn(8, 256)

tropical = TropLinear(256, 512, heads=32, cells=4, code_dim=32)
print(tropical(x).shape)

tropical_tilelang = TropLinear(256, 512, heads=32, cells=4, code_dim=32, backend="tilelang")
print(tropical_tilelang(x).shape)

pairwise = PairwiseLinear(256, 512, tables=16, comparisons=6)
print(pairwise(x).shape)
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

`backend="tilelang"` is an explicit inference backend for `TropLinear`. It fuses hard routing and selected-code accumulation without materializing the full score tensor. Training with `backend="tilelang"` falls back to the torch route path to preserve autograd. If TileLang compilation fails, ensure a CUDA toolkit compatible with the GPU is first on `PATH` and export `CC=/usr/bin/gcc CXX=/usr/bin/g++`.

CPU inference currently uses the torch path. The TileLang wheel used here exposes CUDA compilation, but not LLVM CPU codegen; for CPU deployment, benchmark `torch.compile(layer.eval(), mode="reduce-overhead")` instead of `backend="tilelang"`.

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
  --pairwise-comparisons 6
```

## Scope

This package is deliberately minimal. It does not include the old chamber-affine payload, tropical payload ablations, sparse comparator variants, or experiment orchestration.
