# tropnn

`tropnn` is a minimal teaching-oriented tropical neural layer library.

It exposes one basic building block, `TropLinear`, meant to play the same role for tropical min-face networks that `nn.Linear` plays for dense networks.

The design target is intentionally narrow:

- hard chamber selection at inference time,
- local min-face surrogate during training,
- a small reference implementation that collaborators can read in one sitting,
- an optional Triton score kernel that accelerates the hottest routing path without changing semantics.

This package is intended to be the seed of a standalone repository for collaborators who want to build other neural networks, such as MLPs or RNNs, out of the same `trop_minface` primitive.

## Core Idea

For each group, the layer computes affine scores

$$
i_g(x) = \arg\max_k s_{gk}(Px),
$$

then selects one chamber-affine branch per group and sums them:

$$
y(x) = \sum_{t,g} \bigl(A_{tg,i_g(x)} Px + b_{tg,i_g(x)}\bigr).
$$

Training uses a local min-face surrogate against the runner-up branch,

$$
\tilde y_g(x)
=
F_{g,i_g}(x)
+ U\bigl(s_{g,i_g}(x) - s_{g,j_g}(x)\bigr)
\bigl(F_{g,j_g}(x) - F_{g,i_g}(x)\bigr),
$$

with

$$
U(m) = \frac{0.5}{1 + |m|}.
$$

Inference remains fully hard. Only the training path uses the local surrogate.

## Package Layout

- `__init__.py`: public API
- `module.py`: `TropLinear`
- `functional.py`: stateless tensor operations and min-face logic
- `triton_ops.py`: optional Triton score kernel
- `example_emnist.py`: self-contained EMNIST training example

The reference implementation lives in `functional.py` and `module.py`. Triton is optional and should be treated as an optimization layer, not the source of truth.

## Quick Start

Install from this directory:

```bash
uv pip install -e .
```

Basic usage:

```python
import torch
from tropnn import TropLinear

layer = TropLinear(
    256,
    512,
    tables=16,
    groups=2,
    cells=4,
    rank=32,
)

x = torch.randn(8, 256)
y = layer(x)
print(y.shape)
```

Because `TropLinear` behaves like a basic neural layer, collaborators can stack it directly into MLPs, residual blocks, or recurrent modules.

## EMNIST Example

The package includes a small example that reproduces the `trop_minface` EMNIST experiment used in the parent `lutflow` repository.

Run:

```bash
uv run python -m tropnn.example_emnist \
  --root /path/to/emnist \
  --split digits \
  --epochs 3 \
  --batch-size 512 \
  --lr 1e-3 \
  --tables 16 \
  --hidden-dim 128 \
  --depth 2 \
  --groups 2 \
  --cells 4 \
  --rank 32 \
  --max-train 20000 \
  --max-test 4000 \
  --device cuda \
  --seed 0
```

On the same setup used in the original experiment path, this reproduces:

- epoch 1: `test_loss=0.5054`, `test_acc=0.8468`
- epoch 2: `test_loss=0.3343`, `test_acc=0.8975`
- epoch 3: `test_loss=0.2641`, `test_acc=0.9223`

## Triton

`triton.py` contains an optional score kernel for the grouped affine routing scores. If Triton is not available, or if the environment cannot compile Triton launchers, the package falls back to the pure PyTorch reference path.

Recommended usage:

- `backend="torch"` for correctness and teaching
- `backend="auto"` for practical use on CUDA
- `backend="triton"` only when the environment is known to support Triton compilation

## Reading Order

For collaborators, the recommended reading order is:

1. `functional.py`
2. `module.py`
3. `example_emnist.py`
4. `triton_ops.py`

This keeps the mathematical semantics separate from the acceleration path.

## Scope

This package is deliberately minimal. It does not attempt to include:

- pairwise or sparse comparators,
- experiment naming or benchmark orchestration,
- multiple surrogate families,
- large training frameworks.

The goal is to keep the core layer understandable, reusable, and easy to adapt into new architectures.