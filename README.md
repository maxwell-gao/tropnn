# tropnn
`tropnn` is a minimal teaching-oriented tropical neural layer library.

It exposes one basic building block, `TropLinear`, for tropical min-face networks in the role that `nn.Linear` plays for dense networks.

The scope is narrow: hard chamber selection at inference, a local min-face surrogate during training, a readable reference path, and an optional Triton score kernel for the hottest routing step. Examples and benchmarking stay outside the core layer.

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
- `backend.py`: backend selection and score dispatch boundary
- `backends/triton_scores.py`: optional Triton score kernel
- `layers/base.py`: shared shell for routed linear layers
- `layers/tropical.py`: `TropLinear` concrete implementation
- `module.py`: compatibility re-export for `TropLinear`
- `examples/emnist.py`: self-contained EMNIST training example
- `tools/benchmark.py`: benchmarking helper and CLI

The reference implementation lives in `layers/base.py` and `layers/tropical.py`. `backend.py` is the only place that chooses between reference and accelerated paths. Triton is optional and should be treated as an optimization layer, not the source of truth.

## Quick Start

Install the core library from this directory:

```bash
uv pip install -e .
```

Install with Triton acceleration enabled:

```bash
uv pip install -e ".[triton]"
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
uv run python -m tropnn.examples.emnist \
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
`backends/triton_scores.py` contains an optional score kernel for the grouped affine routing scores. If Triton is not available, or if the environment cannot compile Triton launchers, the package falls back to the pure PyTorch reference path.

- `backend="torch"`: correctness and teaching
- `backend="auto"`: practical CUDA inference
- `backend="triton"`: only when launcher compilation is known to work

## Reading Order

For collaborators, the recommended reading order is `layers/base.py`, `layers/tropical.py`, `backend.py`, `examples/emnist.py`, `backends/triton_scores.py`, then `tools/benchmark.py`.

## Scope
This package is deliberately minimal. It does not attempt to include:

- pairwise or sparse comparators,
- experiment naming or benchmark orchestration,
- multiple surrogate families,
- large training frameworks.