from __future__ import annotations

import argparse
import time

import torch
from torch import Tensor

from ..backend import has_tilelang, has_triton, trop_scores, trop_scores_reference
from ..layers import PairwiseLinear, TropLinear


def _copy_weights(dst: TropLinear, src: TropLinear) -> None:
    with torch.no_grad():
        dst.proj.weight.copy_(src.proj.weight)
        dst.router_weight.copy_(src.router_weight)
        dst.router_bias.copy_(src.router_bias)
        dst.code.copy_(src.code)
        dst.output_proj.weight.copy_(src.output_proj.weight)
        if src.output_proj.bias is not None and dst.output_proj.bias is not None:
            dst.output_proj.bias.copy_(src.output_proj.bias)


def _copy_pairwise_weights(dst: PairwiseLinear, src: PairwiseLinear) -> None:
    with torch.no_grad():
        dst.anchors.copy_(src.anchors)
        dst.thresholds.copy_(src.thresholds)
        dst.lut.copy_(src.lut)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _module_state_bytes(layer: torch.nn.Module) -> int:
    total = 0
    for tensor in list(layer.parameters()) + list(layer.buffers()):
        total += tensor.numel() * tensor.element_size()
    return total


def _bench_forward_peak_bytes(layer: torch.nn.Module, x: Tensor) -> int:
    if x.device.type != "cuda":
        return 0
    layer.eval()
    _sync_if_cuda(x.device)
    baseline = torch.cuda.memory_allocated(x.device)
    torch.cuda.reset_peak_memory_stats(x.device)
    with torch.no_grad():
        output = layer(x)
        _sync_if_cuda(x.device)
    peak = torch.cuda.max_memory_allocated(x.device)
    del output
    _sync_if_cuda(x.device)
    return max(0, int(peak - baseline))


def _bench_train_peak_bytes(layer: torch.nn.Module, x: Tensor) -> int:
    if x.device.type != "cuda":
        return 0
    layer.train()
    layer.zero_grad(set_to_none=True)
    _sync_if_cuda(x.device)
    baseline = torch.cuda.memory_allocated(x.device)
    torch.cuda.reset_peak_memory_stats(x.device)
    loss = layer(x).square().mean()
    loss.backward()
    _sync_if_cuda(x.device)
    peak = torch.cuda.max_memory_allocated(x.device)
    del loss
    layer.zero_grad(set_to_none=True)
    _sync_if_cuda(x.device)
    return max(0, int(peak - baseline))


def _bench_forward(layer: torch.nn.Module, x: Tensor, *, warmup: int, steps: int) -> tuple[Tensor, float]:
    with torch.no_grad():
        for _ in range(warmup):
            layer(x)
        _sync_if_cuda(x.device)
        t0 = time.perf_counter()
        last = None
        for _ in range(steps):
            last = layer(x)
        _sync_if_cuda(x.device)
    assert last is not None
    return last, (time.perf_counter() - t0) * 1000.0 / steps


def _bench_train_step(layer: torch.nn.Module, x: Tensor, *, warmup: int, steps: int) -> float:
    layer.train()
    for _ in range(warmup):
        layer.zero_grad(set_to_none=True)
        loss = layer(x).square().mean()
        loss.backward()
    _sync_if_cuda(x.device)
    t0 = time.perf_counter()
    for _ in range(steps):
        layer.zero_grad(set_to_none=True)
        loss = layer(x).square().mean()
        loss.backward()
    _sync_if_cuda(x.device)
    return (time.perf_counter() - t0) * 1000.0 / steps


def _bench_scores(layer: TropLinear, x: Tensor, *, warmup: int, steps: int) -> tuple[float, float, float]:
    if x.ndim == 2:
        x = x.unsqueeze(1)
    compute_dtype = torch.float32 if x.dtype in {torch.float16, torch.bfloat16} else x.dtype
    z = layer.proj(x).to(compute_dtype)
    router_weight = layer.router_weight.to(dtype=compute_dtype, device=x.device)
    router_bias = layer.router_bias.to(dtype=compute_dtype, device=x.device)

    with torch.no_grad():
        for _ in range(warmup):
            trop_scores_reference(z, router_weight, router_bias)
        _sync_if_cuda(x.device)
        t0 = time.perf_counter()
        ref_out = None
        for _ in range(steps):
            ref_out = trop_scores_reference(z, router_weight, router_bias)
        _sync_if_cuda(x.device)
        ref_ms = (time.perf_counter() - t0) * 1000.0 / steps

        for _ in range(warmup):
            tri_out = trop_scores(z, router_weight, router_bias, backend="auto")
        _sync_if_cuda(x.device)
        t0 = time.perf_counter()
        for _ in range(steps):
            tri_out = trop_scores(z, router_weight, router_bias, backend="auto")
        _sync_if_cuda(x.device)
        tri_ms = (time.perf_counter() - t0) * 1000.0 / steps

    assert ref_out is not None
    return ref_ms, tri_ms, float((ref_out - tri_out).abs().max().item())


def benchmark_trop_linear_auto(
    *,
    batch_size: int = 512,
    in_features: int = 28 * 28,
    out_features: int = 128,
    cells: int = 4,
    heads: int = 32,
    code_dim: int = 32,
    warmup: int = 20,
    steps: int = 100,
    device: str = "cuda",
    seed: int = 0,
    compile_cpu: bool = False,
    pairwise_tables: int = 136,
    comparisons: int = 6,
) -> dict[str, float]:
    dev = torch.device(device)
    torch.manual_seed(seed)
    torch_layer = TropLinear(
        in_features,
        out_features,
        heads=heads,
        cells=cells,
        code_dim=code_dim,
        backend="torch",
        seed=seed,
    ).to(dev)
    auto_layer = TropLinear(
        in_features,
        out_features,
        heads=heads,
        cells=cells,
        code_dim=code_dim,
        backend="auto",
        seed=seed,
    ).to(dev)
    tilelang_layer = TropLinear(
        in_features,
        out_features,
        heads=heads,
        cells=cells,
        code_dim=code_dim,
        backend="tilelang",
        seed=seed,
    ).to(dev)
    _copy_weights(auto_layer, torch_layer)
    _copy_weights(tilelang_layer, torch_layer)
    torch_layer.eval()
    auto_layer.eval()
    tilelang_layer.eval()
    pairwise_torch = PairwiseLinear(
        in_features,
        out_features,
        tables=pairwise_tables,
        comparisons=comparisons,
        backend="torch",
        seed=seed,
    ).to(dev)
    pairwise_tilelang = PairwiseLinear(
        in_features,
        out_features,
        tables=pairwise_tables,
        comparisons=comparisons,
        backend="tilelang",
        seed=seed,
    ).to(dev)
    _copy_pairwise_weights(pairwise_tilelang, pairwise_torch)
    pairwise_torch.eval()
    pairwise_tilelang.eval()
    x = torch.randn(batch_size, in_features, device=dev)
    score_torch_ms, score_auto_ms, score_max_diff = _bench_scores(torch_layer, x, warmup=warmup, steps=steps)
    torch_out, torch_ms = _bench_forward(torch_layer, x, warmup=warmup, steps=steps)
    auto_out, auto_ms = _bench_forward(auto_layer, x, warmup=warmup, steps=steps)
    tilelang_ms = float("nan")
    tilelang_max_diff = float("nan")
    if has_tilelang() and dev.type == "cuda":
        try:
            tilelang_out, tilelang_ms = _bench_forward(tilelang_layer, x, warmup=warmup, steps=steps)
            tilelang_max_diff = float((torch_out - tilelang_out).abs().max().item())
        except RuntimeError as exc:
            print(f"tilelang_unavailable={exc}")
    compiled_cpu_ms = float("nan")
    compiled_cpu_max_diff = float("nan")
    if compile_cpu and dev.type == "cpu":
        compiled_layer = torch.compile(torch_layer, mode="reduce-overhead")
        compiled_out, compiled_cpu_ms = _bench_forward(compiled_layer, x, warmup=warmup, steps=steps)
        compiled_cpu_max_diff = float((torch_out - compiled_out).abs().max().item())
    max_diff = float((torch_out - auto_out).abs().max().item())
    pairwise_torch_out, pairwise_torch_ms = _bench_forward(pairwise_torch, x, warmup=warmup, steps=steps)
    pairwise_tilelang_ms = float("nan")
    pairwise_tilelang_max_diff = float("nan")
    if has_tilelang() and dev.type == "cuda":
        pairwise_tilelang_out, pairwise_tilelang_ms = _bench_forward(pairwise_tilelang, x, warmup=warmup, steps=steps)
        pairwise_tilelang_max_diff = float((pairwise_torch_out - pairwise_tilelang_out).abs().max().item())
    tropical_torch_train_ms = _bench_train_step(torch_layer, x, warmup=max(1, warmup // 4), steps=max(1, steps // 4))
    tropical_tilelang_train_ms = float("nan")
    pairwise_torch_train_ms = _bench_train_step(pairwise_torch, x, warmup=max(1, warmup // 4), steps=max(1, steps // 4))
    pairwise_tilelang_train_ms = float("nan")
    if has_tilelang() and dev.type == "cuda":
        tropical_tilelang_train_ms = _bench_train_step(tilelang_layer, x, warmup=max(1, warmup // 4), steps=max(1, steps // 4))
        pairwise_tilelang_train_ms = _bench_train_step(pairwise_tilelang, x, warmup=max(1, warmup // 4), steps=max(1, steps // 4))
    tropical_torch_forward_peak_bytes = _bench_forward_peak_bytes(torch_layer, x)
    tropical_auto_forward_peak_bytes = _bench_forward_peak_bytes(auto_layer, x)
    tropical_tilelang_forward_peak_bytes = float("nan")
    pairwise_torch_forward_peak_bytes = _bench_forward_peak_bytes(pairwise_torch, x)
    pairwise_tilelang_forward_peak_bytes = float("nan")
    tropical_torch_train_peak_bytes = _bench_train_peak_bytes(torch_layer, x)
    tropical_tilelang_train_peak_bytes = float("nan")
    pairwise_torch_train_peak_bytes = _bench_train_peak_bytes(pairwise_torch, x)
    pairwise_tilelang_train_peak_bytes = float("nan")
    if has_tilelang() and dev.type == "cuda":
        tropical_tilelang_forward_peak_bytes = _bench_forward_peak_bytes(tilelang_layer, x)
        pairwise_tilelang_forward_peak_bytes = _bench_forward_peak_bytes(pairwise_tilelang, x)
        tropical_tilelang_train_peak_bytes = _bench_train_peak_bytes(tilelang_layer, x)
        pairwise_tilelang_train_peak_bytes = _bench_train_peak_bytes(pairwise_tilelang, x)
    return {
        "tropical_state_bytes": float(_module_state_bytes(torch_layer)),
        "pairwise_state_bytes": float(_module_state_bytes(pairwise_torch)),
        "score_torch_ms": score_torch_ms,
        "score_auto_ms": score_auto_ms,
        "score_speedup": score_torch_ms / score_auto_ms,
        "score_max_diff": score_max_diff,
        "torch_ms": torch_ms,
        "auto_ms": auto_ms,
        "tilelang_ms": tilelang_ms,
        "speedup": torch_ms / auto_ms,
        "tilelang_speedup": torch_ms / tilelang_ms,
        "compiled_cpu_ms": compiled_cpu_ms,
        "compiled_cpu_speedup": torch_ms / compiled_cpu_ms,
        "tropical_torch_train_ms": tropical_torch_train_ms,
        "tropical_tilelang_train_ms": tropical_tilelang_train_ms,
        "tropical_tilelang_train_speedup": tropical_torch_train_ms / tropical_tilelang_train_ms,
        "pairwise_torch_ms": pairwise_torch_ms,
        "pairwise_tilelang_ms": pairwise_tilelang_ms,
        "pairwise_tilelang_speedup": pairwise_torch_ms / pairwise_tilelang_ms,
        "pairwise_torch_train_ms": pairwise_torch_train_ms,
        "pairwise_tilelang_train_ms": pairwise_tilelang_train_ms,
        "pairwise_tilelang_train_speedup": pairwise_torch_train_ms / pairwise_tilelang_train_ms,
        "tropical_torch_forward_peak_bytes": float(tropical_torch_forward_peak_bytes),
        "tropical_auto_forward_peak_bytes": float(tropical_auto_forward_peak_bytes),
        "tropical_tilelang_forward_peak_bytes": tropical_tilelang_forward_peak_bytes,
        "tropical_torch_train_peak_bytes": float(tropical_torch_train_peak_bytes),
        "tropical_tilelang_train_peak_bytes": tropical_tilelang_train_peak_bytes,
        "pairwise_torch_forward_peak_bytes": float(pairwise_torch_forward_peak_bytes),
        "pairwise_tilelang_forward_peak_bytes": pairwise_tilelang_forward_peak_bytes,
        "pairwise_torch_train_peak_bytes": float(pairwise_torch_train_peak_bytes),
        "pairwise_tilelang_train_peak_bytes": pairwise_tilelang_train_peak_bytes,
        "max_diff": max_diff,
        "tilelang_max_diff": tilelang_max_diff,
        "pairwise_tilelang_max_diff": pairwise_tilelang_max_diff,
        "compiled_cpu_max_diff": compiled_cpu_max_diff,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TropLinear backend='auto' against backend='torch'.")
    for name, default in (
        ("--batch-size", 512),
        ("--in-features", 28 * 28),
        ("--out-features", 128),
        ("--heads", 32),
        ("--cells", 4),
        ("--code-dim", 32),
        ("--warmup", 20),
        ("--steps", 100),
        ("--pairwise-tables", 136),
        ("--comparisons", 6),
    ):
        parser.add_argument(name, type=int, default=default)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile-cpu", action="store_true", help="Also benchmark torch.compile on CPU inference.")
    parser.add_argument("--num-threads", type=int, default=0, help="Set torch CPU threads when > 0.")
    args = parser.parse_args()

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    print(f"cuda={torch.cuda.is_available()} has_triton={has_triton()} has_tilelang={has_tilelang()} device={args.device}")
    result = benchmark_trop_linear_auto(
        batch_size=args.batch_size,
        in_features=args.in_features,
        out_features=args.out_features,
        heads=args.heads,
        cells=args.cells,
        code_dim=args.code_dim,
        warmup=args.warmup,
        steps=args.steps,
        device=args.device,
        seed=args.seed,
        compile_cpu=args.compile_cpu,
        pairwise_tables=args.pairwise_tables,
        comparisons=args.comparisons,
    )
    for key in (
        "tropical_state_bytes",
        "pairwise_state_bytes",
        "score_torch_ms",
        "score_auto_ms",
        "score_speedup",
        "score_max_diff",
        "torch_ms",
        "auto_ms",
        "tilelang_ms",
        "speedup",
        "tilelang_speedup",
        "compiled_cpu_ms",
        "compiled_cpu_speedup",
        "tropical_torch_train_ms",
        "tropical_tilelang_train_ms",
        "tropical_tilelang_train_speedup",
        "pairwise_torch_ms",
        "pairwise_tilelang_ms",
        "pairwise_tilelang_speedup",
        "pairwise_torch_train_ms",
        "pairwise_tilelang_train_ms",
        "pairwise_tilelang_train_speedup",
        "tropical_torch_forward_peak_bytes",
        "tropical_auto_forward_peak_bytes",
        "tropical_tilelang_forward_peak_bytes",
        "tropical_torch_train_peak_bytes",
        "tropical_tilelang_train_peak_bytes",
        "pairwise_torch_forward_peak_bytes",
        "pairwise_tilelang_forward_peak_bytes",
        "pairwise_torch_train_peak_bytes",
        "pairwise_tilelang_train_peak_bytes",
        "max_diff",
        "tilelang_max_diff",
        "pairwise_tilelang_max_diff",
        "compiled_cpu_max_diff",
    ):
        precision = 6 if "diff" in key else 4
        print(f"{key}={result[key]:.{precision}f}")


if __name__ == "__main__":
    main()
