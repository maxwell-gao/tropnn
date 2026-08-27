from argparse import Namespace

import torch
from tropnn.layers.maddness import FrozenMaddness
from tropnn.tools.maddness_end_to_end_ste_factorial import compile_original_maddness
from tropnn.tools.product_atlas_pc_action_factorial import (
    ARMS,
    anisotropic_teacher,
    cached_action_rows,
    fit_additive_rows,
    fit_seed,
    summarize,
)
from tropnn.tools.random_linear_address_action_factorial import orthogonal_teacher


def _fixture() -> tuple[torch.Tensor, torch.Tensor, object]:
    generator = torch.Generator().manual_seed(19)
    x = torch.randn(512, 8, generator=generator)
    teacher = orthogonal_teacher(8, 0, torch.device("cpu"))
    return x, teacher, compile_original_maddness(x, tables=4)


def test_cached_action_is_reconstruct_then_teacher() -> None:
    x, teacher, compiled = _fixture()
    reconstruction = FrozenMaddness(compiled)(x)
    action = cached_action_rows(compiled, teacher)
    action_compiled = type(compiled)(
        compiled.split_indices,
        compiled.thresholds,
        compiled.encoder_centroids,
        action,
    )
    cached = FrozenMaddness(action_compiled)(x)
    assert torch.allclose(cached, reconstruction @ teacher.T, rtol=1e-6, atol=2e-6)


def test_direct_output_ridge_commutes_with_linear_teacher() -> None:
    x, teacher, compiled = _fixture()
    codes = FrozenMaddness(compiled).hard_codes(x)
    direct = fit_additive_rows(codes, x @ teacher.T, leaves=16, ridge=1.0)
    cached = cached_action_rows(compiled, teacher)
    assert torch.allclose(direct, cached, rtol=2e-5, atol=2e-5)


def test_anisotropic_teacher_has_frozen_spectrum_contract() -> None:
    teacher = anisotropic_teacher(64, 0, torch.device("cpu"))
    singular = torch.linalg.svdvals(teacher.double())
    assert torch.allclose(singular.square().mean(), torch.tensor(1.0, dtype=torch.float64), atol=1e-7)
    assert abs(float(singular.max() / singular.min()) - 16.0) < 1e-5


def test_tiny_factorial_preserves_auxiliary_rows_and_hard_forward() -> None:
    args = Namespace(
        device="cpu",
        dim=8,
        tables=4,
        depth=4,
        compiler_samples=512,
        held_samples=256,
        steps=2,
        batch_size=64,
        lr=0.0001,
        ridge=1.0,
        tau=1.0,
        reconstruction_weight=1.0,
        teacher_mode="anisotropic",
        log_every=1,
    )
    rows, audit, state = fit_seed(0, args)
    assert [row.arm for row in rows] == list(ARMS)
    assert audit["all_hard_replays_exact"] is True
    assert audit["all_finite"] is True
    assert audit["frozen_reconstruction_rows_max_abs_difference"] == 0
    assert audit["orthogonal_task_and_inverse_nmse_max_abs_difference"] is None
    assert torch.equal(
        state["product_free_task_reconstruction.rows"][..., 8:],
        state["product.reconstruction_rows"],
    )
    summary = summarize(rows)
    assert set(summary["arms"]) == set(ARMS)
