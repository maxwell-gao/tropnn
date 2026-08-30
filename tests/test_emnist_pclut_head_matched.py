from argparse import Namespace

import torch
import torch.nn.functional as F
from tropnn.tools.emnist_pclut_head_matched import (
    ClassicPairHead,
    Evaluation,
    _train_pair_heads,
    summarize,
)
from tropnn.tools.product_atlas_pc_action_factorial import fit_additive_rows


def _head(*, rows: torch.Tensor, train_rows: bool, train_thresholds: bool) -> ClassicPairHead:
    return ClassicPairHead(
        8,
        3,
        tables=2,
        comparisons=2,
        seed=7,
        rows=rows,
        trainable_rows=train_rows,
        trainable_thresholds=train_thresholds,
    )


def test_classic_pair_head_hard_code_and_output_replay() -> None:
    rows = torch.randn(2, 4, 3, generator=torch.Generator().manual_seed(1))
    head = _head(rows=rows, train_rows=False, train_thresholds=False)
    x = torch.randn(64, 8, generator=torch.Generator().manual_seed(2))
    hard, codes = head.hard_output(x)
    assert codes.shape == (64, 2)
    assert torch.equal(hard, head.eval()(x))
    assert torch.equal(hard, sum(rows[table, codes[:, table]] for table in range(2)))


def test_compiled_rows_improve_dense_teacher_fit() -> None:
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(4096, 8, generator=generator)
    teacher = torch.randn(8, 3, generator=generator)
    target = x @ teacher
    template = _head(rows=torch.zeros(2, 4, 3), train_rows=False, train_thresholds=False)
    codes = template.hard_codes(x)
    rows = fit_additive_rows(codes, target, 4, ridge=1.0)
    fitted = ClassicPairHead(
        8,
        3,
        tables=2,
        comparisons=2,
        seed=7,
        rows=rows,
        supports=template.supports,
        thresholds=torch.zeros(2, 2),
        trainable_rows=False,
        trainable_thresholds=False,
    )
    assert F.mse_loss(fitted(x), target) < F.mse_loss(torch.zeros_like(target), target)


def test_fixed_and_joint_training_update_expected_parameters() -> None:
    generator = torch.Generator().manual_seed(4)
    x = torch.randn(256, 8, generator=generator)
    labels = torch.randint(3, (256,), generator=generator)
    rows = torch.randn(2, 4, 3, generator=generator) * 0.01
    template = _head(rows=rows, train_rows=False, train_thresholds=False)
    common = dict(
        input_dim=8,
        output_dim=3,
        tables=2,
        comparisons=2,
        seed=7,
        rows=rows,
        supports=template.supports,
        thresholds=torch.zeros(2, 2),
    )
    fixed = ClassicPairHead(**common, trainable_rows=True, trainable_thresholds=False)
    joint = ClassicPairHead(**common, trainable_rows=True, trainable_thresholds=True)
    fixed_rows = fixed.rows.detach().clone()
    joint_rows = joint.rows.detach().clone()
    curves, gradients = _train_pair_heads(
        fixed,
        joint,
        x,
        labels,
        epochs=2,
        batch_size=64,
        lr=0.01,
        seed=0,
        device=torch.device("cpu"),
    )
    assert not torch.equal(fixed.rows, fixed_rows)
    assert not torch.equal(joint.rows, joint_rows)
    assert fixed.thresholds.square().sum() == 0
    assert joint.thresholds.square().sum() > 0
    assert gradients["fixed_rows"] > 0
    assert gradients["joint_rows"] > 0
    assert gradients["joint_thresholds"] > 0
    assert len(curves["pair_joint_route_action"]) == 2


def test_summary_uses_named_matched_effects() -> None:
    rows = []
    for seed in range(3):
        for arm, ce in (
            ("pq_tied_frozen", 0.9),
            ("pq_free_action", 0.7),
            ("grid_tied_frozen", 1.0),
            ("grid_free_action", 0.8),
            ("pair_tied_frozen", 1.0),
            ("pair_free_action", 0.85),
            ("pair_joint_route_action", 0.75),
        ):
            rows.append(Evaluation(seed, arm, ce, 0.8, 3.0, 2.0, 4.0, 0.5, 0.0))
    value = summarize(rows, {seed: (0.73, 0.8) for seed in range(3)})
    assert value["decisions"] == {
        "route_training_materially_helps_pair": True,
        "free_rows_materially_help_pair": True,
        "pair_joint_within_0p02_ce_of_pq": False,
        "pair_joint_materially_beats_grid": True,
    }


def test_formal_defaults_are_explicit() -> None:
    args = Namespace(hidden_dim=64, tables=32, comparisons=4, compiler_samples=32768, epochs=10)
    assert (args.hidden_dim, args.tables, args.comparisons, args.compiler_samples, args.epochs) == (64, 32, 4, 32768, 10)
