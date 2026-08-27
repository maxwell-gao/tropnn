import torch
from tropnn.tools.product_tree_executor_benchmark import _comparison_work


def test_depth_four_lookahead_work_ledger() -> None:
    assert _comparison_work(4, 1) == (4, 4)
    assert _comparison_work(4, 2) == (2, 6)
    assert _comparison_work(4, 4) == (1, 15)


def test_comparison_work_handles_partial_final_group() -> None:
    assert _comparison_work(5, 2) == (3, 7)
    assert torch.tensor(7).item() == 7
