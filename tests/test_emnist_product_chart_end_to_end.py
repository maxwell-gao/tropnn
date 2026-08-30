import torch
from tropnn.tools.emnist_product_chart_end_to_end import make_end_to_end_models


def _source_state(generator: torch.Generator) -> dict[str, torch.Tensor]:
    return {
        "seed0.dense_pretrained.stem.weight": torch.randn(4, 6, generator=generator),
        "seed0.dense_pretrained.stem.bias": torch.randn(4, generator=generator),
    }


def test_released_stems_are_exactly_paired_but_independent() -> None:
    generator = torch.Generator().manual_seed(930)
    centroids = torch.randn(2, 4, 2, generator=generator)
    rows = torch.randn(2, 4, 3, generator=generator)
    models, _error, initial = make_end_to_end_models(
        _source_state(generator),
        0,
        6,
        3,
        centroids,
        rows,
        rank=3,
        temperature=1.0,
    )
    states = [model.stem.state_dict() for model in models.values()]
    for state in states:
        for key in initial:
            assert torch.equal(state[key], initial[key])
    pointers = [model.stem.weight.data_ptr() for model in models.values()]
    assert len(set(pointers)) == len(pointers)


def test_released_stem_receives_gradient_through_hard_forward_surrogate() -> None:
    generator = torch.Generator().manual_seed(931)
    centroids = torch.randn(2, 4, 2, generator=generator)
    rows = torch.randn(2, 4, 3, generator=generator)
    models, _error, _initial = make_end_to_end_models(
        _source_state(generator),
        0,
        6,
        3,
        centroids,
        rows,
        rank=3,
        temperature=1.0,
    )
    model = models["trained_local"]
    x = torch.randn(32, 6, generator=generator)
    hard = model.head.hard_output(model.features(x))[0]
    deployed = model(x)
    assert torch.equal(hard, deployed)
    deployed.square().mean().backward()
    assert model.stem.weight.grad is not None and torch.count_nonzero(model.stem.weight.grad) > 0
    assert model.head.centroids.grad is not None and torch.count_nonzero(model.head.centroids.grad) > 0
    assert model.head.local_maps.grad is not None and torch.count_nonzero(model.head.local_maps.grad) > 0
