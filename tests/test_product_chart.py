import torch
from tropnn.layers.product_chart import ProductChartField


def _centroids() -> torch.Tensor:
    generator = torch.Generator().manual_seed(900)
    return torch.randn(3, 4, 2, generator=generator)


def test_hybrid_product_coordinates_reconstruct_input_exactly() -> None:
    model = ProductChartField(_centroids(), 5, 3, action="constant", surrogate="none", trainable_centroids=False)
    x = torch.randn(2, 7, 6, generator=torch.Generator().manual_seed(901))
    coordinates = model.chart_coordinates(x)
    assert torch.equal(coordinates.codes, model.hard_codes(x))
    torch.testing.assert_close(model.reconstruct(coordinates), x, rtol=0, atol=2e-7)


def test_zero_local_maps_strictly_nest_constant_action() -> None:
    centroids = _centroids()
    constant = ProductChartField(centroids, 5, 3, action="constant", surrogate="none", seed=902)
    local = ProductChartField(centroids, 5, 3, action="local_linear", surrogate="none", seed=999)
    with torch.no_grad():
        local.centroids.copy_(constant.centroids)
        local.offsets.copy_(constant.offsets)
        local.output_basis.copy_(constant.output_basis)
        local.local_maps.zero_()
    x = torch.randn(32, 6, generator=torch.Generator().manual_seed(903))
    assert torch.equal(constant.hard_codes(x), local.hard_codes(x))
    assert torch.equal(constant(x), local(x))


def test_zero_shared_maps_strictly_nest_constant_action() -> None:
    centroids = _centroids()
    constant = ProductChartField(centroids, 5, 3, action="constant", surrogate="none", seed=912)
    shared = ProductChartField(centroids, 5, 3, action="shared_linear", surrogate="none", seed=999)
    with torch.no_grad():
        shared.centroids.copy_(constant.centroids)
        shared.offsets.copy_(constant.offsets)
        shared.output_basis.copy_(constant.output_basis)
        shared.local_maps.zero_()
    x = torch.randn(32, 6, generator=torch.Generator().manual_seed(913))
    assert torch.equal(constant.hard_codes(x), shared.hard_codes(x))
    assert torch.equal(constant(x), shared(x))


def test_soft_pq_changes_backward_but_not_hard_forward() -> None:
    model = ProductChartField(_centroids(), 5, 3, action="local_linear", surrogate="soft_pq", seed=904)
    with torch.no_grad():
        model.local_maps.normal_(std=0.1)
    x = torch.randn(64, 6, generator=torch.Generator().manual_seed(905), requires_grad=True)
    explicit, codes = model.hard_output(x)
    deployed = model(x)
    assert torch.equal(deployed, explicit)
    assert torch.equal(codes, model.hard_codes(x))
    deployed.square().mean().backward()
    assert x.grad is not None and torch.count_nonzero(x.grad) > 0
    assert model.centroids.grad is not None and torch.count_nonzero(model.centroids.grad) > 0
    assert model.offsets.grad is not None and torch.count_nonzero(model.offsets.grad) > 0
    assert model.local_maps.grad is not None and torch.count_nonzero(model.local_maps.grad) > 0
    assert model.output_basis.grad is not None and torch.count_nonzero(model.output_basis.grad) > 0


def test_soft_pq_backward_is_hard_action_plus_soft_mixture_gradient() -> None:
    model = ProductChartField(_centroids(), 5, 3, action="local_linear", surrogate="soft_pq", seed=914)
    with torch.no_grad():
        model.local_maps.normal_(std=0.1)
    actual_x = torch.randn(17, 6, generator=torch.Generator().manual_seed(915), requires_grad=True)
    actual = torch.autograd.grad(model(actual_x).square().sum(), actual_x)[0]

    expected_x = actual_x.detach().clone().requires_grad_(True)
    hard = model.hard_output(expected_x)[0]
    soft = model.soft_output(expected_x)
    # The detached soft value cancels numerically in ``forward`` but not in
    # backward, so the expected gradient contains both paths evaluated with
    # the hard output as the upstream derivative.
    expected = torch.autograd.grad((2 * hard.detach() * (hard + soft)).sum(), expected_x)[0]
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_shared_map_is_constant_lookup_plus_one_global_low_rank_map() -> None:
    generator = torch.Generator().manual_seed(916)
    model = ProductChartField(_centroids(), 5, 3, action="shared_linear", surrogate="none", seed=917)
    with torch.no_grad():
        model.local_maps.normal_(std=0.2)
    x = torch.randn(29, 6, generator=generator)
    output, codes = model.hard_output(x)
    local = x.reshape(-1, model.tables, model.block_width)
    table = torch.arange(model.tables).unsqueeze(0).expand(x.shape[0], -1)
    selected_offsets = model.offsets[table, codes]
    selected_centroids = model.centroids[table, codes]
    maps = model.local_maps[:, 0]
    constant_latent = selected_offsets - torch.einsum("nts,tsr->ntr", selected_centroids, maps)
    global_latent = torch.einsum("nts,tsr->ntr", local, maps)
    expected = (constant_latent + global_latent).sum(dim=1) @ model.output_basis
    torch.testing.assert_close(output, expected, rtol=2e-6, atol=2e-6)


def test_product_chart_has_no_one_layer_cross_block_interaction() -> None:
    model = ProductChartField(_centroids(), 5, 3, action="local_linear", surrogate="none", seed=918)
    with torch.no_grad():
        model.local_maps.normal_(std=0.2)
    base = torch.randn(1, 6, generator=torch.Generator().manual_seed(919))
    delta_a = torch.zeros_like(base)
    delta_b = torch.zeros_like(base)
    delta_a[:, :2] = torch.tensor([[0.7, -0.4]])
    delta_b[:, 2:4] = torch.tensor([[-0.3, 0.6]])
    mixed_difference = model(base + delta_a + delta_b) - model(base + delta_a) - model(base + delta_b) + model(base)
    torch.testing.assert_close(mixed_difference, torch.zeros_like(mixed_difference), rtol=0, atol=5e-7)


def test_rank_sufficient_local_field_exactly_represents_linear_map() -> None:
    generator = torch.Generator().manual_seed(906)
    centroids = torch.randn(2, 4, 2, generator=generator)
    weight = torch.randn(4, 3, generator=generator)
    model = ProductChartField(
        centroids,
        output_dim=3,
        rank=3,
        action="local_linear",
        surrogate="none",
        trainable_centroids=False,
        seed=907,
    )
    with torch.no_grad():
        model.output_basis.copy_(torch.eye(3))
        for table in range(model.tables):
            block_weight = weight[2 * table : 2 * (table + 1)]
            model.local_maps[table].copy_(block_weight.unsqueeze(0).expand(model.codes, -1, -1))
            model.offsets[table].copy_(centroids[table] @ block_weight)
    x = torch.randn(128, 4, generator=torch.Generator().manual_seed(908))
    torch.testing.assert_close(model(x), x @ weight, rtol=2e-6, atol=2e-6)
