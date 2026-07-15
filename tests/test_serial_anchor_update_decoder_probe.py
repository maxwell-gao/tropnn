from __future__ import annotations

import torch

from tropnn.tools.serial_anchor_update_decoder_probe import AnchorRouteUpdate, SerialAnchorUpdateDecoder


def test_signed_anchor_update_is_zero_initialized_and_zero_sum() -> None:
    layer = AnchorRouteUpdate(16, tables=4, comparisons=3, variant="signed", seed=7)
    x = torch.randn(5, 1, 16)
    assert torch.count_nonzero(layer(x)) == 0
    with torch.no_grad():
        layer.payload.normal_()
    output = layer(x)
    torch.testing.assert_close(output.sum(dim=-1), torch.zeros(5, 1), atol=1e-6, rtol=0.0)


def test_two_sided_anchor_update_supports_common_mode() -> None:
    layer = AnchorRouteUpdate(16, tables=1, comparisons=2, variant="two_sided", seed=11)
    with torch.no_grad():
        layer.payload[..., 0].fill_(1.0)
        layer.payload[..., 1].fill_(1.0)
    output = layer(torch.randn(3, 1, 16))
    torch.testing.assert_close(output.sum(dim=-1), torch.full((3, 1), 4.0))


def test_serial_anchor_parameter_counts_match_sparse_budget() -> None:
    signed = SerialAnchorUpdateDecoder(64, depth=16, tables=16, comparisons=5, variant="signed", seed=0)
    two_sided = SerialAnchorUpdateDecoder(
        64, depth=16, tables=16, comparisons=5, variant="two_sided", seed=0
    )
    assert sum(parameter.numel() for parameter in signed.parameters()) == 38_912
    assert sum(parameter.numel() for parameter in two_sided.parameters()) == 77_312


def test_serial_anchor_route_surrogate_propagates_across_blocks() -> None:
    model = SerialAnchorUpdateDecoder(16, depth=4, tables=4, comparisons=3, variant="signed", seed=3)
    with torch.no_grad():
        for block in model.blocks:
            block.payload.normal_(std=0.1)
        model.readout.lut.normal_(std=0.1)
    pair = torch.randn(8, 16)
    model(pair).square().mean().backward()
    assert all(block.payload.grad is not None for block in model.blocks)
    assert all(torch.isfinite(block.payload.grad).all() for block in model.blocks)
