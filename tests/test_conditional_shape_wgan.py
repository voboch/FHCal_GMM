"""The conditional generator must preserve total signal even across geometry gaps."""

import numpy as np
import pytest

from pipeline.gaussian_projection import SectionProjector, section_bounds
from pipeline.projection_geometry import load_projection_geometry


def test_integrated_head_at_narrow_width_outside_active_sections():
    torch = pytest.importorskip("torch")
    from conditional_shape_wgan import LATENT, ShapeGenerator

    lower, upper = section_bounds(load_projection_geometry())
    origin, extent = lower.min(0), upper.max(0) - lower.min(0)
    # Find an in-bounds mean whose Gaussian has tiny mass on the active sections.
    grid = np.stack(np.meshgrid(*[np.linspace(0.001, 0.999, 9)] * 3), axis=-1).reshape(
        -1, 3
    )
    points = origin + grid * extent
    distance = np.maximum(
        np.maximum(lower - points[:, None], points[:, None] - upper), 0
    )
    scale = np.array([2.5, 2.5, 0.175])
    far = np.argmax(np.min(np.sum((distance / scale) ** 2, -1), axis=1))
    projector = SectionProjector(lower, upper, "integrated")
    expected, _ = projector.component_shapes(
        points[far : far + 1], np.log(scale[None, :])
    )
    amplitude = torch.tensor(grid[far], dtype=torch.float32)
    net = ShapeGenerator(torch.full((1, 438), 1 / 438), "gmm", 1)
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.zero_()
        net.prior[0, 1:4] = torch.logit(amplitude)
        net.prior[0, 4:7] = -40
    pred = net(torch.zeros(1, 2), torch.tensor([0]), torch.zeros(1, LATENT))
    np.testing.assert_allclose(pred.detach().numpy(), expected, atol=2e-5)
    np.testing.assert_allclose(pred.sum().item(), 1, atol=2e-6)
    (pred * torch.arange(438)).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters())
