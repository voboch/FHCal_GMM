import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import approx_fprime
from scipy.stats import norm

from analyze_module_energy import conditional_shape
from analyze_gmm_packet_decoder import OBSERVABLES, section_statistics
from conditional_gmm_baseline import sample_packet_fractions
from evaluate_gaussian_projection import read_energy_rows
from pipeline.projection_geometry import legacy_grid_to_modules, load_projection_geometry
from pipeline.gaussian_projection import SectionProjector, fit_mixture, section_bounds, shape_objective
from pipeline.layered_projection import LayeredSectionProjector, SamplingStack


def test_geometry_excludes_hole_and_padded_sections():
    geometry = load_projection_geometry()
    lower, upper = section_bounds(geometry)
    centers = (lower + upper) / 2
    assert len(lower) == int(geometry.layer_mask.sum()) == 438
    assert not np.any(np.all(centers[:, :2] == 0, axis=1))
    encoded = np.zeros((1, 7, 9, 10))
    assert geometry.modules[18].module_id == 19
    encoded[0, 3, 2, 0] = 3
    assert legacy_grid_to_modules(encoded, geometry)[0, 18, 0] == 3
    encoded[0, 3, 2, 7] = 1
    with pytest.raises(ValueError, match="absent physical section"):
        legacy_grid_to_modules(encoded, geometry)


def test_integrals_match_quadrature_for_unequal_boxes_in_tail():
    low = np.array([[-2, -2, 0], [1, 0, 0], [8, -1, 1]], dtype=float)
    high = np.array([[0, 0, 1], [4, 1, 2], [10, 1, 2]], dtype=float)
    mean = np.array([0.5, -0.8, 0.3])
    sigma = np.array([1.0, 0.5, 0.7])
    expected = np.array([
        np.prod([quad(lambda x: norm.pdf(x, loc=mean[j], scale=sigma[j]), lo[j], hi[j])[0]
                 for j in range(3)])
        for lo, hi in zip(low, high)
    ])
    result, _ = SectionProjector(low, high, "integrated").component_shapes(mean[None], np.log(sigma[None]))
    np.testing.assert_allclose(result[0], expected / expected.sum(), atol=0, rtol=1e-8)
    assert result[0, 2] > 0


@pytest.mark.parametrize("method", ["integrated", "center"])
def test_loss_gradient_and_single_component_fit(method):
    low = np.array([[-2, -2, 0], [0, -2, 0], [-2, 0, 1], [0, 0, 1]], dtype=float)
    projector = SectionProjector(low, low + [2, 2, 1], method)
    params = np.array([-1, -0.5, 0.2, 1, 0.3, 1.5, 0.1, -0.2, 0.4, 0.2, 0.1, -0.1, 0.35])
    reference = np.array([0.7, 0.1, 0.2, 0.0])
    _, exact = shape_objective(params, projector, reference, 2)
    numeric = approx_fprime(params, lambda p: shape_objective(p, projector, reference, 2)[0], 1e-6)
    np.testing.assert_allclose(exact, numeric, atol=3e-7, rtol=1e-4)
    target = projector.mixture(np.array([0.1, 0.1, 1.0, 0.0, 0.0, 0.0]), 1)[0]
    fitted = fit_mixture(projector, target, 1, 2, 150, 7, [0.2, 0.2, 0.2])
    assert fitted.loss < 1e-8
    np.testing.assert_allclose(fitted.prediction.sum(), 1)


def test_select_energy_rows_without_reordering(tmp_path):
    path = tmp_path / "response.npz"
    array = np.arange(5 * 7 * 9 * 10, dtype=np.float32).reshape(5, 7, 9, 10)
    np.savez_compressed(path, EnergyDeposit=array)
    np.testing.assert_array_equal(read_energy_rows(path, np.array([4, 0, 2]), 5), array[[4, 0, 2]])


def test_packet_decoder_preserves_mass_and_mean_with_discrete_support():
    components = np.array([[0.25, 0.75, 0.0], [0.0, 0.0, 1.0]])
    weights = np.array([0.6, 0.4])
    draws = sample_packet_fractions(components, weights, 1, np.random.default_rng(42), replicas=4096)
    assert draws.shape == (4096, 3)
    np.testing.assert_allclose(draws.sum(axis=1), 1)
    np.testing.assert_allclose(draws[:, 2], 0.4)
    np.testing.assert_array_equal((draws[:, :2] > 0).sum(axis=1), 1)
    np.testing.assert_allclose(draws.mean(axis=0), weights @ components, atol=0.015)


def test_longitudinal_neighbours_do_not_cross_modules_or_thresholds():
    response = np.array([[0.0, 60.0, 40.0, 0.0, 0.0], [98.0, 0.5, 0.0, 1.5, 0.0]])
    values = dict(zip(OBSERVABLES, section_statistics(response, np.array([True, False, True, True])).T, strict=True))
    np.testing.assert_allclose(values["neighbours_pct"], [0.0, 200 / 3])
    np.testing.assert_allclose(values["neighbours_ge_1pct_pct"], [0.0, 0.0])
    np.testing.assert_allclose(values["weak_share_pct"], [0.0, 0.5])


def test_asymmetric_line_shape_is_a_conditional_cdf():
    x = np.r_[np.linspace(0, 20, 101), np.inf]
    for sigma in (0.2, 2.5):
        low = conditional_shape(x, [5.0, 0.6, sigma], True, order=80)
        high = conditional_shape(x, [5.0, 0.6, sigma], True, order=160)
        np.testing.assert_allclose(low[[0, -1]], [0, 1], atol=1e-12)
        assert np.all(np.diff(low) >= 0)
        np.testing.assert_allclose(low, high, atol=.001)


def test_mass_depth_widens_passive_gaps_and_preserves_section_period():
    stack = SamplingStack()
    geo_lo, geo_hi = stack.active_intervals(2, mass_depth=False)
    mass_lo, mass_hi = stack.active_intervals(2, mass_depth=True)
    np.testing.assert_allclose(mass_hi, geo_hi)
    np.testing.assert_allclose(mass_lo[1] - mass_lo[0], 1)
    np.testing.assert_allclose((geo_hi - geo_lo).sum(axis=-1), .2)
    np.testing.assert_allclose((mass_hi - mass_lo).sum(axis=-1), .424 / 18.584)
    assert np.all(mass_lo > geo_lo)


@pytest.mark.parametrize("settings", [
    {"absorber_cm": -1.6, "absorber_density": -11.35},
    {"scintillator_density": np.inf},
    {"plates_per_section": 6.0},
    {"plates_per_section": True},
])
def test_sampling_stack_rejects_invalid_physical_inputs(settings):
    with pytest.raises(ValueError):
        SamplingStack(**settings)


def test_sampling_stack_rejects_overflow_and_unresolvable_intervals():
    for stack in (SamplingStack(absorber_cm=1e308, scintillator_cm=1e308),
                  SamplingStack(scintillator_cm=1e-300)):
        with pytest.raises(ValueError):
            stack.active_intervals(1, mass_depth=False)
    for sections in (0, 2.0, True):
        with pytest.raises(ValueError, match="Sections"):
            SamplingStack().active_intervals(sections, mass_depth=True)


@pytest.mark.parametrize("mass_depth", [False, True])
def test_active_plate_integral_and_gradient(mass_depth):
    projector = LayeredSectionProjector(load_projection_geometry(), mass_depth=mass_depth)
    parameters = np.array([0., -20., 1.82, np.log(11), np.log(13), np.log(.32)])
    means, log_scales = parameters[:3][None], parameters[3:][None]
    shape, gradient = projector.component_shapes(means, log_scales)
    expected = approx_fprime(parameters, lambda p: projector.component_shapes(p[None, :3], p[None, 3:])[0][0], 1e-6)
    np.testing.assert_allclose(shape.sum(), 1)
    np.testing.assert_allclose(gradient[0], expected, rtol=1e-4, atol=5e-7)
    scales = np.exp(log_scales[0])
    xy = norm.cdf(projector.xy_upper, means[0, :2], scales[:2]) - norm.cdf(projector.xy_lower, means[0, :2], scales[:2])
    depth = np.array([sum(quad(norm.pdf, lo, hi, args=(means[0, 2], scales[2]))[0]
                          for lo, hi in zip(lower, upper, strict=True))
                      for lower, upper in zip(projector.active_lower, projector.active_upper, strict=True)])
    mass = xy.prod(axis=1)[projector.module] * depth[projector.depth]
    np.testing.assert_allclose(shape[0], mass / mass.sum(), atol=1e-12)
