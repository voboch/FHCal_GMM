"""Event-wise shape fits on the effective FHCal section volumes."""

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr, softmax


@dataclass(frozen=True)
class MixtureFit:
    prediction: np.ndarray
    loss: float
    converged: bool
    converged_restarts: int
    lower_width_hits: np.ndarray
    upper_width_hits: np.ndarray
    status: int
    iterations: int
    restart_trace: tuple[dict, ...] = ()


def section_bounds(geometry):
    """XY in cm, depth in section units relative to each module's first section."""
    boxes = []
    for module in geometry.modules:
        for layer in range(module.layers):
            boxes.append((
                [module.x - module.width / 2, module.y - module.height / 2, layer],
                [module.x + module.width / 2, module.y + module.height / 2, layer + 1],
            ))
    boxes = np.asarray(boxes, dtype=np.float64)
    return boxes[:, 0], boxes[:, 1]


class SectionProjector:
    """Condition each component on the physical readout support."""

    def __init__(self, lower, upper, method):
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        if (
            self.lower.ndim != 2 or self.lower.shape != self.upper.shape
            or self.lower.shape[1] != 3 or not len(self.lower)
            or not (np.isfinite(self.lower).all() and np.isfinite(self.upper).all())
            or np.any(self.upper <= self.lower)
        ):
            raise ValueError("Expected nonempty, finite, positive section boxes [C, 3]")
        if method not in {"integrated", "center"}:
            raise ValueError(f"Unknown section projection: {method}")
        self.method = method
        self.centers = (self.lower + self.upper) / 2
        self.extent = self.upper.max(axis=0) - self.lower.min(axis=0)
        self.volumes = np.prod(self.upper - self.lower, axis=1)

    def component_shapes(self, means, log_scales):
        """Return [K,C] probabilities and [K,C,6] parameter derivatives."""
        scale = np.exp(log_scales)[:, None, :]
        mean = means[:, None, :]
        midpoint = (self.centers - mean) / scale
        if self.method == "center":
            log_mass = -0.5 * np.sum(midpoint**2, axis=-1) + np.log(self.volumes)
            shape = softmax(log_mass, axis=1)
            log_gradient = np.concatenate((midpoint / scale, midpoint**2), axis=-1)
        else:
            lo = (self.lower - mean) / scale
            hi = (self.upper - mean) / scale
            # In the upper tail, subtract survival probabilities to avoid 1 - 1.
            axis_mass = np.where(lo >= 0, ndtr(-lo) - ndtr(-hi), ndtr(hi) - ndtr(lo))
            mass = np.prod(axis_mass, axis=-1)
            total = mass.sum(axis=1, keepdims=True)
            if np.any(total <= 0):
                raise ValueError("A component assigns no numerically observable mass to FHCal")
            shape = mass / total
            phi_lo = np.exp(-0.5 * lo**2) / np.sqrt(2 * np.pi)
            phi_hi = np.exp(-0.5 * hi**2) / np.sqrt(2 * np.pi)
            raw_gradients = ((phi_lo - phi_hi) / scale, lo * phi_lo - hi * phi_hi)
            log_gradient = np.concatenate([
                np.divide(gradient, axis_mass, out=np.zeros_like(gradient), where=axis_mass > 0)
                for gradient in raw_gradients
            ], axis=-1)
        mean_gradient = np.sum(shape[..., None] * log_gradient, axis=1, keepdims=True)
        return shape, shape[..., None] * (log_gradient - mean_gradient)

    def mixture(self, parameters, k):
        means, log_scales = parameters[:3*k].reshape(k, 3), parameters[3*k:6*k].reshape(k, 3)
        components, gradients = self.component_shapes(means, log_scales)
        weights = softmax(np.r_[parameters[6*k:], 0.0])
        prediction = weights @ components
        return prediction, components, gradients, weights


def shape_objective(parameters, projector, reference, k):
    predicted, components, gradients, weights = projector.mixture(parameters, k)
    # The shared floor is numerical and keeps the Hellinger derivative finite at zeros.
    root = np.sqrt(predicted + 1e-15)
    difference = root - np.sqrt(reference + 1e-15)
    output_gradient = difference / (2 * root)
    component_gradient = weights[:, None] * np.einsum("c,kcp->kp", output_gradient, gradients)
    weight_gradient = weights[:-1] * ((components[:-1] - predicted) @ output_gradient)
    gradient = np.r_[component_gradient[:, :3].ravel(), component_gradient[:, 3:].ravel(), weight_gradient]
    return 0.5 * float(difference @ difference), gradient


def fit_mixture(projector, reference, k, restarts, maxiter, seed, min_scales, *, trace_restarts=False):
    reference = np.asarray(reference, dtype=np.float64)
    min_scales = np.asarray(min_scales, dtype=np.float64)
    if (reference.shape != (len(projector.lower),) or not np.isfinite(reference).all()
            or np.any(reference < 0) or not np.isclose(reference.sum(), 1.0)):
        raise ValueError("The fit requires normalized nonnegative section fractions")
    if (min(k, restarts, maxiter) < 1 or min_scales.shape != (3,)
            or not np.isfinite(min_scales).all() or np.any(min_scales <= 0)):
        raise ValueError("Invalid mixture fitting settings")
    low, high = projector.lower.min(axis=0), projector.upper.max(axis=0)
    if np.any(min_scales > projector.extent):
        raise ValueError("Minimum scale exceeds the readout extent")
    low_scale, high_scale = np.log(min_scales), np.log(projector.extent)
    limits = list(zip(np.tile(low, k), np.tile(high, k)))
    limits += list(zip(np.tile(low_scale, k), np.tile(high_scale, k)))
    limits += [(-12.0, 12.0)] * (k - 1)
    rng = np.random.default_rng(seed)
    centroid = reference @ projector.centers
    spread = np.sqrt(reference @ (projector.centers - centroid)**2)
    starts = []
    start_hashes = []
    for restart in range(restarts):
        if k == 1 and restart == 0:
            means = centroid[None]
        else:
            selected = rng.choice(len(reference), size=k, replace=np.count_nonzero(reference) < k, p=reference)
            means = projector.centers[selected] + rng.normal(0, min_scales / 4, size=(k, 3))
        width = np.clip(spread / np.sqrt(k) * (1 + 0.4 * (restart % 2)), min_scales, projector.extent)
        initial = np.r_[np.clip(means, low, high).ravel(), np.tile(np.log(width), k), np.zeros(k - 1)]
        if trace_restarts:
            start_hashes.append(sha256(initial.tobytes()).hexdigest())
        starts.append(minimize(
            shape_objective, initial, args=(projector, reference, k), jac=True,
            method="L-BFGS-B", bounds=limits, options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 1e-6},
        ))
    finite = [result for result in starts if np.isfinite(result.fun)]
    if not finite:
        raise RuntimeError("No restart produced a finite objective")
    best = min(finite, key=lambda result: result.fun)
    widths = best.x[3*k:6*k].reshape(k, 3)
    trace = []
    if trace_restarts:
        for start_hash, result in zip(start_hashes, starts, strict=True):
            valid = np.isfinite(result.fun) and np.isfinite(result.x).all()
            tv = float(0.5 * np.abs(projector.mixture(result.x, k)[0] - reference).sum()) if valid else None
            trace.append({
                "initial_parameters_sha256": start_hash,
                "selected": result is best, "success": bool(result.success),
                "status": int(result.status), "message": str(result.message), "iterations": int(result.nit),
                "loss": float(result.fun) if valid else None, "tv": tv,
                "parameters": result.x.tolist() if valid else None,
            })
    return MixtureFit(
        projector.mixture(best.x, k)[0], float(best.fun), bool(best.success),
        sum(bool(result.success) for result in starts),
        np.isclose(widths, low_scale, atol=1e-5).sum(axis=0),
        np.isclose(widths, high_scale, atol=1e-5).sum(axis=0), int(best.status), int(best.nit), tuple(trace),
    )
