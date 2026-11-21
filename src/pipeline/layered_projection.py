"""Integrate section response on the scintillator plates of a sampling stack."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.special import ndtr

from pipeline.gaussian_projection import SectionProjector, section_bounds


@dataclass(frozen=True)
class SamplingStack:
    """Regular Pb-then-scintillator periods, with thicknesses in cm."""

    absorber_cm: float = 1.6
    scintillator_cm: float = 0.4
    absorber_density: float = 11.35
    scintillator_density: float = 1.06
    plates_per_section: int = 6

    def __post_init__(self):
        values = np.array([self.absorber_cm, self.scintillator_cm,
                           self.absorber_density, self.scintillator_density], dtype=float)
        if not (np.isfinite(values).all() and np.all(values > 0)):
            raise ValueError("Plate thicknesses and densities must be finite and positive")
        if not positive_integer(self.plates_per_section):
            raise ValueError("Plates per section must be a positive integer")

    def active_intervals(self, sections, *, mass_depth):
        if not positive_integer(sections):
            raise ValueError("Sections must be a positive integer")
        lengths = np.array([self.absorber_cm, self.scintillator_cm], dtype=float)
        with np.errstate(over="ignore", under="ignore"):
            if mass_depth:
                lengths *= [self.absorber_density, self.scintillator_density]
            pitch = lengths.sum()
            section_pitch = pitch * self.plates_per_section
        if not np.isfinite(section_pitch) or np.any(lengths <= 0):
            raise ValueError("Effective plate thicknesses and pitch must be finite and positive")
        period = np.arange(sections * self.plates_per_section).reshape(sections, self.plates_per_section)
        upper = (period + 1) / self.plates_per_section
        lower = upper - lengths[1] / section_pitch
        if np.any(lower >= upper):
            raise ValueError("Active plate width is below the depth coordinate precision")
        return lower, upper


def positive_integer(value):
    return isinstance(value, Integral) and not isinstance(value, bool) and value > 0


def normal_intervals(lower, upper, mean, scale):
    lo, hi = (lower - mean) / scale, (upper - mean) / scale
    mass = np.where(lo >= 0, ndtr(-lo) - ndtr(-hi), ndtr(hi) - ndtr(lo))
    phi_lo = np.exp(-0.5 * lo**2) / np.sqrt(2 * np.pi)
    phi_hi = np.exp(-0.5 * hi**2) / np.sqrt(2 * np.pi)
    return mass, (phi_lo - phi_hi) / scale, lo * phi_lo - hi * phi_hi


def safe_ratio(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b > 0)


class LayeredSectionProjector(SectionProjector):
    """Use active subintervals in depth while keeping one output per readout section."""

    def __init__(self, geometry, *, mass_depth, stack=SamplingStack()):
        super().__init__(*section_bounds(geometry), "integrated")
        self.module, self.depth = np.nonzero(geometry.layer_mask)
        _, starts = np.unique(self.module, return_index=True)
        self.xy_lower, self.xy_upper = self.lower[starts, :2], self.upper[starts, :2]
        self.active_lower, self.active_upper = stack.active_intervals(geometry.max_layers, mass_depth=mass_depth)

    def component_shapes(self, means, log_scales):
        scale = np.exp(log_scales)
        xy_mass, xy_location, xy_width = normal_intervals(
            self.xy_lower, self.xy_upper, means[:, None, :2], scale[:, None, :2],
        )
        depth_values = normal_intervals(
            self.active_lower, self.active_upper, means[:, 2, None, None], scale[:, 2, None, None],
        )
        z_mass, z_location, z_width = (x.sum(axis=-1) for x in depth_values)
        mass = xy_mass.prod(axis=-1)[:, self.module] * z_mass[:, self.depth]
        total = mass.sum(axis=-1, keepdims=True)
        if np.any(total <= 0):
            raise ValueError("A component assigns no numerically observable mass to scintillator")
        shape = mass / total
        gradient = np.concatenate((
            safe_ratio(xy_location, xy_mass)[:, self.module],
            safe_ratio(z_location, z_mass)[:, self.depth, None],
            safe_ratio(xy_width, xy_mass)[:, self.module],
            safe_ratio(z_width, z_mass)[:, self.depth, None],
        ), axis=-1)
        weighted_gradient = shape[..., None] * gradient
        return shape, weighted_gradient - shape[..., None] * weighted_gradient.sum(axis=1, keepdims=True)
