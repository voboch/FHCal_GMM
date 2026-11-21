"""Archived detector geometry for the FHCal Gaussian projection pilot."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


GEOMETRY_PATH = Path(__file__).resolve().parents[1] / "gaussian_projection" / "geometry_muons_19.json"


@dataclass(frozen=True)
class ModuleGeometry:
    module_id: int
    x: float
    y: float
    z: float
    width: float
    height: float
    layers: int
    legacy_row: int
    legacy_col: int


@dataclass(frozen=True)
class ProjectionGeometry:
    modules: tuple
    max_layers: int
    source_commit: str
    source_symbol: str
    sha256: str

    @property
    def layer_mask(self):
        counts = np.array([module.layers for module in self.modules])
        return np.arange(self.max_layers) < counts[:, None]


def load_projection_geometry(path=GEOMETRY_PATH):
    payload = path.read_bytes()
    metadata = json.loads(payload)
    modules = tuple(ModuleGeometry(**row) for row in metadata["modules"])
    max_layers = metadata["max_layers"]
    identifiers = [module.module_id for module in modules]
    positions = [(module.legacy_row, module.legacy_col) for module in modules]
    if identifiers != sorted(set(identifiers)) or len(set(positions)) != len(modules):
        raise ValueError("Expected ordered unique module IDs and grid slots")
    if any(not (0 <= row < 7 and 0 <= col < 9) for row, col in positions):
        raise ValueError("Invalid legacy grid slot")
    if any(not (0 < m.layers <= max_layers and m.width > 0 and m.height > 0) for m in modules):
        raise ValueError("Invalid module section count or footprint")
    return ProjectionGeometry(
        modules, max_layers, metadata["source_commit"], metadata["source_symbol"], sha256(payload).hexdigest(),
    )


def legacy_grid_to_modules(energy, geometry):
    """Apply the frozen map and reject response outside its physical support."""
    if energy.ndim != 4 or energy.shape[1:] != (7, 9, geometry.max_layers):
        raise ValueError(f"Expected [events, 7, 9, {geometry.max_layers}]")
    rows, cols = zip(*((m.legacy_row, m.legacy_col) for m in geometry.modules))
    occupied = np.zeros((7, 9), dtype=bool)
    occupied[rows, cols] = True
    if np.any(energy[:, ~occupied, :] != 0):
        raise ValueError("Unmapped legacy slots contain response")
    physical = energy[:, rows, cols, :]
    if not np.isfinite(physical).all() or np.any(physical < 0) or np.any(physical[:, ~geometry.layer_mask] != 0):
        raise ValueError("Invalid response or response in an absent physical section")
    if not np.allclose(energy.sum(axis=(1, 2, 3)), physical.sum(axis=(1, 2)), rtol=1e-5, atol=1e-6):
        raise ValueError("Legacy-to-physical conversion failed the sum check")
    return physical
