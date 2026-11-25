"""Extract physical FHCal sections, reserving fresh reference rows per direction."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from conditional_gmm_baseline import digest
from evaluate_gaussian_projection import read_energy_rows
from evaluate_single_particle_tails import inputs, N
from pipeline.projection_geometry import (
    legacy_grid_to_modules,
    load_projection_geometry,
)


def prepare(species, archive, output, reports):
    prior = Path("reports/single_particle") / f"{species}_split.json"
    manifest = json.loads(prior.read_text())
    geometry = load_projection_geometry()
    directions, historical, expected = inputs(species)
    assert digest(archive) == expected == manifest["archive_sha256"]
    assert geometry.sha256 == manifest["geometry_sha256"]
    with np.load(archive) as source:
        pdg, momentum, vertex, totals = [
            source[key]
            for key in (
                "ParticlePDG",
                "ParticleMomentum",
                "ParticlePoint",
                "TotalEnergySci",
            )
        ]
    rng = np.random.default_rng(241123)
    selected, arrays = [], []
    excluded = set(historical)
    for group in manifest["split"]:
        for ids in group.values():
            excluded.update(ids)
    for direction, group in zip(directions, manifest["split"], strict=True):
        same = np.flatnonzero(
            (pdg == (-13 if species == "muon" else 2212))
            & (momentum == direction).all(axis=1)
        )
        old = np.intersect1d(same, sorted(excluded))
        available = np.setdiff1d(same, old)
        candidates = rng.choice(available, min(N + 1536, len(available)), replace=False)
        ids = np.r_[old, candidates]
        grids = read_energy_rows(archive, ids, len(pdg))
        sections = legacy_grid_to_modules(grids, geometry)[:, geometry.layer_mask]
        del grids
        np.testing.assert_allclose(
            sections.sum(axis=1), totals[ids], rtol=1e-5, atol=1e-6
        )
        seen, keep = set(), []
        for i, row in enumerate(ids):
            key = hashlib.blake2b(
                vertex[row].tobytes() + sections[i].tobytes(), digest_size=20
            ).digest()
            if i >= len(old) and key not in seen:
                keep.append(i)
            seen.add(key)
            if len(keep) == N:
                break
        if len(keep) != N:
            raise ValueError("Too few fresh unique records")
        selected.append(ids[keep].tolist())
        for role in ("fit", "threshold"):
            if not np.array_equal(old[np.searchsorted(old, group[role])], group[role]):
                raise ValueError(f"Unmatched {role} indices in direction {direction}")
        arrays.append(
            np.stack(
                [
                    sections[np.searchsorted(old, group["fit"])],
                    sections[np.searchsorted(old, group["threshold"])],
                    sections[keep],
                ]
            ).astype(np.float32)
        )
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f"{species}_sections.npz", sections=np.stack(arrays))
    (reports / f"{species}_reference.json").write_text(
        json.dumps(
            dict(
                archive_sha256=expected,
                prior_split_sha256=digest(prior),
                geometry_sha256=geometry.sha256,
                directions=directions.tolist(),
                new_test=selected,
                selection_seed=241123,
            )
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("species", choices=["muon", "proton"])
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    prepare(
        args.species,
        args.archive,
        Path("outputs/conditional_shapes"),
        Path("reports/conditional_shapes"),
    )
