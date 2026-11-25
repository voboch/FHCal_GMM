"""Score fixed generated module counts on previously unused archive rows."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

import numpy as np

from conditional_gmm_baseline import digest
from evaluate_gaussian_projection import read_energy_rows
from evaluate_single_particle_tails import inputs, N
from pipeline.projection_geometry import (
    legacy_grid_to_modules,
    load_projection_geometry,
)

REPORT = Path("reports/generated_total_replication")
SEED = 261124


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def score_species(species, archive, output):
    directions, exclusions, expected = inputs(species)
    geometry = load_projection_geometry()
    split_path = Path("reports/single_particle") / f"{species}_split.json"
    test_path = Path("reports/conditional_shapes") / f"{species}_reference.json"
    forecast_path = Path("reports/generated_total") / f"{species}_modules.csv"
    prior = json.loads(split_path.read_text())
    fixed = json.loads(test_path.read_text())
    with forecast_path.open() as stream:
        forecasts = list(csv.DictReader(stream))
    assert (
        digest(archive)
        == prior["archive_sha256"]
        == fixed["archive_sha256"]
        == expected
    )
    assert prior["geometry_sha256"] == fixed["geometry_sha256"] == geometry.sha256
    assert digest(split_path) == fixed["prior_split_sha256"]
    for group in prior["split"]:
        for indices in group.values():
            exclusions.update(indices)
    for indices in fixed["new_test"]:
        exclusions.update(indices)
    with np.load(archive) as data:
        pdg, momenta, vertices, totals = (
            data[key]
            for key in (
                "ParticlePDG",
                "ParticleMomentum",
                "ParticlePoint",
                "TotalEnergySci",
            )
        )
    module_idx = np.nonzero(geometry.layer_mask)[0]
    starts = np.flatnonzero(np.r_[True, np.diff(module_idx) != 0])
    positions = {module.module_id: i for i, module in enumerate(geometry.modules)}

    def maps(indices):
        grids = read_energy_rows(archive, indices, len(pdg))
        values = legacy_grid_to_modules(grids, geometry)[:, geometry.layer_mask]
        np.testing.assert_allclose(
            values.sum(axis=1), totals[indices], rtol=1e-5, atol=1e-6
        )
        return values

    def key(row, values):
        return hashlib.blake2b(
            vertices[row].tobytes() + values.tobytes(), digest_size=20
        ).digest()

    manifest, module_rows, group_rows = [], [], []
    species_code = -13 if species == "muon" else 2212
    for g, momentum in enumerate(directions, start=1):
        group = np.flatnonzero(
            (pdg == species_code) & (momenta == momentum).all(axis=1)
        )
        previous = np.intersect1d(group, list(exclusions))
        available = np.setdiff1d(group, previous)
        order = np.random.default_rng([SEED, abs(species_code), g]).permutation(
            available
        )
        past = maps(previous)
        seen = {key(row, values) for row, values in zip(previous, past, strict=True)}
        old_test = np.searchsorted(previous, fixed["new_test"][g - 1])
        np.testing.assert_array_equal(previous[old_test], fixed["new_test"][g - 1])
        prior_sums = np.add.reduceat(
            past[old_test].astype(np.float32), starts, axis=1, dtype=np.float64
        )
        del past
        selected, selected_maps = [], []
        for start in range(0, len(order), N + 512):
            block = order[start : start + N + 512]
            for row, values in zip(block, maps(block), strict=True):
                signature = key(row, values)
                if signature not in seen:
                    seen.add(signature)
                    selected.append(int(row))
                    selected_maps.append(values.astype(np.float32))
                if len(selected) == N:
                    break
            if len(selected) == N:
                break
        if not selected:
            raise ValueError(f"No unused distinct records for {species} direction {g}")
        observed = np.add.reduceat(
            np.stack(selected_maps), starts, axis=1, dtype=np.float64
        )
        n = len(selected)
        manifest.append(
            dict(
                direction=g,
                momentum=momentum.tolist(),
                total_indices=len(group),
                excluded_indices=len(previous),
                candidate_indices=len(available),
                selected=selected,
            )
        )
        by_method = {}
        for record in forecasts:
            if int(record["direction"]) != g:
                continue
            module, threshold = int(record["module"]), float(record["q95"])
            idx = positions[module]
            assert int((prior_sums[:, idx] > threshold).sum()) == int(
                record["ref_above"]
            )
            old_n = int(record["n"])
            if old_n != N or not 0 <= int(record["model_above"]) <= old_n:
                raise ValueError(
                    f"Invalid generated counts for {species}, direction {g}"
                )
            new_above = int((observed[:, idx] > threshold).sum())
            old_fraction = int(record["ref_above"]) / old_n
            model_fraction = int(record["model_above"]) / old_n
            new_fraction = new_above / n
            method, seed = record["method"], int(record["seed"])
            errors = [
                100 * abs(a - b)
                for a, b in [
                    (model_fraction, new_fraction),
                    (model_fraction, old_fraction),
                    (old_fraction, new_fraction),
                ]
            ]
            by_method.setdefault((method, seed), []).append(errors)
            module_rows.append(
                dict(
                    direction=g,
                    module=module,
                    method=method,
                    seed=seed,
                    q95=threshold,
                    original_n=old_n,
                    original_above=record["ref_above"],
                    model_above=record["model_above"],
                    new_n=n,
                    new_above=new_above,
                )
            )
        for (method, seed), errors in by_method.items():
            model_new, model_old, controls = np.mean(errors, axis=0)
            group_rows.append(
                dict(
                    direction=g,
                    n=n,
                    eligible=len(errors),
                    method=method,
                    seed=seed,
                    model_new_pp=model_new,
                    model_old_pp=model_old,
                    controls_pp=controls,
                )
            )
    write_rows(output / f"{species}_modules.csv", module_rows)
    write_rows(output / f"{species}_directions.csv", group_rows)
    receipt = dict(
        archive_sha256=expected,
        selection_seed=SEED,
        geometry_sha256=geometry.sha256,
        split_sha256=digest(split_path),
        previous_test_sha256=digest(test_path),
        forecasts_sha256=digest(forecast_path),
        groups=manifest,
    )
    (output / f"{species}_reference.json").write_text(json.dumps(receipt) + "\n")
    full = [r for r in group_rows if r["n"] == N]
    summary = dict(
        full_directions=sorted({r["direction"] for r in full}),
        records_by_direction=[len(m["selected"]) for m in manifest],
        control_to_control_mae_pp=mean(r["controls_pp"] for r in full),
    )
    for method in ("empirical", "direct", "gmm"):
        rows = [r for r in full if r["method"] == method]
        summary[method] = {
            field: mean(r[field] for r in rows)
            for field in ("model_old_pp", "model_new_pp")
        }
        summary[method]["model_new_by_seed_pp"] = {
            str(seed): mean(r["model_new_pp"] for r in rows if r["seed"] == seed)
            for seed in sorted({r["seed"] for r in rows})
        }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--muon", type=Path, required=True)
    parser.add_argument("--proton", type=Path, required=True)
    args = parser.parse_args()
    REPORT.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=REPORT) as staging:
        output = Path(staging)
        summary = {}
        for species in ("muon", "proton"):
            summary[species] = score_species(species, getattr(args, species), output)
            print(species, summary[species], flush=True)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        for artifact in sorted(output.iterdir()):
            artifact.replace(REPORT / artifact.name)
