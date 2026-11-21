"""Relate archived FHCal fit errors to measured responses and recorded inputs."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from audit_gaussian_projection_optimizer import fit_rows, read_run
from evaluate_gaussian_projection import read_energy_rows
from pipeline.projection_geometry import legacy_grid_to_modules, load_projection_geometry


def rank_correlation(x, y):
    rx, ry = rankdata(x, axis=-1), rankdata(y, axis=-1)
    rx -= rx.mean(axis=-1, keepdims=True)
    ry -= ry.mean(axis=-1, keepdims=True)
    numerator = (rx * ry).sum(axis=-1)
    norm = np.sqrt((rx**2).sum(axis=-1) * (ry**2).sum(axis=-1))
    return np.divide(numerator, norm, out=np.full_like(numerator, np.nan), where=norm > 0)


def percentile_interval(values):
    return np.nanquantile(values, [0.025, 0.975]).tolist()


def check_run(reference, run, k, *, common_rows):
    for key in ("source_sha256", "geometry_provenance", "runtime"):
        if reference[key] != run[key]:
            raise ValueError(f"Fit run changed {key}")
    if run["settings"]["components"] != [k]:
        raise ValueError("Unexpected component count")
    excluded = {"components"} if common_rows else {"components", "events_per_repeat", "seeds"}
    for key in reference["settings"].keys() - excluded:
        if reference["settings"][key] != run["settings"][key]:
            raise ValueError(f"Fit run changed {key}")
    if common_rows and reference["reference"] != run["reference"]:
        raise ValueError("Fit run changed the rows")
    if reference["selection"]["pdg"] != run["selection"]["pdg"]:
        raise ValueError("Fit run changed the recorded species")


def analyze(npz_path, fit_paths, direction_paths):
    receipts = {}
    runs = {}
    for k, path in zip((4, 6, 8), fit_paths, strict=True):
        runs[k], receipts[path.name] = read_run(path)
        check_run(runs[4], runs[k], k, common_rows=True)
    reference = runs[4]
    with npz_path.open("rb") as stream:
        hash_input = sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hash_input.update(block)
    geometry = load_projection_geometry()
    if hash_input.hexdigest() != reference["source_sha256"] or geometry.sha256 != reference["geometry_provenance"]["sha256"]:
        raise ValueError("Unexpected NPZ or geometry version")

    indices = reference["reference"]["selected_event_indices"]
    with np.load(npz_path, allow_pickle=False) as data:
        vertex, momentum, total = (data[key] for key in ("ParticlePoint", "ParticleMomentum", "TotalEnergySci"))
    raw = read_energy_rows(npz_path, np.array(indices), reference["selection"]["archive_rows"])
    sections = legacy_grid_to_modules(raw, geometry)[:, geometry.layer_mask]
    sums = sections.sum(axis=1)
    np.testing.assert_allclose(sums, total[indices], rtol=1e-5, atol=1e-6)
    if np.any(sums <= 0):
        raise ValueError("Expected positive observed totals")
    fractions = sections / sums[:, None]
    depth = np.nonzero(geometry.layer_mask)[1] + 0.5
    depth_mean = fractions @ depth
    threshold = reference["settings"]["occupancy_fraction_threshold"]
    observables = {
        "total": sums,
        "positive_sections": (sections > 0).sum(axis=1),
        "sections_ge_1pct": (fractions >= threshold).sum(axis=1),
        "effective_sections": 1 / (fractions**2).sum(axis=1),
        "depth_mean": depth_mean,
        "depth_rms": np.sqrt((fractions * (depth - depth_mean[:, None])**2).sum(axis=1)),
        **{f"vertex_{axis}": vertex[indices, j] for j, axis in enumerate("xyz")},
    }
    by_k = {k: fit_rows(run) for k, run in runs.items()}
    tv = {f"tv_{k}": np.array([fits[i, "integrated"]["tv"] for i in indices]) for k, fits in by_k.items()}
    tv["gain_4_to_8"] = tv["tv_4"] - tv["tv_8"]
    rng = np.random.default_rng(19)
    draws = rng.integers(0, len(indices), size=(1000, len(indices)))
    correlations = {}
    for name, values in observables.items():
        correlations[name] = {
            metric: {"rho": float(rank_correlation(values, target)),
                     "interval": percentile_interval(rank_correlation(values[draws], target[draws]))}
            for metric, target in tv.items()
        }
    tails = {
        label: {"n": int(mask.sum()), **{metric: float(values[mask].mean()) for metric, values in tv.items()}}
        for label, mask in (("under_31", observables["positive_sections"] < 31),
                            ("at_least_31", observables["positive_sections"] >= 31))
    }
    events = [
        {"event_index": i, **{name: column[j].item() for name, column in (observables | tv).items()}}
        for j, i in enumerate(indices)
    ]

    direction_runs = [runs[6]]
    for path in direction_paths:
        run, receipts[path.name] = read_run(path)
        check_run(reference, run, 6, common_rows=False)
        direction_runs.append(run)
    vectors = np.unique(momentum, axis=0)
    direction_runs.sort(key=lambda run: run["selection"]["momentum"])
    if [run["selection"]["momentum"] for run in direction_runs] != vectors.tolist():
        raise ValueError("Expected one fit run for each exact recorded momentum")
    groups = []
    draws = np.random.default_rng(19).integers(0, 48, size=(1000, 48))
    for j, run in enumerate(direction_runs, 1):
        rows = [r for r in run["event_metrics"] if r["projection"] == "integrated" and r["seed"] == 19]
        row_ids = [row["event_index"] for row in rows]
        if len(set(row_ids)) != 48 or not np.all(momentum[row_ids] == np.asarray(run["selection"]["momentum"])):
            raise ValueError("Expected 48 distinct seed-19 rows from each momentum group")
        scores = np.array([row["tv"] for row in rows])
        groups.append({
            "group": j, "recorded_momentum": run["selection"]["momentum"],
            "archive_rows": run["selection"]["momentum_group_rows"], "selected_rows": row_ids,
            "tv_mean": float(scores.mean()), "tv_mean_interval": percentile_interval(scores[draws].mean(axis=1)),
            "median_total_signal": float(np.median(total[row_ids])),
            "successful_fits": sum(row["converged"] for row in rows),
        })
    return {
        "input_sha256": hash_input.hexdigest(), "geometry_sha256": geometry.sha256,
        "fit_exports_sha256": receipts, "bootstrap_draws": 1000, "section_fraction_threshold": threshold,
        "events": events, "correlations": correlations, "occupancy_slices": tails, "direction_groups": groups,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    for k in (4, 6, 8):
        parser.add_argument(f"--fit-{k}", type=Path, required=True)
    parser.add_argument("--direction-run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.input, (args.fit_4, args.fit_6, args.fit_8), args.direction_run)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
