"""Join matched iteration-budget replays with measured section occupancy."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from evaluate_gaussian_projection import METHODS, read_energy_rows
from pipeline.projection_geometry import legacy_grid_to_modules, load_projection_geometry


def read_run(path):
    raw = path.read_bytes()
    return json.loads(raw), sha256(raw).hexdigest()


def fit_rows(run):
    fits = {(row["event_index"], row["projection"]): row for row in run["event_metrics"]}
    expected = {(i, method) for i in run["reference"]["selected_event_indices"] for method in METHODS}
    if fits.keys() != expected or len(fits) != len(run["event_metrics"]):
        raise ValueError("Expected one fit per selected row and operator")
    return fits


def inspect_fits(run):
    fits = fit_rows(run)
    for row in fits.values():
        trace = row["restart_trace"]
        if len(trace) != run["settings"]["restarts"] or row["k"] != run["settings"]["components"][0]:
            raise ValueError("Unexpected restart count or component count")
        selected, = (start for start in trace if start["selected"])
        if any(selected[key] != row[key] for key in ("tv", "loss")):
            raise ValueError("Selected restart differs from the reported fit")
    return fits


def summarize_components(reference_run, reference_fits, comparison_paths):
    indices = reference_run["reference"]["selected_event_indices"]
    settings = reference_run["settings"]
    rng = np.random.default_rng(settings["seeds"][0])
    draws = rng.integers(0, len(indices), size=(settings["bootstrap_resamples"], len(indices)))
    runs = [(reference_run, reference_fits)]
    checksums = {}
    for path in comparison_paths:
        run, digest = read_run(path)
        for key in ("source_sha256", "selection", "geometry_provenance", "reference", "runtime"):
            if reference_run[key] != run[key]:
                raise ValueError(f"Component scan changed {key}")
        if any(settings[key] != run["settings"][key] for key in settings if key != "components"):
            raise ValueError("Component scan changed the fitting budget")
        runs.append((run, fit_rows(run)))
        checksums[path.name] = digest

    summaries = []
    for run, fits in runs:
        k, = run["settings"]["components"]
        for method in METHODS:
            rows = [fits[i, method] for i in indices]
            if any(row["k"] != k for row in rows):
                raise ValueError("Component count differs from the run settings")
            tv = np.array([row["tv"] for row in rows])
            delta = tv - np.array([reference_fits[i, method]["tv"] for i in indices])
            summaries.append({
                "k": k, "projection": method, "shape_coordinates": 7*k - 1,
                "tv_mean": float(tv.mean()),
                "loss_mean": float(np.mean([row["loss"] for row in rows])),
                "success": sum(row["converged"] for row in rows),
                "median_selected_iterations": float(np.median([row["optimizer_iterations"] for row in rows])),
                "paired_tv_mean": float(delta.mean()),
                "paired_tv_interval_95": np.quantile(delta[draws].mean(axis=1), [0.025, 0.975]).tolist(),
            })
    if len({row["k"] for row in summaries}) != len(runs):
        raise ValueError("Component scan requires distinct component counts")
    return {"exports": checksums, "summaries": summaries}


def audit(input_path, baseline_path, replay_path, comparison_paths=()):
    baseline, baseline_hash = read_run(baseline_path)
    replay, replay_hash = read_run(replay_path)
    for key in ("source_sha256", "selection", "geometry_provenance", "reference", "runtime"):
        if baseline[key] != replay[key]:
            raise ValueError(f"Replay changed {key}")
    settings = [dict(run["settings"]) for run in (baseline, replay)]
    limits = [s.pop("maxiter") for s in settings]
    if limits[0] >= limits[1] or settings[0] != settings[1] or len(settings[0]["components"]) != 1:
        raise ValueError("Replay must increase only the iteration limit for a single component count")

    with input_path.open("rb") as stream:
        data_hash = sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            data_hash.update(block)
    if data_hash.hexdigest() != baseline["source_sha256"]:
        raise ValueError("Input NPZ differs from the replay source")
    geometry = load_projection_geometry()
    if geometry.sha256 != baseline["geometry_provenance"]["sha256"]:
        raise ValueError("Geometry differs from the replay source")
    indices = baseline["reference"]["selected_event_indices"]
    raw = read_energy_rows(input_path, np.asarray(indices), baseline["selection"]["archive_rows"])
    sections = legacy_grid_to_modules(raw, geometry)[:, geometry.layer_mask]
    totals = sections.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError("The diagnostic requires nonempty responses")
    threshold = settings[0]["occupancy_fraction_threshold"]
    before, after = inspect_fits(baseline), inspect_fits(replay)
    rows = []
    for i, reference, total in zip(indices, sections, totals, strict=True):
        row = {
            "event_index": i,
            "positive_sections": int(np.count_nonzero(reference > 0)),
            "sections_at_fraction_threshold": int(np.count_nonzero(reference / total >= threshold)),
            "operators": {},
        }
        for method in METHODS:
            pair = (before[i, method], after[i, method])
            starts = [[r["initial_parameters_sha256"] for r in fit["restart_trace"]] for fit in pair]
            if starts[0] != starts[1]:
                raise ValueError(f"Replay changed the initial parameters for row {i}, {method}")
            row["operators"][method] = {key: [fit[key] for fit in pair] for key in ("tv", "loss", "converged")}
        rows.append(row)
    result = {
        "baseline_sha256": baseline_hash, "replay_sha256": replay_hash,
        "npz_sha256": data_hash.hexdigest(), "geometry_sha256": geometry.sha256,
        "k": settings[0]["components"][0], "iteration_limits": limits,
        "section_fraction_threshold": threshold, "n_sections": int(geometry.layer_mask.sum()),
        "rows": rows,
    }
    if comparison_paths:
        result["component_scan"] = summarize_components(replay, after, comparison_paths)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--compare-components", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input, args.baseline, args.replay, args.compare_components)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
