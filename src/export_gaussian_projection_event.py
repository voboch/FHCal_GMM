"""Replay archived integrated fits and export section fractions for illustration."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from evaluate_gaussian_projection import assess_shape, fit_seed, read_energy_rows
from pipeline.gaussian_projection import SectionProjector, fit_mixture, section_bounds
from pipeline.projection_geometry import legacy_grid_to_modules, load_projection_geometry


PILOT = Path(__file__).resolve().parent / "gaussian_projection"
ARCHIVE = "width_sensitivity.json"
K = 4
GALLERY_QUANTILES = (0.1, 0.5, 0.9)


def select_examples(archive, quantiles, occupancy=None):
    rows = [
        row for row in archive["event_metrics"]
        if row["projection"] == "integrated" and row["k"] == K
    ]
    keys = [row["event_index"] for row in rows]
    if not rows or len(set(keys)) != len(rows) or len(rows) != archive["pooled"]["n_events"]:
        raise ValueError("Expected one archived integrated K=4 fit per row")
    values = [row["tv"] for row in rows]
    targets = np.quantile(values, quantiles, method="linear")
    selected = [
        (quantile, float(target), min(rows, key=lambda row: (abs(row["tv"] - target), row["event_index"])))
        for quantile, target in zip(quantiles, targets, strict=True)
    ]
    if occupancy is not None:
        busiest = min(rows, key=lambda row: (-occupancy[row["event_index"]], row["event_index"]))
        selected.append((None, None, busiest))
    if len({row["event_index"] for _, _, row in selected}) != len(selected):
        raise ValueError("Gallery criteria must select distinct example rows")
    return selected


def capture(input_path, *, gallery=False, max_occupancy_example=False, pilot_dir=PILOT):
    evidence = json.loads((pilot_dir / "evidence.json").read_text())
    raw = (pilot_dir / ARCHIVE).read_bytes()
    archive_hash = sha256(raw).hexdigest()
    if archive_hash != evidence["exports"][ARCHIVE]:
        raise ValueError("Unexpected archived width sensitivity results")
    archive = json.loads(raw)
    quantiles = GALLERY_QUANTILES if gallery else (0.5,)
    settings = archive["settings"]
    with input_path.open("rb") as stream:
        h = sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    if h.hexdigest() != archive["source_sha256"]:
        raise ValueError("NPZ checksum does not match the archive")
    geometry = load_projection_geometry()
    if geometry.sha256 != evidence["geometry_snapshot_sha256"]:
        raise ValueError("Geometry snapshot does not match the archive")
    if max_occupancy_example:
        indices = archive["reference"]["selected_event_indices"]
    else:
        selection = select_examples(archive, quantiles)
        indices = [row["event_index"] for _, _, row in selection]
    raw_events = read_energy_rows(input_path, np.array(indices), archive["selection"]["archive_rows"])
    physical_events = legacy_grid_to_modules(raw_events, geometry)
    by_index = dict(zip(indices, physical_events, strict=True))
    occupancy = None
    if max_occupancy_example:
        counts = np.count_nonzero(physical_events[:, geometry.layer_mask] > 0, axis=1)
        occupancy = {i: int(count) for i, count in zip(indices, counts, strict=True)}
        selection = select_examples(archive, quantiles[:-1], occupancy)
    projector = SectionProjector(*section_bounds(geometry), "integrated")
    examples = [
        {
            "quantile": quantile, "target_tv": target,
            **replay_fit(by_index[row["event_index"]], geometry.layer_mask, projector, settings, row),
        }
        for quantile, target, row in selection
    ]
    provenance = {
        "archive": ARCHIVE, "archive_sha256": archive_hash, "source_sha256": h.hexdigest(),
        "geometry_sha256": geometry.sha256, "scope": archive["scope"],
    }
    n = archive["pooled"]["n_events"]
    if gallery:
        rule = "closest_to_linear_quantile_tv"
        if occupancy is not None:
            rule += "; last_example_max_positive_sections"
            for event in examples:
                event["positive_sections"] = occupancy[event["event_index"]]
        return {
            **provenance, "selection": rule + "; ties_by_ascending_npz_row",
            "n_rows": n, "projection": "integrated", "k": K, "min_scales": settings["min_scales"],
            "examples": examples,
        }
    example, = examples
    return {
        **provenance,
        "selection": "closest_to_median_tv; ties_by_ascending_npz_row",
        "event_index": example["event_index"], "seed": example["seed"],
        "n_rows": n, "median_tv": example["target_tv"],
        "projection": "integrated", "k": K, "min_scales": settings["min_scales"],
        **{key: example[key] for key in ("converged", "metrics", "reference_fractions", "fit_fractions")},
    }


def replay_fit(physical, mask, projector, settings, selected):
    row_index = selected["event_index"]
    total = float(physical.sum())
    if total <= 0:
        raise ValueError("An empty event has no section fractions")
    reference = physical[mask] / total
    fit = fit_mixture(
        projector, reference, K, settings["restarts"], settings["maxiter"],
        fit_seed(selected["seed"], row_index, K), settings["min_scales"],
    )
    module_index, depth_index = np.nonzero(mask)
    metrics = assess_shape(reference, fit.prediction, module_index, depth_index, projector.centers[:, :2])
    for key in metrics:
        np.testing.assert_allclose(metrics[key], selected[key], rtol=1e-7, atol=1e-10, err_msg=f"{row_index}: {key}")
    if fit.converged != selected["converged"] or fit.status != selected["optimizer_status"]:
        raise ValueError(f"Optimizer result differs from the archived fit for row {row_index}")
    return {
        "event_index": row_index, "seed": selected["seed"],
        "converged": bool(fit.converged), "metrics": metrics,
        "reference_fractions": reference.tolist(), "fit_fractions": fit.prediction.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Unpacked archived NPZ")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gallery", action="store_true", help="Export the 10th, 50th and 90th TV percentile examples")
    parser.add_argument("--max-occupancy-example", action="store_true",
                        help="Replace the 90th percentile gallery example with the most occupied row")
    parser.add_argument("--pilot-dir", type=Path, default=PILOT, help="Directory of archived results")
    args = parser.parse_args()
    if args.max_occupancy_example and not args.gallery:
        parser.error("--max-occupancy-example requires --gallery")
    export = capture(args.input, gallery=args.gallery, max_occupancy_example=args.max_occupancy_example,
                     pilot_dir=args.pilot_dir)
    args.output.write_text(json.dumps(export, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for event in export["examples"] if args.gallery else [export]:
        print(f"NPZ row {event['event_index']}: TV={event['metrics']['tv']:.6f}")


if __name__ == "__main__":
    main()
