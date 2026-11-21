"""Compare full-section and active-plate projections on the frozen proton split."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import partial
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.stats import wasserstein_distance

from analyze_gmm_packet_decoder import DIRECTORY, OBSERVABLES, distances, named, section_statistics
from conditional_gmm_baseline import digest, sample_packet_fractions
from evaluate_gaussian_projection import fit_seed, read_energy_rows
from evaluate_gmm_packet_decoder import VALIDATION_ROWS, bank_probabilities, reference_rows
from pipeline.gaussian_projection import SectionProjector, fit_mixture, section_bounds
from pipeline.layered_projection import LayeredSectionProjector, SamplingStack
from pipeline.projection_geometry import legacy_grid_to_modules, load_projection_geometry


MODES = ("full_section", "active_length", "active_mass_depth")
NEIGHBOURS = (OBSERVABLES.index("neighbours_pct"), OBSERVABLES.index("neighbours_ge_1pct_pct"))


def refit_row(projector, settings, row):
    index, response = row
    k = settings["components"]
    total = response.sum()
    if total <= 0:
        return {"row": int(index), "parameters": [0.0] * (7 * k - 1), "tv": None, "loss": None,
                "status": -1, "success": False, "iterations": 0, "message": "No observed signal", "start": None}
    fit = fit_mixture(projector, response / total, k, settings["restarts"], settings["maxiter"],
                      fit_seed(settings["seed"], int(index), k), settings["min_scales"], trace_restarts=True)
    (start, trace), = [(i, trace) for i, trace in enumerate(fit.restart_trace, 1) if trace["selected"]]
    return {"row": int(index), "start": start, "parameters": trace["parameters"], "tv": trace["tv"],
            "loss": trace["loss"], "status": trace["status"], "success": trace["success"],
            "iterations": trace["iterations"], "message": trace["message"],
            "initial_parameters_sha256": trace["initial_parameters_sha256"]}


def simulated_statistics(bank, projector, calibration, contiguous):
    _, components, weights = bank_probabilities(bank, projector)
    rng = np.random.default_rng(calibration["seed"] + 2)
    stats = []
    for p, w, e in zip(components, weights, bank["energy"], strict=True):
        fraction = sample_packet_fractions(p, w, calibration["selected_packets_per_component"], rng,
                                           replicas=calibration["replicas_per_code"])
        response = fraction * e[None, :, None]
        np.testing.assert_allclose(response.sum(axis=-1), np.broadcast_to(e, response.shape[:2]))
        stats.append(section_statistics(response, contiguous).transpose(1, 0, 2))
    return np.asarray(stats)


def neighbours_w1(reference, response):
    return np.mean([
        [wasserstein_distance(a[:, m], b[..., m].ravel()) for m in NEIGHBOURS]
        for a, b in zip(reference, response, strict=True)
    ], axis=0)


def contrast_intervals(reference, statistics, calibration):
    rng = np.random.default_rng(calibration["seed"] + 9)
    group = np.arange(len(reference))[:, None]
    shape = statistics[0].shape[:2]
    samples = []
    for _ in range(calibration["bootstrap_resamples"]):
        bank = group, rng.integers(shape[1], size=shape)
        ref = reference[group, rng.integers(reference.shape[1], size=reference.shape[:2])]
        full, length, mass = [neighbours_w1(ref, method[bank]) for method in statistics]
        samples.append([mass - full, mass - length])
    values = np.quantile(samples, (0.025, 0.975), axis=0).transpose(1, 2, 0)
    return {key: {OBSERVABLES[m]: v.tolist() for m, v in zip(NEIGHBOURS, interval, strict=True)}
            for key, interval in zip(("mass_minus_full", "mass_minus_length"), values, strict=True)}


def evaluate(source, output, workers):
    calibration_path = DIRECTORY / "stochastic_decoder.json"
    calibration = json.loads(calibration_path.read_text())
    receipt_path = DIRECTORY / "results.json"
    receipt = json.loads(receipt_path.read_text())
    bank_path = DIRECTORY / "bank.npz"
    if digest(bank_path) != calibration["bank_sha256"] or digest(receipt_path) != calibration["baseline_receipt_sha256"]:
        raise ValueError("Unexpected historical bank or receipt")
    with np.load(bank_path, allow_pickle=False) as data:
        bank = {name: data[name] for name in ("rows", "momenta", "energy", "parameters", "fit_tv", "optimizer_status")}
    bank["components"] = receipt["settings"]["components"]
    geometry = load_projection_geometry()
    indices, reference, _ = reference_rows(source, bank, receipt, geometry)
    np.testing.assert_array_equal(indices[:, VALIDATION_ROWS:], [r["holdout"] for r in calibration["split"]])
    with np.load(source, allow_pickle=False) as data:
        n = len(data["ParticlePDG"])
    flat_rows = bank["rows"].ravel()
    raw = legacy_grid_to_modules(read_energy_rows(source, flat_rows, n), geometry)[:, geometry.layer_mask]
    np.testing.assert_allclose(raw.sum(axis=-1), bank["energy"].ravel())
    jobs = list(zip(flat_rows, raw, strict=True))
    stack = SamplingStack()
    projectors = (SectionProjector(*section_bounds(geometry), "integrated"),
                  LayeredSectionProjector(geometry, mass_depth=False, stack=stack),
                  LayeredSectionProjector(geometry, mass_depth=True, stack=stack))
    module, depth = np.nonzero(geometry.layer_mask)
    contiguous = (module[:-1] == module[1:]) & (np.diff(depth) == 1)
    reference_stats = section_statistics(reference[:, VALIDATION_ROWS:], contiguous)
    all_statistics, summaries, replay_fits = [], {}, {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for mode, projector in zip(MODES, projectors, strict=True):
            candidate = bank.copy()
            if mode != "full_section":
                print(f"Refitting {mode}: {len(jobs)} rows", flush=True)
                fits = list(pool.map(partial(refit_row, projector, receipt["settings"]), jobs, chunksize=2))
                replay_fits[mode] = fits
                candidate["parameters"] = np.asarray([fit["parameters"] for fit in fits]).reshape(bank["parameters"].shape)
                candidate["fit_tv"] = np.asarray([fit["tv"] or 0.0 for fit in fits]).reshape(bank["fit_tv"].shape)
                candidate["optimizer_status"] = np.asarray([fit["status"] for fit in fits]).reshape(bank["optimizer_status"].shape)
            stats = simulated_statistics(candidate, projector, calibration, contiguous)
            w1 = distances(reference_stats, stats)
            summaries[mode] = {"fit_tv_mean": float(candidate["fit_tv"].mean()),
                               "successful_fits": int((candidate["optimizer_status"] == 0).sum()),
                               "w1": named(w1.mean(axis=0)),
                               "w1_by_direction": [named(group) for group in w1],
                               "mean": named(stats.mean(axis=(0, 1, 2)))}
            all_statistics.append(stats)
    archived = json.loads((DIRECTORY / "decoder_analysis.json").read_text())
    for metric, value in summaries["full_section"]["w1"].items():
        np.testing.assert_allclose(value, archived["w1_macro"]["stochastic"][metric], atol=1e-12)
    result = {"source_sha256": calibration["source_sha256"], "bank_sha256": calibration["bank_sha256"],
              "geometry_sha256": geometry.sha256, "calibration_sha256": digest(calibration_path),
              "stack": asdict(stack), "order": "absorber_then_scintillator", "settings": receipt["settings"],
              "n_bank": len(flat_rows), "n_holdout": int(reference_stats.shape[0] * reference_stats.shape[1]),
              "packets_per_component": calibration["selected_packets_per_component"],
              "fit_telemetry": replay_fits,
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
              "fit_code_sha256": digest(Path(__file__).with_name("pipeline") / "gaussian_projection.py"),
              "decoder_code_sha256": digest(Path(__file__).with_name("conditional_gmm_baseline.py")),
              "replicas_per_code": calibration["replicas_per_code"], "results": summaries,
              "reference_mean": named(reference_stats.mean(axis=(0, 1))),
              "paired_ci95_neighbours": contrast_intervals(reference_stats, all_statistics, calibration),
              "evaluator_sha256": digest(Path(__file__)),
              "projector_sha256": digest(Path(__file__).with_name("pipeline") / "layered_projection.py")}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Exact proton archive NPZ")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", default=4, type=int)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.output.exists() and not args.output.is_file():
        parser.error("--output must name a file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evaluate(args.input, args.output, args.workers)


if __name__ == "__main__":
    main()
