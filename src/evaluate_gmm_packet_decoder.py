"""Calibrate and score the stochastic decoder on rows unused by the proton pilot."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

from conditional_gmm_baseline import METRICS, PILOT, digest, observables, sample_packet_fractions
from evaluate_gaussian_projection import read_energy_rows
from pipeline.gaussian_projection import SectionProjector, section_bounds
from pipeline.projection_geometry import legacy_grid_to_modules, load_projection_geometry


VALIDATION_ROWS, TEST_ROWS, REPLICAS, SEED, BOOTSTRAPS = 128, 256, 32, 14717, 1000
PACKET_GRID = (4, 8, 16, 32, 64, 128, 256, 512, 1024)
GUARDS = {"sections_ge_1pct": 1.0, "max_share_pp": 2.0, "depth_centroid": 0.10}


def reference_rows(source, bank, receipt, geometry):
    analysis = PILOT / "event_dependence.json"
    evidence = json.loads((PILOT / "evidence.json").read_text())
    if (digest(source) != receipt["source_sha256"] or geometry.sha256 != receipt["geometry_sha256"]
            or digest(analysis) != evidence["exports"][analysis.name]):
        raise ValueError("Input, geometry or pilot provenance changed")
    history = json.loads(analysis.read_text())
    excluded = ({row["event_index"] for row in history["events"]}
                | {i for group in history["direction_groups"] for i in group["selected_rows"]}
                | {i for group in receipt["groups"] for field in ("bank_rows", "holdout_rows") for i in group[field]})
    with np.load(source, allow_pickle=False) as data:
        pdg, momenta, vertices, archived_total = (data[name] for name in
                                                  ("ParticlePDG", "ParticleMomentum", "ParticlePoint", "TotalEnergySci"))
    rng = np.random.default_rng(SEED)
    eligible = np.setdiff1d(np.flatnonzero(pdg == 2212), sorted(excluded), assume_unique=True)
    indices = np.stack([
        rng.choice(eligible[np.all(momenta[eligible] == p, axis=1)], VALIDATION_ROWS + TEST_ROWS, replace=False)
        for p in bank["momenta"]
    ])
    bank_rows = bank["rows"].ravel()
    all_rows = np.r_[indices.ravel(), bank_rows]
    raw = read_energy_rows(source, all_rows, len(pdg))
    flat = legacy_grid_to_modules(raw, geometry)[:, geometry.layer_mask]
    np.testing.assert_allclose(flat.sum(axis=1), archived_total[all_rows], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(flat[len(indices.ravel()):].sum(axis=1), bank["energy"].ravel())
    records = np.column_stack((momenta[all_rows], vertices[all_rows], flat))
    if len(np.unique(records, axis=0)) != len(records):
        raise ValueError("Duplicate records in the new split or bank")
    response = flat[:indices.size].reshape(*indices.shape, flat.shape[-1])
    return indices, response, len(excluded)


def bank_probabilities(bank, projector):
    mixtures = [[projector.mixture(code, bank["components"]) for code in group] for group in bank["parameters"]]
    fractions = np.array([[entry[0] for entry in group] for group in mixtures])
    components = np.array([[entry[1] for entry in group] for group in mixtures])
    weights = np.array([[entry[3] for entry in group] for group in mixtures])
    return fractions, components, weights


def sampled_observables(components, weights, energy, depth, packets, seed):
    rng = np.random.default_rng(seed)
    groups = []
    for component, weight, total in zip(components, weights, energy, strict=True):
        generated = sample_packet_fractions(component, weight, packets, rng, replicas=REPLICAS) * total[None, :, None]
        np.testing.assert_allclose(generated.sum(axis=-1), np.broadcast_to(total, (REPLICAS, len(total))))
        groups.append(observables(generated, depth).transpose(1, 0, 2))
    return np.asarray(groups)


def scores(reference, synthetic):
    """W1 of each observable for a generated distribution stratified by direction."""
    return np.array([
        [wasserstein_distance(a, b) for a, b in zip(ref.T, sample.reshape(-1, len(METRICS)).T, strict=True)]
        for ref, sample in zip(reference, synthetic, strict=True)
    ])


def named(values):
    return {metric: value.tolist() for metric, value in zip(METRICS, values, strict=True)}


def choose_packets(reference, smooth, components, weights, energy, depth):
    baseline = scores(reference, smooth).mean(axis=0)
    indexed_guards = [(METRICS.index(metric), limit) for metric, limit in GUARDS.items()]
    candidates = []
    for packets in PACKET_GRID:
        candidate = sampled_observables(components, weights, energy, depth, packets, SEED + 1)
        metrics = scores(reference, candidate).mean(axis=0)
        violation = sum(max(0.0, (metrics[i] - baseline[i]) / limit - 1) for i, limit in indexed_guards)
        candidates.append({"packets": packets, "guard_violation": float(violation), "w1_macro": named(metrics)})
    best = min(candidates, key=lambda record: (record["guard_violation"], record["w1_macro"]["positive_sections"],
                                               record["packets"]))
    return best, candidates, named(baseline)


def paired_interval(reference, smooth, stochastic):
    rng = np.random.default_rng(SEED + 3)
    groups = np.arange(len(reference))[:, None]
    contrasts = []
    for _ in range(BOOTSTRAPS):
        bank_draw = groups, rng.integers(smooth.shape[1], size=smooth.shape[:2])
        ref = reference[groups, rng.integers(reference.shape[1], size=reference.shape[:2])]
        contrasts.append((scores(ref, stochastic[bank_draw]) - scores(ref, smooth[bank_draw])).mean(axis=0))
    return named(np.quantile(contrasts, (0.025, 0.975), axis=0).T)


def evaluate(source, bank_path, output):
    receipt_path = bank_path.with_name("results.json")
    receipt = json.loads(receipt_path.read_text())
    bank_sha = digest(bank_path)
    if bank_sha != receipt["bank_sha256"]:
        raise ValueError("Bank digest differs from the baseline receipt")
    with np.load(bank_path, allow_pickle=False) as data:
        bank = {name: data[name] for name in ("rows", "momenta", "energy", "parameters")}
    bank["components"] = receipt["settings"]["components"]
    geometry = load_projection_geometry()
    projector = SectionProjector(*section_bounds(geometry), "integrated")
    depth = projector.centers[:, -1]
    indices, reference, n_excluded = reference_rows(source, bank, receipt, geometry)
    obs = observables(reference, depth)
    validation, holdout = obs[:, :VALIDATION_ROWS], obs[:, VALIDATION_ROWS:]
    fractions, components, weights = bank_probabilities(bank, projector)
    energy = bank["energy"]
    smooth = observables(fractions * energy[..., None], depth)
    best, grid, validation_baseline = choose_packets(validation, smooth, components, weights, energy, depth)
    packets = best["packets"]
    print(f"Selected {packets} packets per component on validation", flush=True)
    stochastic = sampled_observables(components, weights, energy, depth, packets, SEED + 2)
    baseline, sampled = scores(holdout, smooth), scores(holdout, stochastic)
    result = {
        "scope": "held-out NPZ rows, fixed proton mixture bank and recorded directions",
        "bank_sha256": bank_sha, "baseline_receipt_sha256": digest(receipt_path),
        "source_sha256": receipt["source_sha256"], "geometry_sha256": geometry.sha256,
        "evaluator_sha256": digest(Path(__file__)),
        "decoder_sha256": digest(Path(__file__).with_name("conditional_gmm_baseline.py")),
        "seed": SEED, "replicas_per_code": REPLICAS, "bootstrap_resamples": BOOTSTRAPS,
        "excluded_rows": n_excluded,
        "split": [{"direction": i, "validation": rows[:VALIDATION_ROWS].tolist(),
                   "holdout": rows[VALIDATION_ROWS:].tolist()} for i, rows in enumerate(indices, 1)],
        "validation_guards": GUARDS,
        "selected_packets_per_component": packets, "selected_meets_guards": best["guard_violation"] == 0,
        "validation_smooth_w1": validation_baseline, "validation_grid": grid,
        "holdout_w1_macro": {"smooth": named(baseline.mean(axis=0)), "stochastic": named(sampled.mean(axis=0))},
        "holdout_w1_by_direction": [{"direction": i, "smooth": named(a), "stochastic": named(b)}
                                    for i, (a, b) in enumerate(zip(baseline, sampled, strict=True), 1)],
        "holdout_stochastic_minus_smooth_ci95": paired_interval(holdout, smooth, stochastic),
        "mean_observable": {"reference": named(holdout.mean(axis=(0, 1))), "smooth": named(smooth.mean(axis=(0, 1))),
                            "stochastic": named(stochastic.mean(axis=(0, 1, 2)))},
        "empty_reference": {"validation": int((validation[..., 0] == 0).sum()),
                            "holdout": int((holdout[..., 0] == 0).sum())},
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Exact proton archive NPZ")
    parser.add_argument("bank", type=Path, help="Prototype bank NPZ with results.json next to it")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    evaluate(args.input, args.bank, args.output)


if __name__ == "__main__":
    main()
