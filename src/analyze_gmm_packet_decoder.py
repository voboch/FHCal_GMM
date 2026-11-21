"""Compare section occupancy and longitudinal neighbours on the fixed decoder evaluation."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

from conditional_gmm_baseline import ACTIVE_SHARE, PILOT, digest, observables, sample_packet_fractions
from evaluate_gaussian_projection import read_energy_rows
from evaluate_gmm_packet_decoder import VALIDATION_ROWS, bank_probabilities, reference_rows, scores
from pipeline.gaussian_projection import SectionProjector, section_bounds
from pipeline.projection_geometry import legacy_grid_to_modules, load_projection_geometry


DIRECTORY = PILOT / "conditional_baseline"
OBSERVABLES = ("positive_sections", "sections_ge_1pct", "weak_sections", "weak_share_pct",
               "neighbours_pct", "neighbours_ge_1pct_pct")
METHODS = ("reference", "raw_bank", "smooth", "stochastic")
CONTRASTS = (("smooth_minus_raw_bank", "smooth", "raw_bank"),
             ("stochastic_minus_smooth", "stochastic", "smooth"))


def section_statistics(response, contiguous):
    total = response.sum(axis=-1, keepdims=True)
    share = np.divide(response, total, out=np.zeros_like(response), where=total > 0)
    positive = response > 0
    strong = share >= ACTIVE_SHARE
    weak = positive & ~strong

    def paired_share(mask):
        pairs = mask[..., :-1] & mask[..., 1:] & contiguous
        paired = np.zeros_like(mask)
        paired[..., :-1] |= pairs
        paired[..., 1:] |= pairs
        count = mask.sum(axis=-1)
        return np.divide(100 * paired.sum(axis=-1), count, out=np.zeros(count.shape), where=count > 0)

    return np.stack((positive.sum(axis=-1), strong.sum(axis=-1), weak.sum(axis=-1),
                     100 * np.where(weak, share, 0).sum(axis=-1), paired_share(positive),
                     paired_share(strong)), axis=-1)


def named(values):
    return dict(zip(OBSERVABLES, values.tolist(), strict=True))


def distances(reference, sample):
    return np.array([
        [wasserstein_distance(a, b) for a, b in zip(ref.T, model.reshape(-1, len(OBSERVABLES)).T, strict=True)]
        for ref, model in zip(reference, sample, strict=True)
    ])


def compare(statistics, draws, seed):
    reference = statistics["reference"]
    distances_by_direction = {method: distances(reference, statistics[method]) for method in METHODS[1:]}
    rng = np.random.default_rng(seed)
    group = np.arange(len(reference))[:, None]
    bank_shape = statistics["raw_bank"].shape[:2]
    bootstrap = []
    for _ in range(draws):
        bank = group, rng.integers(bank_shape[1], size=bank_shape)
        ref = reference[group, rng.integers(reference.shape[1], size=reference.shape[:2])]
        measured = {method: distances(ref, statistics[method][bank]).mean(axis=0) for method in METHODS[1:]}
        bootstrap.append([measured[a] - measured[b] for _, a, b in CONTRASTS])
    intervals = np.quantile(bootstrap, (0.025, 0.975), axis=0).transpose(1, 2, 0)
    group_means = {method: values.reshape(len(reference), -1, len(OBSERVABLES)).mean(axis=1)
                   for method, values in statistics.items()}
    distributions = {}
    for i, metric in enumerate(OBSERVABLES[:2]):
        distributions[metric] = {}
        for method, array in statistics.items():
            value, count = np.unique(array[..., i].astype(int), return_counts=True)
            distributions[metric][method] = {"value": value.tolist(), "count": count.tolist()}
    return {
        "means": {method: named(values.mean(axis=0)) for method, values in group_means.items()},
        "mean_by_direction": {method: [named(row) for row in values] for method, values in group_means.items()},
        "w1_macro": {method: named(values.mean(axis=0)) for method, values in distances_by_direction.items()},
        "w1_by_direction": {method: [named(row) for row in values] for method, values in distances_by_direction.items()},
        "contrast_w1_ci95": {label: named(interval) for (label, _, _), interval in zip(CONTRASTS, intervals, strict=True)},
        "occupancy_distributions": distributions,
    }


def illustrative_example(rows, raw, smooth, stochastic, geometry):
    group = 2
    bank = raw[group]
    shares = bank / bank.sum(axis=-1, keepdims=True)
    counts = (shares >= ACTIVE_SHARE).sum(axis=-1)
    i = np.lexsort((rows[group], abs(counts - np.median(counts))))[0]
    module, depth = np.nonzero(geometry.layer_mask)
    totals = np.bincount(module, weights=bank[i])
    strongest = np.argsort(-totals, kind="stable")[:4]
    panels = (bank[i], smooth[group, i], *stochastic[group, i, :2])
    arrays = []
    for response in panels:
        grid = [[None] * geometry.max_layers for _ in strongest]
        fraction = 100 * response / response.sum()
        for j, selected in enumerate(strongest):
            for layer, value in zip(depth[module == selected], fraction[module == selected], strict=True):
                grid[j][layer] = float(value)
        arrays.append(grid)
    return {
        "direction": group + 1, "row": int(rows[group, i]), "total": float(bank[i].sum()),
        "module_ids": [geometry.modules[j].module_id for j in strongest],
        "section_shares_pct": arrays,
        "n_positive_all_sections": [int((array > 0).sum()) for array in panels],
    }


def analyze(source, directory):
    calibration_path = directory / "stochastic_decoder.json"
    receipt_path = directory / "results.json"
    bank_path = directory / "bank.npz"
    calibration = json.loads(calibration_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    for path, key in [(bank_path, "bank_sha256"), (receipt_path, "baseline_receipt_sha256"),
                      (Path(__file__).with_name("evaluate_gmm_packet_decoder.py"), "evaluator_sha256"),
                      (Path(__file__).with_name("conditional_gmm_baseline.py"), "decoder_sha256")]:
        if digest(path) != calibration[key]:
            raise ValueError(f"Unexpected calibration dependency: {path.name}")
    with np.load(bank_path, allow_pickle=False) as data:
        bank = {name: data[name] for name in ("rows", "momenta", "energy", "parameters")}
    bank["components"] = receipt["settings"]["components"]
    geometry = load_projection_geometry()
    projector = SectionProjector(*section_bounds(geometry), "integrated")
    indices, reference, excluded = reference_rows(source, bank, receipt, geometry)
    expected = [item["validation"] + item["holdout"] for item in calibration["split"]]
    np.testing.assert_array_equal(indices, expected)
    if excluded != calibration["excluded_rows"]:
        raise ValueError("Excluded rows changed")
    reference = reference[:, VALIDATION_ROWS:]
    with np.load(source, allow_pickle=False) as data:
        n_rows = len(data["ParticlePDG"])
    raw = read_energy_rows(source, bank["rows"].ravel(), n_rows)
    raw = legacy_grid_to_modules(raw, geometry)[:, geometry.layer_mask].reshape(*bank["rows"].shape, -1)
    np.testing.assert_allclose(raw.sum(axis=-1), bank["energy"])
    fractions, components, weights = bank_probabilities(bank, projector)
    smooth = fractions * bank["energy"][..., None]
    rng = np.random.default_rng(calibration["seed"] + 2)
    stochastic = np.stack([
        (sample_packet_fractions(p, w, calibration["selected_packets_per_component"], rng,
                                 replicas=calibration["replicas_per_code"]) * e[None, :, None]).transpose(1, 0, 2)
        for p, w, e in zip(components, weights, bank["energy"], strict=True)
    ])
    for method, response in [("smooth", smooth), ("stochastic", stochastic)]:
        values = scores(observables(reference, projector.centers[:, -1]), observables(response, projector.centers[:, -1]))
        expected = [tuple(row[method].values()) for row in calibration["holdout_w1_by_direction"]]
        np.testing.assert_allclose(values, expected, atol=1e-12, rtol=1e-12)
    module, layer = np.nonzero(geometry.layer_mask)
    contiguous = (module[:-1] == module[1:]) & (np.diff(layer) == 1)
    stats = {method: section_statistics(response, contiguous)
             for method, response in zip(METHODS, (reference, raw, smooth, stochastic), strict=True)}
    return {
        "source_sha256": calibration["source_sha256"], "geometry_sha256": geometry.sha256,
        "calibration_sha256": digest(calibration_path), "analysis_code_sha256": digest(Path(__file__)),
        "n_holdout_per_direction": reference.shape[1], "n_bank_per_direction": raw.shape[1],
        "replicas_per_code": calibration["replicas_per_code"], "active_share": ACTIVE_SHARE,
        "bootstrap_resamples": calibration["bootstrap_resamples"],
        "example": illustrative_example(bank["rows"], raw, smooth, stochastic, geometry),
        **compare(stats, calibration["bootstrap_resamples"], calibration["seed"] + 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Exact proton archive NPZ")
    parser.add_argument("--directory", type=Path, default=DIRECTORY, help="Committed bank and calibration")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.input, args.directory)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
