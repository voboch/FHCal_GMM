"""Measure total and per-module visible energy and test an asymmetric muon line shape."""

import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_hermitenorm, roots_legendre
from scipy.stats import landau, norm, wasserstein_distance

from conditional_gmm_baseline import digest, sample_packet_fractions
from evaluate_gaussian_projection import read_energy_rows
from evaluate_gmm_packet_decoder import VALIDATION_ROWS, bank_probabilities, reference_rows
from pipeline.gaussian_projection import SectionProjector, section_bounds
from pipeline.projection_geometry import legacy_grid_to_modules, load_projection_geometry


PILOT = Path(__file__).resolve().parent / "gaussian_projection"
BANK = PILOT / "proton_pilot/conditional_baseline"
MUON_SEED, MUON_TRAIN, MUON_TEST = 2371, 2048, 4096


def kinematics(momentum):
    return {"p": momentum.tolist(), "norm": float(np.linalg.norm(momentum)),
            "angle_z_deg": float(np.degrees(np.arctan2(np.linalg.norm(momentum[:2]), momentum[2])))}


def describe(values, edges=None):
    positive = values[values > 0]
    summary = {"n": len(values), "n_positive": len(positive), "p_zero": float((values == 0).mean()),
               "mean": float(values.mean()), "q10_q50_q90_positive": np.quantile(positive, [.1, .5, .9]).tolist()
               if len(positive) else None}
    if edges is not None:
        summary["positive_histogram"] = histogram(positive, edges)
    return summary


def survival(values, grid):
    return (1 - np.searchsorted(np.sort(values), grid, side="right") / len(values)).tolist()


def histogram(values, edges):
    counts, _ = np.histogram(values, bins=edges)
    return {"bins": counts.tolist(), "above": int((values > edges[-1]).sum())}


def column_summaries(columns, edges=None):
    reference = columns["reference"]
    result = {}
    for name, values in columns.items():
        measured = describe(values, edges)
        if name != "reference":
            measured["w1"] = float(wasserstein_distance(reference, values))
        result[name] = measured
    return result


def proton_study(source, geometry):
    calibration = json.loads((BANK / "stochastic_decoder.json").read_text())
    receipt = json.loads((BANK / "results.json").read_text())
    if (digest(BANK / "bank.npz") != calibration["bank_sha256"]
            or digest(BANK / "results.json") != calibration["baseline_receipt_sha256"]):
        raise ValueError("Unexpected bank or baseline receipt")
    with np.load(BANK / "bank.npz", allow_pickle=False) as data:
        bank = {name: data[name] for name in ("rows", "momenta", "energy", "parameters")}
    bank["components"] = receipt["settings"]["components"]
    indices, reference, _ = reference_rows(source, bank, receipt, geometry)
    np.testing.assert_array_equal(indices[:, VALIDATION_ROWS:], [r["holdout"] for r in calibration["split"]])
    with np.load(source, allow_pickle=False) as data:
        n = len(data["ParticlePDG"])
    raw = legacy_grid_to_modules(read_energy_rows(source, bank["rows"].ravel(), n), geometry).sum(axis=-1)
    raw = raw.reshape(*bank["rows"].shape, -1)
    np.testing.assert_allclose(raw.sum(axis=-1), bank["energy"], rtol=1e-12, atol=1e-12)
    projector = SectionProjector(*section_bounds(geometry), "integrated")
    fractions, components, weights = bank_probabilities(bank, projector)
    membership = np.eye(len(geometry.modules))[np.flatnonzero(geometry.layer_mask.ravel()) // geometry.max_layers]
    reference = reference[:, VALIDATION_ROWS:] @ membership
    smooth = (fractions * bank["energy"][..., None]) @ membership
    rng = np.random.default_rng(calibration["seed"] + 2)
    stochastic = np.stack([
        (sample_packet_fractions(p, w, calibration["selected_packets_per_component"], rng,
                                 replicas=calibration["replicas_per_code"]) * total[None, :, None]) @ membership
        for p, w, total in zip(components, weights, bank["energy"], strict=True)
    ])
    np.testing.assert_allclose(stochastic.sum(axis=-1), np.broadcast_to(bank["energy"][:, None], stochastic.shape[:-1]))
    np.testing.assert_allclose(smooth.sum(axis=-1), bank["energy"])
    methods = {"reference": reference, "raw_bank": raw, "smooth": smooth,
               "stochastic": stochastic.reshape(len(raw), -1, raw.shape[-1])}
    modules = [m.module_id for m in geometry.modules]
    ranked = ((raw[2] / bank["energy"][2, :, None]) >= .01).sum(axis=0)
    chosen = np.lexsort((modules, -ranked))[:2]
    full_grid = np.linspace(0, np.quantile(reference.sum(axis=-1), .995), 161)
    module_edges = np.r_[0, np.geomspace(.001, full_grid[-1], 33)]
    groups, survival_curves = [], []
    for i, p in enumerate(bank["momenta"]):
        totals = {name: value[i].sum(axis=-1) for name, value in methods.items()}
        groups.append({"direction": i + 1, **kinematics(p), "total": column_summaries(totals), "modules": {
            str(module): column_summaries({name: value[i, :, j] for name, value in methods.items()}, module_edges)
            for j, module in enumerate(modules)}})
        survival_curves.append({name: survival(value, full_grid) for name, value in totals.items()})
    illustrated = []
    for j in chosen:
        samples = {name: value[2, :, j] for name, value in methods.items()}
        grid = np.linspace(0, max(np.quantile(array, .99) for array in samples.values()), 161)
        illustrated.append({"id": modules[j], "grid": grid.tolist(),
                            "survival": {name: survival(value, grid) for name, value in samples.items()}})
    return {"source_sha256": calibration["source_sha256"], "calibration_sha256": digest(BANK / "stochastic_decoder.json"),
            "n_holdout": int(reference.shape[0] * reference.shape[1]), "holdout_rows": indices[:, VALIDATION_ROWS:].tolist(),
            "groups": groups, "total_grid": full_grid.tolist(), "total_survival": survival_curves,
            "module_histogram_edges": module_edges.tolist(),
            "illustrated_modules_direction_3": illustrated}


@lru_cache
def quadrature(order):
    nodes, weights = roots_hermitenorm(order)
    return nodes, weights / np.sqrt(2 * np.pi)


@lru_cache
def landau_nodes(order):
    nodes, weights = roots_legendre(order)
    return landau.ppf((1 + nodes) / 2), weights / 2


def line_shape(values, parameters, asymmetric, *, density=False, order=80):
    loc, scale = parameters[:2]
    distribution = landau if asymmetric else norm
    evaluate = distribution.pdf if density else distribution.cdf
    if not asymmetric:
        return evaluate(values, loc=loc, scale=scale)
    if parameters[2] > 2 * scale:
        nodes, weights = landau_nodes(2 * order)
        gaussian = norm.pdf if density else norm.cdf
        return gaussian(values[..., None], loc=loc + scale * nodes, scale=parameters[2]) @ weights
    nodes, weights = quadrature(order)
    return evaluate(values[..., None] - parameters[2] * nodes, loc=loc, scale=scale) @ weights


def conditional_shape(values, parameters, asymmetric, *, density=False, order=80):
    low = line_shape(np.asarray(0.0), parameters, asymmetric, order=order)
    y = line_shape(np.asarray(values), parameters, asymmetric, density=density, order=order)
    return (y if density else y - low) / max(1.0 - low, 1e-250)


def fit_lines(train, test):
    median = np.median(train)
    span = np.subtract(*np.quantile(train, [.75, .25]))
    if span <= 0:
        raise ValueError("A positive IQR is required for the muon line fit")
    edges = np.r_[0, np.unique(np.quantile(train, np.linspace(.01, .99, 49))), np.inf]
    counts, _ = np.histogram(train, edges)
    test_bins = np.searchsorted(edges, test, side="right") - 1
    x = np.linspace(0, np.quantile(train, .995), 480)
    models = {}
    losses = {}
    for name, asymmetric in (("normal", False), ("landau_gauss", True)):
        def physical(p):
            return np.r_[median + p[0] * span, span * np.exp(p[1:])]

        def probabilities(p, order):
            return np.maximum(np.diff(conditional_shape(edges, physical(p), asymmetric, order=order)), 1e-250)

        def objective(p):
            return float(-counts @ np.log(probabilities(p, 80)) / counts.sum())

        bounds = [(-6, 6), (np.log(.005), np.log(10))]
        starts = [[0.0, -.5]]
        if asymmetric:
            bounds.append((np.log(.004), np.log(4)))
            starts = [[-.45, -1.4, np.log(sigma)] for sigma in (.06, .25, .6)]
        fits = [minimize(objective, start, method="L-BFGS-B", bounds=bounds) for start in starts]
        fit = min((candidate for candidate in fits if candidate.success), key=lambda candidate: candidate.fun)
        parameters = physical(fit.x)
        probabilities_high = probabilities(fit.x, 160)
        quadrature_delta = float(np.abs(probabilities_high - probabilities(fit.x, 80)).max())
        if quadrature_delta > .001:
            raise ValueError(f"Muon line quadrature differs by {quadrature_delta}")
        losses[name] = -np.log(probabilities_high[test_bins])
        models[name] = {"parameters": parameters.tolist(), "test_nll": float(losses[name].mean()),
                        "max_quadrature_delta": quadrature_delta,
                        "pdf": conditional_shape(x, parameters, asymmetric, density=True, order=160).tolist(),
                        "survival": (1 - conditional_shape(x, parameters, asymmetric, order=160)).tolist()}
    difference = losses["landau_gauss"] - losses["normal"]
    rng = np.random.default_rng(MUON_SEED + 1)
    interval = np.quantile([rng.choice(difference, size=len(difference), replace=True).mean() for _ in range(1000)],
                           [.025, .975])
    hist_edges = np.linspace(0, x[-1], 60)
    return {"n_train_positive": len(train), "n_test_positive": len(test), "plot_grid": x.tolist(),
            "hist_edges": hist_edges.tolist(), "test_histogram_positive": histogram(test, hist_edges),
            "test_survival_positive": survival(test, x), "models": models,
            "contrast_test_nll": float(difference.mean()), "contrast_ci95": interval.tolist()}


def muon_study(source, geometry):
    receipt = PILOT / "pilot_results.json"
    evidence = json.loads((PILOT / "evidence.json").read_text())
    if digest(receipt) != evidence["exports"][receipt.name] or geometry.sha256 != evidence["geometry_snapshot_sha256"]:
        raise ValueError("Unexpected muon pilot receipt or geometry")
    archived = json.loads(receipt.read_text())
    source_hash = digest(source)
    if source_hash != archived["source_sha256"]:
        raise ValueError("Unexpected muon archive")
    with np.load(source, allow_pickle=False) as data:
        momentum, pdg, total = (data[key] for key in ("ParticleMomentum", "ParticlePDG", "TotalEnergySci"))
    momenta = sorted(np.unique(momentum[pdg == -13], axis=0), key=lambda p: kinematics(p)["angle_z_deg"])
    available = np.setdiff1d(np.flatnonzero(pdg == -13), archived["reference"]["selected_event_indices"], assume_unique=True)
    rng = np.random.default_rng(MUON_SEED)
    groups = np.stack([rng.choice(available[np.all(momentum[available] == p, axis=1)], MUON_TRAIN + MUON_TEST,
                                  replace=False) for p in momenta])
    energies = legacy_grid_to_modules(read_energy_rows(source, groups.ravel(), len(pdg)), geometry).sum(axis=-1)
    energies = energies.reshape(*groups.shape, -1)
    totals = energies.sum(axis=-1)
    np.testing.assert_allclose(totals, total[groups], rtol=1e-5, atol=1e-6)
    train, test = energies[:, :MUON_TRAIN], energies[:, MUON_TRAIN:]
    j = int(np.argmax(np.count_nonzero(train[0] > 0, axis=0)))
    found = test[0, :, j]
    fit = fit_lines(train[0, train[0, :, j] > 0, j], found[found > 0])
    grid = np.linspace(0, np.quantile(totals[:, MUON_TRAIN:], .995), 161)
    module_edges = np.r_[0, np.geomspace(.001, grid[-1], 33)]
    return {"source_sha256": source_hash, "seed": MUON_SEED, "n_train_per_direction": MUON_TRAIN,
            "n_test_per_direction": MUON_TEST, "total_grid": grid.tolist(), "module_histogram_edges": module_edges.tolist(),
            "groups": [{**kinematics(p), "total": describe(total_group), "total_survival": survival(total_group, grid),
                        "fit_rows": indices[:MUON_TRAIN].tolist(), "holdout_rows": indices[MUON_TRAIN:].tolist(),
                        "modules": {str(module.module_id): describe(array[:, i], module_edges)
                                    for i, module in enumerate(geometry.modules)}}
                       for p, array, total_group, indices in zip(momenta, test, totals[:, MUON_TRAIN:], groups, strict=True)],
            "selected_module": geometry.modules[j].module_id, "line_shape": fit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protons", type=Path, required=True)
    parser.add_argument("--muons", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    geometry = load_projection_geometry()
    data = {"geometry_sha256": geometry.sha256, "code_sha256": digest(Path(__file__)),
            "protons": proton_study(args.protons, geometry), "muons": muon_study(args.muons, geometry)}
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
