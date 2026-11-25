"""One-factor archive study of muon and proton upper tails by recorded direction."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from analyze_module_energy import BANK
from conditional_gmm_baseline import digest
from evaluate_gaussian_projection import read_energy_rows
from muon_tail_bank import fit_bank
from pipeline.projection_geometry import (
    legacy_grid_to_modules,
    load_projection_geometry,
)

PILOT = BANK.parent.parent
N = 4096
MIN_POSITIVE = 500
SEED = 61123
ROLES = ("fit", "threshold", "test")
DEFAULT_CACHE = Path("outputs/single_particle")


def read_json(path):
    return json.loads(path.read_text())


def inputs(species):
    historical = read_json(PILOT / "energy_response.json")
    if species == "muon":
        groups = historical["muons"]["groups"]
        excluded = set(
            read_json(PILOT / "pilot_results.json")["reference"][
                "selected_event_indices"
            ]
        )
        for group in groups:
            excluded.update(group["fit_rows"] + group["holdout_rows"])
        return (
            np.array([g["p"] for g in groups]),
            excluded,
            historical["muons"]["source_sha256"],
        )
    receipt = read_json(BANK / "results.json")
    diagnostic = read_json(BANK.parent / "event_dependence.json")
    excluded = {r["event_index"] for r in diagnostic["events"]}
    for g in diagnostic["direction_groups"]:
        excluded.update(g["selected_rows"])
    for g in receipt["groups"]:
        excluded.update(g["bank_rows"] + g["holdout_rows"])
    for g in read_json(BANK / "stochastic_decoder.json")["split"]:
        excluded.update(g["validation"] + g["holdout"])
    return (
        np.array([g["momentum"] for g in receipt["groups"]]),
        excluded,
        receipt["source_sha256"],
    )


def build_split(path, species, geo):
    momenta, historical, expected = inputs(species)
    if digest(path) != expected:
        raise ValueError(f"Unexpected {species} archive")
    with np.load(path, allow_pickle=False) as f:
        pdg, p, vertex, total = (
            f[k]
            for k in (
                "ParticlePDG",
                "ParticleMomentum",
                "ParticlePoint",
                "TotalEnergySci",
            )
        )
    n = len(p)
    rng = np.random.default_rng(SEED + (1 if species == "proton" else 0))
    modules, depths = np.nonzero(geo.layer_mask)
    selectors = np.column_stack(
        [
            np.ones(len(modules)),
            np.eye(geo.max_layers)[depths],
            np.eye(len(geo.modules))[modules],
        ]
    )
    selected, projections = [], []
    for momentum in momenta:
        same = np.flatnonzero(
            (pdg == (-13 if species == "muon" else 2212)) & (p == momentum).all(axis=1)
        )
        old = np.intersect1d(same, list(historical))
        required = 3 * N + (128 if species == "muon" else 0)
        candidates = rng.choice(np.setdiff1d(same, old), required + 512, replace=False)
        ids = np.r_[old, candidates]
        sections = legacy_grid_to_modules(read_energy_rows(path, ids, n), geo)[
            :, geo.layer_mask
        ]
        np.testing.assert_allclose(
            sections.sum(axis=-1), total[ids], rtol=1e-5, atol=1e-6
        )
        fingerprints = set()
        keep = []
        for j, index in enumerate(ids):
            h = hashlib.blake2b(
                vertex[index].tobytes() + sections[j].tobytes(), digest_size=20
            ).digest()
            if j >= len(old) and h not in fingerprints:
                keep.append(j)
            fingerprints.add(h)
            if len(keep) == required:
                break
        if len(keep) != required:
            raise ValueError("Insufficient unique physical records")
        taken = ids[keep]
        selected.append(
            {key: taken[i * N : (i + 1) * N].tolist() for i, key in enumerate(ROLES)}
        )
        if species == "muon":
            selected[-1]["bank"] = taken[3 * N :].tolist()
        projections.append(sections[keep[: 3 * N]] @ selectors)
    return momenta, selected, np.stack(projections), expected, selectors


def prepare(path, species, output, workers, cache):
    cache.mkdir(parents=True, exist_ok=True)
    geo = load_projection_geometry()
    momenta, split, arrays, sha, selectors = build_split(path, species, geo)
    if species == "muon":
        bank = fit_bank(
            path,
            np.array([g["bank"] for g in split]),
            geo,
            output / "muon_bank.npz",
            workers,
        )
    else:
        from plot_tail_atlas import load_samples

        loaded, _, _ = load_samples(path, geo)
        bank = {k: v for k, v in loaded.items() if k != "reference"}
    projected = {k: v @ selectors for k, v in bank.items()}
    np.savez_compressed(
        cache / (species + "_samples.npz"), data=arrays, momenta=momenta, **projected
    )
    (output / (species + "_split.json")).write_text(
        json.dumps(
            dict(
                seed=SEED + int(species == "proton"),
                archive_sha256=sha,
                geometry_sha256=geo.sha256,
                excluded_count=len(inputs(species)[1]),
                split=split,
                momenta=momenta.tolist(),
                roles=list(ROLES),
            )
        )
        + "\n"
    )


def interval(hits, size, alpha=0.05):
    if not size:
        return None, None
    low = beta.ppf(alpha / 2, hits, size - hits + 1) if hits else 0.0
    high = beta.ppf(1 - alpha / 2, hits + 1, size - hits) if hits < size else 1.0
    return float(low), float(high)


def mc_error(values, cut, method, species):
    """Conditional Monte Carlo SE for P(above | positive), holding the training set fixed."""
    if cut is None or not np.any(values > 0):
        return None
    positive = values > 0
    estimate = np.sum(values > cut) / positive.sum()
    influence = ((values > cut) - estimate * positive) / positive.mean()
    if method == "flow_shape":
        return float(influence.std(ddof=1) / np.sqrt(len(values)))
    if method == "stochastic":
        strata = (
            influence.reshape(128, 16)
            if species == "muon"
            else influence.reshape(32, 32).T
        )
        return float(
            np.sqrt(strata.var(axis=1, ddof=1).sum() / strata.shape[1]) / len(strata)
        )
    return 0.0


def summarize(species, direction, labels, samples, calibration, quantile, min_positive):
    rows = []
    for j, label in enumerate(labels):
        positive = calibration[:, j][calibration[:, j] > 0]
        eligible = len(positive) >= min_positive
        cut = float(np.quantile(positive, quantile)) if len(positive) else None
        for method, data in samples.items():
            x = data[:, j]
            npos = int((x > 0).sum())
            hits = int((x > cut).sum()) if cut is not None else None
            cond = hits / npos if npos and hits is not None else None
            lo, hi = (
                interval(hits, npos)
                if method == "reference" and hits is not None
                else (None, None)
            )
            rows.append(
                dict(
                    species=species,
                    direction=direction,
                    observable=label,
                    method=method,
                    eligible=eligible,
                    calib_positive=len(positive),
                    threshold=cut,
                    n=len(x),
                    n_positive=npos,
                    n_above=hits,
                    p_zero=1 - npos / len(x),
                    tail_all=hits / len(x) if hits is not None else None,
                    tail_pos=cond,
                    ref_cond_low95=lo,
                    ref_cond_high95=hi,
                    conditional_mc_se=mc_error(x, cut, method, species)
                    if method in ("flow_shape", "stochastic")
                    else None,
                )
            )
    return rows


def score(species, output, cache):
    from conditional_tail_flow import generate_flow
    from plot_single_particle_tails import render_species

    with np.load(cache / (species + "_samples.npz"), allow_pickle=False) as f:
        fit, calib, test = np.split(f["data"], 3, axis=1)
        momenta = f["momenta"]
        banks = {key: f[key] for key in ("raw_bank", "smooth", "stochastic")}
    geo = load_projection_geometry()
    labels = (
        ["Детектор"]
        + [f"Слой {i + 1}" for i in range(geo.max_layers)]
        + [f"Модуль {m.module_id}" for m in geo.modules]
    )
    generated, info = generate_flow(fit, SEED, output / (species + "_flow"))
    (output / (species + "_flow.json")).write_text(
        json.dumps(info, indent=2, allow_nan=False) + "\n"
    )
    pooled = fit.reshape(-1, len(labels))
    rows, groups = [], []
    for i, p in enumerate(momenta):
        samples = {
            "reference": test[i],
            "conditional_raw": fit[i],
            "pooled_raw": pooled,
            **{k: v[i] for k, v in banks.items()},
            "flow_shape": generated[i],
        }
        for name, matrix in samples.items():
            if not np.isfinite(matrix).all() or matrix.min() < 0:
                raise ValueError(name)
            np.testing.assert_allclose(
                matrix[:, 0], matrix[:, 1:11].sum(axis=1), atol=1e-10
            )
            np.testing.assert_allclose(
                matrix[:, 0], matrix[:, 11:].sum(axis=1), atol=1e-10
            )
        groups.append(
            dict(direction=i + 1, momentum=p.tolist(), labels=labels, samples=samples)
        )
        rows.extend(
            summarize(species, i + 1, labels, samples, calib[i], 0.95, MIN_POSITIVE)
        )
    with (output / (species + "_tails.csv")).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    render_species(output / (species + "_distributions.pdf"), species, groups, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("species", choices=("muon", "proton"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument("--prepare-only", action="store_true")
    stages.add_argument("--score-only", action="store_true")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.score_only:
        prepare(args.source, args.species, args.output, args.workers, args.cache)
    if not args.prepare_only:
        score(args.species, args.output, args.cache)


if __name__ == "__main__":
    main()
