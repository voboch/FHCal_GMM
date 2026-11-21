"""Fit, evaluate, and sample a direction-conditioned FHCal mixture bank."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from hashlib import file_digest, sha256
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.stats import wasserstein_distance

from evaluate_gaussian_projection import fit_seed, read_energy_rows
from pipeline.gaussian_projection import SectionProjector, fit_mixture, section_bounds
from pipeline.projection_geometry import legacy_grid_to_modules, load_projection_geometry


PILOT = Path(__file__).resolve().parent / "gaussian_projection" / "proton_pilot"
K, BANK_SIZE, HOLDOUT_SIZE = 6, 32, 256
RESTARTS, MAXITER, MIN_SCALES = 3, 800, (2.5, 2.5, 0.175)
SPLIT_SEED, BOOTSTRAPS, ACTIVE_SHARE = 20261120, 1000, 0.01
METRICS = ("total", "sections_ge_1pct", "positive_sections", "max_share_pp", "depth_centroid")
METHODS = ("pooled_raw", "conditional_raw", "conditional_gmm")


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def load_splits(source, geometry):
    evidence = json.loads((PILOT / "evidence.json").read_text())
    analysis = PILOT / "event_dependence.json"
    source_hash = digest(source)
    if (source_hash != evidence["input_npz_sha256"]
            or geometry.sha256 != evidence["geometry_snapshot_sha256"]
            or digest(analysis) != evidence["exports"][analysis.name]):
        raise ValueError("This baseline requires the versioned proton archive, geometry and diagnostic selection")
    prior = json.loads(analysis.read_text())
    excluded = sorted({row["event_index"] for row in prior["events"]}
                      | {i for group in prior["direction_groups"] for i in group["selected_rows"]})
    with np.load(source, allow_pickle=False) as data:
        pdg, momentum, point, recorded_total = (data[name] for name in
                                                ("ParticlePDG", "ParticleMomentum", "ParticlePoint", "TotalEnergySci"))
    protons = np.flatnonzero(pdg == 2212)
    directions = np.unique(momentum[protons], axis=0)
    if len(directions) != 5 or not np.isfinite(point[protons]).all():
        raise ValueError("Expected five recorded proton directions and finite vertices")
    available = np.setdiff1d(protons, excluded, assume_unique=True)
    rng = np.random.default_rng(SPLIT_SEED)
    selected = np.stack([
        rng.choice(available[np.all(momentum[available] == p, axis=1)], BANK_SIZE + HOLDOUT_SIZE, replace=False)
        for p in directions
    ])
    indices = selected.ravel()
    raw = read_energy_rows(source, indices, len(pdg))
    flat = legacy_grid_to_modules(raw, geometry)[:, geometry.layer_mask]
    np.testing.assert_allclose(flat.sum(axis=1), recorded_total[indices], rtol=1e-5, atol=1e-6)
    signatures = [sha256(r.tobytes()).digest()
                  for r in np.column_stack((momentum[indices], point[indices], flat))]
    if len(set(signatures)) != len(indices):
        raise ValueError("Found duplicate rows in the selected bank and holdout")
    response = flat.reshape(*selected.shape, flat.shape[-1])
    return directions, selected, response, source_hash, excluded


def fit_row(projector, row):
    index, response = row
    total = response.sum()
    if total == 0:
        return np.zeros(7 * K - 1), 0, -1
    fit = fit_mixture(projector, response / total, K, RESTARTS, MAXITER,
                      fit_seed(SPLIT_SEED, int(index), K), MIN_SCALES, trace_restarts=True)
    selected, = [start for start in fit.restart_trace if start["selected"]]
    return np.asarray(selected["parameters"]), selected["tv"], fit.status


def sample_packet_fractions(components, weights, packets, rng, *, replicas=None):
    """Sample equal-energy deposits within each component on the physical sections."""
    if packets < 1:
        raise ValueError("Packets per component must be positive")
    size = None if replicas is None else (replicas, *components.shape[:-1])
    counts = rng.multinomial(packets, components, size=size)
    return np.einsum("...k,...ki->...i", weights, counts) / packets


def decode(projector, parameters, totals, *, packets=None, rng=None):
    response = np.zeros((len(totals), len(projector.lower)))
    weights = np.zeros((len(totals), K))
    for i in np.flatnonzero(totals > 0):
        fractions, components, _, weights[i] = projector.mixture(parameters[i], K)
        if packets is not None:
            fractions = sample_packet_fractions(components, weights[i], packets, rng)
        response[i] = fractions * totals[i]
    np.testing.assert_allclose(response.sum(axis=1), totals, rtol=1e-12, atol=1e-12)
    return response, weights


def observables(response, depth):
    totals = response.sum(axis=-1)
    fractions = np.divide(response, totals[..., None], out=np.zeros_like(response), where=totals[..., None] > 0)
    return np.stack((totals, (fractions >= ACTIVE_SHARE).sum(axis=-1), (response > 0).sum(axis=-1),
                     100 * fractions.max(axis=-1), fractions @ depth), axis=-1)


def distribution_scores(bank_raw, bank_gmm, holdout):
    def scores(raw, gmm, reference):
        pooled = raw.reshape(-1, len(METRICS))
        return np.asarray([
            [[wasserstein_distance(a, b) for a, b in zip(ref.T, candidate.T, strict=True)]
             for candidate in (pooled, raw_group, gmm_group)]
            for ref, raw_group, gmm_group in zip(reference, raw, gmm, strict=True)
        ])

    def as_dict(values, names):
        return {name: dict(zip(METRICS, row.tolist(), strict=True))
                for name, row in zip(names, values, strict=True)}

    measured = scores(bank_raw, bank_gmm, holdout)
    rng = np.random.default_rng(SPLIT_SEED + 1)
    groups = np.arange(len(holdout))[:, None]
    contrasts = []
    for _ in range(BOOTSTRAPS):
        draw = groups, rng.integers(BANK_SIZE, size=bank_raw.shape[:2])
        reference = holdout[groups, rng.integers(HOLDOUT_SIZE, size=holdout.shape[:2])]
        pooled, raw, gmm = scores(bank_raw[draw], bank_gmm[draw], reference).mean(axis=0)
        contrasts.append((raw - pooled, gmm - raw))
    interval = np.quantile(contrasts, (0.025, 0.975), axis=0).transpose(1, 2, 0)
    return {
        "w1_by_direction": [as_dict(group, METHODS) for group in measured],
        "w1_macro": as_dict(measured.mean(axis=0), METHODS),
        "macro_contrast_ci95": as_dict(interval, ("conditional_raw_minus_pooled_raw", "conditional_gmm_minus_conditional_raw")),
    }


def train(source, output, workers):
    geometry = load_projection_geometry()
    projector = SectionProjector(*section_bounds(geometry), "integrated")
    directions, selected, response, input_hash, excluded = load_splits(source, geometry)
    bank_rows, holdout_rows = selected[:, :BANK_SIZE], selected[:, BANK_SIZE:]
    bank_raw, holdout = response[:, :BANK_SIZE], response[:, BANK_SIZE:]
    n_channels = bank_raw.shape[-1]
    flat_raw = bank_raw.reshape(-1, n_channels)
    totals = flat_raw.sum(axis=-1)
    jobs = list(zip(bank_rows.ravel(), flat_raw, strict=True))
    fit = partial(fit_row, projector)
    print(f"Fitting {len(jobs)} bank rows with {workers} worker(s)", flush=True)
    if workers == 1:
        fits = list(map(fit, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            fits = list(executor.map(fit, jobs))
    parameters = np.stack([row[0] for row in fits])
    tv = np.array([row[1] for row in fits])
    status = np.array([row[2] for row in fits])
    bank_fit, _ = decode(projector, parameters, totals)
    reconstructed_tv = np.divide(np.abs(flat_raw - bank_fit).sum(axis=-1), 2 * totals,
                                 out=np.zeros_like(totals), where=totals > 0)
    np.testing.assert_allclose(tv, reconstructed_tv, rtol=1e-12, atol=1e-12)
    depth = projector.centers[:, -1]
    raw_obs = observables(bank_raw, depth)
    fit_obs = observables(bank_fit.reshape(bank_raw.shape), depth)
    holdout_obs = observables(holdout, depth)
    np.testing.assert_allclose(fit_obs[..., 0], raw_obs[..., 0], rtol=1e-12, atol=1e-12)
    results = distribution_scores(raw_obs, fit_obs, holdout_obs)
    output.mkdir(parents=True, exist_ok=True)
    bank = output / "bank.npz"
    np.savez_compressed(bank, geometry_sha256=geometry.sha256, momenta=directions,
                        rows=bank_rows, energy=totals.reshape(bank_rows.shape),
                        parameters=parameters.reshape(*bank_rows.shape, -1),
                        fit_tv=tv.reshape(bank_rows.shape), optimizer_status=status.reshape(bank_rows.shape))
    result = {
        "scope": "held-out NPZ rows; exact recorded direction; physical event IDs not verified",
        "source_sha256": input_hash, "geometry_sha256": geometry.sha256, "bank_sha256": digest(bank),
        "code_sha256": digest(Path(__file__)),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "settings": {"pdg": 2212, "components": K, "restarts": RESTARTS, "maxiter": MAXITER,
                     "min_scales": MIN_SCALES, "seed": SPLIT_SEED, "bootstrap_draws": BOOTSTRAPS,
                     "active_share": ACTIVE_SHARE, "n_sections": n_channels},
        "groups": [{"direction": i, "momentum": p.tolist(), "bank_rows": rows.tolist(), "holdout_rows": held.tolist(),
                    "holdout_mean_total": float(reference[:, 0].mean())}
                   for i, (p, rows, held, reference) in enumerate(zip(directions, bank_rows, holdout_rows, holdout_obs,
                                                                       strict=True), 1)],
        "excluded_diagnostic_count": len(excluded),
        "empty_rows": {"bank": int((totals == 0).sum()), "holdout": int((holdout_obs[..., 0] == 0).sum())},
        "fits": {"n_nonempty": int((totals > 0).sum()), "successful": int((status == 0).sum()),
                 "optimizer_status_counts": {str(int(code)): int((status == code).sum()) for code in np.unique(status)},
                 "mean_train_tv_nonempty": float(tv[totals > 0].mean())},
        **results,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {bank} and {output / 'results.json'}", flush=True)


def generate(bank, direction, events, seed, output, *, packets=None):
    geometry = load_projection_geometry()
    projector = SectionProjector(*section_bounds(geometry), "integrated")
    with np.load(bank, allow_pickle=False) as data:
        if str(data["geometry_sha256"]) != geometry.sha256:
            raise ValueError("Bank and section geometry differ")
        momenta = data["momenta"]
        if not 1 <= direction <= len(momenta):
            raise ValueError(f"Direction must be between 1 and {len(momenta)}")
        group = direction - 1
        rows = data["rows"][group]
        rng = np.random.default_rng(seed)
        draw = rng.integers(len(rows), size=events)
        totals = data["energy"][group, draw]
        parameters = data["parameters"][group, draw]
    response, weights = decode(projector, parameters, totals, packets=packets, rng=rng)
    module, depth = np.nonzero(geometry.layer_mask)
    module_ids = np.array([entry.module_id for entry in geometry.modules])
    np.savez_compressed(output, direction=direction, momentum=momenta[group], seed=seed, source_row=rows[draw],
                        energy=totals, weights=weights, centers=parameters[:, :3*K].reshape(-1, K, 3),
                        widths=np.exp(parameters[:, 3*K:6*K]).reshape(-1, K, 3), section_energy=response,
                        module_id=module_ids[module], layer=depth + 1, geometry_sha256=geometry.sha256,
                        decoder="smooth" if packets is None else "stochastic",
                        packets_per_component=0 if packets is None else packets)
    print(f"Wrote {events} responses for direction {direction} to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train", help="Build the bank and evaluate it on held-out archive rows")
    fit.add_argument("input", type=Path)
    fit.add_argument("--output-dir", required=True, type=Path)
    fit.add_argument("--workers", type=int, default=1)
    sample = commands.add_parser("generate", help="Sample section responses and GMM parameters without the source NPZ")
    sample.add_argument("bank", type=Path)
    sample.add_argument("--direction", required=True, type=int)
    sample.add_argument("--events", type=int, default=100)
    sample.add_argument("--seed", type=int, default=SPLIT_SEED)
    sample.add_argument("--output", required=True, type=Path)
    sample.add_argument("--decoder", choices=("smooth", "stochastic"), default="smooth")
    sample.add_argument("--packets-per-component", type=int,
                        help="Positive packet count for stochastic decoding; default from the calibrated bank")
    args = parser.parse_args()
    if args.command == "train":
        if args.workers < 1:
            parser.error("--workers must be positive")
        train(args.input, args.output_dir, args.workers)
    else:
        if args.events < 1 or args.seed < 0 or args.output.suffix != ".npz":
            parser.error("Use a positive --events, nonnegative --seed and an .npz output path")
        packets = args.packets_per_component
        if args.decoder == "smooth" and packets is not None:
            parser.error("--packets-per-component requires --decoder stochastic")
        if args.decoder == "stochastic" and packets is None:
            calibration = args.bank.with_name("stochastic_decoder.json")
            if not calibration.is_file():
                parser.error("No calibration for this bank. Pass --packets-per-component explicitly")
            tuning = json.loads(calibration.read_text())
            if tuning["bank_sha256"] != digest(args.bank):
                parser.error("Packet calibration belongs to a different bank")
            packets = tuning["selected_packets_per_component"]
        if packets is not None and packets < 1:
            parser.error("--packets-per-component must be positive")
        generate(args.bank, args.direction, args.events, args.seed, args.output, packets=packets)


if __name__ == "__main__":
    main()
