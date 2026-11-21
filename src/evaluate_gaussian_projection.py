"""Measure FHCal representation error on observed single-particle-format events."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
from subprocess import PIPE, run
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import scipy
from numpy.lib.format import read_array_header_1_0, read_array_header_2_0, read_magic

from pipeline.projection_geometry import legacy_grid_to_modules, load_projection_geometry
from pipeline.gaussian_projection import SectionProjector, fit_mixture, section_bounds

METHODS = ("integrated", "center")
METRICS = (
    "tv", "depth_w1", "zero_channel_mass", "active_section_count_delta",
    "transverse_tv", "lateral_width_error", "fit_seconds",
)
ACTIVE_FRACTION = 0.01
BOOTSTRAP_RESAMPLES = 1000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Unpacked legacy NPZ")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdg", type=int, required=True)
    parser.add_argument("--momentum", type=float, nargs=3, help="Exact px py pz; default most frequent")
    parser.add_argument("--vertex", type=float, nargs=3, help="Optional exact generation vertex")
    parser.add_argument("--primary-count-key", help="Optional verified NPZ key used to select count == 1")
    parser.add_argument("--events-per-repeat", type=int, default=48)
    parser.add_argument("--seeds", type=int, nargs="+", default=[19, 816])
    parser.add_argument("--components", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--trace-restarts", action="store_true", help="Archive optimizer diagnostics for every restart")
    parser.add_argument("--min-scales", type=float, nargs=3, default=[5.0, 5.0, 0.35])
    return parser.parse_args()


def read_energy_rows(path, indices, n):
    """Read only the requested C-ordered records of the large EnergyDeposit array."""
    headers = {(1, 0): read_array_header_1_0, (2, 0): read_array_header_2_0}
    with ZipFile(path) as archive, archive.open("EnergyDeposit.npy") as stream:
        version = read_magic(stream)
        if version not in headers:
            raise ValueError(f"Unsupported energy NPY version {version}")
        shape, fortran, dtype = headers[version](stream)
        if shape != (n, 7, 9, 10) or fortran or dtype.kind not in "fiu":
            raise ValueError(f"Expected C-ordered numeric legacy energy [N,7,9,10], got {shape}")
        beginning = stream.tell()
        row_bytes = int(np.prod(shape[1:])) * dtype.itemsize
        energy = np.empty((len(indices), *shape[1:]), dtype=np.float64)
        for i in np.argsort(indices):
            stream.seek(beginning + int(indices[i]) * row_bytes)
            energy[i] = np.frombuffer(stream.read(row_bytes), dtype=dtype).reshape(shape[1:])
    return energy


def sample_events(args):
    with np.load(args.input, allow_pickle=False) as data:
        pdg, point, momentum = (data[key] for key in ("ParticlePDG", "ParticlePoint", "ParticleMomentum"))
        n = len(pdg)
        if pdg.shape != (n,) or point.shape != (n, 3) or momentum.shape != (n, 3):
            raise ValueError("Expected a single recorded PDG, vertex and momentum per row")
        if not all(np.isfinite(array).all() for array in (pdg, point, momentum)):
            raise ValueError("Input conditions must be finite")
        eligible = pdg == args.pdg
        if args.primary_count_key:
            count = data[args.primary_count_key]
            if count.shape != (n,) or not np.isfinite(count).all():
                raise ValueError("Primary count must be a finite scalar per row")
            eligible &= count == 1
        if args.vertex is not None:
            eligible &= np.all(point == args.vertex, axis=1)
        if not eligible.any():
            raise ValueError("No eligible events with the requested PDG and vertex")
        momentum_values, counts = np.unique(momentum[eligible], axis=0, return_counts=True)
        selected_momentum = momentum_values[counts.argmax()] if args.momentum is None else np.array(args.momentum)
        choices = np.flatnonzero(eligible & np.all(momentum == selected_momentum, axis=1))
        if len(choices) < len(args.seeds) * args.events_per_repeat:
            raise ValueError("Not enough events for disjoint repeats in the requested momentum group")
        available = choices.copy()
        groups = []
        for seed in args.seeds:
            selected = np.random.default_rng(seed).choice(available, size=args.events_per_repeat, replace=False)
            groups.append(selected)
            available = np.setdiff1d(available, selected, assume_unique=True)
        indices = np.concatenate(groups)
        archived_total = data["TotalEnergySci"][indices] if "TotalEnergySci" in data else None
    return indices, archived_total, {
        "archive_rows": n, "pdg": args.pdg, "momentum": selected_momentum.tolist(),
        "momentum_group_rows": len(choices), "vertex_cut": args.vertex,
        "vertex_min": point[choices].min(axis=0).tolist(), "vertex_max": point[choices].max(axis=0).tolist(),
        "unique_vertices_in_group": len(np.unique(point[choices], axis=0)),
        "primary_count_key": args.primary_count_key,
        "primary_multiplicity": "filtered_from_count_field" if args.primary_count_key else "not_verified_from_source",
    }


def fit_seed(repeat_seed, row_index, components):
    return int(np.random.SeedSequence([repeat_seed, row_index, components]).generate_state(1)[0])


def assess_shape(reference, predicted, module_index, depth_index, xy):
    def lateral_width(distribution):
        center = distribution @ xy
        return np.sqrt(distribution @ np.sum((xy - center)**2, axis=1))

    depth_delta = np.bincount(depth_index, weights=predicted - reference)
    transverse_delta = np.bincount(module_index, weights=predicted - reference)
    return {
        "tv": float(0.5 * np.abs(predicted - reference).sum()),
        "depth_w1": float(np.abs(np.cumsum(depth_delta)[:-1]).sum()),
        "zero_channel_mass": float(predicted[reference == 0].sum()),
        "active_section_count_delta": int(np.count_nonzero(predicted >= ACTIVE_FRACTION))
        - int(np.count_nonzero(reference >= ACTIVE_FRACTION)),
        "transverse_tv": float(0.5 * np.abs(transverse_delta).sum()),
        "lateral_width_error": float(lateral_width(predicted) - lateral_width(reference)),
    }


def summarize(records, components, seed):
    """Independent-row bootstrap within repeat; pair errors by original NPZ row."""
    rng = np.random.default_rng(seed)
    keys = list(dict.fromkeys(row["event_index"] for row in records))
    draws = rng.integers(0, len(keys), size=(BOOTSTRAP_RESAMPLES, len(keys)))
    fits = {}
    for kind in METHODS:
        for k in components:
            rows = [r for r in records if r["projection"] == kind and r["k"] == k]
            fits[kind, k] = rows
    summaries = []
    for (kind, k), rows in fits.items():
        entry = {
            "projection": kind, "k": k, "n": len(rows), "shape_coordinates": 7 * k - 1,
            "best_optimizer_success": sum(r["converged"] for r in rows),
            "optimizer_successes_all_restarts": sum(r["converged_restarts"] for r in rows),
            "fits_at_width_limit": sum(any(r["lower_width_hits"]) or any(r["upper_width_hits"]) for r in rows),
            "lower_width_hits_xyz": np.sum([r["lower_width_hits"] for r in rows], axis=0).tolist(),
            "upper_width_hits_xyz": np.sum([r["upper_width_hits"] for r in rows], axis=0).tolist(),
        }
        for metric in METRICS:
            values = np.array([r[metric] for r in rows])
            entry[metric] = {
                "mean": float(values.mean()), "median": float(np.median(values)),
                "p90": float(np.quantile(values, 0.90)),
            }
        summaries.append(entry)
    paired = []
    for k in components:
        models = fits["integrated", k], fits["center", k]
        pair = np.array([[r["tv"] for r in model] for model in models])
        if [r["event_index"] for r in models[0]] != [r["event_index"] for r in models[1]]:
            raise RuntimeError("Paired fits have different event order")
        delta = pair[0] - pair[1]
        paired.append({
            "contrast": "integrated_minus_center", "k": k, "tv_mean": float(delta.mean()),
            "tv_interval_95": np.quantile(delta[draws].mean(axis=1), [0.025, 0.975]).tolist(),
        })
        if k != components[0]:
            for kind in METHODS:
                base = np.array([r["tv"] for r in fits[kind, components[0]]])
                delta = np.array([r["tv"] for r in fits[kind, k]]) - base
                paired.append({
                    "contrast": f"{kind}_minus_k{components[0]}", "k": k,
                    "tv_mean": float(delta.mean()),
                    "tv_interval_95": np.quantile(delta[draws].mean(axis=1), [0.025, 0.975]).tolist(),
                })
    return {"n_events": len(keys), "fits": summaries, "paired_tv": paired}


def main():
    args = parse_args()
    if (args.events_per_repeat < 2 or len(set(args.seeds)) != len(args.seeds)
            or len(set(args.components)) != len(args.components) or min(args.components) < 1
            or min(args.seeds) < 0 or min(args.restarts, args.maxiter) < 1
            or not np.isfinite(args.min_scales).all() or min(args.min_scales) <= 0):
        raise ValueError("Use positive fit settings, unique nonnegative seeds and at least two events per repeat")
    args.components.sort()
    indices, archived_total, selection = sample_events(args)
    energy = read_energy_rows(args.input, indices, selection["archive_rows"])
    geometry = load_projection_geometry()
    mask = geometry.layer_mask
    physical = legacy_grid_to_modules(energy, geometry)
    lower, upper = section_bounds(geometry)
    projectors = {method: SectionProjector(lower, upper, method) for method in METHODS}
    xy = (lower[:, :2] + upper[:, :2]) / 2
    module_index, depth_index = np.nonzero(mask)
    observations = physical[:, mask]
    total = observations.sum(axis=1)
    records = []
    for repeat, seed in enumerate(args.seeds):
        group = range(repeat * args.events_per_repeat, (repeat + 1) * args.events_per_repeat)
        for i in group:
            if total[i] == 0:
                continue
            reference = observations[i] / total[i]
            for k in args.components:
                init_seed = fit_seed(seed, int(indices[i]), k)
                for method, projector in projectors.items():
                    start = perf_counter()
                    fit = fit_mixture(
                        projector, reference, k, args.restarts, args.maxiter, init_seed, args.min_scales,
                        trace_restarts=args.trace_restarts,
                    )
                    seconds = perf_counter() - start
                    scores = assess_shape(reference, fit.prediction, module_index, depth_index, xy)
                    records.append({
                        "seed": seed, "event_index": int(indices[i]), "k": k, "projection": method,
                        **scores, "fit_seconds": float(seconds), "loss": fit.loss,
                        "converged": fit.converged, "converged_restarts": fit.converged_restarts,
                        "lower_width_hits": fit.lower_width_hits.tolist(),
                        "upper_width_hits": fit.upper_width_hits.tolist(),
                        "optimizer_status": fit.status, "optimizer_iterations": fit.iterations,
                    })
                    if args.trace_restarts:
                        records[-1]["restart_trace"] = fit.restart_trace
    for seed in args.seeds:
        if not any(r["seed"] == seed for r in records):
            raise ValueError(f"No nonempty events available to fit for repeat {seed}")
    source_digest = hashlib.sha256()
    with args.input.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            source_digest.update(block)
    revision = run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
        stdout=PIPE, stderr=PIPE, text=True, check=False,
    )
    result = {
        "code_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "scope": "in_sample_representation",
        "source": args.input.name, "source_sha256": source_digest.hexdigest(), "selection": selection,
        "geometry": "student_muons_19_mapping_applied_to_legacy_grid",
        "geometry_provenance": {
            "sha256": geometry.sha256, "source_commit": geometry.source_commit,
            "source_symbol": geometry.source_symbol,
        },
        "coordinates": {"transverse": "cm", "depth": "section index from local front"},
        "settings": {
            "components": args.components, "restarts": args.restarts, "maxiter": args.maxiter,
            "events_per_repeat": args.events_per_repeat, "seeds": args.seeds, "min_scales": args.min_scales,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "occupancy_fraction_threshold": ACTIVE_FRACTION,
        },
        "runtime": {"python": platform.python_version(), "machine": platform.machine(),
                    "numpy": np.__version__, "scipy": scipy.__version__},
        "reference": {
            "n_sections": int(mask.sum()), "n_selected": len(indices), "empty_selected": int((total == 0).sum()),
            "selected_event_indices": indices.tolist(),
            "observed_zero_section_fraction_nonempty": float((observations[total > 0] == 0).mean()),
            "max_abs_section_sum_minus_archived_total": (
                float(np.max(np.abs(total - archived_total))) if archived_total is not None else None
            ),
        },
        "repeats": {
            str(seed): summarize([r for r in records if r["seed"] == seed], args.components, seed)
            for seed in args.seeds
        },
        "pooled": summarize(records, args.components, args.seeds[0]),
        "event_metrics": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Wrote {args.output}: {len(indices)} selected events, {len(records)} fits")


if __name__ == "__main__":
    main()
