"""Evaluate a training-donor occupancy count on two frozen FHCal generators."""

import csv
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.stats import wasserstein_distance

from conditional_gmm_baseline import digest
from conditional_shape_wgan import CACHE, MODELS, REPORT, SEEDS, generate
from evaluate_conditional_shapes import empirical
from pipeline.projection_geometry import load_projection_geometry


OUTPUT = Path("reports/section_sparsity")
METRICS = ("section_rate_mae_pp", "count_w1", "tail_mae_pp")
METHODS = ("empirical", "direct", "direct_topk", "gmm", "gmm_topk")


def keep_largest(rows, counts, totals):
    # Stable ordering preserves ascending geometry index for tied signals.
    order = np.argsort(-rows, axis=-1, kind="stable")
    retained = np.arange(rows.shape[-1])[None, :] < counts[:, None]
    mask = np.zeros_like(rows, dtype=bool)
    np.put_along_axis(mask, order, retained, axis=1)
    kept = np.where(mask, rows, 0).astype(np.float64)
    kept_sums = kept.sum(axis=1)
    if not np.array_equal(kept_sums > 0, totals > 0):
        raise ValueError("Inconsistent mask and total signal")
    scale = np.divide(totals, kept_sums, out=np.zeros_like(kept_sums), where=kept_sums > 0)
    kept *= scale[:, None]
    np.testing.assert_allclose(kept.sum(axis=1), totals, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal((kept > 0).sum(axis=1), np.minimum(counts, (rows > 0).sum(axis=1)))
    return kept


def direction_metrics(rows, totals, counts, reference, reference_modules, cuts, starts):
    if not np.isfinite(rows).all() or (rows < 0).any():
        raise ValueError("Invalid generated map")
    np.testing.assert_allclose(rows.sum(axis=1), totals, rtol=2e-5, atol=1e-6)
    positive = rows > 0
    ref_positive = reference > 0
    occupancy = positive.sum(axis=1)
    model_modules = np.add.reduceat(rows, starts, axis=1, dtype=np.float64)
    ref_rate = (reference_modules > cuts).mean(axis=0)
    model_rate = (model_modules > cuts).mean(axis=0)
    eligible = np.isfinite(cuts)
    return dict(
        n=len(rows), eligible_modules=int(eligible.sum()),
        section_rate_mae_pp=100 * np.abs(positive.mean(axis=0) - ref_positive.mean(axis=0)).mean(),
        count_w1=wasserstein_distance(occupancy, ref_positive.sum(axis=1)),
        tail_mae_pp=100 * np.abs(model_rate[eligible] - ref_rate[eligible]).mean(),
        under_donor_count=int((occupancy < counts).sum()),
    )


def evaluate(species):
    with np.load(CACHE / f"{species}_sections.npz") as source:
        train, calibration, reference = source["sections"].transpose(1, 0, 2, 3)
    geo = load_projection_geometry()
    module_ids = np.nonzero(geo.layer_mask)[0]
    starts = np.flatnonzero(np.r_[True, np.diff(module_ids) != 0])
    total = reference.sum(axis=-1)
    cal = np.add.reduceat(calibration, starts, axis=-1, dtype=np.float64)
    ref_modules = np.add.reduceat(reference, starts, axis=-1, dtype=np.float64)
    cuts = [[np.quantile(col[col > 0], .95) if np.count_nonzero(col > 0) >= 500 else np.nan
             for col in group.T] for group in cal]
    detail = []
    for seed in SEEDS:
        donors = empirical(train, total, seed)
        counts = (donors > 0).sum(axis=-1)
        for method in ("empirical", "direct", "gmm"):
            predictions = donors if method == "empirical" else generate(species, method, seed, total)
            for group, signals in enumerate(predictions):
                variants = {method: signals}
                if method != "empirical":
                    variants[method + "_topk"] = keep_largest(signals, counts[group], total[group])
                for name, rows in variants.items():
                    metrics = direction_metrics(rows, total[group], counts[group], reference[group],
                                                ref_modules[group], np.asarray(cuts[group]), starts)
                    detail.append(dict(species=species, method=name, seed=seed, direction=group + 1, **metrics))
    summary = {}
    for method in METHODS:
        group = [r for r in detail if r["method"] == method]
        summary[method] = {}
        for metric in METRICS:
            by_seed = [np.mean([r[metric] for r in group if r["seed"] == seed]) for seed in SEEDS]
            summary[method][metric] = dict(mean=np.mean(by_seed), by_seed=by_seed)
    old = json.loads((REPORT / "summary.json").read_text())[species]
    for method in ("empirical", "direct", "gmm"):
        for metric, prior_name in [("count_w1", "positive_w1"), ("tail_mae_pp", "tail_mae_pp")]:
            np.testing.assert_allclose(summary[method][metric]["by_seed"], old[method][prior_name])
    return detail, summary


def main():
    torch.set_num_threads(1)
    detail, summary = [], {}
    for species in ("muon", "proton"):
        rows, summary[species] = evaluate(species)
        detail.extend(rows)
    OUTPUT.mkdir(exist_ok=True, parents=True)
    with (OUTPUT / "directions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(detail[0]))
        writer.writeheader()
        writer.writerows(detail)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    sources = [Path(__file__).relative_to(Path.cwd()), OUTPUT / "PROTOCOL.md", REPORT / "summary.json",
               Path("src/conditional_shape_wgan.py"), Path("src/evaluate_conditional_shapes.py"),
               Path("src/pipeline/gaussian_projection.py"), Path("src/pipeline/projection_geometry.py")]
    for species in ("muon", "proton"):
        sources.extend([CACHE / f"{species}_sections.npz", REPORT / f"{species}_reference.json"])
        sources.extend(MODELS / f"{species}_{method}_{seed}.pt"
                       for method in ("direct", "gmm") for seed in SEEDS)
    receipt = dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                   scipy=scipy.__version__, seeds=SEEDS, neighbours=16,
                   geometry_sha256=load_projection_geometry().sha256,
                   input_sha256={str(path): digest(path) for path in sources},
                   export_sha256={name: digest(OUTPUT / name) for name in ("summary.json", "directions.csv")})
    (OUTPUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
