"""Score paired section maps at fixed reference totals and recorded directions."""

import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import wasserstein_distance

from conditional_shape_wgan import CACHE, REPORT, SEEDS, generate
from analyze_gmm_packet_decoder import section_statistics
from pipeline.projection_geometry import load_projection_geometry


def empirical(draws, totals, seed):
    rng = np.random.default_rng(seed + 913)
    out = np.zeros_like(draws)
    for group, (rows, values) in enumerate(zip(draws, totals, strict=True)):
        row_totals = rows.sum(-1)
        order = rng.permutation(np.flatnonzero(row_totals > 0))
        order = order[np.argsort(row_totals[order], kind="stable")]
        logs = np.log(row_totals[order])
        k = min(len(order), 16)
        cut = (logs[:-k] + logs[k:]) / 2
        pos = values > 0
        index = np.searchsorted(cut, np.log(values[pos])) + rng.integers(
            k, size=pos.sum()
        )
        donor = order[index]
        out[group, pos] = rows[donor] * (values[pos] / row_totals[donor])[:, None]
    return out


def write_csv(name, rows):
    with (REPORT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def evaluate(species):
    with np.load(CACHE / f"{species}_sections.npz") as f:
        train, cal, test = f["sections"].transpose(1, 0, 2, 3)
    totals = test.sum(-1)
    geo = load_projection_geometry()
    modules, _ = np.nonzero(geo.layer_mask)
    starts = np.flatnonzero(np.r_[True, np.diff(modules) != 0])
    contiguous = modules[:-1] == modules[1:]
    calibration = np.add.reduceat(cal, starts, axis=-1, dtype=np.float64)
    real_modules = np.add.reduceat(test, starts, axis=-1, dtype=np.float64)
    real_stats = section_statistics(test, contiguous)
    thresholds = []
    for rows in calibration:
        thresholds.append(
            [
                (i, np.quantile(col[col > 0], 0.95))
                for i, col in enumerate(rows.T)
                if (col > 0).sum() >= 500
            ]
        )
    details, by_channel, matched = [], [], {}
    for seed in SEEDS:
        for kind in ("empirical", "direct", "gmm"):
            model = (
                empirical(train, totals, seed)
                if kind == "empirical"
                else generate(species, kind, seed, totals)
            )
            assert np.isfinite(model).all() and (model >= 0).all()
            np.testing.assert_allclose(model.sum(-1), totals, rtol=2e-5, atol=1e-6)
            response = np.add.reduceat(model, starts, axis=-1, dtype=np.float64)
            stats = section_statistics(model, contiguous)
            contrasts = []
            for g, cuts in enumerate(thresholds):
                indices = [m for m, t in cuts]
                limits = np.array([t for m, t in cuts])
                above = response[g][:, indices] > limits
                ref = real_modules[g][:, indices] > limits
                diff = above.astype(np.int8) - ref.astype(np.int8)
                contrasts.append(diff)
                for j, (m, t) in enumerate(cuts):
                    by_channel.append(
                        dict(
                            seed=seed,
                            method=kind,
                            direction=g + 1,
                            module=geo.modules[m].module_id,
                            threshold=t,
                            n=len(ref),
                            ref_above=int(ref[:, j].sum()),
                            model_above=int(above[:, j].sum()),
                            ref_positive=int((real_modules[g, :, m] > 0).sum()),
                            model_positive=int((response[g, :, m] > 0).sum()),
                        )
                    )
                row = dict(
                    seed=seed,
                    method=kind,
                    direction=g + 1,
                    n=len(ref),
                    eligible=len(indices),
                    tail_mae_pp=100 * np.abs(diff.mean(0)).mean(),
                )
                for key, col in [("occupied", 1), ("neighbours", 5), ("positive", 0)]:
                    a, b = real_stats[g, :, col], stats[g, :, col]
                    row[f"{key}_w1"] = wasserstein_distance(a, b)
                    row[f"{key}_ref_mean"] = a.mean()
                    row[f"{key}_model_mean"] = b.mean()
                details.append(row)
            if kind != "empirical":
                matched[(kind, seed)] = contrasts
    # Shared row bootstrap for the signed difference in absolute tail errors.
    rng = np.random.default_rng(83291)
    bootstrap = np.zeros(400)
    for g in range(len(totals)):
        n = len(totals[g])
        weights = (
            rng.multinomial(n, np.full(n, 1 / n), size=len(bootstrap)).astype(
                np.float64
            )
            / n
        )
        for seed in SEEDS:
            for method, sign in [("gmm", 1), ("direct", -1)]:
                gap = weights @ matched[(method, seed)][g].astype(np.float64)
                bootstrap += (
                    sign * 100 * np.abs(gap).mean(1) / (len(totals) * len(SEEDS))
                )
    write_csv(f"{species}_directions.csv", details)
    write_csv(f"{species}_modules.csv", by_channel)
    summary = {}
    for method in ["empirical", "direct", "gmm"]:
        summary[method] = {
            key: [
                np.mean(
                    [
                        row[key]
                        for row in details
                        if row["method"] == method and row["seed"] == s
                    ]
                )
                for s in SEEDS
            ]
            for key in [
                "tail_mae_pp",
                "occupied_w1",
                "neighbours_w1",
                "positive_w1",
                "positive_model_mean",
                "positive_ref_mean",
            ]
        }
    gaps = np.subtract(summary["gmm"]["tail_mae_pp"], summary["direct"]["tail_mae_pp"])
    summary["gmm_minus_direct"] = dict(
        by_seed=gaps.tolist(),
        mean=gaps.mean(),
        paired_bootstrap_percentiles=np.quantile(bootstrap, [0.025, 0.975]).tolist(),
    )
    return summary


def plot(results):
    metrics = [
        ("tail_mae_pp", "Module tail MAE [pp]"),
        ("occupied_w1", "Occupancy W1 [sections]"),
        ("neighbours_w1", "Longitudinal neighbour W1 [pp]"),
    ]
    names = ["Empirical", "Direct WGAN", "GMM WGAN"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.1), layout="constrained")
    for i, (particle, results_row) in enumerate(results.items()):
        for ax, (key, label) in zip(axes[i], metrics, strict=True):
            for y, (method, color) in enumerate(
                zip(
                    ("empirical", "direct", "gmm"),
                    ("#456", "teal", "#944675"),
                    strict=True,
                )
            ):
                data = results_row[method][key]
                ax.plot([min(data), max(data)], [y, y], color=color)
                ax.scatter(data, [y - 0.07, y, y + 0.07], s=27, color=color)
            ax.set_xlim(left=0)
            ax.set_ylim(2.5, -0.5)
            ax.set_yticks(range(3), names if ax == axes[i, 0] else [])
            ax.set_xlabel(label)
            ax.set_title(
                ("Positive muons" if particle == "muon" else "Protons")
                if ax == axes[i, 0]
                else ""
            )
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="x", alpha=0.2)
    fig.savefig(REPORT / "fixed_total_comparison.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    torch.set_num_threads(1)
    results = {species: evaluate(species) for species in ["muon", "proton"]}
    (REPORT / "summary.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )
    plot(results)
