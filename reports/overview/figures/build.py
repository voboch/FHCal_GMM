"""Render FHCal slide figures from the archived samples and fitted generators."""
from contextlib import chdir
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors, ticker

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from conditional_tail_flow import replay_flow
from conditional_shape_wgan import SEEDS, generate
from evaluate_conditional_shapes import empirical
from plot_single_particle_tails import signal_bins, percent_per_bin
from pipeline.projection_geometry import load_projection_geometry

REPORT = ROOT / "reports/single_particle"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "path", "svg.hashsalt": "fhcal-summary"})
STYLES = [
    ("Transport MC", "#171717", "-"),
    ("Empirical (N = 4096)", "#007e76", "--"),
    ("CFM + empirical shape", "#6153a8", "-."),
]


def signal_spectra(species, direction, module):
    with np.load(ROOT / "outputs/single_particle" / f"{species}_samples.npz") as z:
        train, _, reference = np.split(z["data"], 3, axis=1)
        flow = replay_flow(train, REPORT / f"{species}_flow/conditional_tail_flow.pt")
        d = direction - 1
        samples = (reference[d], train[d], flow[d])
    with (REPORT / f"{species}_tails.csv").open() as f:
        records = {r["observable"]: r for r in csv.DictReader(f)
                   if r["direction"] == str(direction) and r["method"] == "reference"}
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.7), sharey=True)
    fig.subplots_adjust(left=.080, right=.992, top=.78, bottom=.24, wspace=.15)
    panels = [(f"Модуль {module}", f"Module {module}", 10 + module),
              ("Слой 2", "Layer 2", 2), ("Детектор", "Full detector", 0)]
    peak = 0.
    for ax, (label, title, index) in zip(axes, panels, strict=True):
        t = float(records[label]["threshold"])
        positive = [sample[:, index][sample[:, index] > 0] for sample in samples]
        edges = signal_bins(positive, t)
        ax.axvline(t, color="#756f64", ls=":", lw=1.6)
        for values, (name, color, linestyle) in zip(positive, STYLES, strict=True):
            frequencies = percent_per_bin(values, edges)
            peak = max(peak, frequencies.max())
            ax.stairs(frequencies, edges, color=color, lw=1.8,
                      ls=linestyle, label=name)
        ax.set(title=title, xlim=(0, edges[-1]))
        ax.set_xlabel("Recorded signal [simulation units]")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.grid(axis="y", alpha=.20)
    axes[0].set_ylim(0, 1.08 * peak)
    axes[0].set_ylabel("Positive entries per bin [%]")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncols=len(STYLES), frameon=False, fontsize=11.5, columnspacing=1.1)
    fig.savefig(HERE / f"{species}.svg", metadata={"Date": None})
    plt.close(fig)


def wgan_spectra(species, direction, module):
    report = ROOT / "reports/conditional_shapes"
    with np.load(ROOT / "outputs/conditional_shapes" / f"{species}_sections.npz") as archive:
        train, calibration, test = archive["sections"].transpose(1, 0, 2, 3)
    with (report / f"{species}_modules.csv").open() as stream:
        checks = {(int(row["seed"]), row["method"]): row for row in csv.DictReader(stream)
                  if int(row["direction"]) == direction and int(row["module"]) == module}
    d = direction - 1
    geo = load_projection_geometry()
    module_idx, layer_idx = np.nonzero(geo.layer_mask)
    masks = (module_idx == [m.module_id for m in geo.modules].index(module), layer_idx == 1)

    def project(maps):
        return [maps[:, mask].sum(axis=1, dtype=np.float64) for mask in masks]

    reference = project(test[d])
    thresholds = [np.quantile(v[v > 0], .95) for v in project(calibration[d])]
    totals = test.sum(axis=-1)
    samples = {"Transport MC": [reference], "Empirical shape": [], "Direct WGAN-GP": [], "GMM WGAN-GP": []}
    for seed in SEEDS:
        for name, key in [("Empirical shape", "empirical"), ("Direct WGAN-GP", "direct"),
                          ("GMM WGAN-GP", "gmm")]:
            with chdir(ROOT):
                maps = empirical(train, totals, seed) if key == "empirical" else generate(species, key, seed, totals)
            np.testing.assert_allclose(maps[d].sum(axis=-1), totals[d], rtol=2e-5, atol=1e-6)
            values = project(maps[d])
            row = checks[seed, key]
            assert np.isclose(thresholds[0], float(row["threshold"]))
            assert np.count_nonzero(values[0] > thresholds[0]) == int(row["model_above"])
            assert np.count_nonzero(reference[0] > thresholds[0]) == int(row["ref_above"])
            samples[name].append(values)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.7))
    fig.subplots_adjust(left=.075, right=.99, top=.77, bottom=.22, wspace=.19)
    styles = [("#171717", "-"), ("#007e76", "--"), ("#225da0", "-."), ("#ac357d", (0, (6, 2)))]
    for j, ax in enumerate(axes):
        edges = signal_bins([run[j] for runs in samples.values() for run in runs], thresholds[j])
        for (name, runs), (color, style) in zip(samples.items(), styles, strict=True):
            hist = np.mean([percent_per_bin(run[j], edges) for run in runs], axis=0)
            np.testing.assert_allclose(hist.sum(), 100.)
            ax.stairs(hist, edges, color=color, ls=style, lw=1.9, label=name)
        ax.axvline(thresholds[j], color="#756f64", ls=":", lw=1.5)
        ax.set(xlim=(0, edges[-1]), ylim=(0, None), xlabel="Recorded signal [simulation units]",
               title=f"Module {module}" if j == 0 else "Layer 2")
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("All entries per bin [%]")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncols=4, frameon=False)
    fig.savefig(HERE / f"{species}_wgan.svg", metadata={"Date": None})
    plt.close(fig)


def total_comparison():
    """Compare the same three spatial generators under two choices of total signal."""
    base = ROOT / "reports"
    fixed = json.loads((base / "conditional_shapes/summary.json").read_text())
    generated = json.loads((base / "generated_total/summary.json").read_text())
    methods = ["empirical", "direct", "gmm"]
    labels = ["Empirical shape", "Direct WGAN-GP", "GMM WGAN-GP"]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 3.65), sharex=True, sharey=True)
    fig.subplots_adjust(left=.15, right=.986, top=.72, bottom=.22, wspace=.13)
    rightmost = 0.
    for ax, species, title in zip(axes, ("muon", "proton"), ("Positive muons", "Protons"), strict=True):
        for j, method in enumerate(methods):
            observed = fixed[species][method]["tail_mae_pp"]
            sampled = generated[species][method]["by_seed_pp"]
            for runs, shift, color, marker, name in [
                (observed, +.13, "#515761", "o", "Observed total"),
                (sampled, -.13, "#116e73", "D", "CFM-generated total"),
            ]:
                average = np.mean(runs)
                rightmost = max(rightmost, max(runs))
                spread = [[average - min(runs)], [max(runs) - average]]
                ax.errorbar(average, 2-j+shift, xerr=spread, color=color, fmt=marker,
                            ms=6, lw=1.8, capsize=3, label=name if j == 0 else None)
        ax.set_title(title)
        ax.set(ylim=(-.46, 2.46), yticks=[2, 1, 0], yticklabels=labels)
        ax.set_xlabel("Mean absolute error in exceedance rate [pp]")
        ax.grid(axis="x", alpha=.2)
    axes[0].set_xlim(0, 1.06 * rightmost)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncols=2, frameon=False)
    fig.savefig(HERE / "generated_total.svg", metadata={"Date": None})
    plt.close(fig)


def event_map():
    base = ROOT / "src/gaussian_projection/proton_pilot"
    item = json.loads((base / "example_events.json").read_text())["examples"][-1]
    mask = load_projection_geometry().layer_mask
    panels = []
    for key in ("reference_fractions","fit_fractions"):
        panel = np.full(mask.shape, np.nan)
        panel[mask] = np.array(item[key])*100
        panels.append(panel)
    fig, axes = plt.subplots(2, 1, figsize=(13.8,3.9), sharex=True)
    fig.subplots_adjust(left=.082,right=.83,bottom=.18,top=.89,hspace=.48)
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#d8dadd")
    scale = colors.PowerNorm(.45, vmin=0, vmax=max(np.nanmax(v) for v in panels))
    for ax, arr, label in zip(axes, panels, ["Transport MC", "Gaussian mixture fit (K = 4)"], strict=True):
        im = ax.imshow(arr.T, origin="lower", aspect="auto", extent=(.5,54.5,.5,10.5),
                       interpolation="nearest", cmap=cmap, norm=scale)
        ax.set_title(label, loc="left",fontsize=14, pad=2)
        ax.set_yticks([1,5,10]); ax.set_ylabel("Layer ID")
    axes[-1].set_xticks([1,5,10,15,20,25,30,35,40,45,50,54])
    axes[-1].set_xlabel("Module ID")
    bar = fig.colorbar(im,cax=fig.add_axes([.865,.22,.016,.59]),ticks=[0,1,5,10,20])
    bar.set_label("Fraction of total signal [%]")
    fig.savefig(HERE/"event.svg",metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    event_map()
    signal_spectra("muon",2,8)
    signal_spectra("proton",3,26)
    torch.set_num_threads(1)
    wgan_spectra("muon", 2, 8)
    wgan_spectra("proton", 3, 26)
    total_comparison()
