"""Render scientific figures from the archived proton decoder analysis."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FuncFormatter, NullLocator
import numpy as np

from pipeline.layered_projection import SamplingStack
from plot_gaussian_projection import PILOT, decimal_label, save, set_style


DIRECTORY = PILOT / "proton_pilot" / "conditional_baseline"
STYLES = {
    "reference": ("Симуляция, контроль", "#1d2327", "-", "o"),
    "raw_bank": ("Исходный банк", "#687278", ":", "s"),
    "smooth": ("Гладкий", "#a34d1e", "--", "D"),
    "stochastic": ("Стохастический", "#225f9c", "-.", "o"),
}


def occupancy(analysis, output):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.48), sharey=True)
    fig.subplots_adjust(left=.10, right=.98, top=.88, bottom=.30, wspace=.19)
    data = analysis["occupancy_distributions"]
    titles = ("(a) Ненулевые секции", "(b) Секции с долей ≥ 1 %")
    max_positive = max(max(data["positive_sections"][method]["value"]) for method in STYLES if method != "smooth")
    limit = 10 * np.ceil(max_positive / 10)
    for ax, (metric, distributions), title in zip(axes, data.items(), titles, strict=True):
        ax.set_title(title, loc="left")
        ax.set_xlabel("Число секций в строке")
        ax.set_ylim(0, 1.03)
        upper = limit if metric == "positive_sections" else max(max(group["value"]) for group in distributions.values()) + 1
        for method, (label, color, line, _) in STYLES.items():
            distribution = distributions[method]
            x = np.r_[0, distribution["value"], max(upper, distribution["value"][-1])]
            y = np.r_[0, np.cumsum(distribution["count"]) / sum(distribution["count"]), 1]
            ax.step(x, y, where="post", color=color, linestyle=line, linewidth=1.3, label=label)
        ax.set_xlim(0, upper)
    low, high = min(data["positive_sections"]["smooth"]["value"]), max(data["positive_sections"]["smooth"]["value"])
    if low > limit:
        axes[0].text(.05, .94, f"Гладкий вне шкалы\n{low}–{high} секций", transform=axes[0].transAxes,
                     ha="left", va="top", fontsize=7, color=STYLES["smooth"][1])
    axes[0].set_ylabel("Накопленная доля откликов")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: decimal_label(value, 1)))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.53, .01), frameon=False, ncol=4)
    save(fig, output, "occupancy_cdf")


def direction_bias(analysis, output):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.63))
    fig.subplots_adjust(left=.10, right=.98, top=.87, bottom=.30, wspace=.34)
    means = analysis["mean_by_direction"]
    direction = np.arange(1, len(means["reference"]) + 1)
    panels = (("sections_ge_1pct", "(a) Секции с долей ≥ 1 %", "Разность, секции"),
              ("neighbours_pct", "(b) Секции с продольным соседом", "Разность долей, п.п."))
    for ax, (metric, title, unit) in zip(axes, panels, strict=True):
        ref = np.array([row[metric] for row in means["reference"]])
        ax.axhline(0, color=STYLES["reference"][1], linewidth=.7)
        for offset, (method, (label, color, _, marker)) in zip((-.16, 0, .16), list(STYLES.items())[1:], strict=True):
            values = np.asarray([row[metric] for row in means[method]])
            ax.plot(direction + offset, values - ref, color=color, marker=marker, markerfacecolor="none",
                    linestyle="none", markersize=5, label=label)
        ax.set_title(title, loc="left")
        ax.set_xlim(.55, len(direction) + .45)
        ax.set_xticks(direction)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("Группа направления")
        ax.set_ylabel(unit)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.53, .02), frameon=False, ncol=3)
    save(fig, output, "direction_bias")


def example_maps(analysis, output):
    example = analysis["example"]
    arrays = np.asarray(example["section_shares_pct"], dtype=float)
    active_layers = np.flatnonzero(np.isfinite(arrays[0]).any(axis=0))[-1] + 1
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 3.63), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.98, bottom=.23, top=.93, wspace=.15, hspace=.35)
    colors = ["#ffffff", "#dce9f2", "#a1c0da", "#538bb9", "#155184"]
    cmap = ListedColormap(colors)
    cmap.set_bad("#c0c0c0")
    norm = BoundaryNorm(np.arange(-.5, 5), len(colors))
    labels = ("(a) Исходная строка", "(b) Гладкий", "(c) Стохастический, реализация 1",
              "(d) Стохастический, реализация 2")
    for i, (ax, values, label) in enumerate(zip(axes.flat, arrays, labels, strict=True)):
        values = values[:, :active_layers].T
        intervals = np.searchsorted([0.0, 0.1, 1.0, 10.0], np.nan_to_num(values), side="right")
        intervals[values == 0] = 0
        grid = np.ma.masked_where(~np.isfinite(values), intervals)
        rows, columns = values.shape
        image = ax.pcolormesh(np.arange(columns + 1), np.arange(rows + 1), grid, cmap=cmap,
                              norm=norm, edgecolors="#c6cbd0", linewidth=.35)
        ax.set_xticks(np.arange(columns) + .5, labels=example["module_ids"])
        ax.set_yticks(np.arange(rows) + .5, labels=np.arange(1, rows + 1))
        ax.set_title(f"{label}  ·  {example['n_positive_all_sections'][i]} ненул.", loc="left")
        ax.tick_params(which="both", top=False, right=False)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        if i % 2 == 0:
            ax.set_ylabel("Секция по глубине")
        if i >= 2:
            ax.set_xlabel("Номер модуля")
    cax = fig.add_axes((.23, .078, .60, .034))
    colorbar = fig.colorbar(image, cax=cax, orientation="horizontal", ticks=np.arange(len(colors)))
    colorbar.set_ticklabels(["0", "(0; 0,1)", "[0,1; 1)", "[1; 10)", "≥ 10"])
    colorbar.set_label("Доля полного сигнала в секции, %", labelpad=1)
    cax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True, length=0)
    save(fig, output, "section_maps")


def calibration_scan(calibration, output):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.35))
    fig.subplots_adjust(left=.10, right=.98, top=.87, bottom=.31, wspace=.28)
    rows = calibration["validation_grid"]
    counts = [row["packets"] for row in rows]
    selected = calibration["selected_packets_per_component"]
    metrics = ("positive_sections", "sections_ge_1pct")
    titles = ("(a) Ненулевые секции", "(b) Секции с долей ≥ 1 %")
    for ax, metric, title in zip(axes, metrics, titles, strict=True):
        ax.set_title(title, loc="left")
        ax.set_xscale("log", base=2)
        ax.set_xticks(counts, labels=counts)
        values = [row["w1_macro"][metric] for row in rows]
        ax.plot(counts, values, color=STYLES["stochastic"][1], marker="o", markerfacecolor="white",
                linewidth=1.1, markersize=4)
        rejected = [i for i, row in enumerate(rows) if row["guard_violation"] > 0]
        ax.plot([counts[i] for i in rejected], [values[i] for i in rejected], color="#a34d1e",
                marker="x", linestyle="none", markersize=5)
        ax.plot(selected, values[counts.index(selected)], color=STYLES["stochastic"][1], marker="o", markersize=5)
        ax.set_xlabel("Порций на компоненту")
        ax.set_ylabel(r"$W_1$, секции")
    baseline = calibration["validation_smooth_w1"][metrics[1]]
    axes[1].axhline(baseline, color=STYLES["smooth"][1], linestyle="--", linewidth=.8)
    axes[1].axhline(baseline + calibration["validation_guards"][metrics[1]], color="#565e63", linestyle=":", linewidth=1)
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: decimal_label(value, 0)))
    save(fig, output, "packet_calibration")


def layered_response(data, output):
    fig, (geometry, contrast) = plt.subplots(1, 2, figsize=(7.1, 2.45))
    fig.subplots_adjust(left=.11, right=.985, bottom=.29, top=.84, wspace=.70)
    stack = SamplingStack(**data["stack"])
    for y, mass_depth in enumerate((True, False)):
        lower, upper = stack.active_intervals(1, mass_depth=mass_depth)
        geometry.broken_barh([(0, 1)], (y - .18, .36), facecolors="#dadde0")
        geometry.broken_barh(list(zip(lower[0], (upper - lower)[0], strict=True)), (y - .18, .36),
                             facecolors="#225f9c", linewidth=.30, edgecolors="#225f9c")
    geometry.set_title("(a) Активные пластины одной секции", loc="left")
    geometry.set_yticks([0, 1], ["Массовая", "Линейная"])
    geometry.set_ylim(-.5, 1.5)
    geometry.set_xlim(0, 1)
    geometry.set_xlabel("Глубина, доли секции")
    geometry.xaxis.set_major_formatter(FuncFormatter(lambda v, _: decimal_label(v, 1)))
    geometry.yaxis.set_minor_locator(NullLocator())

    modes = ("full_section", "active_length")
    keys = ("mass_minus_full", "mass_minus_length")
    metrics = ("neighbours_pct", "neighbours_ge_1pct_pct")
    contrast.axvline(0, color="#687278", linewidth=.8)
    for i, (control, key) in enumerate(zip(modes, keys, strict=True)):
        for j, metric in enumerate(metrics):
            effect = data["results"]["active_mass_depth"]["w1"][metric] - data["results"][control]["w1"][metric]
            lower, upper = data["paired_ci95_neighbours"][key][metric]
            y = 1 - i + (j - .5) * .27
            color, marker = (("#225f9c", "o"), ("#a34d1e", "s"))[j]
            contrast.hlines(y, lower, upper, color=color, linewidth=1)
            contrast.plot(effect, y, marker=marker, color=color, markersize=4.4, linestyle="none",
                          label=("Ненулевые", "С долей ≥ 1 %")[j] if i == 0 else None)
    contrast.set_title("(b) Ошибка продольного соседства", loc="left")
    contrast.set_yticks([0, 1], ["Масса − пластины\nв линейной глубине", "Масса − целая\nсекция"])
    contrast.set_ylim(-.39, 1.39)
    contrast.set_xlabel(r"Разность $W_1$, п.п.")
    contrast.xaxis.set_major_formatter(FuncFormatter(lambda v, _: decimal_label(v, 1)))
    contrast.yaxis.set_minor_locator(NullLocator())
    fig.legend(*contrast.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.76, .009), ncol=2, frameon=False)
    save(fig, output, "layered_projection")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DIRECTORY)
    args = parser.parse_args()
    data = json.loads((args.directory / "decoder_analysis.json").read_text())
    calibration = (args.directory / "stochastic_decoder.json").read_bytes()
    if sha256(calibration).hexdigest() != data["calibration_sha256"]:
        raise ValueError("Analysis and calibration exports differ")
    set_style()
    target = args.directory / "decoder_figures"
    target.mkdir(exist_ok=True)
    occupancy(data, target)
    direction_bias(data, target)
    example_maps(data, target)
    calibration_scan(json.loads(calibration), target)
    layered = args.directory / "layered_projection.json"
    if layered.is_file():
        payload = json.loads(layered.read_text())
        if payload["calibration_sha256"] != sha256(calibration).hexdigest():
            raise ValueError("Layered projection used a different calibration")
        layered_response(payload, target)


if __name__ == "__main__":
    main()
