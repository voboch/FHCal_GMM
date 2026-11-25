"""Conditional energy distributions for each instrumented module, depth and detector total."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


METHODS = {
    "reference": ("Transport MC", "#161616", "-", "MC"),
    "conditional_raw": ("Empirical, same direction", "#007e76", "--", "E"),
    "pooled_raw": ("Empirical, pooled directions", "#797079", ":", "P"),
    "raw_bank": ("Empirical subset", "#cd7100", "-.", "S"),
    "smooth": ("Integrated GMM", "#1669af", (0, (3, 1, 1, 1)), "G"),
    "stochastic": ("GMM + multinomial sampling", "#b52e76", (0, (7, 2)), "M"),
    "flow_shape": ("CFM + empirical shape", "#6153a8", (0, (4, 2, 1, 2)), "F"),
}
SPECIES_TITLES = {
    "proton": "Протоны", "protons": "Протоны", "2212": "Протоны",
    "muon": "Мюоны", "muons": "Мюоны", "-13": "Мюоны", "13": "Мюоны",
}
HISTOGRAM_BINS = 40


def _eligible(value):
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def signal_bins(samples, threshold=None):
    """Use common equal-width bins that cover all observations and the marked cut."""
    maximum = max(float(values.max()) for values in samples if len(values))
    right = max(maximum, threshold or 0.)
    return np.linspace(0, right if right > 0 else 1., HISTOGRAM_BINS + 1)


def percent_per_bin(values, edges):
    counts, _ = np.histogram(values, bins=edges)
    if counts.sum() != values.size:
        raise ValueError("Histogram range does not cover the sample")
    return counts * (100. / values.size)


def english_label(label):
    return label.replace("Детектор", "Full detector").replace("Слой", "Layer").replace("Модуль", "Module")


def _plot_panel(fig, slot, label, samples, rows):
    ax = fig.add_subplot(slot)
    ref = rows["reference"]
    threshold = float(ref["threshold"]) if ref["threshold"] not in (None, "") else None
    edges = signal_bins(samples.values(), threshold)
    for name, (_, color, style, _) in METHODS.items():
        ax.stairs(percent_per_bin(samples[name], edges), edges, color=color, ls=style, lw=1.2)
    if threshold is not None:
        ax.axvline(threshold, color="#705828", lw=1., ls=":")
    ax.set(xlim=(0, edges[-1]), ylim=(0, None), xlabel="Recorded signal [simulation units]",
           ylabel="Entries per bin [%]")
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{english_label(label)}   |   positive MC signals: {ref['n_positive']}/{ref['n']}",
                 fontsize=10, loc="left")


def render_species(pdf_path: Path, species: str, groups: list[dict], summary: list[dict]):
    """Plot ordinary signal histograms for every direction and observable."""
    plt.switch_backend("Agg")
    if not groups:
        raise ValueError(f"No momentum directions for {species}")
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    rows = {(int(row["direction"]), row["observable"], row["method"]): row
            for row in summary if row["species"] == species}
    species_title = {"muon": "Positive muons", "proton": "Protons"}.get(species, species)
    handles = [Line2D([], [], color=color, ls=style, lw=1.4, label=f"{code}   {name}")
               for name, color, style, code in METHODS.values()]
    with plt.rc_context({"pdf.fonttype": 42}), PdfPages(
        pdf_path, metadata={"Title": f"FHCal · {species_title} · Signal histograms"}
    ) as pdf:
        for group in groups:
            direction = int(group["direction"])
            labels = group["labels"]
            arrays = group["samples"]
            indices = [[0, *range(1, 6)], list(range(6, 11))]
            indices.extend(list(range(i, min(i + 6, len(labels)))) for i in range(11, len(labels), 6))
            for page in indices:
                fig = plt.figure(figsize=(13.3, 8.75))
                slots = fig.add_gridspec(3, 2, left=.045, right=.983, bottom=.167, top=.785,
                                         hspace=.76, wspace=.24)
                fig.text(.045, .952, f"{species_title}  /  direction {direction}  /  "
                         f"{english_label(labels[page[0]])} – {english_label(labels[page[-1]])}", fontsize=15, weight="bold")
                p = ", ".join(f"{value:.3f}" for value in group["momentum"])
                sizes = " · ".join(f"{METHODS[name][3]} {len(arrays[name])}" for name in METHODS)
                fig.text(.045, .914, f"Recorded momentum ({p}). Entries per method: {sizes}.",
                         fontsize=9)
                fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.04, .897), ncol=4,
                           frameon=False, fontsize=8.5, handlelength=2.7, columnspacing=2.2)
                for index, column in enumerate(page):
                    values = {name: arrays[name][:, column] for name in METHODS}
                    comparison = {name: rows[direction, labels[column], name] for name in METHODS}
                    _plot_panel(fig, slots[index // 2, index % 2], labels[column], values, comparison)
                fig.text(.045, .07, "Each histogram sums to 100% across 40 equal-width bins. Zero signals enter the first bin. "
                         "Bin edges are shared by all methods within a panel.", fontsize=9)
                fig.text(.045, .038, "The dotted vertical line marks the 95th percentile of positive calibration signals, "
                         "when defined. Signal values use the original simulation scale.", fontsize=9)
                pdf.savefig(fig)
                plt.close(fig)


def plot_tail_comparison(directory: Path, output: Path):
    """Compare the positive-response exceedance rate at each fixed calibration cut."""
    import csv

    cuts = (("muon", 2, "Модуль 8"), ("proton", 3, "Модуль 26"),
            ("proton", 4, "Детектор"), ("proton", 5, "Детектор"))
    order = ("reference", "raw_bank", "smooth", "stochastic", "conditional_raw", "flow_shape")
    names = ("Симуляция", "Малый банк", "Гладкая смесь", "Порционная смесь",
             "Большой условный банк", "FM суммы + форма соседа")
    rows = {}
    for species in {cut[0] for cut in cuts}:
        with (directory / f"{species}_tails.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                rows[species, int(row["direction"]), row["observable"], row["method"]] = row
    plt.switch_backend("Agg")
    fig, axes = plt.subplots(2, 2, figsize=(10.3, 6.9), sharex=True, sharey=True)
    fig.subplots_adjust(left=.209, right=.979, top=.83, bottom=.188, wspace=.15, hspace=.53)
    fig.text(.028, .95, "Хвост положительного энерговыделения", fontsize=18, weight="bold")
    fig.text(.028, .906, "Доля превышений среди откликов E > 0, %. Порог — 95-й процентиль положительных откликов на калибровке.",
             fontsize=11)
    upper = max(100 * float(rows[*cut, method]["tail_pos"]) for cut in cuts for method in order)
    limit = np.ceil((upper + 2) / 5) * 5
    for ax, cut in zip(axes.flat, cuts, strict=True):
        species, direction, observable = cut
        reference = rows[*cut, "reference"]
        threshold = float(reference["threshold"])
        ax.set_title(f"{SPECIES_TITLES[species]} · направление {direction} · {observable.lower()}",
                     loc="left", fontsize=11, weight="bold", pad=20)
        ax.text(0, 1.045, f"Порог {threshold:.2f}".replace(".", ",") +
                f"   ·   в симуляции {reference['n_above']} из {reference['n_positive']} выше порога",
                transform=ax.transAxes, fontsize=8.5, color="#414141")
        ax.axvline(5, color="#86724f", lw=1.25, linestyle=(0, (3, 3)), zorder=1)
        for i, method in enumerate(order):
            row = rows[*cut, method]
            n_pos, n_above = int(row["n_positive"]), int(row["n_above"])
            if not _eligible(row["eligible"]) or n_pos <= 0:
                raise ValueError(f"Unavailable tail estimate for {cut}, {method}")
            y = len(order) - i - 1
            rate = 100 * n_above / n_pos
            if not np.isclose(rate, 100 * float(row["tail_pos"])):
                raise ValueError(f"Inconsistent positive-response rate for {cut}, {method}")
            color = METHODS[method][1]
            last = rate
            if method == "reference":
                low = 100 * float(row["ref_cond_low95"])
                last = 100 * float(row["ref_cond_high95"])
                ax.errorbar(rate, y, xerr=[[rate - low], [last - rate]], color=color,
                            lw=1.3, capsize=3.3, zorder=3)
            ax.plot(rate, y, "o", ms=5.8, color=color, zorder=4)
            ax.annotate(f"{rate:.2f}".replace(".", ","), (last, y), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=8.4, color=color,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": .1})
        ax.set(yticks=list(range(len(order) - 1, -1, -1)), yticklabels=names,
               xlim=(-.6, limit), ylim=(-.50, len(order) - .48),
               xticks=np.arange(0, limit + .01, 5))
        ax.tick_params(axis="y", labelsize=9, length=0)
        ax.tick_params(axis="x", labelsize=8, labelbottom=True)
        ax.grid(axis="y", alpha=.13, lw=.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.text(.028, .099, "Пунктир — номинальные 5 %. Отрезки у симуляции — поточечные 95%-интервалы "
             "Клоппера–Пирсона при независимости строк и фиксированном пороге.", fontsize=8.4)
    fig.text(.028, .064, "Для генераторов показаны оценки без интервалов. Порционные повторы используют "
             "одни и те же записи малого банка. Пороги даны в единицах экспорта.", fontsize=8.4)
    fig.text(.028, .03, "Срезы иллюстрируют расхождения. Остальные модули, слои и направления показаны "
             "в атласах распределений.", fontsize=8.4)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot tail probabilities from the two single-particle CSV files.")
    parser.add_argument("directory", type=Path, help="Directory containing muon_tails.csv and proton_tails.csv")
    parser.add_argument("--output", type=Path, help="Default: tail_comparison.png in the input directory")
    arguments = parser.parse_args()
    plot_tail_comparison(arguments.directory, arguments.output or arguments.directory / "tail_comparison.png")
