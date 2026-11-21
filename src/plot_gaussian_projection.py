"""Build the FHCal scientific figures and report table from the archived fits."""

import argparse
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
from matplotlib.ticker import FuncFormatter, NullLocator

from pipeline.projection_geometry import load_projection_geometry


PILOT = Path(__file__).resolve().parent / "gaussian_projection"
WIDTH_RUNS = (
    ("pilot_results.json", "Исходные минимумы ширин"),
    ("width_sensitivity.json", "Уменьшенные минимумы ширин"),
)
METHODS = {
    "integrated": ("Интеграл по секции", "#245b92", "o", "-"),
    "center": ("Центр × объём", "#262626", "s", "--"),
}


def read_verified_export(name, evidence, pilot_dir):
    payload = (pilot_dir / name).read_bytes()
    if sha256(payload).hexdigest() != evidence["exports"][name]:
        raise ValueError(f"Unexpected version of {name}")
    return json.loads(payload)


def load_exports(evidence, pilot_dir):
    runs = []
    for name, _ in WIDTH_RUNS:
        run = read_verified_export(name, evidence, pilot_dir)
        ks = run["settings"]["components"]
        fits = {(row["projection"], row["k"]): row for row in run["pooled"]["fits"]}
        if fits.keys() != set(product(METHODS, ks)):
            raise ValueError(f"Incomplete fit table in {name}")
        if any(v["n"] != run["pooled"]["n_events"] for v in fits.values()):
            raise ValueError(f"Fit counts disagree in {name}")
        run["fits"] = fits
        runs.append(run)

    a, b = runs
    for key in ("scope", "source_sha256", "selection", "reference", "geometry"):
        if a[key] != b[key]:
            raise ValueError(f"Cannot compare runs with different {key}")
    fit_settings = [{k: v for k, v in r["settings"].items() if k != "min_scales"} for r in runs]
    if fit_settings[0] != fit_settings[1]:
        raise ValueError("The width comparison also changed the fit settings")
    return runs


def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.65,
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 260,
    })


def plot_methods(ax, run, metric, factor=1):
    ks = run["settings"]["components"]
    for method, (label, color, marker, style) in METHODS.items():
        values = [factor * run["fits"][method, k][metric]["mean"] for k in ks]
        ax.plot(ks, values, color=color, linestyle=style, marker=marker,
                markersize=3.1 if method == "center" else 5.4,
                linewidth=0.95, markerfacecolor="none", label=label)


def format_axes(axes, run):
    ks = run["settings"]["components"]
    for ax in axes:
        ax.set_xticks(ks, labels=ks)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(min(ks) - .22, max(ks) + .22)
        ax.set_xlabel("Число компонент " + r"$K$")
        ax.tick_params(which="major", length=3.5)
        ax.tick_params(which="minor", length=2)


def set_context(fig, run, detail=None):
    n = run["pooled"]["n_events"]
    pdg = str(run["selection"]["pdg"]).replace("-", "−")
    fig.text(.09, .94, f"BM@N FHCal · симуляция · PDG {pdg}, общий импульс · подгонки по {n} строкам",
             fontsize=8, ha="left")
    if detail:
        fig.text(.09, .867, detail, fontsize=8, ha="left")


def decimal_label(value, digits):
    return f"{value:.{digits}f}".replace(".", ",").replace("-", "−")


def save(fig, output_dir, stem):
    for ext in ("pdf", "png"):
        metadata = {"CreationDate": None, "ModDate": None} if ext == "pdf" else None
        fig.savefig(output_dir / f"{stem}.{ext}", metadata=metadata)
    plt.close(fig)


def plot_width_comparison(runs, output_dir):
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(7.1, 2.66))
    fig.subplots_adjust(left=.09, right=.985, bottom=.22, top=.81, wspace=.15)
    set_context(fig, runs[0])
    for i, (ax, run, (_, title)) in enumerate(zip(axes, runs, WIDTH_RUNS)):
        plot_methods(ax, run, "tv")
        ax.set_title(f"({chr(97+i)}) {title}", loc="left", pad=5)
    format_axes(axes, runs[0])
    maximum = max(row["tv"]["mean"] for run in runs for row in run["fits"].values())
    axes[0].set_ylim(0, maximum * 1.11)
    axes[0].set_ylabel(r"Средняя TV долей по секциям")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: decimal_label(value, 1)))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.52, .008), frameon=False, ncol=2)
    save(fig, output_dir, "tv_by_components")


def plot_paired_and_zeros(run, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9))
    fig.subplots_adjust(left=.11, right=.985, bottom=.21, top=.755, wspace=.42)
    set_context(fig, run, "Уменьшенные минимумы ширин · 95%-е парные бутстреп-интервалы на панели (a)")
    paired, zeros = axes
    paired.set_title("(a) Парная разность ошибок", loc="left", pad=5)
    rows = {row["k"]: row for row in run["pooled"]["paired_tv"]
            if row["contrast"] == "integrated_minus_center"}
    ks = run["settings"]["components"]
    means = np.array([rows[k]["tv_mean"] for k in ks])
    bounds = np.array([rows[k]["tv_interval_95"] for k in ks]).T
    paired.axhline(0, linestyle="--", color="#707070", linewidth=.75, zorder=1)
    paired.errorbar(ks, 100 * means, yerr=100 * np.vstack((means - bounds[0], bounds[1] - means)),
                    linestyle="none", marker="o", markersize=3.8, color="#245b92",
                    linewidth=.95, capsize=2.4, zorder=3)
    paired.set_ylabel(r"$\mathrm{TV}_{\mathrm{int}}-\mathrm{TV}_{\mathrm{ctr}}$, п.п.")
    paired.yaxis.set_major_formatter(FuncFormatter(lambda value, _: decimal_label(value, 1)))
    zeros.set_title("(b) Сигнал в секциях с нулём в референсе", loc="left", pad=5)
    plot_methods(zeros, run, "zero_channel_mass", factor=100)
    zeros.set_ylim(bottom=-.18)
    zeros.axhline(0, color="#707070", linewidth=.55, zorder=1)
    zeros.set_ylabel("Доля суммарного сигнала, %")
    format_axes(axes, run)
    handles, labels = zeros.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.52, .008), frameon=False, ncol=2)
    save(fig, output_dir, "paired_tv_and_zero_signal")


def event_map(ax, modules, values, labels, norm, cmap, title):
    boxes = [Rectangle((m.x - m.width / 2, m.y - m.height / 2), m.width, m.height) for m in modules]
    patch = PatchCollection(boxes, cmap=cmap, norm=norm, edgecolor="#555555", linewidth=.3)
    patch.set_array(values)
    ax.set_facecolor("#e2e2e2")
    ax.add_collection(patch, autolim=True)
    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, loc="left", pad=5)
    for i in labels:
        m = modules[i]
        label = ax.text(m.x, m.y, str(m.module_id), ha="center", va="center", fontsize=7)
        label.set_path_effects([patheffects.withStroke(linewidth=1.7, foreground="white")])
    return patch


def section_percentages(event, mask):
    reference = np.zeros(mask.shape)
    fitted = np.zeros(mask.shape)
    reference[mask] = 100 * np.asarray(event["reference_fractions"])
    fitted[mask] = 100 * np.asarray(event["fit_fractions"])
    np.testing.assert_allclose([reference.sum(), fitted.sum()], 100)
    np.testing.assert_allclose(np.abs(fitted - reference).sum() / 200, event["metrics"]["tv"])
    np.testing.assert_allclose(fitted[(reference == 0) & mask].sum() / 100, event["metrics"]["zero_channel_mass"])
    return reference, fitted


def plot_event_examples(gallery, output_dir):
    geometry = load_projection_geometry()
    if geometry.sha256 != gallery["geometry_sha256"]:
        raise ValueError("Unexpected geometry for the event examples")
    examples = [
        (event, *section_percentages(event, geometry.layer_mask))
        for event in gallery["examples"]
    ]
    largest_module_delta = max(np.abs((fit - ref).sum(axis=1)).max() for _, ref, fit in examples)
    delta_max = max(0.1, np.ceil(10 * largest_module_delta) / 10)
    for event, ref, fit in examples:
        plot_event_example(event, geometry, gallery["k"], ref, fit, delta_max, output_dir)


def plot_event_example(event, geometry, k, reference, fitted, delta_max, output_dir):
    residual = fitted - reference
    top = np.argsort(-reference.sum(axis=1), kind="stable")[:2]
    shown = sorted(i for i in top if reference[i].any())
    if not shown:
        raise ValueError("No recorded response to show")
    xy_reference = reference.sum(axis=1)
    xy_fitted = fitted.sum(axis=1)
    xy_residual = residual.sum(axis=1)
    shared = Normalize(0, 100)
    centered = TwoSlopeNorm(vmin=-delta_max, vcenter=0, vmax=delta_max)

    fig = plt.figure(figsize=(7.1, 4.55))
    layout = fig.add_gridspec(2, 1, left=.12, right=.96, bottom=.16, top=.84,
                              height_ratios=(1.40, 1), hspace=.48)
    maps = layout[0].subgridspec(2, 3, height_ratios=(1, .075), hspace=.62, wspace=.24)
    profiles = layout[1].subgridspec(2, len(shown), height_ratios=(2.0, 1), hspace=.16, wspace=.37)
    if event["quantile"] is None:
        selection = f"{event['positive_sections']} ненулевых секций"
        stem = "example_event_max_occupancy"
    else:
        percentile = round(100 * event["quantile"])
        selection = f"около {percentile}-го процентиля TV"
        stem = "example_event_projection" if percentile == 50 else f"example_event_q{percentile}"
    fig.text(.11, .962,
             f"BM@N FHCal · симуляция · строка NPZ № {event['event_index']} · {selection}",
             va="top", fontsize=8)
    tv = decimal_label(event["metrics"]["tv"], 4)
    fig.text(.11, .925, f"K = {k} · TV = {tv} по {geometry.layer_mask.sum()} секциям · сумма отклика задана",
             va="top", fontsize=8)
    inputs = (
        ("(a) Симуляция", xy_reference, shared, "Blues"),
        ("(b) Интегральная смесь", xy_fitted, shared, "Blues"),
        ("(c) Смесь − референс", xy_residual, centered, "RdBu_r"),
    )
    for j, (title, values, norm, cmap) in enumerate(inputs):
        ax = fig.add_subplot(maps[0, j])
        patch = event_map(ax, geometry.modules, values, shown, norm, cmap, title)
        ax.set_xticks([-60, 0, 60])
        ax.set_yticks([-40, 0, 40])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlabel(r"$x$, см", labelpad=1)
        if j == 0:
            ax.set_ylabel(r"$y$, см")
        else:
            ax.tick_params(labelleft=False)
        if j != 0:
            cax = fig.add_subplot(maps[1, :2] if j == 1 else maps[1, 2])
            scale = fig.colorbar(patch, cax=cax, orientation="horizontal")
            ticks = [0, shared.vmax / 2, shared.vmax] if j == 1 else [-delta_max, 0, delta_max]
            scale.set_ticks(ticks, labels=[decimal_label(t, 0 if j == 1 else 2) for t in ticks])
            scale.solids.set_rasterized(False)
            scale.solids.set_edgecolor("face")
            scale.set_label("Доля сигнала в модуле, %" if j == 1 else "Разность долей, п.п.", labelpad=0)
            cax.tick_params(axis="x", which="both", top=False, labeltop=False, labelbottom=True, pad=1)
            cax.xaxis.set_minor_locator(NullLocator())

    section_max = max(reference[shown].max(), fitted[shown].max())
    section_delta = max(np.abs(residual[shown]).max(), np.finfo(float).eps)
    for j, i in enumerate(shown):
        profile = fig.add_subplot(profiles[0, j])
        delta = fig.add_subplot(profiles[1, j], sharex=profile)
        module = geometry.modules[i]
        edges = np.arange(module.layers + 1) + .5
        profile.stairs(reference[i, :module.layers], edges, color="#262626", label="Симуляция (референс)")
        profile.stairs(fitted[i, :module.layers], edges, color="#245b92", linestyle="--",
                       label="Интегральная смесь")
        profile.set_title(f"({chr(100+j)}) Модуль {module.module_id}", loc="left", pad=4)
        profile.set_ylim(-.04 * section_max, 1.10 * section_max)
        profile.tick_params(labelbottom=False)
        delta.axhline(0, color="#777777", linewidth=.55)
        delta.stairs(residual[i, :module.layers], edges, color="#245b92")
        delta.set_xticks(np.arange(module.layers) + 1)
        delta.xaxis.set_minor_locator(NullLocator())
        delta.yaxis.set_minor_locator(NullLocator())
        delta.set_ylim(-1.13 * section_delta, 1.13 * section_delta)
        delta.yaxis.set_major_formatter(FuncFormatter(lambda value, _: decimal_label(value, 1)))
        delta.set_xlabel("Номер продольной секции", labelpad=2)
        if j == 0:
            profile.set_ylabel("Доля сигнала, %")
            delta.set_ylabel("Остаток, п.п.")
            handles, labels = profile.get_legend_handles_labels()
        else:
            profile.tick_params(labelleft=False)
            delta.tick_params(labelleft=False)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.53, .018),
               frameon=False, ncol=2)
    save(fig, output_dir, stem)


def plot_optimizer_diagnostics(diagnostic, run, output_dir):
    rows = diagnostic["rows"]
    positive = np.array([row["positive_sections"] for row in rows])
    significant = [row["sections_at_fraction_threshold"] for row in rows]
    before, after = diagnostic["iteration_limits"]
    tv = np.array([row["operators"]["integrated"]["tv"] for row in rows])
    success = np.array([row["operators"]["integrated"]["converged"][0] for row in rows])
    delta = 100 * (tv[:, 1] - tv[:, 0])
    threshold = decimal_label(100 * diagnostic["section_fraction_threshold"], 0)

    fig, (occupancy, replay) = plt.subplots(1, 2, figsize=(7.1, 3.0))
    fig.subplots_adjust(left=.10, right=.985, bottom=.30, top=.755, wspace=.34)
    set_context(fig, run, f"K = {diagnostic['k']} · уменьшенные минимумы ширин · интеграл на панели (b)")
    bins = np.arange(-.5, positive.max() + 5.5, 5)
    occupancy.hist(positive, bins=bins, histtype="step", color="#245b92", label="Ненулевой сигнал")
    occupancy.hist(significant, bins=bins, histtype="step", color="#262626", linestyle="--",
                   label=f"Доля ≥ {threshold} %")
    occupancy.set_title("(a) Занятость по референсу", loc="left", pad=5)
    occupancy.set_xlabel("Число секций")
    occupancy.set_ylabel("Число строк")
    replay.axhline(0, linestyle="--", linewidth=.7, color="#777777", zorder=1)
    replay.scatter(positive[success], delta[success], s=13, facecolors="none", edgecolors="#245b92",
                   linewidths=.8, label=f"Успех при {before}", zorder=3)
    replay.scatter(positive[~success], delta[~success], s=14, marker="x", color="#a74e38",
                   linewidths=.8, label=f"Лимит при {before}", zorder=3)
    replay.set_title("(b) Изменение TV при повторе", loc="left", pad=5)
    replay.set_xlabel("Число ненулевых секций")
    replay.set_ylabel(rf"$\mathrm{{TV}}_{{{after}}}-\mathrm{{TV}}_{{{before}}}$, п.п.")
    replay.yaxis.set_major_formatter(FuncFormatter(lambda value, _: decimal_label(value, 1)))
    for ax in (occupancy, replay):
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.32), ncol=2, frameon=False,
                  columnspacing=.7, handletextpad=.35)
    save(fig, output_dir, "occupancy_and_optimizer")


def plot_component_scan(diagnostic, run, output_dir):
    summaries = diagnostic["component_scan"]["summaries"]
    ks = sorted({row["k"] for row in summaries})
    scan = {
        "settings": {"components": ks},
        "fits": {(row["projection"], row["k"]): {"tv": {"mean": row["tv_mean"]}} for row in summaries},
    }
    fig, ax = plt.subplots(figsize=(6.5, 2.72))
    fig.subplots_adjust(left=.13, right=.985, bottom=.23, top=.74)
    set_context(fig, run, f"Уменьшенные минимумы ширин · лимит {diagnostic['iteration_limits'][-1]} итераций")
    plot_methods(ax, scan, "tv")
    format_axes([ax], scan)
    ax.set_ylabel("Средняя TV долей по секциям")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: decimal_label(value, 2)))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    save(fig, output_dir, "components_extended")


def plot_event_dependence(analysis, run, output_dir):
    rows = analysis["events"]
    active = [r["positive_sections"] for r in rows]
    significant = [r["sections_ge_1pct"] for r in rows]
    fig, (absolute, gain) = plt.subplots(1, 2, figsize=(7.1, 2.92))
    fig.subplots_adjust(left=.10, right=.985, bottom=.24, top=.76, wspace=.33)
    set_context(fig, run, "Интегральная смесь · уменьшенные минимумы · 800 итераций на старт")
    for k, color, marker in ((4, "#245b92", "o"), (8, "#262626", "x")):
        absolute.scatter(active, [r[f"tv_{k}"] for r in rows], s=13, color=color, marker=marker,
                         linewidths=.7, alpha=.69, label=f"K = {k}")
    absolute.set_title("(a) Ошибка и ненулевые секции", loc="left", pad=5)
    absolute.set_xlabel("Число ненулевых секций")
    absolute.set_ylabel("TV по секциям")
    absolute.set_ylim(bottom=0)
    absolute.legend(frameon=False, loc="upper left", ncol=2, handletextpad=.1, columnspacing=.6)
    gain.axhline(0, linestyle="--", linewidth=.7, color="#777777")
    gain.scatter(significant, [r["gain_4_to_8"] for r in rows], s=13, color="#245b92", alpha=.69)
    gain.set_title("(b) Выигрыш при K = 8", loc="left", pad=5)
    gain.set_xlabel("Число секций с долей ≥ 1 %")
    gain.set_ylabel(r"$\mathrm{TV}_4-\mathrm{TV}_8$")
    for ax in (absolute, gain):
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: decimal_label(value, 2)))
    save(fig, output_dir, "event_dependence")


def plot_direction_groups(analysis, output_dir):
    groups = analysis["direction_groups"]
    means = np.array([row["tv_mean"] for row in groups])
    lower, upper = np.array([row["tv_mean_interval"] for row in groups]).T
    fig, ax = plt.subplots(figsize=(6.5, 2.52))
    fig.subplots_adjust(left=.14, right=.98, bottom=.23, top=.75)
    fig.text(.1, .945, "BM@N FHCal · симуляция · PDG 2212 · по 48 строк на направление", fontsize=8)
    fig.text(.1, .858, "Интеграл · K = 6 · уменьшенные минимумы ширин · 800 итераций", fontsize=8)
    ids = [row["group"] for row in groups]
    ax.errorbar(ids, means, yerr=[means - lower, upper - means], color="#245b92",
                linestyle="none", marker="o", markersize=3.8, linewidth=.95, capsize=2.4)
    ax.set_xticks(ids)
    ax.set_xlim(.65, len(ids) + .35)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(0, 1.12 * upper.max())
    ax.set_ylabel("Средняя TV по секциям")
    ax.set_xlabel("Группа записанного направления")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: decimal_label(value, 2)))
    save(fig, output_dir, "momentum_groups")


def replace_report_table(source, name, rows):
    start = f"<!-- BEGIN CANONICAL {name} TABLE -->"
    end = f"<!-- END CANONICAL {name} TABLE -->"
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError(f"RESULTS.md must have one generated {name} table")
    before, tail = source.split(start, maxsplit=1)
    _, after = tail.split(end, maxsplit=1)
    return before + start + "\n\n" + "\n".join(rows) + "\n\n" + end + after


def update_report_tables(runs, gallery, path):
    source = path.read_text()
    header = "| $K$ | Интеграл, исходные | Центр × объём, исходные | Интеграл, уменьшенные | Центр × объём, уменьшенные |"
    band = "| ---: | ---: | ---: | ---: | ---: |"
    rows = [header, band]
    for k in runs[0]["settings"]["components"]:
        means = [decimal_label(run["fits"][method, k]["tv"]["mean"], 4)
                 for run in runs for method in METHODS]
        rows.append(f"| {k} | {' | '.join(means)} |")
    rendered = replace_report_table(source, "TV", rows)
    with_occupancy = any(event["quantile"] is None for event in gallery["examples"])
    columns = "Процентиль TV | Строка NPZ"
    example_band = band
    if with_occupancy:
        columns = "Критерий отбора | Строка NPZ | Ненулевые секции"
        example_band = "| --- | ---: | ---: | ---: | ---: | ---: |"
    examples = [
        f"| {columns} | TV по секциям | Сигнал в нулевых секциях, % | Макс. отклонение секции, п.п. |",
        example_band,
    ]
    for event in gallery["examples"]:
        if event["quantile"] is None:
            label = "Макс. занятость"
        else:
            label = f"{round(100 * event['quantile'])}-й"
            if with_occupancy:
                label += " процентиль TV"
        entry = f"| {label} | {event['event_index']} |"
        if with_occupancy:
            entry += f" {event['positive_sections']} |"
        tv = decimal_label(event["metrics"]["tv"], 4)
        leakage = decimal_label(100 * event["metrics"]["zero_channel_mass"], 2)
        residual = np.asarray(event["fit_fractions"]) - np.asarray(event["reference_fractions"])
        maximum = decimal_label(100 * np.abs(residual).max(), 2)
        examples.append(f"{entry} {tv} | {leakage} | {maximum} |")
    rendered = replace_report_table(rendered, "EXAMPLES", examples)
    if rendered != source:
        path.write_text(rendered)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-only", action="store_true")
    parser.add_argument("--pilot-dir", type=Path, default=PILOT, help="Directory of archived results and report")
    args = parser.parse_args()
    evidence = json.loads((args.pilot_dir / "evidence.json").read_text())
    runs = load_exports(evidence, args.pilot_dir)
    gallery = read_verified_export("example_events.json", evidence, args.pilot_dir)
    if gallery["archive_sha256"] != evidence["exports"]["width_sensitivity.json"]:
        raise ValueError("Event examples do not match the width-sensitivity archive")
    update_report_tables(runs, gallery, args.pilot_dir / "RESULTS.md")
    if not args.table_only:
        set_style()
        output_dir = args.pilot_dir / "figures"
        output_dir.mkdir(exist_ok=True)
        plot_width_comparison(runs, output_dir)
        plot_paired_and_zeros(runs[1], output_dir)
        plot_event_examples(gallery, output_dir)
        if "optimizer_diagnostics.json" in evidence["exports"]:
            diagnostic = read_verified_export("optimizer_diagnostics.json", evidence, args.pilot_dir)
            plot_optimizer_diagnostics(diagnostic, runs[1], output_dir)
            if "component_scan" in diagnostic:
                plot_component_scan(diagnostic, runs[1], output_dir)
        if "event_dependence.json" in evidence["exports"]:
            analysis = read_verified_export("event_dependence.json", evidence, args.pilot_dir)
            plot_event_dependence(analysis, runs[1], output_dir)
            plot_direction_groups(analysis, output_dir)


if __name__ == "__main__":
    main()
