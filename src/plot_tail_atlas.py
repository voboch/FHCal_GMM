"""Replay the proton decoder on its archived split and plot every energy projection."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import beta

from analyze_module_energy import BANK
from conditional_gmm_baseline import digest, sample_packet_fractions
from evaluate_gaussian_projection import read_energy_rows
from evaluate_gmm_packet_decoder import bank_probabilities, reference_rows
from pipeline.gaussian_projection import SectionProjector, section_bounds
from pipeline.projection_geometry import legacy_grid_to_modules, load_projection_geometry


STYLES = {
    "reference": ("Симуляция", "#141414", "-"),
    "raw_bank": ("Исходный банк", "#c36200", "--"),
    "smooth": ("Гладкий", "#0072a2", ":"),
    "stochastic": ("Порционный", "#ad4482", "-."),
}


def load_samples(source, geometry):
    calibration = json.loads((BANK / "stochastic_decoder.json").read_text())
    receipt = json.loads((BANK / "results.json").read_text())
    if (digest(BANK / "bank.npz") != calibration["bank_sha256"]
            or digest(BANK / "results.json") != calibration["baseline_receipt_sha256"]):
        raise ValueError("Bank or calibration receipt changed")
    with np.load(BANK / "bank.npz", allow_pickle=False) as data:
        bank = {name: data[name] for name in ("rows", "momenta", "energy", "parameters")}
    bank["components"] = receipt["settings"]["components"]
    indices, reference, _ = reference_rows(source, bank, receipt, geometry)
    split = calibration["split"]
    np.testing.assert_array_equal(indices, [s["validation"] + s["holdout"] for s in split])
    reference = reference[:, len(split[0]["validation"]):]
    with np.load(source, allow_pickle=False) as data:
        size = len(data["ParticlePDG"])
    raw = legacy_grid_to_modules(read_energy_rows(source, bank["rows"].ravel(), size), geometry)
    raw = raw[:, geometry.layer_mask].reshape(*bank["rows"].shape, -1)
    energy = bank["energy"]
    np.testing.assert_allclose(raw.sum(axis=-1), energy)
    projector = SectionProjector(*section_bounds(geometry), "integrated")
    fractions, components, weights = bank_probabilities(bank, projector)
    rng = np.random.default_rng(calibration["seed"] + 2)
    methods = dict(reference=reference, raw_bank=raw, smooth=fractions * energy[..., None])
    draws = [sample_packet_fractions(p, w, calibration["selected_packets_per_component"], rng,
                                     replicas=calibration["replicas_per_code"]) * total[None, :, None]
             for p, w, total in zip(components, weights, energy, strict=True)]
    methods["stochastic"] = np.stack(draws).reshape(len(energy), -1, raw.shape[-1])
    for name in ("raw_bank", "smooth", "stochastic"):
        totals = methods[name].sum(axis=-1)
        np.testing.assert_allclose(totals, np.tile(energy, (1, totals.shape[1] // energy.shape[1])))
    return methods, bank, calibration


def project(methods, geometry, bank):
    modules, layers = np.nonzero(geometry.layer_mask)
    module_selector = np.eye(len(geometry.modules))[modules]
    layer_selector = np.eye(geometry.max_layers)[layers]
    labels = ["Детектор"] + [f"Слой {i + 1}" for i in range(geometry.max_layers)]
    labels += [f"Модуль {m.module_id}" for m in geometry.modules]
    projections = {}
    for name, sections in methods.items():
        by_module = sections @ module_selector
        by_layer = sections @ layer_selector
        total = sections.sum(axis=-1)
        np.testing.assert_allclose(by_module.sum(axis=-1), total)
        np.testing.assert_allclose(by_layer.sum(axis=-1), total)
        # Both decoders preserve the sampled total exactly in the model.
        # Use its stored value after checking the projection up to rounding.
        if name != "reference":
            n = bank["energy"].shape[1]
            total = np.tile(bank["energy"], (1, total.shape[1] // n))
        projections[name] = np.concatenate((total[..., None], by_layer, by_module), axis=-1)
    return labels, projections


def draw_panel(axes, samples, title):
    left, right = axes
    ref = samples["reference"]
    pos = ref[ref > 0]
    xmax = max(np.max(values) for values in samples.values())
    zoom = np.quantile(pos, .9) if len(pos) else xmax
    for name, values in samples.items():
        _, color, style = STYLES[name]
        x, counts = np.unique(values, return_counts=True)
        prob = counts.cumsum() / len(values)
        upper = xmax * 1.04 if xmax > 0 else 1.0
        left.step(np.r_[0, x, upper], np.r_[0, prob, 1], where="post",
                  color=color, ls=style, lw=1.05)
        # Include the last strictly positive step through the observed maximum.
        before_max = 1 - prob[:-1]
        last_step = before_max[-1] if len(before_max) else 1.0
        right.step(np.r_[0, x], np.r_[1, before_max, last_step],
                   where="post", color=color, ls=style, lw=1.05)
        right.plot(x[-1], .00055, marker="v", color=color, markersize=3, clip_on=False)
    left.set(xlim=(0, zoom if zoom > 0 else 1.0), ylim=(-.035, 1.035), yticks=[0, .5, 1])
    if xmax > 0:
        right.set_xlim(0, xmax * 1.04)
    else:
        right.set_xlim(0, 1)
    right.set(yscale="log", ylim=(.00046, 1.35), yticks=[.001, .01, .1, 1])
    left.set_title(f"{title}  ·  E>0: {len(pos)}/{len(ref)}", loc="left", fontsize=9)
    for ax in axes:
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
        ax.tick_params(labelsize=7, pad=2)
        ax.grid(axis="y", alpha=.22)
        ax.spines[["top", "right"]].set_visible(False)



def plot_support(groups, output):
    fig, ax = plt.subplots(figsize=(7.7, 4.0))
    fig.subplots_adjust(left=.12, right=.97, top=.82, bottom=.28)
    fractions = []
    for row in groups:
        hits, size = row["reference_above"], row["reference_rows"]
        rate = hits / size
        lo = beta.ppf(.025, hits, size - hits + 1) if hits else 0
        hi = beta.ppf(.975, hits + 1, size - hits) if hits < size else 1
        group = row["direction"]
        ax.errorbar(group, 100 * rate, yerr=[[100 * (rate - lo)], [100 * (hi - rate)]],
                    fmt="o", color="#15496a", capsize=4)
        ax.annotate(f"{hits}/{size}", (group, 100 * hi), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9)
        fractions.append(hi)
    ax.axhline(0, color="#a54324", ls="--", label="Банк и оба декодера: 0 %")
    ax.set(xlim=(.6, len(groups) + .4), ylim=(-.9, 100 * max(fractions) + 3.8),
           xticks=[r["direction"] for r in groups], xlabel="Направление протона",
           ylabel="Доля строк выше максимума, %")
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    fig.suptitle("Полный сигнал выше максимума своего банка", y=.96, fontsize=15)
    fig.text(.045, .09, "Порог — максимум 32 банковских сумм в каждом направлении. Отрезки — поточечные 95%-интервалы", fontsize=8)
    fig.text(.045, .045, "Клоппера–Пирсона при допущении независимых строк. Архивный контроль использован для разведки.", fontsize=8)
    fig.savefig(output, dpi=155)
    plt.close(fig)


def write_atlas(source, output):
    plt.switch_backend("Agg")
    plt.rcParams["pdf.fonttype"] = 42
    output.mkdir(parents=True, exist_ok=True)
    geometry = load_projection_geometry()
    methods, bank, calibration = load_samples(source, geometry)
    labels, projections = project(methods, geometry, bank)
    summary = []
    maxima = []
    groups = [[0, *range(1, 6)], list(range(6, 11))]
    groups += [list(range(i, min(i + 6, len(labels)))) for i in range(11, len(labels), 6)]
    handles = [Line2D([], [], color=color, ls=style, label=label) for label, color, style in STYLES.values()]
    with PdfPages(output / "proton_tail_atlas.pdf", metadata={"Title": "FHCal — распределения энергии протонов"}) as pdf:
        for g, p in enumerate(bank["momenta"]):
            bound = float(bank["energy"][g].max())
            reference_total = projections["reference"][g, :, 0]
            above = int((reference_total > bound).sum())
            if above != int((reference_total > bound + 1e-5).sum()):
                raise ValueError("An exceedance lies within the numerical precision check")
            maxima.append(dict(direction=g + 1, bank_max=bound, reference_max=float(reference_total.max()),
                               reference_above=above, reference_rows=len(reference_total), numerical_guard=1e-5))
            for group in groups:
                fig = plt.figure(figsize=(12.6, 8.4))
                slots = fig.add_gridspec(3, 2, left=.052, right=.985, bottom=.135, top=.805,
                                         hspace=.58, wspace=.22)
                fig.text(.052, .954, f"Протоны  /  направление {g + 1}  /  " + labels[group[0]] +
                         " — " + labels[group[-1]], fontsize=15, weight="bold")
                fig.text(.052, .912, "Записанный импульс (" + ", ".join(f"{v:.3f}" for v in p) +
                         ") в единицах экспорта. Референс: 256 строк. Банк: 32 записи. Порционный декодер: 32 повтора на запись.", fontsize=9)
                fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.046, .885),
                           ncol=4, frameon=False, fontsize=9)
                for j, index in enumerate(group):
                    values = {name: data[g, :, index] for name, data in projections.items()}
                    pair = slots[j // 2, j % 2].subgridspec(1, 2, wspace=.28)
                    draw_panel([fig.add_subplot(pair[0, k]) for k in range(2)], values, labels[index])
                    for name, arr in values.items():
                        pos = arr[arr > 0]
                        summary.append(dict(direction=g + 1, observable=labels[index], method=name,
                                            rows=len(arr), parent_rows=len(arr) if name == "reference" else len(bank["energy"][g]),
                                            n_positive=len(pos), p_zero=float((arr == 0).mean()),
                                            positive_q90=float(np.quantile(pos, .9)) if len(pos) else "",
                                            max_energy=float(arr.max())))
                fig.text(.052, .075, "В каждой паре слева P(E ≤ x) до q90 положительного референса (при отсутствии попаданий — до общего максимума), справа P(E > x) на полной шкале.", fontsize=8.3)
                fig.text(.052, .051, "x — сигнал в единицах экспорта. Треугольник под кривой отмечает максимум выборки, где превышений уже 0. Число E>0 в заголовке относится к симуляции.", fontsize=8.3)
                fig.text(.052, .026, "Сумма банка и обоих декодеров совпадает по построению. Слои 1–7 суммируют 54 модуля, слои 8–10 — 20. Контроль ранее просматривался.", fontsize=8.3)
                pdf.savefig(fig)
                plt.close(fig)
    plot_support(maxima, output / "total_support.png")
    with (output / "proton_tail_counts.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (output / "proton_total_support.json").write_text(json.dumps({
        "source_sha256": calibration["source_sha256"], "bank_sha256": calibration["bank_sha256"],
        "geometry_sha256": geometry.sha256, "decoder_seed": calibration["seed"] + 2, "groups": maxima,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_atlas(args.source, args.output)
