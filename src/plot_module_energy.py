"""Plot the archived energy-response comparison at event and module level."""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_gaussian_projection import PILOT, decimal_label, save, set_style


PROTON = PILOT / "proton_pilot"
DECODER = PROTON / "conditional_baseline"
METHODS = {
    "reference": ("Симуляция", "#20252a", "-"),
    "raw_bank": ("Исходный банк", "#636f76", ":"),
    "smooth": ("Гладкий", "#a34d1e", "--"),
    "stochastic": ("Стохастический", "#215f9a", "-."),
}


def curve(ax, x, y, label, color, linestyle, *, log=False, continuous=False):
    y = np.asarray(y)
    if log:
        y = np.where(y > 0, y, np.nan)
    ax.plot(x, y, color=color, linestyle=linestyle, label=label, linewidth=1.05,
            drawstyle="default" if continuous else "steps-post")


def axes_for_energy(count):
    fig, axes = plt.subplots(1, count, figsize=(7.1, 2.61), squeeze=False)
    fig.subplots_adjust(left=.097, right=.982, bottom=.29, top=.86, wspace=.42)
    return fig, axes[0]


def survival_axis(ax, label, *, log=False):
    ax.set_xlabel(label)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(.001, 1.15)
    else:
        ax.set_ylim(0, 1.03)


def legend(fig, handles, labels, columns):
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.53, .012),
               frameon=False, ncol=columns)


def proton_reference(data):
    fig, axes = axes_for_energy(2)
    styles = [("#20252a", "-"), ("#636f76", ":"), ("#215f9a", "-."), ("#a34d1e", "--"), ("#447458", (0, (5, 1, 1, 1)))]
    for group, (color, line) in enumerate(styles, 1):
        curve(axes[0], data["total_grid"], data["total_survival"][group - 1]["reference"],
              f"Гр. {group}", color, line, log=True)
    axes[0].set_title("(a) Полный сигнал по направлениям", loc="left")
    axes[0].set_ylabel(r"$P(E > e)$")
    for module, (color, line) in zip(data["illustrated_modules_direction_3"], styles[2:4], strict=True):
        curve(axes[1], module["grid"], module["survival"]["reference"],
              f"Модуль {module['id']}", color, line, log=True)
    axes[1].set_title("(b) Модули, группа 3", loc="left")
    for ax in axes:
        survival_axis(ax, "Сигнал, ед. экспорта", log=True)
    handles, labels = axes[0].get_legend_handles_labels()
    extra, descriptions = axes[1].get_legend_handles_labels()
    legend(fig, handles + extra, labels + descriptions, 7)
    save(fig, PROTON / "figures", "energy_reference")


def proton_decoder(data):
    fig, axes = axes_for_energy(3)
    totals = data["total_survival"][2]
    curve(axes[0], data["total_grid"], totals["reference"], *METHODS["reference"], log=True)
    curve(axes[0], data["total_grid"], totals["raw_bank"], "Банк и декодеры", "#636f76", "--", log=True)
    axes[0].set_title("(a) Полный сигнал", loc="left")
    axes[0].set_ylabel(r"$P(E > e)$")
    axes[0].legend(loc="lower left", fontsize=7, frameon=False)
    for i, (ax, module) in enumerate(zip(axes[1:], data["illustrated_modules_direction_3"], strict=True), 1):
        for name, style in METHODS.items():
            curve(ax, module["grid"], module["survival"][name], *style, log=True)
        ax.set_title(f"({chr(97 + i)}) Модуль {module['id']}", loc="left")
    for ax in axes:
        survival_axis(ax, "Сигнал, ед. экспорта", log=True)
    legend(fig, *axes[1].get_legend_handles_labels(), 4)
    save(fig, DECODER / "decoder_figures", "energy_decoder")


def muon_energy(data):
    fig, axes = axes_for_energy(3)
    for group, (color, line) in zip(data["groups"], [("#20252a", "-"), ("#215f9a", "-."), ("#a34d1e", "--")], strict=True):
        curve(axes[0], data["total_grid"], group["total_survival"],
              decimal_label(group["angle_z_deg"], 2) + "°", color, line, log=True)
    axes[0].set_title("(a) Полный сигнал", loc="left")
    axes[0].set_ylabel(r"$P(E > e)$")
    axes[0].legend(loc="lower left", frameon=False)
    survival_axis(axes[0], "Сигнал, ед. экспорта", log=True)
    fit = data["line_shape"]
    edges = np.array(fit["hist_edges"])
    density = np.array(fit["test_histogram_positive"]["bins"]) / (fit["n_test_positive"] * np.diff(edges))
    axes[1].stairs(density, edges, color="#20252a", label="Симуляция", linewidth=1)
    curve(axes[2], fit["plot_grid"], fit["test_survival_positive"], "Симуляция", "#20252a", "-", log=True)
    for key, (label, color, line) in {"normal": ("Гаусс", "#a34d1e", "--"),
                                     "landau_gauss": ("Ландау ⊗ Гаусс", "#215f9a", "-.")}.items():
        curve(axes[1], fit["plot_grid"], fit["models"][key]["pdf"], label, color, line, continuous=True)
        curve(axes[2], fit["plot_grid"], fit["models"][key]["survival"], label, color, line, log=True, continuous=True)
    axes[1].set_title(f"(b) Модуль {data['selected_module']}, форма", loc="left")
    axes[1].set_ylabel("Плотность, 1/ед. экспорта")
    axes[1].set_xlabel("Сигнал при E > 0")
    axes[2].set_title(f"(c) Модуль {data['selected_module']}, хвост", loc="left")
    axes[2].set_ylabel(r"$P(E > e\mid E > 0)$")
    survival_axis(axes[2], "Сигнал, ед. экспорта", log=True)
    legend(fig, *axes[2].get_legend_handles_labels(), 3)
    save(fig, PILOT / "figures", "energy_muon")


def main():
    data = json.loads((PILOT / "energy_response.json").read_text())
    set_style()
    proton_reference(data["protons"])
    proton_decoder(data["protons"])
    muon_energy(data["muons"])


if __name__ == "__main__":
    main()
