"""Evaluate the released conditional flow with its direction retained or marginalized."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

from analyze_module_energy import conditional_shape
from conditional_tail_flow import replay_flow
from evaluate_single_particle_tails import (
    DEFAULT_CACHE,
    MIN_POSITIVE,
    PILOT,
    interval,
    summarize,
)
from pipeline.projection_geometry import load_projection_geometry


def evaluate(directory, species, cache):
    with np.load(cache / f"{species}_samples.npz") as data:
        fit, threshold, test = np.split(data["data"], 3, axis=1)
        banks = {name: data[name] for name in ("raw_bank", "smooth", "stochastic")}
    generated = replay_flow(fit, directory / f"{species}_flow/conditional_tail_flow.pt")
    pooled = generated.reshape(-1, generated.shape[-1])
    comparisons = []
    for group, ref in enumerate(test):
        cuts = {}
        for j in range(11, 65):
            pos = threshold[group, :, j]
            pos = pos[pos > 0]
            if len(pos) >= MIN_POSITIVE:
                cuts[j] = np.quantile(pos, 0.95)
        scores = {}
        for name, values in [
            ("correct_direction", generated[group]),
            ("marginalized_direction", pooled),
        ]:
            differences = [
                abs(np.mean(ref[:, j] > t) - np.mean(values[:, j] > t))
                for j, t in cuts.items()
            ]
            scores[name] = dict(
                w1_total=float(wasserstein_distance(ref[:, 0], values[:, 0])),
                module_tail_all_mae_pp=float(100 * np.mean(differences))
                if differences
                else None,
                module_positive_mass_mae_pp=float(
                    100
                    * np.abs(
                        (ref[:, 11:] > 0).mean(axis=0)
                        - (values[:, 11:] > 0).mean(axis=0)
                    ).mean()
                ),
            )
        comparisons.append(
            dict(direction=group + 1, eligible_modules=len(cuts), **scores)
        )
    geo = load_projection_geometry()
    labels = (
        ["Детектор"]
        + [f"Слой {i + 1}" for i in range(geo.max_layers)]
        + [f"Модуль {m.module_id}" for m in geo.modules]
    )
    q99 = []
    for i, ref in enumerate(test):
        models = {
            "reference": ref,
            "conditional_raw": fit[i],
            "pooled_raw": fit.reshape(-1, len(labels)),
            **{k: v[i] for k, v in banks.items()},
            "flow_shape": generated[i],
        }
        q99.extend(summarize(species, i + 1, labels, models, threshold[i], 0.99, 2500))
    with (directory / f"{species}_q99.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(q99[0]))
        writer.writeheader()
        writer.writerows(q99)
    result = dict(species=species, directional_control=comparisons)
    anchor = 0 if species == "muon" else 2
    module = 16 if species == "muon" else 26
    shared = []
    for label in ("Детектор", f"Модуль {module}"):
        j = labels.index(label)
        cal = threshold[anchor, :, j]
        common_cut = float(np.quantile(cal[cal > 0], 0.95))
        for g in range(len(test)):
            if g == anchor:
                continue
            cells = [
                x[:, j] > common_cut
                for x in (generated[g], generated[anchor], test[g], test[anchor])
            ]
            limits = [interval(int(x.sum()), len(x), alpha=0.05 / 4) for x in cells]
            rates = [float(x.mean()) for x in cells]
            # Union of four binomial intervals covers this contrast with probability >= .95
            # under the row model, conditional on fixed cut and trained generator.
            lo = limits[0][0] - limits[1][1] - limits[2][1] + limits[3][0]
            hi = limits[0][1] - limits[1][0] - limits[2][0] + limits[3][1]
            shared.append(
                dict(
                    observable=label,
                    anchor=anchor + 1,
                    direction=g + 1,
                    threshold=common_cut,
                    counts=[int(x.sum()) for x in cells],
                    sample_sizes=[len(x) for x in cells],
                    reference_effect_pp=100 * (rates[2] - rates[3]),
                    model_effect_pp=100 * (rates[0] - rates[1]),
                    difference_ci95_pp=[100 * lo, 100 * hi],
                )
            )
    result["shared_threshold_directional_effects"] = shared
    if species == "muon":
        old = json.loads((PILOT / "energy_response.json").read_text())["muons"]
        module = old["selected_module"]
        ids = [m.module_id for m in load_projection_geometry().modules]
        j = 11 + ids.index(module)
        cal = threshold[0, :, j]
        t = float(np.quantile(cal[cal > 0], 0.95))
        obs = test[0, :, j]
        n = int((obs > 0).sum())
        hits = int((obs > t).sum())
        estimates = {}
        for name, model in old["line_shape"]["models"].items():
            estimates[name] = float(
                1
                - conditional_shape(
                    np.array([t]),
                    model["parameters"],
                    name == "landau_gauss",
                    order=160,
                )[0]
            )
        result["historical_muon_line_shape"] = dict(
            module=module,
            direction=1,
            threshold=t,
            positive=n,
            above=hits,
            positive_tail=hits / n,
            reference_ci95=interval(hits, n),
            fixed_positive_models=estimates,
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    directory = args.directory
    metrics = [evaluate(directory, name, args.cache) for name in ("muon", "proton")]
    (directory / "conditioning.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    )
