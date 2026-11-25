"""Compare full FHCal generators with identical CFM totals and held-out reference maps."""

import csv
import hashlib
import json
import platform
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from analyze_gmm_packet_decoder import section_statistics
from conditional_shape_wgan import CACHE, MODELS, SEEDS, generate
from conditional_tail_flow import replay_flow
from evaluate_conditional_shapes import empirical
from pipeline.projection_geometry import load_projection_geometry

REPORT = Path("reports/generated_total")
PREVIOUS = Path("reports/conditional_shapes")
CFM = Path("reports/single_particle")
METHODS = ("empirical", "direct", "gmm")
N = 4096
BOOTSTRAP = 400
BOOTSTRAP_SEED = 241124


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def independent_sample_interval(real, samples):
    """Sample transport and generation independently; keep common generated indices."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    contrast = np.zeros(BOOTSTRAP)
    probs = np.full(N, 1 / N)
    for direction, reference in enumerate(real):
        ref_w = rng.multinomial(N, probs, size=BOOTSTRAP) / N
        gen_w = rng.multinomial(N, probs, size=BOOTSTRAP) / N
        ref_rate = ref_w @ reference.astype(float)
        for seed in SEEDS:
            for method, sign in (("direct", -1), ("gmm", 1)):
                gen_rate = gen_w @ samples[method, seed][direction].astype(float)
                contrast += sign * np.abs(gen_rate - ref_rate).mean(axis=1)
    contrast *= 100 / (len(real) * len(SEEDS))
    return np.quantile(contrast, [.025, .975]).tolist()


def evaluate(species, output):
    projection_cache = Path("outputs/single_particle") / f"{species}_samples.npz"
    section_cache = CACHE / f"{species}_sections.npz"
    flow = CFM / f"{species}_flow/conditional_tail_flow.pt"
    flow_receipt = CFM / f"{species}_flow.json"
    if json.loads(flow_receipt.read_text())["checkpoint_sha256"] != digest(flow):
        raise ValueError("CFM checkpoint differs from training receipt")
    ref_file = PREVIOUS / f"{species}_reference.json"
    split_file = CFM / f"{species}_split.json"
    ref = json.loads(ref_file.read_text())
    split = json.loads(split_file.read_text())
    if ref["prior_split_sha256"] != digest(split_file) or ref["archive_sha256"] != split["archive_sha256"]:
        raise ValueError("Split provenance differs")
    geo = load_projection_geometry()
    if geo.sha256 != ref["geometry_sha256"]:
        raise ValueError("Detector geometry differs")
    for old, new in zip(split["split"], ref["new_test"], strict=True):
        if len(set(new)) != N or any(set(ids).intersection(new) for ids in old.values()):
            raise ValueError("Control identities overlap with a prior split")
    with np.load(projection_cache) as f:
        training_projections = np.split(f["data"], 3, axis=1)[0]
    with np.load(section_cache) as f:
        train, cal, test = f["sections"].transpose(1, 0, 2, 3)
    np.testing.assert_allclose(train.sum(-1, dtype=np.float64), training_projections[:, :, 0], rtol=2e-6, atol=1e-8)
    totals = np.ascontiguousarray(replay_flow(training_projections, flow)[:, :N, 0], dtype=np.float32)
    assert totals.shape == test.shape[:2] and np.isfinite(totals).all() and (totals >= 0).all()
    modules, _ = np.nonzero(geo.layer_mask)
    starts = np.flatnonzero(np.r_[True, modules[1:] != modules[:-1]])
    contiguous = modules[:-1] == modules[1:]

    def sums(maps):
        return np.add.reduceat(maps, starts, axis=-1, dtype=np.float64)

    calibration = sums(cal)
    reference = sums(test)
    ref_stats = section_statistics(test, contiguous)
    selected = [np.flatnonzero((rows > 0).sum(axis=0) >= 500) for rows in calibration]
    thresholds = [np.asarray([np.quantile(rows[:, i][rows[:, i] > 0], .95) for i in indices])
                  for rows, indices in zip(calibration, selected, strict=True)]
    above_ref = [rows[:, idx] > cuts for rows, idx, cuts in zip(reference, selected, thresholds, strict=True)]
    total_rows = []
    for d, row in enumerate(cal.sum(-1, dtype=np.float64)):
        threshold = np.quantile(row[row > 0], .95)
        total_rows.append(dict(direction=d+1, threshold=threshold, n=N,
                               ref_above=int((test[d].sum(-1, dtype=np.float64) > threshold).sum()),
                               model_above=int((totals[d] > threshold).sum())))
    inputs = [projection_cache, section_cache, flow, flow_receipt, ref_file, split_file, REPORT / "PROTOCOL.md",
              Path("src/evaluate_generated_total.py"), Path("src/conditional_shape_wgan.py"), Path("src/conditional_tail_flow.py"),
              Path("src/evaluate_conditional_shapes.py"), Path("src/analyze_gmm_packet_decoder.py")]
    channel_rows, direction_rows, model_samples = [], [], {}
    for method in METHODS:
        for seed in SEEDS:
            if method == "empirical":
                maps = empirical(train, totals, seed)
            else:
                inputs.append(MODELS / f"{species}_{method}_{seed}.pt")
                maps = generate(species, method, seed, totals)
            if not np.isfinite(maps).all() or (maps < 0).any():
                raise ValueError(f"Invalid generated signals: {species}, {method}, seed {seed}")
            np.testing.assert_allclose(maps.sum(-1), totals, rtol=2e-5, atol=1e-6)
            generated = sums(maps)
            stats = section_statistics(maps, contiguous)
            indicators = []
            for d, indices in enumerate(selected):
                above = generated[d][:, indices] > thresholds[d]
                indicators.append(above)
                ref_count = above_ref[d].sum(0)
                gen_count = above.sum(0)
                direction_rows.append(dict(direction=d+1, seed=seed, method=method, eligible=len(indices),
                                           tail_mae_pp=100 * np.abs(gen_count - ref_count).mean()/N,
                                           occupied_w1=wasserstein_distance(ref_stats[d,:,1], stats[d,:,1]),
                                           neighbours_w1=wasserstein_distance(ref_stats[d,:,5], stats[d,:,5])))
                for j, m in enumerate(indices):
                    channel_rows.append(dict(direction=d+1, module=geo.modules[m].module_id, method=method,
                                             seed=seed, q95=thresholds[d][j], n=N,
                                             ref_above=ref_count[j], model_above=gen_count[j]))
            model_samples[method, seed] = indicators
    summary = {}
    for method in METHODS:
        rows = [r for r in direction_rows if r["method"] == method]
        by_seed = [float(np.mean([r["tail_mae_pp"] for r in rows if r["seed"] == seed])) for seed in SEEDS]
        summary[method] = {metric: float(np.mean([r[metric] for r in rows]))
                           for metric in ("tail_mae_pp", "occupied_w1", "neighbours_w1")}
        summary[method]["by_seed_pp"] = by_seed
    summary["gmm_minus_direct"] = dict(
        mean_pp=summary["gmm"]["tail_mae_pp"] - summary["direct"]["tail_mae_pp"],
        bootstrap_95_pp=independent_sample_interval(above_ref, model_samples))
    summary["total_tail_mae_pp"] = float(np.mean([100*abs(r["model_above"]-r["ref_above"])/N for r in total_rows]))
    summary["eligible_module_directions"] = sum(map(len, selected))
    dump_csv(output / f"{species}_modules.csv", channel_rows)
    dump_csv(output / f"{species}_directions.csv", direction_rows)
    dump_csv(output / f"{species}_totals.csv", total_rows)
    receipt = dict(input_sha256={str(p):digest(p) for p in inputs},
                   total_sequence_sha256=hashlib.sha256(totals.tobytes()).hexdigest(),
                   rows_per_direction=N, direction_vectors=ref["directions"], shape_seeds=SEEDS,
                   bootstrap_repeats=BOOTSTRAP, bootstrap_seed=BOOTSTRAP_SEED,
                   numpy=np.__version__, torch=str(torch.__version__), python=platform.python_version())
    return summary, receipt


if __name__ == "__main__":
    torch.set_num_threads(1)
    results, receipts = {}, {}
    with TemporaryDirectory(dir=REPORT) as temporary:
        output = Path(temporary)
        for species in ("muon", "proton"):
            results[species], receipts[species] = evaluate(species, output)
        for name, data in (("summary", results), ("receipt", receipts)):
            (output / f"{name}.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        for path in output.iterdir():
            path.replace(REPORT / path.name)
