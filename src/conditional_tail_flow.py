"""Conditional flow matching for positive total signal, with empirical shower shapes.

One fit covers the discrete directions of one particle species. Each row holds
the total, ten layer sums and 54 module sums. All three views of a generated
shower share a single shape donor and a single generated total.
"""

from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch import nn


DRAW_COUNT = 8192
NEIGHBORS = 16
HIDDEN = 64
BATCH = 256
UPDATES = 1800
LEARNING_RATE = 0.001
ODE_STEPS = 48


class ConditionalVelocity(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.register_buffer("directions", torch.eye(groups))
        self.layers = nn.Sequential(
            nn.Linear(groups + 2, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, 1),
        )

    def forward(self, position, time, direction):
        inputs = torch.cat((position[:, None], time[:, None], self.directions[direction]), dim=1)
        return self.layers(inputs)[:, 0]


def _projections(train):
    values = np.asarray(train, dtype=np.float64, order="C")
    if values.ndim != 3 or min(values.shape[:2]) < 1 or values.shape[2] != 65:
        raise ValueError("Expected [directions, train rows, 1 total + 10 layers + 54 modules]")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Energy projections must be finite and nonnegative")
    total = values[..., 0]
    for view in (values[..., 1:11], values[..., 11:]):
        np.testing.assert_allclose(view.sum(axis=-1), total, rtol=2e-6, atol=1e-12)
        if np.any(view[total == 0] != 0):
            raise ValueError("A zero-energy training row has nonzero projections")
    return values


def _fit(energy, seed):
    groups, n = energy.shape
    means = np.zeros(groups)
    scales = np.zeros(groups)
    counts = np.count_nonzero(energy > 0, axis=1)
    standardized = np.zeros_like(energy, dtype=np.float32)
    for group, row in enumerate(energy):
        positive = row[row > 0]
        if positive.size:
            logs = np.log(positive)
            means[group] = logs.mean()
            if np.ptp(logs) > 0:
                scales[group] = logs.std()
                standardized[group, :positive.size] = (logs - means[group]) / scales[group]

    # Initialization and minibatches have local RNG state. The outer process
    # can train the two species in either order without changing a fit.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = ConditionalVelocity(groups)
    local_rng = torch.Generator().manual_seed(seed)
    counts_t = torch.from_numpy(counts)
    target_pool = torch.from_numpy(standardized)
    active = torch.from_numpy(np.flatnonzero(scales > 0))
    history = []
    if len(active):
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        for step in range(UPDATES):
            direction = active[torch.randint(len(active), (BATCH,), generator=local_rng)]
            index = (torch.rand(BATCH, generator=local_rng) * counts_t[direction]).long()
            data = target_pool[direction, index]
            noise = torch.randn(BATCH, generator=local_rng)
            t = torch.rand(BATCH, generator=local_rng)
            position = torch.lerp(noise, data, t)
            loss = (model(position, t, direction) - (data - noise)).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if (step + 1) % 300 == 0:
                history.append(float(loss.detach()))
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite fitted velocity parameters")
    return model.eval(), {
        "seed": int(seed), "directions": groups, "train_rows_per_direction": n,
        "positive_counts": counts.tolist(), "zero_probability": (1 - counts / n).tolist(),
        "log_mean": means.tolist(), "log_scale": scales.tolist(),
        "hidden_units": HIDDEN, "updates": UPDATES if len(active) else 0,
        "batch_size": BATCH, "learning_rate": LEARNING_RATE, "ode_midpoint_steps": ODE_STEPS,
        "neighbors": NEIGHBORS, "samples_per_direction": DRAW_COUNT,
        "train_loss_every_300_steps": history, "cpu_threads": 1, "torch_version": str(torch.__version__),
        "numpy_version": np.__version__, "shape_distance": "absolute log total signal",
    }


def _positive_draws(model, group, mean, scale, count, rng):
    if scale == 0:
        return np.full(count, np.exp(mean))
    state = torch.from_numpy(rng.standard_normal(count).astype(np.float32))
    directions = torch.full((count,), group, dtype=torch.long)
    dt = 1.0 / ODE_STEPS
    with torch.inference_mode():
        for i in range(ODE_STEPS):
            time = torch.full_like(state, i * dt)
            middle = state + (dt / 2) * model(state, time, directions)
            state = state + dt * model(middle, time + dt / 2, directions)
    with np.errstate(over="raise", under="raise", invalid="raise"):
        energy = np.exp(mean + scale * state.numpy().astype(np.float64))
    if not np.isfinite(energy).all() or (energy <= 0).any():
        raise FloatingPointError("Generated positive energy outside representable range")
    return energy


def _sample(model, train, info):
    draws = np.zeros((len(train), DRAW_COUNT, train.shape[-1]), dtype=np.float64)
    children = np.random.SeedSequence([info["seed"], 1]).spawn(len(train))
    for group, (rows, stream) in enumerate(zip(train, children, strict=True)):
        rng = np.random.default_rng(stream)
        selected = rng.random(DRAW_COUNT) >= info["zero_probability"][group]
        if not selected.any():
            continue
        requested = _positive_draws(model, group, info["log_mean"][group], info["log_scale"][group],
                                    int(selected.sum()), rng)
        pool = rng.permutation(np.flatnonzero(rows[:, 0] > 0))
        ordered = pool[np.argsort(rows[pool, 0], kind="stable")]
        log_energy = np.log(rows[ordered, 0])
        k = min(NEIGHBORS, len(ordered))
        # Between adjacent k-neighbour windows only their two outside
        # endpoints differ. Their midpoint is the switch in log-energy space.
        boundaries = (log_energy[:-k] + log_energy[k:]) * 0.5
        start = np.searchsorted(boundaries, np.log(requested))
        donor = ordered[start + rng.integers(k, size=len(requested))]
        fractions = rows[donor, 1:] / rows[donor, 0, None]
        draws[group, selected, 0] = requested
        draws[group, selected, 1:] = requested[:, None] * fractions
    return _projections(draws)


def _train_digest(train):
    return sha256(train.tobytes(order="C")).hexdigest()


def generate_flow(train: np.ndarray, seed: int, output: Path) -> tuple[np.ndarray, dict]:
    """Fit on train alone, generate the projections, and save weights in output.

    Use a separate output directory per fit. The checkpoint contains model
    weights, per-direction normalization and the fitted zero-energy hurdle.
    Replaying projected showers also requires the original training donors.
    """
    train = _projections(train)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "conditional_tail_flow.pt"
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        model, info = _fit(train[..., 0], int(seed))
        info["train_sha256_float64"] = _train_digest(train)
        info["source_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
        generated = _sample(model, train, info)
    finally:
        torch.set_num_threads(threads)
    with NamedTemporaryFile(dir=output, suffix=".pt", delete=False) as handle:
        stage = Path(handle.name)
    try:
        with stage.open("wb") as handle:
            torch.save({"state_dict": model.state_dict(), "settings": info}, handle)
        stage.replace(checkpoint)
    finally:
        stage.unlink(missing_ok=True)
    return generated, {**info, "checkpoint": checkpoint.name, "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest()}


def replay_flow(train: np.ndarray, checkpoint: Path) -> np.ndarray:
    """Replay generation from a saved checkpoint without training."""
    train = _projections(train)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    info = saved["settings"]
    if (list(train.shape[:2]) != [info["directions"], info["train_rows_per_direction"]]
            or info["train_sha256_float64"] != _train_digest(train)
            or info["source_sha256"] != sha256(Path(__file__).read_bytes()).hexdigest()):
        raise ValueError("Checkpoint requires its original training rows and implementation")
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            model = ConditionalVelocity(len(train))
        model.load_state_dict(saved["state_dict"])
        return _sample(model.eval(), train, info)
    finally:
        torch.set_num_threads(threads)
