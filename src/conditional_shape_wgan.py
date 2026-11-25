"""Paired conditional WGAN-GP experiment using a fixed observed total per draw."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pipeline.gaussian_projection import section_bounds
from pipeline.projection_geometry import load_projection_geometry

CACHE = Path("outputs/conditional_shapes")
REPORT = Path("reports/conditional_shapes")
MODELS = REPORT / "models"
SEEDS = (173, 283, 397)
STEPS, BATCH, CRITIC_STEPS = 800, 64, 3
LATENT, WIDTH = 16, 96


def mlp(inputs, outputs):
    return nn.Sequential(
        nn.Linear(inputs, WIDTH),
        nn.LeakyReLU(0.2),
        nn.Linear(WIDTH, WIDTH),
        nn.LeakyReLU(0.2),
        nn.Linear(WIDTH, outputs),
    )


class ShapeGenerator(nn.Module):
    def __init__(self, mean_fractions, kind, k):
        super().__init__()
        self.kind, self.k = kind, k
        dims = 438 if kind == "direct" else 7 * k
        self.net = mlp(LATENT + len(mean_fractions) + 1, dims)
        if kind == "direct":
            # A one-count pseudocount on the scale of 4096 unit-normalized maps.
            prior = (mean_fractions + 1 / (4096 * 438)).log()
        else:
            low, high = section_bounds(load_projection_geometry())
            extent = high.max(0) - low.min(0)
            origin = low.min(0)
            centers = ((low + high) / 2 - origin) / extent
            # Unique axis intervals keep the differentiable box integral small.
            for axis in range(3):
                pairs = np.stack(
                    [
                        (low[:, axis] - origin[axis]) / extent[axis],
                        (high[:, axis] - origin[axis]) / extent[axis],
                    ],
                    axis=1,
                )
                intervals, index = np.unique(pairs, axis=0, return_inverse=True)
                self.register_buffer(
                    f"edges{axis}", torch.tensor(intervals, dtype=torch.float32)
                )
                self.register_buffer(
                    f"index{axis}", torch.tensor(index, dtype=torch.long)
                )
            self.register_buffer(
                "floor_width",
                torch.tensor([2.5, 2.5, 0.175] / extent, dtype=torch.float32),
            )
            # Place centres at energy-weighted depth quantiles in the training map.
            priors = []
            for fractions in mean_fractions.numpy():
                order = np.argsort(centers[:, 2], kind="stable")
                indices = order[
                    np.searchsorted(
                        np.cumsum(fractions[order]), (np.arange(k) + 0.5) / k
                    )
                ]
                loc = torch.tensor(centers[indices], dtype=torch.float32).clamp(
                    0.001, 0.999
                )
                init = torch.cat(
                    [torch.zeros(k, 1), torch.logit(loc), torch.full((k, 3), -3.0)],
                    dim=1,
                )
                priors.append(init.flatten())
            prior = torch.stack(priors)
        self.register_buffer("prior", prior)

    def forward(self, cond, group, noise):
        raw = self.net(torch.cat([cond, noise], dim=-1)) + self.prior[group]
        if self.kind == "direct":
            return raw.softmax(-1)
        comp = raw.reshape(-1, self.k, 7)
        weights = comp[:, :, 0].softmax(-1)
        means = comp[:, :, 1:4].sigmoid()
        widths = self.floor_width + F.softplus(comp[:, :, 4:7])
        log_mass = 0.0
        for axis in range(3):
            edges = getattr(self, f"edges{axis}")
            scale = widths[:, :, axis, None].double()
            lo = (edges[:, 0] - means[:, :, axis, None]) / scale
            hi = (edges[:, 1] - means[:, :, axis, None]) / scale
            # Reflect right-tail intervals, then subtract their CDFs in log space.
            log_hi = torch.special.log_ndtr(torch.where(lo >= 0, -lo, hi))
            log_lo = torch.special.log_ndtr(torch.where(lo >= 0, -hi, lo))
            interval = log_hi + torch.log(-torch.expm1(log_lo - log_hi))
            log_mass = (
                log_mass
                + interval.index_select(-1, getattr(self, f"index{axis}")).float()
            )
        normalized = log_mass.softmax(-1)
        return torch.sum(weights[:, :, None] * normalized, dim=1)


def conditions(total, mean, sd, group, groups):
    z = (total.clamp_min(1e-20).log() - mean[group]) / sd[group]
    return torch.cat([F.one_hot(group, groups).float(), z[:, None]], dim=1)


def train(species, kind, seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    with np.load(CACHE / f"{species}_sections.npz") as archive:
        train_map = torch.from_numpy(archive["sections"][:, 0].copy())
    totals = train_map.sum(-1)
    groups = len(totals)
    fractions = train_map / totals[:, :, None].clamp_min(1e-20)
    priors = fractions.sum(1) / (totals > 0).sum(1)[:, None]
    positive = [row[row > 0].log() for row in totals]
    mean = torch.stack([x.mean() for x in positive])
    sd = torch.stack([x.std(unbiased=False).clamp_min(1e-6) for x in positive])
    k = 4 if species == "muon" else 6
    gen = ShapeGenerator(priors, kind, k)
    torch.manual_seed(seed + 1)
    critic = mlp(438 + groups + 1, 1)
    params = dict(lr=0.0003, betas=(0.0, 0.9))
    opt_g = torch.optim.Adam(gen.parameters(), **params)
    opt_d = torch.optim.Adam(critic.parameters(), **params)
    rand = torch.Generator().manual_seed(seed + 2)
    nonzero = torch.nonzero(totals > 0)

    def batch():
        rows = nonzero[torch.randint(len(nonzero), (BATCH,), generator=rand)]
        g, i = rows.T
        c = conditions(totals[g, i], mean, sd, g, groups)
        return g, c, fractions[g, i], torch.randn(BATCH, LATENT, generator=rand)

    def score(x, c):
        return critic(torch.cat([x, c], dim=-1))

    start = time.monotonic()
    history = []
    for step in range(1, STEPS + 1):
        for _ in range(CRITIC_STEPS):
            g, c, real, z = batch()
            with torch.no_grad():
                fake = gen(c, g, z)
            alpha = torch.rand(BATCH, 1, generator=rand)
            mix = torch.lerp(real, fake, alpha).requires_grad_(True)
            (grad,) = torch.autograd.grad(score(mix, c).sum(), mix, create_graph=True)
            penalty = (grad.norm(2, dim=1) - 1).square().mean()
            gap = score(real, c).mean() - score(fake, c).mean()
            loss_d = 10 * penalty - gap
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()
        g, c, _, z = batch()
        loss_g = -score(gen(c, g, z), c).mean()
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()
        if step % 200 == 0:
            history.append(
                dict(
                    step=step,
                    critic_gap=float(gap.detach()),
                    gp=float(penalty.detach()),
                    generator_loss=float(loss_g.detach()),
                )
            )
    checkpoint = dict(
        generator=gen.state_dict(),
        priors=priors,
        mean=mean,
        sd=sd,
        species=species,
        kind=kind,
        seed=seed,
        k=k,
    )
    MODELS.mkdir(parents=True, exist_ok=True)
    path = MODELS / f"{species}_{kind}_{seed}.pt"
    torch.save(checkpoint, path)
    receipt = dict(
        species=species,
        kind=kind,
        seed=seed,
        steps=STEPS,
        critic_steps=STEPS * CRITIC_STEPS,
        seconds=time.monotonic() - start,
        generator_parameters=sum(p.numel() for p in gen.parameters()),
        critic_parameters=sum(p.numel() for p in critic.parameters()),
        history=history,
        torch_version=str(torch.__version__),
    )
    (REPORT / f"{path.stem}.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )
    print(path.stem, round(receipt["seconds"], 1), flush=True)


@torch.no_grad()
def generate(species, kind, seed, totals):
    saved = torch.load(
        MODELS / f"{species}_{kind}_{seed}.pt", map_location="cpu", weights_only=True
    )
    net = ShapeGenerator(saved["priors"], kind, saved["k"])
    net.load_state_dict(saved["generator"])
    net.eval()
    rand = torch.Generator().manual_seed(seed + 913)
    result = []
    for d, values in enumerate(totals):
        chunks = []
        for total in torch.from_numpy(values).split(128):
            g = torch.full((len(total),), d)
            cond = conditions(total, saved["mean"], saved["sd"], g, len(totals))
            fractions = net(cond, g, torch.randn(len(g), LATENT, generator=rand))
            np.testing.assert_allclose(fractions.sum(-1).numpy(), 1, atol=2e-5)
            chunks.append((total[:, None] * fractions).numpy())
        result.append(np.concatenate(chunks))
    return np.stack(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("species", choices=["muon", "proton"])
    parser.add_argument("kind", choices=["gmm", "direct"])
    args = parser.parse_args()
    for seed in SEEDS:
        train(args.species, args.kind, seed)
