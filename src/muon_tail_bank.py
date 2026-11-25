"""Fit the muon K=4 bank and decode spatial showers on explicitly selected rows."""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import scipy

from conditional_gmm_baseline import digest, sample_packet_fractions
from evaluate_gaussian_projection import fit_seed, read_energy_rows
from pipeline.gaussian_projection import SectionProjector, fit_mixture, section_bounds
from pipeline.projection_geometry import legacy_grid_to_modules


K, RESTARTS, MAXITER = 4, 3, 200
MIN_SCALES = (2.5, 2.5, 0.175)
FIT_SEED, PACKET_SEED = 20261123, 20261124
PACKETS, REPLICAS = 128, 16


def _fit_row(projector, job):
    row, response = job
    total = response.sum()
    if total == 0:
        return np.zeros(7 * K - 1), -1
    result = fit_mixture(projector, response / total, K, RESTARTS, MAXITER,
                         fit_seed(FIT_SEED, int(row), K), MIN_SCALES, trace_restarts=True)
    best, = [start for start in result.restart_trace if start['selected']]
    return np.asarray(best['parameters']), result.status


def _array_digest(*arrays):
    checksum = sha256()
    for array in arrays:
        checksum.update(str(array.shape).encode())
        checksum.update(array.dtype.str.encode())
        checksum.update(np.ascontiguousarray(array).tobytes())
    return checksum.hexdigest()


def fit_bank(source: Path, rows, geometry, output: Path, workers: int):
    """Return raw_bank, smooth [G,N,C] and stochastic [G,N*16,C].

    Consecutive groups of 16 stochastic entries share a code. The 128-packet
    setting is transferred from the proton pilot without muon recalibration.
    ``output`` is the checkpoint NPZ file. Existing checkpoints must match
    the data, row order, geometry, fitter and payload to be reused.
    """
    source, output, rows = Path(source), Path(output), np.asarray(rows)
    if (rows.ndim != 2 or rows.shape[0] != 3 or rows.shape[1] == 0 or
            rows.dtype.kind not in 'iu' or rows.size != np.unique(rows).size or workers < 1):
        raise ValueError('Expected unique integer muon row IDs in three groups and positive workers')
    rows = rows.astype(np.int64)
    with np.load(source, allow_pickle=False) as archive:
        pdg = archive['ParticlePDG']
        if rows.min() < 0 or rows.max() >= pdg.size or not np.all(pdg[rows] == -13):
            raise ValueError('Bank contains invalid row IDs or particles other than PDG -13')
        momenta = archive['ParticleMomentum'][rows]
        if not np.all(momenta == momenta[:, :1]) or len(np.unique(momenta[:, 0], axis=0)) != 3:
            raise ValueError('Each bank group must represent one distinct recorded momentum')
        recorded = archive['TotalEnergySci'][rows]
    grids = read_energy_rows(source, rows.ravel(), len(pdg))
    raw = legacy_grid_to_modules(grids, geometry)[:, geometry.layer_mask].reshape(*rows.shape, -1)
    energy = raw.sum(axis=-1)
    np.testing.assert_allclose(energy, recorded, rtol=1e-5, atol=1e-6)
    project = SectionProjector(*section_bounds(geometry), 'integrated')
    base = Path(__file__).resolve().parent
    metadata = json.dumps(dict(
        source_sha256=digest(source), geometry_sha256=geometry.sha256,
        components=K, restarts=RESTARTS, maxiter=MAXITER, min_scales=MIN_SCALES,
        seed=FIT_SEED, packet_seed=PACKET_SEED, packets=PACKETS, replicas=REPLICAS,
        code_sha256={p: digest(base / p) for p in (
            'muon_tail_bank.py', 'conditional_gmm_baseline.py', 'evaluate_gaussian_projection.py',
            'pipeline/gaussian_projection.py', 'pipeline/projection_geometry.py')},
        numpy=np.__version__, scipy=scipy.__version__), sort_keys=True)
    if output.exists():
        with np.load(output, allow_pickle=False) as bank:
            parameters, status = bank['parameters'], bank['status']
            if (str(bank['metadata']) != metadata or not np.array_equal(rows, bank['rows']) or
                    not np.array_equal(energy, bank['energy']) or
                    str(bank['payload_sha256']) != _array_digest(rows, energy, parameters, status)):
                raise ValueError(f'Checkpoint inputs, optimizer or checksum differ: {output}')
    else:
        fit = partial(_fit_row, project)
        jobs = zip(rows.ravel(), raw.reshape(rows.size, -1), strict=True)
        if workers == 1:
            fitted = list(map(fit, jobs))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                fitted = list(pool.map(fit, jobs, chunksize=4))
        parameters = np.stack([code for code, _ in fitted]).reshape(*rows.shape, 7 * K - 1)
        status = np.array([status for _, status in fitted]).reshape(rows.shape)
        output.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=output.parent, suffix='.npz', delete=False) as temp:
            temp_path = Path(temp.name)
        try:
            np.savez_compressed(temp_path, rows=rows, energy=energy, parameters=parameters, status=status,
                                momenta=momenta[:, 0], metadata=metadata,
                                payload_sha256=_array_digest(rows, energy, parameters, status))
            temp_path.replace(output)
        finally:
            temp_path.unlink(missing_ok=True)
    smooth = np.zeros_like(raw)
    stochastic = np.zeros((*rows.shape, REPLICAS, raw.shape[-1]))
    for group, index in zip(*np.nonzero(energy), strict=True):
        probability, components, _, weights = project.mixture(parameters[group, index], K)
        total = energy[group, index]
        smooth[group, index] = total * probability
        rng = np.random.default_rng([PACKET_SEED, int(rows[group, index])])
        stochastic[group, index] = total * sample_packet_fractions(components, weights, PACKETS, rng, replicas=REPLICAS)
    np.testing.assert_allclose(smooth.sum(axis=-1), energy, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(stochastic.sum(axis=-1), np.repeat(energy[..., None], REPLICAS, axis=-1),
                               rtol=1e-12, atol=1e-12)
    return {'raw_bank': raw, 'smooth': smooth, 'stochastic': stochastic.reshape(len(rows), -1, raw.shape[-1])}
