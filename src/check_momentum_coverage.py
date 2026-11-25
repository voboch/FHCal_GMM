"""Audit archived particle conditions and test whether |p| predicts proton E_vis.

Run with scikit-learn installed: python src/check_momentum_coverage.py SOURCE_DATA_DIR
The largest nested NPZ uses about 3.7 GB of temporary disk during the audit.
"""
import argparse
from contextlib import contextmanager
from hashlib import blake2b, sha1, sha256
from itertools import combinations
import json
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryFile
from zipfile import ZipFile

import numpy as np
import scipy
from scipy.spatial import cKDTree
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from pipeline.projection_geometry import load_projection_geometry


SOURCE_COMMIT = '200744156f5eaeeca9da503b53232febf3186cca'
BLOBS = {
    'fhcal_data.npz': 'da694b42afcea4ced533cf23d3d46b8986bdcde4',
    'fhcal_data3.npz': '33b088e357ad0a93a497ba604f4c6dbc13e93aa1',
    'fhcal_data_side_modules81619.zip': 'f4c7155b857b8993642fae052854992298451827',
    'fhcal_data_side_modules81619_180.zip': '4257fefbef843a29be628b55381d2a18a5750eec',
    'fhcal_data_side_modules81619_new.zip': '7a5a0b4cb3a67d75d109344f24c8d954d5d11cec',
    'fhcal_train_90000.zip': 'dabb866674611fa47b65f5b309a821dbb2bd8add',
    'fhcal_data_muons_81619.zip': 'dabeffa422b7c34e5d57a3e4a94e85cd21adedf3',
    'fhcal_data_muons_81619_large.zip': '0faea481a1a5ef75892a872d10bd1c7c416c5117',
}
SEED = 20261123
QUANTILES = [0, .25, .5, .75, 1]


@contextmanager
def open_npz(path):
    blob = sha1(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            blob.update(chunk)
    if blob.hexdigest() != BLOBS[path.name]:
        raise ValueError(f'Unexpected source revision for {path.name}')
    with ZipFile(path) as outer:
        if path.suffix == '.npz':
            yield outer
        else:
            member, = outer.namelist()
            with outer.open(member) as source, TemporaryFile() as unpacked:
                copyfileobj(source, unpacked, 4 << 20)
                unpacked.seek(0)
                with ZipFile(unpacked) as inner:
                    yield inner


def inspect_source(path, mask):
    with open_npz(path) as archive:
        fields = sorted(Path(name).stem for name in archive.namelist())
        arrays = []
        for key in ('ParticlePDG', 'ParticleMomentum', 'ParticlePoint', 'TotalEnergySci'):
            with archive.open(key + '.npy') as stream:
                arrays.append(np.lib.format.read_array(stream, allow_pickle=False))
        pdg, momentum, vertex, target = arrays
        metadata = np.column_stack((pdg, momentum, vertex, target)).astype('<f8')
        if not np.isfinite(metadata).all():
            raise ValueError('Nonfinite metadata')
        ids = np.empty(len(target), dtype='V16')
        invalid_sum = short_sum = invalid_mask = 0
        max_delta = 0.
        with archive.open('EnergyDeposit.npy') as stream:
            version = np.lib.format.read_magic(stream)
            header = {(1, 0): np.lib.format.read_array_header_1_0,
                      (2, 0): np.lib.format.read_array_header_2_0}[version]
            shape, fortran, dtype = header(stream)
            if fortran or shape[0] != len(target):
                raise ValueError(f'Unexpected energy layout in {path.name}')
            cells = int(np.prod(shape[1:]))
            for start in range(0, len(target), 1024):
                stop = min(len(target), start + 1024)
                packet = stream.read((stop - start) * cells * dtype.itemsize)
                rows = np.frombuffer(packet, dtype=dtype).reshape(stop-start, cells).astype('<f8')
                if not np.isfinite(rows).all() or np.any(rows < 0):
                    raise ValueError('Invalid energy response')
                sums = rows.sum(axis=1)
                max_delta = max(max_delta, float(np.abs(sums - target[start:stop]).max()))
                close = np.isclose(sums, target[start:stop], rtol=1e-5, atol=1e-6)
                invalid_sum += int(np.count_nonzero(~close))
                short_sum += int(np.count_nonzero((~close) & (sums < target[start:stop])))
                if shape[1:] == mask.shape:
                    invalid_mask += int(np.any(rows[:, ~mask.ravel()] != 0, axis=1).sum())
                ids[start:stop] = [blake2b(row, key=blake2b(meta, digest_size=16).digest(), digest_size=16).digest()
                                   for row, meta in zip(rows, metadata[start:stop], strict=True)]
    magnitude = np.linalg.norm(momentum, axis=1)
    direction = momentum / magnitude[:, None]
    rays, lab = np.unique(np.round(direction, 6), axis=0, return_inverse=True)
    pmin = np.full(lab.max()+1, np.inf)
    pmax = np.full_like(pmin, -np.inf)
    np.minimum.at(pmin, lab, magnitude)
    np.maximum.at(pmax, lab, magnitude)
    support, counts = np.unique(momentum, axis=0, return_counts=True)
    keys = np.unique(ids)
    summary = dict(n=len(target), fields=fields, pdg=np.unique(pdg).tolist(), section_shape=shape[1:],
                   n_unique_vertices=len(np.unique(vertex, axis=0)), n_unique_momenta=len(support),
                   target_positive_rows=int(np.count_nonzero(target > 0)),
                   theta_deg_range=np.degrees(np.arccos(np.clip([direction[:,2].max(), direction[:,2].min()], -1, 1))).tolist(),
                   rounded_unit_vectors=rays.tolist() if len(rays) <= 10 else None,
                   n_directions_rounded_6dp=len(pmin), directions_with_magnitude_span_gt_1e_4=int(sum(pmax-pmin > 1e-4)),
                   abs_p_range=[float(magnitude.min()), float(magnitude.max())],
                   p_vectors_and_counts=[[v.tolist(), int(n)] for v, n in zip(support, counts)] if len(support)<=10 else None,
                   target_sum_mismatches=invalid_sum, section_sum_lower_among_mismatches=short_sum, max_abs_target_sum_residual=max_delta,
                   nonzero_outside_geometry=invalid_mask if shape[1:] == mask.shape else None,
                   exact_duplicate_rows=len(ids)-len(keys))
    return summary, keys, (direction, magnitude, vertex, target)


def prediction_study(direction, magnitude, vertex, target):
    # All rows are forward; two transverse direction cosines specify the unit vector.
    if not np.all(direction[:, 2] > 0):
        raise ValueError('Signed forward direction required')
    x = np.column_stack((direction[:, :2], vertex))
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(target))
    train, test = np.split(order, [len(order)*3//4])
    options = dict(loss='absolute_error', learning_rate=.08, max_iter=100, max_leaf_nodes=15,
                   min_samples_leaf=30, l2_regularization=1., early_stopping=False, random_state=SEED)
    errors = []
    with threadpool_limits(limits=1):
        for features in (x, np.column_stack((x, magnitude))):
            model = HistGradientBoostingRegressor(**options).fit(features[train], target[train])
            errors.append(np.abs(model.predict(features[test])-target[test]))
    baseline, with_p = errors
    gain = baseline - with_p
    bootstrap = [rng.choice(gain, len(gain), replace=True).mean() for _ in range(2000)]
    # Describe the |p| support around each holdout row in angular and vertex space.
    scales = np.subtract(*np.quantile(x[train], [.75, .25], axis=0))
    distance, neighbours = cKDTree(x[train] / scales).query(x[test] / scales, k=64)
    local_p = magnitude[train[neighbours]]
    p_edges = np.quantile(magnitude[train], QUANTILES)
    bins = np.searchsorted(p_edges[1:-1], magnitude[test])
    present = [np.any(np.searchsorted(p_edges[1:-1], local_p) == i, axis=1) for i in range(4)]
    theta = np.degrees(np.arccos(np.clip(direction[:, 2], -1, 1)))
    theta_edges = np.quantile(theta[train], QUANTILES)
    angle_bins = np.searchsorted(theta_edges[1:-1], theta[test])
    cross_counts = np.zeros((4,4), dtype=int)
    np.add.at(cross_counts, (bins, angle_bins), 1)
    by_p = []
    for i in range(4):
        use = bins == i
        by_p.append(dict(n=int(use.sum()), mean_evis=float(target[test[use]].mean()),
                         mae_without_p=float(baseline[use].mean()), mae_with_p=float(with_p[use].mean())))
    return dict(source='fhcal_data.npz', n_train=len(train), n_holdout=len(test), seed=SEED,
                split='numpy.default_rng(seed).permutation(N); first floor(3*N/4) train, remainder holdout',
                split_sha256={name:sha256(idx.astype('<i8').tobytes()).hexdigest() for name,idx in [('train',train),('holdout',test)]},
                features=['px/|p|','py/|p|','ParticlePoint_x','ParticlePoint_y','ParticlePoint_z'],
                added_feature='|p|', model='HistGradientBoostingRegressor', fixed_hyperparameters=options,
                mae_without_p=float(baseline.mean()), mae_with_p=float(with_p.mean()),
                mae_reduction=float(gain.mean()), paired_bootstrap_ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
                bootstrap='2000 paired holdout row resamples with both fitted models fixed; conditional on independent rows',
                training_p_quartile_edges=p_edges.tolist(), training_theta_deg_quartile_edges=theta_edges.tolist(),
                heldout_p_by_theta_quartile_counts=cross_counts.tolist(), heldout_by_training_p_quartile=by_p,
                vertex_coordinate_quartiles=np.quantile(vertex,QUANTILES,axis=0).tolist(),
                overlap=dict(k=64, scaling='train IQR per baseline feature, Euclidean metric',
                             p64_neighbour_radius_quantiles=np.quantile(distance[:,-1],QUANTILES).tolist(),
                             fraction_queries_with_all_training_p_quartiles=float(np.all(present, axis=0).mean()),
                             local_p_iqr_quantiles=np.quantile(np.subtract(*np.quantile(local_p,[.75,.25],axis=1)),QUANTILES).tolist()),
                scope='Predictive value for archived positive proton rows. ParticlePoint is the recorded vertex. Entry points, physical units, event IDs, primary count and generation weights are unverified. No causal interpretation or generator validation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_data_dir', type=Path)
    parser.add_argument('--output', type=Path, default=Path('reports/single_particle/momentum_coverage.json'))
    args = parser.parse_args()
    geometry = load_projection_geometry()
    mask = np.zeros((7,9,geometry.max_layers), dtype=bool)
    for module in geometry.modules:
        mask[module.legacy_row,module.legacy_col,:module.layers] = True
    coverage, keys = {}, {}
    for name in BLOBS:
        row, keys[name], data = inspect_source(args.source_data_dir / name, mask)
        coverage[name] = row
        if name == 'fhcal_data.npz':
            if row['target_sum_mismatches'] or row['nonzero_outside_geometry'] != 0 or row['exact_duplicate_rows']:
                raise ValueError('Unverified prediction data')
            prediction = prediction_study(*data)
    overlap = {a+' | '+b: int(np.intersect1d(keys[a], keys[b], assume_unique=True).size)
               for a,b in combinations(BLOBS, 2) if coverage[a]['section_shape'] == coverage[b]['section_shape']}
    # Compare distinct archived discrete settings across files without double-counting their rows.
    proton_settings = {}
    for row in coverage.values():
        if row['pdg'] == [2212] and row['p_vectors_and_counts']:
            for vector, _ in row['p_vectors_and_counts']:
                p = np.linalg.norm(vector)
                ray = tuple(np.round(vector / p, 6))
                proton_settings.setdefault(ray, set()).add(round(float(p), 4))
    discrete_settings = [dict(unit_vector=list(ray), abs_p=sorted(levels)) for ray, levels in sorted(proton_settings.items())]
    report = dict(discrete_proton_conditions_across_files=discrete_settings,
                  source_commit=SOURCE_COMMIT, source_git_blobs=BLOBS, geometry_sha256=geometry.sha256,
                  evaluator_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  versions=dict(numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
                  target_sum_tolerance=dict(rtol=1e-5,atol=1e-6), archives=coverage,
                  common_distinct_full_rows=overlap, identity='128-bit BLAKE2b of the full float64 section vector keyed by the hash of PDG, momentum, vertex and total; pairs compared only for matching section layouts.',
                  prediction=prediction)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n', encoding='utf-8')
    print(f"Holdout MAE without / with |p|: {prediction['mae_without_p']:.3f} / {prediction['mae_with_p']:.3f}")


if __name__ == '__main__':
    main()
