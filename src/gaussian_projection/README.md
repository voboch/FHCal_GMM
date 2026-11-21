# FHCal Gaussian mixture representation pilot

This pilot fits each event's **observed** section fractions with a small diagonal Gaussian mixture. The observed total enters normalization. The task is to measure the fit quality of a section representation on rows with one recorded PDG, vertex and momentum. The selected group shares PDG and momentum. Primary multiplicity in the original simulation remains to be checked. The pilot does not infer constituent showers.

## Predictions and protocol

- Increasing K from 1 to 2 and 4 should decrease typical shape errors on the sampled rows in this event-wise fit. Use disjoint CPU repeats of 48 rows, with the same rows and optimizer budgets for all arms in each repeat. An informative engineering change is about 0.02 absolute mean total variation, corresponding to 2% of response mass shifted in the fit. This is a pilot scale for interpreting intervals. A physical tolerance requires an application and independent simulation provenance.
- At fixed K, compare independently fitted section integrals and center-density times section volume. Integration should matter most for narrow profiles and mixed module sizes. The direction of the difference after both operators are optimized is uncertain. Report both errors and their paired difference.
- Diagonal Gaussians assign positive response to instrumented sections observed at zero. Report the mass assigned to those sections, and the signed change in section count at a threshold of 1% of the event total. This relative threshold makes the occupancy check independent of the unknown response units and selects sections material to the single-event profile.
- Use the archived effective rectangular module footprints and one arbitrary depth unit per longitudinal section, so depth W1 is in section-index units. Fix XY minimum widths to 5 cm (a third of the smaller module pitch), and depth minimum width to 0.35 section (allows within-section localization without vanishing kernels). Record boundary hits. Three starts and a 200-iteration cap keep each CPU fit bounded. After systematic width-bound contact, the sensitivity replay halved both transverse and depth minima on the same rows with fixed optimizer and seeds. It measures their combined effect. The separate effects remain open.
- Use the most frequent exact stored momentum among rows with PDG −13 in the combined muon export. Generation vertices vary. The file has a scalar PDG and one vertex/momentum per row. Generator primary counts and event IDs are unavailable in this export. ROOT verification is documented for the standalone sample for module 19. Retain original zero-based row indices, record source checksum, and test channel sums and missing sections on the selected rows.
- Both arms minimize the same squared Hellinger loss on section fractions. The main reported score is total variation (half the L1 distance on section fractions). Also report transverse module TV, 1D Wasserstein distance over longitudinal section marginals, error in transverse RMS width, zero-channel mass, and fit wall time. Use 1,000 row-bootstrap replays for paired TV interval estimates, on the assumption that selected rows are independent. Repeat-wise estimates provide a sensitivity check on the sampled rows.

## Run locally

Use the independent CPU environment in `requirements-pilot-lock.txt` from the repository root. The historical exports record their actual versions. The portable evaluator uses only NumPy and SciPy. It consumes a versioned snapshot of the section map and checks the physical support and event totals.

```sh
export FHCAL_SOURCE=../fhcal-source
mkdir -p data outputs
git -C "$FHCAL_SOURCE" show 2007441:GAN_project/data/fhcal_data_muons_81619.zip > data/muon_archive.zip
unzip -p data/muon_archive.zip fhcal_data_muons_81619.npz > data/fhcal_data_muons_81619.npz
replay_dir=$(mktemp -d outputs/muon-replay.XXXXXX)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python -m pytest -q
python src/evaluate_gaussian_projection.py data/fhcal_data_muons_81619.npz \
  --pdg -13 --output "$replay_dir/pilot_results.json"
python src/evaluate_gaussian_projection.py data/fhcal_data_muons_81619.npz \
  --pdg -13 --min-scales 2.5 2.5 0.175 --output "$replay_dir/width_sensitivity.json"
```

For data with a verified primary-count field, pass `--primary-count-key NAME` to retain rows with count one. `--momentum PX PY PZ` and `--vertex X Y Z` select exact stored conditions when needed.

Each component is normalized over instrumented sections before mixing. Its weight is a share of the **registered** response. Both operators minimize half the sum of squared differences of the square-root section fractions, with a common numerical floor of 1e-15 inside each square root. TV is reported from the resulting prediction.

[Read the measured results](RESULTS.md) and the
[scope of the results](../../docs/experiment_state.md).

## Replay the illustrated events

The report illustrates low, median and upper-tail fit error within the
archived sample. For the integrated K=4 fits with halved width bounds,
select the rows closest to the 10th, 50th and 90th empirical TV percentiles
(linear interpolation). Ties use the lower zero-based NPZ row index.
Re-use the archived optimizer settings and seed for each row. Acceptance
requires the same archived shape metrics and optimizer status. Selection
uses fit error, so these examples illustrate that error distribution.
From the repository root, replay them after preparing the NPZ above:

```sh
python src/export_gaussian_projection_event.py data/fhcal_data_muons_81619.npz \
  --gallery --output "$replay_dir/example_events.json"
```

The exporter checks the archive and NPZ digests, geometry, fit metrics and
optimizer status. Plotting reads the committed section fractions without
reopening the original NPZ. Omit `--gallery` to reproduce the original
median-only `example_event.json` export.
