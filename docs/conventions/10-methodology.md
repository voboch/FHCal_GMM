# Methodology — how a claim becomes a result

Derived from `HIC-NeuralFusion/.claude/agents/paper-critic.md`,
`docs/submission_checklist.md`, `docs/weekly_status_*.md`,
`docs/open_decisions_20260929.md`,
`HGNDRecoGNN/docs/HGND_PROGRESS_GUIDE.md`,
`HGNDRecoGNN/docs/plotting_style_protocol.md` §9,
`HGNDRecoGNN/AGENTS.md`.

Both projects arrived at the same discipline by making the same mistakes. This
is that discipline, stated once.

---

## 1. The claim lifecycle

A number moves through five states. Skipping a state is how the withdrawals in
both projects happened.

```
  measured  →  reproduced  →  attributed  →  canonical  →  published
```

| state | the bar |
|---|---|
| **measured** | a number exists, with its sample size and the run that produced it |
| **reproduced** | the same number from a second seed, or a second evaluation of the same weights, with the spread reported |
| **attributed** | the mechanism is instrumented, not inferred — see §3 |
| **canonical** | it is in the project's single results table, and nothing else computes it independently — see §5 |
| **published** | it is in a draft with a caption or sentence that states its uncertainty and its provenance |

A result quoted in conversation may be *measured*. A result in a weekly status
should be *reproduced*. A result in a paper or thesis sentence must be
*canonical*.

## 2. Predictions are committed before the measurement

Where a change is expected to move a number, **write the expected direction and
rough size down first**, in the plan document, then measure. HIC-NeuralFusion
does this (`Expose the energy ledger for evaluation, with predictions committed
first`) and it is the cheapest available protection against reading a result
backwards.

`docs/open_decisions_20260929.md` is the model shape for this. Each open
decision carries: **what** it is, **evidence it is the right target** (with the
numbers), **size** of the expected effect, **cost** in jobs and minutes, **the
risk that decides the implementation**, and **what it does not fix**. The last
two fields are the load-bearing ones.

## 3. Instrument the call; do not infer from outputs

The repeated failure in both projects is diagnosing a mechanism from its
downstream numbers. HIC-NeuralFusion withdrew four consecutive diagnoses of the
charge-closure bug this way (`weekly_status_20261002.md`):

1. a species-id collapse that could not happen (no PDG code to lose);
2. fractional-count rounding — real, fixed, and a **no-op** on the live path;
3. "the function is not on the eval path" — the tracer was armed after the loop
   it was meant to watch;
4. "the clamp can be short by at most one proton" — accepted from a review
   without checking.

The pattern in 1, 2 and 4 is inference from outputs, or deference, instead of
instrumenting the call. The rule:

- **A flag in an invocation string is evidence of intent, not of effect.**
  Verify against the effective config the run actually used
  (`live_status.json` in HIC-NeuralFusion, the run config in HGND). Silent
  no-ops have bitten HIC-NeuralFusion twice.
- **A mechanism is attributed when a test pins it**, ideally parametrised
  (short-by-`n` ⇒ overshoots by `n`), not when a story fits the numbers.
- **A finding handed to you by a review is a hypothesis**, including findings
  from an automated critic. Check it.

## 4. Statistical rigour

### 4.1 Uncertainty: measure the clustering factor, then state it

The naive rule is "always bootstrap at event level, because per-particle errors
treat tracks in an event as independent and understate the error". That is
right for per-event yields — the HGND band single-species yields have an
event-clustered error **2.31×** the per-particle value. It is *not* right
everywhere: for the HGND spectral-hardness ratio `R` the measured inflation is
**1.01 / 1.09 / 0.98**, consistent with unity, because the numerator is rare
(~0.37 high-energy neutrons per event) and carries little intra-event
correlation.

**The shared rule is therefore: measure the clustering factor for the estimator
you are quoting, state it in the caption, and use the event-level bootstrap
where it exceeds unity.** Not "always bootstrap"; not "per-particle is fine".
State the number of resamples.
(`HGNDRecoGNN/results/spectral_hardness/event_bootstrap_check.txt`, 400 resamples.)

### 4.2 Pair everything that can be paired

Where two arms share events — the usual case in both projects, since the same
events are transported, or the same clusters re-scored — report the **paired
difference with its own bootstrap**, not two separately errored curves for the
reader to difference by eye. Paired tests are what made both projects'
strongest results legible:

- HIC-NeuralFusion quotes cost "paired by evaluation seed on fixed weights".
- HGND's central result is a **paired displacement** (`−8.66 % ± 2.21 %`,
  3.91σ) between true and reconstructed cluster energy on the *same selected
  clusters*.

### 4.3 Single-seed numbers are flagged, not quoted

A single-seed headline is a finding to be strengthened, not a result. Three
seeds, with the spread, or say "single seed" in the sentence.

### 4.4 Significant figures track the eval variance

Do not quote `KS = 0.02345` from a 500-event evaluation where the fourth digit
is noise. Round to what the measured spread supports.

### 4.5 Sample size travels with the number

A KS value without its sample size is not interpretable; below ~1000 events the
tails are underestimated. Write `KS = 0.011 (n = 2000 events)`.

## 5. One number, one source

**Every result-dependent number in a manuscript or thesis comes from one
generated table.** HIC-NeuralFusion's is `docs/canonical_results.json` →
`canonical_results.tex`, imported by the manuscript; HGND's is the per-dataset
sensitivity summary CSV that feeds `paper/main.tex`.

Consequences, both learned the hard way:

- A number recomputed in a notebook for a talk is a **second source** and will
  drift. Export it from the canonical table or add it to the table.
- When the release definition changes, the table is regenerated and every
  dependent figure and sentence is re-checked. A pooled figure must not be
  quoted for a release that is a switch between two specialists.
- The canonical table records **provenance per entry**: which run, which
  checkpoint, which seed, which event count, which date.

## 6. Fair-baseline discipline

A comparison is admissible only if the arms are commensurable.

- **Same preprocessing on both sides.** HGND's largest artefact was a
  `StandardScaler` fitted **per dataset** while `eToF` carries the neutron
  energy: the harder sample got a 1.9 % wider scale and part of the effect
  being measured was divided out before the network saw it. Rebuilding the
  caches with one shared scaler, on the *identical checkpoint*, moved the
  differential response from **−30.7 ± 8.6 MeV (3.55σ)** to
  **−11.4 ± 9.8 MeV (1.16σ)** and flipped the reconstructed sign to agree with
  truth. No retraining was involved. **A preprocessing statistic fitted
  per-arm is a leak of the thing being measured.**
- **Same jitter, window, population, estimator and sample size.** Two arms
  measured on different populations, sizes or estimators must not be plotted or
  tabulated as if commensurable (`plotting_style_protocol.md` §9.7). Where they
  must be shown together, group by population and say so in the caption;
  comparisons are meaningful within a block only.
- **Report the baseline's own distance from truth.** "45 % better than the
  prior" means little if the prior is already at `KS = 0.02`.
- **One ablation at a time.** A change that removes two components jointly is
  not evidence about either.

## 7. The physics-artefact sweep

Run this over the raw per-species, per-component tables of any run before
scoring a claim from it. Six review passes in HIC-NeuralFusion audited claims
against evidence and never audited the evidence against physics; three large
artefacts sat in the summary CSVs the whole time.

1. **Degenerate distributions.** A generated standard deviation of exactly
   `0.0` against a reference with real spread is a failure, not a success.
   *Seen:* generated neutrons `153.0 ± 0.00` vs reference `143.8 ± 4.67`.
2. **Suspiciously exact agreement.** `wasserstein = 0.0` usually means the
   quantity is copied through from the reference, not predicted. A pass-through
   must never be reported as model performance. *Seen:* every meson species.
3. **Conserved-quantity signatures.** Totals equal to a fixed budget
   (`153 + 109 = 262 = A(Xe) + A(Cs)`) are legitimate by construction but must
   not be sold as a learned distributional match — they show up as a delta
   function where the reference has spread.
4. **Symmetry violations.** `v₁` is antisymmetric in rapidity; forward and
   backward spectators must roughly cancel. A single-signed `v₁` on both sides
   is unphysical. *Seen:* `+0.030` spectators, `+0.022` participants, against
   references at `~0`.
5. **Prior beats model, per channel.** Compare *both* the prior and the model
   against the reference for every species and harmonic. If the prior is
   closer, the model degrades that channel and no aggregate may hide it.
   *Seen:* participant `v₁`, prior `−0.0006` vs generated `+0.0224` against a
   reference of `−0.0007`.
6. **Per-species regressions under a healthy aggregate.** *Seen:* kaons
   degraded (`KS 0.020 → 0.040`), pions unchanged, only nucleons improved —
   while the aggregate looked healthy at `0.012`.
7. **One-signed bias.** `bias == −mae` to the digit means every bin sits below
   the reference: a systematic offset, not scatter. High `corr` with large
   `bias` is "right shape, wrong offset" (fixable); `corr ≈ 0` means the
   observable is not modelled at all.
8. **Trajectory vs endpoint.** Read the full training history. A metric whose
   trajectory minimum is far better than its last value means the number
   depends entirely on checkpoint selection.
9. **Did the run finish?** Cross-check the recorded status and step against the
   length of the eval history. An aborted seed averaged into a multi-seed mean
   invalidates the mean *and* its spread.
10. **Did the flags apply?** See §3.

For HGND the same sweep reads: efficiency spanning a factor **39.6** in
`E_kin` means a ratio whose denominator sits below 1 GeV is a ratio of a 2 %
efficiency — check what the denominator is actually made of before quoting the
ratio.

## 8. Domain-response errors are not fixed by statistics

HGND's decisive finding: the GNN energy estimate changed the *sign* of the
neutron response on the same selected clusters. "More events would make the
wrong answer more precise." When the estimator responds differently to the two
samples being compared, the effect is a systematic, and precision is not the
remedy — a shared-preprocessing fix, a calibration check, or a different
estimator is.

**Any cross-sample comparison passed through a learned estimator must show that
the estimator responds identically to both samples**, or report the differential
response as a systematic with its size.

## 9. Selections must be the thing they claim to be

HGND's `θ ∈ [8.9°, 13.1°)` band, used for every truth-level number, **is not the
detector**: the front face is a rectangle offset to negative `x` covering 20 % of
azimuth, and because the generator holds the reaction plane fixed with a
strongly energy-dependent `v₁`, the azimuth cannot be weighted away. The
corrected window roughly doubles the isovector signal.

**An acceptance written as a simple interval is a hypothesis about the
geometry.** Check it against the geometry before the numbers built on it are
quoted — and where the generator fixes the reaction plane, azimuthal acceptance
does not factorise.

## 10. Withdrawal is part of the record

Both projects keep withdrawn claims, with the reason, in the source record
and version history — not deleted. Correct any affected current reader-facing
claim. A new report need not reproduce every withdrawn row; include the
withdrawal visibly when it changes the interpretation of the current result.
HIC-NeuralFusion's weekly status opens with "four of my
own claims were withdrawn this week"; `submission_checklist.md` keeps a
"**Superseded** — two diagnoses proposed, both rejected" block.

- A withdrawn claim is recorded with: what was claimed, what was actually true,
  and **what kind of mistake it was** (inference from outputs, deference,
  stale memory, metric defect).
- A `REOPENED` status is a legitimate state for a checklist item and must not be
  quietly flipped back to done.
- **Negative results are results.** A failed ablation belongs in the record; its
  absence is a referee objection.

## 11. Every fixed parameter carries its justification

Promoted from `HGNDRecoGNN/AGENTS.md`, unchanged in substance. **No chosen
constant may appear without its justification attached, at the point of use.**
An energy threshold, a score cut, a binning, a fit range, a rejection criterion,
a `b` window, a loss weight, a warmup length: each arrives with both of —

1. **A plot, whenever the choice is visually representable.** If the parameter
   was picked by optimising something, show the objective *over* the parameter,
   not only its argmax. A scan surface makes "a broad optimum" checkable and
   exposes the case where the optimum is a spike, a boundary artefact or a tie.
   If the parameter follows from a physical feature, plot the feature.
2. **A comment saying why, in the code and in the prose.** State what the
   parameter trades off and what would change if it moved. "Chosen on
   development data" is provenance, not justification.

The failure mode this prevents has occurred more than once in HGND: a
polar-angle band inherited without checking it against the detector geometry
(§9 above), an energy denominator placed where the detector records two per cent
of what arrives, and `~2.2 GeV` quoted as a window centre with nothing behind
it. Each read as settled because it was written down.

The worked example to copy: `scripts/hardness_threshold_opt.py` searches the
centre and half-gap, `scripts/plot_hardness_surface.py` draws response and
significance over the whole scanned plane with the unordered region masked, and
the spectra figure shows the isovector signal crossing unity near 1.8–2.0 GeV —
the physical reason the optimum sits just above it. The number, the surface it
came from, and the physics that explains the surface are all in the note.

For an artifact consuming an already checked result, link the existing
justification; do not repeat every scan in its main sequence. Keep a choice
and its limitation visible when they change the claim. Put detailed scans
in the methods or appendix. An inherited parameter without a scan remains
identified as inherited; rewriting a report does not authorise a new scan.
New scientific choices still follow the requirements above.

**A parameter whose only justification is convention says so in those words**,
so a reader sees it is unexamined rather than assuming it was derived.
HIC-NeuralFusion's default loss weights, ramp schedules and the 20-step Heun
solver are the obvious candidates: most are inherited, and the thesis should say
which.

## 12. Truth and reconstruction are never mixed silently

Every figure, table and distribution states whether it is **MC truth** or
**reconstructed** — in the caption *and* on the panel itself (HGND:
`plot_b_full.mark_provenance`).

- **Truth** uses information available only in simulation: the generated nucleon
  list, true kinetic energy, the truth-matched cluster label. It describes what
  a perfect detector would measure.
- **Reconstructed** uses only what the pipeline derives from hits.
- A figure travels without its caption into talks and slides, which is why the
  panel carries the label too. Mixing the two silently in one figure made an
  earlier HGND draft read as though the detector had measured the truth-level
  response.

For HIC-NeuralFusion the same distinction is **reference / prior / model**: a
quantity copied through from UrQMD or AAMCC is not a model output (§7.2), and a
property of the prior sampler is not a property of the learned field
(honesty rule 6).

## 13. The honesty rules

These have each been violated at least once in these projects.
(`plotting_style_protocol.md` §9, promoted here unchanged.)

1. A figure must not be drawn without its reference arm.
2. A caption must not claim an improvement that one of its panels contradicts.
3. Pseudorapidity must not be labelled rapidity.
4. A mixed-species curve must not be labelled by a single species.
5. Per-particle errors must not be presented as event-level errors.
6. A quantity that is a property of the prior must not be presented as a
   property of the learned model.
7. Two arms measured on different populations, sample sizes or estimators must
   not be plotted as if commensurable.

Two more, from the subsequent year of both projects:

8. A number from a superseded data generation or a superseded release must not
   be quoted as current (HGND `_v2`; HIC pre-cap release numbers).
9. A preprocessing statistic must not be fitted per-arm in a between-arm
   comparison.
