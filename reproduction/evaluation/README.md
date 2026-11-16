# Evaluation records

These records accompany the manuscript's evaluation protocols and results.
The original `v0.2.0-submission` tag and the earlier evidence directories remain
historical snapshots. No model is trained and no new model inference is performed
here.

## Recompute the primary comparison

From the repository root, with NumPy and SciPy installed:

```bash
python reproduction/evaluation/rescore_primary.py --check --output primary_recomputed.json
python reproduction/evaluation/rescore_primary.py --policy ts-control --check
```

The default uses KmerFormer 250M's validation-fitted TS-MSP cutoff,
MetaTransformer 250M's original raw-MSP threshold of 0.5, and genus-aware
Kraken2's native declines. It removes every read declined by any method, leaving
98,272 identical rows. The optional control gives MetaTransformer an additional
temperature-scaled, validation-fitted rule; that is a different output policy.
Both commands verify input hashes, then recompute read accuracy, errors,
McNemar counts, 1,000 paired read-bootstrap intervals, abundance, detection and
1,000 paired community-bootstrap intervals against the stored results.

Temperature fitting minimizes validation NLL. Cutoff selection separately targets
99.065% validation coverage, derived from Kraken2's coverage on the primary test
pool. This target therefore uses test information. The main comparison keeps
MetaTransformer's original output rule, a choice made after both versions had
been run; the temperature-scaled control is the second command above. This is an exploratory
comparison of specified pipelines on their jointly accepted subset.

Detection communities are redrawn from the retained pool: 200 sparse communities
of 1,000 reads and 60 selected genera out of 120, seed 42. The 98 partition samples
serve abundance estimation. Smaller gaps than the full-pool analysis reflect a
changed population and new communities. Interpolated sensitivity at exactly
95% or 99% specificity can include randomized decisions between thresholds;
it differs from the earlier best realizable threshold satisfying a specificity
constraint. Bootstrap intervals condition on fixed model predictions.

## Files and scope

- `primary_scores.npz`: labels, deterministic genus predictions, KF TS-MSP,
  MT raw/TS-MSP, and Kraken2 genus predictions. Selected arrays retain source
  dtypes and ordering. The sources do not contain read IDs; equality of ordered
  labels is checked, and original row alignment is inherited from the experiment.
- `protocol.json`: temperatures, cutoffs, target provenance and sampling rules.
- `primary_reference.json`, `primary_ts_control.json`: stored primary results.
- `primary_sensitivity.json`: the reported KF-validation-rule and Kraken2-confidence
  sensitivity results. These are summary records; alternate Kraken2 runs are not
  recomputed by the primary scorer.
- `all_read_metrics.json`: revised all-read and Kraken2-classified-subset metrics,
  with the interpolated and earlier realizable detection operating points.
- `uncertainty_reported_values.json`, `strand_reported_values.json`: the values used
  in the uncertainty and strand-control tables. These are summaries; they do not
  provide the models/features needed to repeat MC dropout, Laplace or retraining.
- `kraken_genus_aware/`: predictions from the October 30 genus-aware rerun on the
  primary pool, ANI ladder, L3 and log-normal community. Earlier
  `reproduction/ladder/` Kraken2 arrays are the species-only comparison.
- `manifest.json`: original and delivered checksums and transformations.
- `scorer_provenance.json`: source identity of the extracted sample-level scorer.

The scorer reuses the numerical functions of the archived sample-level and
common-read analyses. The two methods' probabilities are already calibrated
where stated; validation logits, training pools and checkpoints are external
inputs. See
[the asset inventory](../../docs/MISSING_ASSETS.md) for access limitations and
[the evaluation guide](../../docs/EVALUATION.md) for interpretation.
