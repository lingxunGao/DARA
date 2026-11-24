# Experimental provenance and limits

The release implements the current no-response-output-gate model. Historical
gated Japan results (for example inflow MAE 617.738) are not the result of this
no-output-gate configuration and must not be relabeled as its performance.

## Legacy and target-disjoint split sizes

| Dataset | Original train / validation / test | Purged train / validation / test |
|---|---|---|
| Japan inflow | 1750 / 583 / 584 | 1745 / 578 / 584 |
| Japan outflow | 1750 / 583 / 584 | 1745 / 578 / 584 |
| Florida | 1030 / 343 / 344 | 1025 / 338 / 344 |

Purging keeps all original test windows, removes earlier windows that overlap
the next split's prediction targets, and refits preprocessing. Input histories
may overlap legitimately. The original selected metrics below were produced
with the original split and repeated test inspection; they are exploratory,
single-seed results, not fresh confirmation or a guaranteed outcome of retraining.

## Florida fixed o019 checkpoint replay, seed 0

Online window 12, learning rate 0.0003, anchor 0.1. The source checkpoint is from
the seed-zero rerun of the `loss_0017` recipe. The independent release replay
matches its online predictions exactly on the verified original runtime.

| Scope | MAE | RMSE |
|---|---:|---:|
| All test times, all regions | 189.786 | 551.921 |
| Disaster times, all regions | 202.020 | 602.247 |
| Disaster times, affected regions | 315.517 | 724.558 |
| All test times, other regions | 166.884 | 530.130 |
| Disaster times, other regions | 171.798 | 565.233 |

The full six-row comparison is in `reported_results/florida_o019_ablation.csv`.
Not all ablations are worse: shared response parameters slightly improve overall
MAE; freezing the full model improves disaster-time other-region MAE. These
outcomes are retained. A model can outperform a baseline table while an ablation
still improves a subset of metrics. No claim of universal component benefit is made.

Japanese validation-selected recipe snapshots have all-test MAE/RMSE:
inflow `n059_joint_random`: 589.861 / 1863.027;
outflow `n043_joint_random`: 615.894 / 2004.088.
Their six-row comparisons are also included. Other Japan candidates selected
using test thresholds exist in the research history; they are not presented as
unseen-test validation in this release.

## Interpretation

- Compare all methods using matched splits, scoring masks, units, data availability,
  and feedback timestamps. Access to observed feedback can improve ordinary baselines.
- Offline checkpoints are chosen on validation. Past selection of recipes/online
  settings using test metrics makes the associated reported results exploratory.
- Three regional scopes are evaluated retrospectively. Affected-region labels
  and future observed weather are not supplied to the model.
- Default weather persistence uses only the last observed weather. Retrospective
  observation products can have operational release delays not represented here.
- Temporal windows overlap within a split; they are not independent events.
- One seed does not estimate a standard deviation. Multiple seeds and untouched
  held-out events are needed for stronger empirical claims.
- Keeping a formula family fixed while learning its parameters is a modeling
  assumption, not evidence that a disaster phenomenon has a unique mathematical law.
