# Disaster Flow Response

Standalone release candidate for hourly disaster-related regional flow forecasting:
**6 observed frames → 6 predicted frames**. The current model combines a
**STAEformer + GRU backbone**, a **context-conditioned structured response branch**,
and **causal online adaptation of the shared linear prediction layer**.

[中文说明](README.zh-CN.md) · [Data format](docs/DATA_FORMAT.md) ·
[Protocol and reported results](docs/EXPERIMENT_PROTOCOL.md) ·
[Third-party attribution](THIRD_PARTY_NOTICES.md) · [License status](LICENSE_STATUS.md)

This archive contains source code, fixed recipes, synthetic-data generation,
training/evaluation/ablation commands, and tests. Original datasets, trained
weights, credentials, experiment caches, baseline implementations, and historical
hyperparameter-search launchers are not included. A synthetic example runs without
access to the original workspace. This is a preparation package for publication;
license selection and third-party redistribution permission remain to be resolved
as described in `LICENSE_STATUS.md`.

## Model

1. Encode six hourly flow observations and calendar features with STAEformer.
2. Encode the most recent `short_steps` observations with a GRU and fuse its
   features into STAEformer before the shared multi-step linear readout.
3. Build regional context from historical flow, deviations from a training-fitted
   profile, observed weather, lagged social counts, calendars, and cross-region
   messages.
4. Produce a bounded response `r` from weather inhibition, persistence/recovery,
   social anticipation, and learned supplementary temporal curves. Response
   parameters depend on regional context and observed evidence.
5. Forecast with `prediction = max(0, backbone * (1 + r))`.

There is **no response-output gate** in this version. The internal feature-fusion
gate between STAEformer and GRU is a separate part of the backbone. Some unused
gate tensors and compatibility initialization classes remain in the state dict
to preserve historical checkpoint keys, parameter initialization, and numerical
reproduction; they do not gate the response during prediction.

Offline training jointly optimizes the backbone and response branch. During
online evaluation, only independent increments to the final shared readout
weight and bias are optimized; encoder/GRU/context/response parameters remain
fixed. The effective readout is `W_checkpoint + delta_W`, `b_checkpoint + delta_b`.
The response values still change with the input evidence. This is not an extra
output-residual model. Forecasts already issued are immutable.

The numerical functions are researcher-specified. LLMs assisted offline literature
and qualitative-pattern organization; no LLM service is called during model
training or forecasting, and no claim is made that an LLM discovered the formulas
or learned their numerical parameters.

## Installation

Use Python 3.8 or newer and install PyTorch appropriate to your CUDA/CPU setup.
The original verification environment was Python 3.8, NumPy 1.24.3, and PyTorch
2.4.1+cu121. Compatibility requirements are broader; exact floating-point
reproduction across versions/hardware is not promised.

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

All commands also work directly from this unpacked directory without the editable
installation. The only runtime dependencies are NumPy and PyTorch.

## Runnable example without private data

From the repository root:

```bash
python -m disaster_response.demo --output demo_data

python -m disaster_response.cli train \
  --config demo_data/config.json --data demo_data/observations.npz \
  --output runs/demo_train --device cpu

python -m disaster_response.cli evaluate \
  --config demo_data/config.json --data demo_data/observations.npz \
  --checkpoint runs/demo_train/demo/trials/full/model.pt \
  --output runs/demo_eval --device cpu

python -m unittest discover -s tests -v
```

The example uses a small architecture, two training epochs, synthetic observations,
and target-disjoint (`purged`) splits. It verifies execution, not research accuracy.

## Fixed recipes for the three datasets

| Recipe | Dataset | Offline source | Online window / learning rate / anchor |
|---|---|---|---|
| `configs/florida_o019.json` | Florida hurricane, `us` | `loss_0017`, seed 0 | 12 / 0.0003 / 0.1 |
| `configs/japan_inflow_validation_selected.json` | Japan typhoon inflow, `japan` | `n059_joint_random`, seed 0 | 72 / 0.0001 / 1 |
| `configs/japan_outflow_validation_selected.json` | Japan typhoon outflow, `japan_outflow` | `n043_joint_random`, seed 0 | 72 / 0.0001 / 1 |
| `configs/japan_inflow_original.json` | Japan inflow | original no-output-gate recipe | 48 / 0.0001 / 0.1 |
| `configs/japan_outflow_original.json` | Japan outflow | original no-output-gate recipe | 48 / 0.0001 / 0.1 |

The Japanese validation-selected recipes are reproducible snapshots, not a claim
that the author has selected them as the final paper configurations. The Florida
recipe was selected using previously inspected test metrics. See the protocol note
before interpreting any reported result.

With locally authorized original-format data, for example:

```bash
python -m disaster_response.cli train \
  --config configs/florida_o019.json --raw-root raw_datas \
  --output runs/florida_train --device cuda:0

python -m disaster_response.cli evaluate \
  --config configs/florida_o019.json --raw-root raw_datas \
  --checkpoint runs/florida_train/us/trials/full/model.pt \
  --output runs/florida_eval --device cuda:0
```

Change the recipe for the Japanese datasets. The checkpoint subdirectory is
`japan/trials/full/model.pt` for inflow and `japan_outflow/trials/full/model.pt`
for outflow. A portable NPZ dataset can be supplied with `--data` instead of
`--raw-root`; its schema is in [DATA_FORMAT.md](docs/DATA_FORMAT.md).

Training writes `model.pt` (validation-selected), `latest.pt` (resumable optimizer,
scheduler, RNG and epoch state), validation metrics, and a data/config manifest.
Rerunning the same training command resumes the last saved epoch or skips an
already completed matching recipe. Change output directories when changing a
recipe or split. Do not run concurrent writers against the same output directory.

Evaluation writes predictions, frozen predictions, targets, issue origins,
`feedback_audit.json`, `final_readout_delta.pt`, and `results_wide_cn.csv`.
The final online delta represents the end of this evaluation stream, not a
replacement for reinitializing adaptation at the start of another evaluation.

## Complete model plus five ablations

```bash
python -m disaster_response.ablate \
  --config configs/florida_o019.json --raw-root raw_datas \
  --output runs/florida_ablations --device cuda:0
```

This trains four offline models and produces six rows:

| Version | Change |
|---|---|
| `full_online` | Full response model with shared-readout online adaptation |
| `without_response` | Backbone only, with the same online adaptation |
| `without_regional_adaptation` | Shared response parameters, keeping regional physical inputs |
| `without_online` | The full checkpoint evaluated without online updates |
| `response_mlp` | Bounded MLP response with the same context and multiplicative fusion |
| `backbone_frozen` | Backbone-only checkpoint evaluated without updates |

Each family uses the same seed, backbone initialization, optimization recipe,
training budget, validation selection criterion, and online settings. Each model
selects its own checkpoint on validation. Structured-response-specific shape
regularization does not apply to the MLP/backbone controls. The MLP is not
parameter-count matched. No test-score filter removes unfavorable ablations.

For a quick ablation execution check, use `demo_data/config.json` and
`--data demo_data/observations.npz` instead.

## Split and online-feedback protocol

Historical recipes explicitly set `split: "original"`: windows are generated
first, then split chronologically 60/20/20. This leaves **five target hours shared
across adjacent split boundaries**. It is retained only to reproduce that experiment.

For target-disjoint runs, add `--split purged` to **both training and evaluation**
and use a new output directory. Purging removes the last five training and last
five validation windows while keeping the test windows. All preprocessing is
refit on retained training inputs. Training an original-split checkpoint and
evaluating it with a purged-split flag does not repair training leakage and is
rejected by the checkpoint/data checks. Historical inputs may legitimately overlap
between splits; future prediction targets do not in the purged policy.

At a forecast issued after observing hour `t`, only labels with timestamp `<= t`
may update the readout. A past six-step prediction contributes individual lead
labels as they arrive. Predictions for `t+1` through `t+6` are issued after that
update. The feedback buffer is measured in observed hours, not mini-batches.
One Adam step uses node-normalized Smooth L1 loss, an anchor to the original
forecast on the same revealed examples, and gradient clipping. No future labels
are supplied to the response branch. Caching historical features is a replay
optimization and does not change label availability.

## Five evaluation scopes

Each row reports MAE and RMSE in original flow units:

1. All test times, all regions.
2. Fixed disaster times, all regions.
3. Fixed disaster times, selected affected regions.
4. All test times, the complement of selected affected regions.
5. Fixed disaster times, the complement of selected affected regions.

Zero-valued targets are masked, matching the research protocol. A disaster window
is included only if all six input and six target hours lie inside the event
interval. Affected-region labels are used only for retrospective evaluation.
The complement does not establish that a region experienced no disaster impact.

## Source map and verification

- `model.py`, `hybrid_backbone.py`, `staeformer.py`: model and backbone.
- `knowledge_response.py`: current structured response library.
- `data.py`, `datasets.py`: causal input builder, training-only preprocessing, loaders.
- `training.py`: isolated original joint-training kernel.
- `backbone_online_adaptation.py`: shared-readout adaptation and label replay.
- `cli.py`, `ablate.py`, `inference.py`: user-facing entry points.
- `contextual_response.py`, `response_shape.py`: compatibility initialization helpers.

The release was checked against six existing checkpoints for strict loading and
identical parameter initialization. Its Florida frozen and online predictions on
all 344 test windows matched the original runtime exactly. Synthetic training,
evaluation, all six ablation rows, target-disjoint splits, and future-label
perturbation checks were also tested. See `docs/VALIDATION_REPORT.json`.

STAEformer should be cited when using this backbone. Its original implementation
is attributed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No citation for
an unpublished project paper or invented author list is supplied.
