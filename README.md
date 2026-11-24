# DARA

Disaster-aware flow forecasting with an STAEformer–GRU backbone,
context-conditioned responses, and online readout adaptation.
The model uses six observed hours to predict the next six hours.

## Install

```bash
pip install -r requirements.txt
```

## 1. Offline training

Jointly train the backbone and response modules on historical observations:

```bash
python main.py train --data data/train.npz --output runs/offline --device cpu
```

The final 20% of observations are reserved for chronological validation,
with no overlapping training/validation targets. Preprocessing is fitted only
on training inputs. The checkpoint with the lowest validation loss is saved
as `runs/offline/model.pt`. Use `--epochs` to set the training budget.

## 2. Online prediction and update

Load the offline checkpoint and replay subsequent hourly observations:

```bash
python main.py online --data data/online.npz --checkpoint runs/offline/model.pt --output runs/online --device cpu
```

The backbone encoder and response modules stay frozen. As true observations
arrive, only the shared prediction layer's weight and bias increments are
updated, before issuing the next forecast. Future labels are never used.
Each invocation starts a new replay from the offline checkpoint; it is not a
live data service. Outputs are `predictions.npz` and `readout_delta.pt`.

## Data

Supply your own NPZ files with these fields (`T`: hours, `N`: nodes):

| Field | Shape / meaning |
| --- | --- |
| `flow` | `[T, N]`, nonnegative observed flow |
| `weather` | `[T, N, 2]`, wind (m/s) and rainfall (mm/hour) |
| `tweet_counts` | `[T, N]`, social counts from the preceding hour |
| `times` | `[T]`, consecutive hourly timestamp strings |
| `tweet_times` | `[T]`, exactly one hour before `times` |
| `node_names` | `[N]`, unique strings in the same order in both files |

Use at least two nodes and 48 hours for offline training with the default split.
Online data needs at least 12 hours; the first six are initial history, and
all forecast targets must follow the offline training/validation period.
Future weather defaults to persistence of the latest observation.
Use `--device cuda:0` for GPU execution and `python main.py train --help` or
`python main.py online --help` for runtime options.

## Attribution

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[LICENSE_STATUS.md](LICENSE_STATUS.md) for source attribution and licensing status.
