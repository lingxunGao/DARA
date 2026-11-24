# DARA

PyTorch implementation of a disaster-aware flow forecasting model: an
STAEformer–GRU backbone, context-conditioned response functions, and online
adaptation of the shared prediction layer. Six observed steps predict six future steps.

## Installation

```bash
pip install -r requirements.txt
```

## Model interface

```python
from disaster_response.model import ModelConfig, StructuredDisasterModel

model = StructuredDisasterModel(ModelConfig(nodes=num_nodes), preprocessing)
prediction, details = model(inputs)  # prediction: [batch, 6, nodes, 1]
```

`preprocessing` contains `node_scale` (positive values, `[nodes]`), `graph`
(nonnegative adjacency, `[nodes, nodes]`), `weather_threshold` (`[2]`),
`mean` (scalar), and `std` (positive scalar), fitted on training data.

`inputs` is a dictionary of finite floating-point tensors on the model's device:

| Keys | Shape |
| --- | --- |
| `flow`, `anomaly`, `history_reference`, `future_reference` | `[batch, 6, nodes]` |
| `weather`, `future_weather`, `social`, `excess` | `[batch, 6, nodes, 2]` |
| `history_calendar`, `future_calendar` | `[batch, 6, 4]` |

Weather channels are wind (m/s) and rainfall (mm/hour). Calendar channels are
sine/cosine pairs for hour of day and day of week. Future weather must be
available at prediction time. Online readout adaptation is implemented in
`disaster_response/backbone_online_adaptation.py`.

## Attribution

The backbone includes code adapted from STAEformer. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[LICENSE_STATUS.md](LICENSE_STATUS.md) for attribution and licensing status.
