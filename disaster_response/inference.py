"""Raw six-hour inference using only saved preprocessing and available history."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .data import build_inputs
from .model import StructuredDisasterModel


def load_model(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if 'experiment_model' in saved:
        raise ValueError('Historical diagnostic checkpoints are not supported')
    model = StructuredDisasterModel(saved['config'], saved['preprocessing'])
    model.load_state_dict(saved['state_dict'])
    return model.to(device).eval(), saved['preprocessing']


@torch.no_grad()
def predict(model, preprocessing, *, flow, weather, times, node_names, tweet_counts,
            tweet_times, tweet_available_at=None, forecast=None):
    if not np.array_equal(node_names, preprocessing['node_names']):
        raise ValueError('Inference node order differs from the checkpoint')
    inputs = build_inputs(flow, weather, times, preprocessing, tweet_counts=tweet_counts,
                          tweet_times=tweet_times, tweet_available_at=tweet_available_at, forecast=forecast)
    device = next(model.parameters()).device
    prediction, details = model({k: v.to(device) for k, v in inputs.items()})
    return prediction.cpu().numpy(), {k: v.cpu().numpy() for k, v in details.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True, help='Timestamped NPZ; see README')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    model, prep = load_model(args.checkpoint, args.device)
    with np.load(args.input, allow_pickle=False) as f:
        kwargs = {k: f[k] for k in ('flow', 'weather', 'times', 'node_names', 'tweet_counts', 'tweet_times')}
        if 'tweet_available_at' in f:
            kwargs['tweet_available_at'] = f['tweet_available_at']
        forecast_keys = ('forecast_weather', 'forecast_issued_at', 'forecast_valid_times')
        if any(k in f for k in forecast_keys):
            if not all(k in f for k in forecast_keys):
                raise ValueError('Forecast needs weather, issued_at and valid_times')
            kwargs['forecast'] = {k[len('forecast_'):]: f[k] for k in forecast_keys}
        prediction, details = predict(model, prep, **kwargs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, prediction=prediction, **details)


if __name__ == '__main__':
    main()
