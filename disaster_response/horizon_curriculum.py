"""Training-only near-to-far loss weighting for direct multi-horizon forecasts."""
import torch


def horizon_weights(epoch, horizon, end_epoch, floor=.25):
    if epoch < 1 or horizon < 1 or end_epoch < 2 or not 0 < floor <= 1:
        raise ValueError('Invalid horizon curriculum configuration')
    progress = min(1., (epoch-1)/(end_epoch-1))
    active = 1+(horizon-1)*progress
    return [floor+(1-floor)*min(1., max(0., active-h)) for h in range(horizon)]


def weighted_mask_mean(values, mask, weights):
    """Normalize by valid weighted mass, including groups with missing targets."""
    if not bool(mask.any()):
        return values.sum()*0
    shape = [1, len(weights)]+[1]*(values.ndim-2)
    mass = values.new_tensor(weights).reshape(shape).expand_as(values)
    return (values[mask]*mass[mask]).sum()/mass[mask].sum()


def balanced_prediction_loss(prediction, target, group, scale, weights):
    error = (prediction[..., 0]-target[..., 0]).abs()/scale
    valid = target[..., 0] != 0
    losses = [weighted_mask_mean(error, valid & (group == g), weights)
              for g in range(3) if bool((valid & (group == g)).any())]
    balanced = torch.stack(losses).mean() if losses else error.sum()*0
    return .5*weighted_mask_mean(error, valid, weights)+.5*balanced
