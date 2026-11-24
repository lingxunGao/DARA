"""Compatibility initialization for released response checkpoints."""
import torch
from torch import nn
from torch.nn import functional as F
from .model import ResponseBank

class SharedRateBank(ResponseBank):
    """Positive rates integrated exactly over eight physical-excess intervals.

    In each weather channel, x is thresholded excess in physical-scale units.
    Dose = sum_channel integral_0^x slope(context) * exp(rate_network) dz.
    A shared MLP reads interval intensity and channel identity; a bounded
    context adjustment allows different curves in different activity states.
    The last interval extends to infinity, so inputs are never clipped at a
    training maximum. Zero initialization recovers the original linear dose.
    Monotonicity holds only with context and spatial weights held fixed.
    """
    def __init__(self, cfg, threshold):
        super().__init__(cfg, threshold)
        lower = torch.tensor([0., .25, .5, 1., 1.5, 2., 3., 4.])
        self.register_buffer('interval_lower', lower)
        self.register_buffer('interval_width', torch.cat((lower.diff(), torch.tensor([float('inf')]))))
        identity = torch.eye(4)[:, None].expand(-1, len(lower), -1)
        level = torch.log1p(lower)[None, :, None].expand(4, -1, -1)
        self.register_buffer('rate_inputs', torch.cat((identity, level), -1))
        self.rate_network = nn.Sequential(nn.Linear(5, 16), nn.GELU(), nn.Linear(16, 1))
        self.rate_condition = nn.Linear(cfg.hidden, 4*len(lower))
        nn.init.zeros_(self.rate_network[-1].weight)
        nn.init.zeros_(self.rate_network[-1].bias)
        nn.init.zeros_(self.rate_condition.weight)
        nn.init.zeros_(self.rate_condition.bias)

    def parameters_for(self, context):
        p = super().parameters_for(context)
        global_log_rate = self.rate_network(self.rate_inputs).squeeze(-1)
        adjustment = .25*torch.tanh(self.rate_condition(context)).reshape(*context.shape[:-1], 4, -1)
        p['rate_multiplier'] = torch.exp(2*torch.tanh(global_log_rate + adjustment))
        return p

    def instantaneous(self, weather, parameters):
        excess = F.relu((weather-parameters['threshold'][:, None])/self.weather_scale)
        lengths = torch.minimum(F.relu(excess[..., None]-self.interval_lower), self.interval_width)
        integrated = (lengths*parameters['rate_multiplier'][:, None]).sum(-1)
        dose = (integrated*parameters['slope'][:, None]).sum(-1)
        return parameters['amplitude'][:, None]*(-torch.expm1(-dose))

