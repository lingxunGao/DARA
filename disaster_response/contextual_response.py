"""Shared piecewise responses, history memory and a context/horizon gate.

All inputs are issue-time available. Monotonicity is conditional on fixed
context and graph, and is not a claim about total predicted flow.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .response_shape import SharedRateBank
from .model import ResponseBank


class SharedConstant(nn.Module):
    """Learnable global offset replacing a context-conditioned parameter head."""
    def __init__(self, bias):
        super().__init__()
        self.bias = nn.Parameter(bias.detach().clone())

    def forward(self, context):
        return self.bias.expand(*context.shape[:-1], -1)


class ContextualBank(SharedRateBank):
    def __init__(self, cfg, threshold):
        super().__init__(cfg, threshold)
        count = 8 if cfg.response_curve == 'piecewise' else 1
        self.memory_rates = nn.Parameter(torch.zeros(4, count))
        self.warning_rates = nn.Parameter(torch.zeros(count))
        self.warning_threshold = nn.Parameter(torch.tensor(-2.))
        self.memory_amplitude = nn.Parameter(torch.tensor(-1.))
        self.duration = nn.Linear(cfg.hidden, 2)
        nn.init.zeros_(self.duration.weight)
        nn.init.zeros_(self.duration.bias)
        if cfg.response_curve == 'single':
            del self.rate_network, self.rate_condition
        if not cfg.condition_response:
            for name in ('condition', 'anticipation', 'duration', 'rate_condition'):
                if hasattr(self, name):
                    setattr(self, name, SharedConstant(getattr(self, name).bias))

    def parameters_for(self, context):
        if self.cfg.response_curve == 'single':
            return ResponseBank.parameters_for(self, context)
        return super().parameters_for(context)

    def instantaneous(self, weather, parameters):
        if self.cfg.response_curve == 'single':
            return ResponseBank.instantaneous(self, weather, parameters)
        return super().instantaneous(weather, parameters)

    def forward(self, context, history_weather, future_weather, social):
        p = self.parameters_for(context)
        extra = .25*torch.tanh(self.duration(context))
        amplitude = torch.sigmoid(self.memory_amplitude+extra[..., 0])
        rates = torch.exp(2*torch.tanh(self.memory_rates))

        def memory_input(weather):
            excess = F.relu((weather-p['threshold'][:, None])/self.weather_scale)
            if self.cfg.response_curve == 'single':
                dose = (excess*rates[:, 0]).sum(-1)
            else:
                lengths = torch.minimum(F.relu(excess[..., None]-self.interval_lower), self.interval_width)
                dose = (lengths*rates).sum((-1, -2))
            return amplitude[:, None]*(-torch.expm1(-dose))

        hist = memory_input(history_weather)
        future = memory_input(future_weather)
        rho = p['retention'] if self.cfg.memory else torch.zeros_like(p['retention'])
        state = torch.zeros_like(hist[:, 0])
        for value in hist.unbind(1):
            state = self.advance(value, state, rho)
        initial_state = state
        instant = self.instantaneous(future_weather, p)
        persistent, total = [], []
        for immediate, write in zip(instant.unbind(1), future.unbind(1)):
            carry = rho*state
            persistent.append((1-immediate)*carry)
            total.append(immediate+(1-immediate)*carry)
            state = self.advance(write, state, rho)
        suppression = torch.stack(total, 1)
        warning = F.relu(social[:, -1, :, 1]-F.softplus(self.warning_threshold))
        if self.cfg.response_curve == 'single':
            dose = warning*torch.exp(2*torch.tanh(self.warning_rates[0]))
        else:
            lengths = torch.minimum(F.relu(warning[..., None]-self.interval_lower), self.interval_width)
            dose = (lengths*torch.exp(2*torch.tanh(self.warning_rates))).sum(-1)
        warning_strength = -torch.expm1(-dose)
        tau = 1+11*torch.sigmoid(extra[..., 1])
        steps = torch.arange(1, 7, device=context.device, dtype=context.dtype)
        anticipation = (p['anticipation'][:, None]*warning_strength[:, None]
                        *torch.exp(-steps[None, :, None]/tau[:, None]))
        return anticipation-suppression, dict(instantaneous=instant, suppression=suppression,
            persistent=torch.stack(persistent, 1), anticipation=anticipation,
            initial_state=initial_state, retention=rho, parameter_offset=p['offset'])


class ContextualGate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.condition_response = cfg.condition_response
        self.horizon = nn.Embedding(cfg.horizon, 8)
        self.net = nn.Sequential(nn.Linear(cfg.hidden+12, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, 1))
        self.node_bias = nn.Parameter(torch.full((cfg.nodes,), -1.))

    def forward(self, local, indirect, anomaly, social, context):
        features = torch.stack((local, indirect, anomaly, social), -1).clamp_min(0)
        evidence = -torch.expm1(-features)
        if self.condition_response:
            context_input = context
        else:
            context_input = torch.zeros_like(context)
        h = context_input[:, None].expand(-1, local.shape[1], -1, -1)
        step = self.horizon.weight[None, :, None].expand(h.shape[0], -1, h.shape[2], -1)
        weight = torch.sigmoid(self.net(torch.cat((h, step, evidence), -1)).squeeze(-1)
                               +self.node_bias[None, None])
        # Flow deviation informs the weight, but cannot activate a response alone.
        applicability = -torch.expm1(-(features[..., 0]+features[..., 1]+features[..., 3]))
        return weight*applicability
