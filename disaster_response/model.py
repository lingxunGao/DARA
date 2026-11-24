"""Two predictive paths. All public predictions are in original flow units.

Weather channel order: wind (m/s), precipitation (mm/hour). The context
encoder never predicts a flow residual. Shape guarantees are conditional on
fixed context/graph; no monotonicity is imposed on the final flow forecast.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    nodes: int
    history: int = 6
    horizon: int = 6
    hidden: int = 48
    heads: int = 4
    layers: int = 2
    dropout: float = .1
    response: str = 'structured'
    cross_region: bool = True
    gate: bool = False
    memory: bool = True
    social: bool = True
    isolate_history: bool = False
    anticipation_bound: float = .3
    backbone: str = 'stae_gru'
    short_steps: int = 3
    gru_hidden: int = 64
    stae_layers: int = 3
    fist_dim: int = 64
    fist_heads: int = 8
    fist_layers: int = 2
    fist_ff: int = 2048
    fist_edges: int = 274
    fist_fan: bool = True
    contextual_response: bool = True
    response_curve: str = 'piecewise'
    response_library: str = 'knowledge_hybrid'
    context_fusion: str = 'original'
    condition_response: bool = True
    shared_response_parameters: bool = False
    mlp_response_bound: float = 1.
    dynamic_mode: str = 'flow'
    cross_region_topk: int = 0
    nonnegative_output: bool = True
    locality_mode: str = 'original'
    mechanism_mode: str = 'none'
    operator_mode: str = 'none'
    operator_static: bool = False
    spatial_experts: str = 'none'
    gate_parameter_mode: str = 'fixed'
    impact_support: str = 'none'

    def __post_init__(self):
        if (self.history,self.horizon)!=(6,6) or self.backbone!='stae_gru' or self.gate or not self.contextual_response:
            raise ValueError('This release supports the current no-output-gate STAE-GRU 6-to-6 family only')
        if (self.context_fusion!='original' or self.locality_mode!='original'
                or self.mechanism_mode!='none' or self.operator_mode!='none'
                or self.spatial_experts!='none' or self.gate_parameter_mode!='fixed'
                or self.impact_support!='none' or self.operator_static):
            raise ValueError('Historical diagnostic modules are not part of this release')
        if self.response_library != ('knowledge_hybrid' if self.response=='structured' else 'manual'):
            raise ValueError('Use knowledge_hybrid for structured responses and manual for MLP/backbone ablations')
        if self.shared_response_parameters and (self.response != 'structured'
                or self.response_library != 'knowledge_hybrid'):
            raise ValueError('Shared response parameters require the knowledge_hybrid response')
        if not 0 < self.mlp_response_bound <= 1:
            raise ValueError('MLP response bound must be in (0, 1]')
        if self.impact_support not in ('none','sparse','weak','temporal','spatial'):
            raise ValueError('Unknown impact support mode')
        if self.impact_support != 'none' and (self.backbone != 'stae_gru' or not self.gate
                or self.response_library != 'knowledge_hybrid' or not self.contextual_response
                or self.gate_parameter_mode != 'fixed' or self.locality_mode != 'original'
                or self.spatial_experts != 'none' or self.context_fusion != 'original'
                or self.mechanism_mode != 'none' or self.operator_mode != 'none'
                or (self.history,self.horizon)!=(6,6)):
            raise ValueError('Impact support requires original 6-to-6 KDRF')
        if self.gate_parameter_mode not in ('fixed', 'dynamic', 'static_basis'):
            raise ValueError('Unknown gate parameter mode')
        if self.gate_parameter_mode != 'fixed' and (
                not self.gate or self.backbone != 'stae_gru'
                or self.response_library != 'knowledge_hybrid' or not self.contextual_response
                or self.context_fusion != 'original' or self.locality_mode != 'original'
                or self.mechanism_mode != 'none' or self.operator_mode != 'none'
                or self.spatial_experts != 'none' or (self.history, self.horizon) != (6, 6)):
            raise ValueError('Dynamic gate requires original 6-to-6 KDRF with output gate enabled')
        if self.spatial_experts not in ('none', 'selective', 'dense', 'sparse'):
            raise ValueError('Unknown spatial expert mode')
        if self.spatial_experts != 'none' and (
                self.backbone != 'stae_gru' or self.response_library != 'knowledge_hybrid'
                or not self.contextual_response or not self.cross_region
                or self.context_fusion != 'original' or self.locality_mode != 'original'
                or self.mechanism_mode != 'none' or self.operator_mode != 'none'
                or (self.history, self.horizon) != (6, 6)):
            raise ValueError('Spatial experts require the original 6-to-6 KDRF configuration')
        if self.operator_static and self.operator_mode == 'none':
            raise ValueError('Static-operator control requires an operator mode')
        if self.operator_mode not in ('none', 'bank', 'spatial', 'full', 'anchored'):
            raise ValueError('Unknown dynamic operator mode')
        if self.operator_mode != 'none' and (self.backbone != 'stae_gru'
                or self.response_library != 'knowledge_hybrid' or not self.contextual_response
                or self.context_fusion != 'original' or self.locality_mode != 'original'
                or self.mechanism_mode != 'none' or not self.cross_region):
            raise ValueError('Dynamic operators require the original STAE-GRU knowledge-hybrid model')
        if self.mechanism_mode not in ('none', 'params', 'temporal', 'graph', 'all'):
            raise ValueError('Unknown response mechanism mode')
        if self.mechanism_mode != 'none' and (self.response_library != 'knowledge_hybrid'
                or not self.contextual_response or not self.cross_region or self.cross_region_topk
                or self.context_fusion != 'original' or self.locality_mode != 'original'):
            raise ValueError('Response mechanisms require the original cross-region hybrid-knowledge setup')
        if self.history < 3 or self.horizon < 1:
            raise ValueError('History must be >=3 and horizon positive')
        if (self.history, self.horizon) != (6, 6) and (
                self.backbone not in ('staeformer', 'stae_gru') or
                self.context_fusion != 'original' or self.locality_mode != 'original' or
                self.response_library != 'knowledge_hybrid' or not self.contextual_response):
            raise ValueError('Variable lengths currently support STAE + original context + knowledge_hybrid only')
        if self.locality_mode not in ('original', 'local_only', 'local_cross', 'selective_cross'):
            raise ValueError('Unknown locality mode')
        if self.locality_mode != 'original' and self.context_fusion != 'original':
            raise ValueError('Locality experiment requires original context fusion')
        if self.nodes < 2 or self.hidden < 4 or self.heads < 1 or self.hidden % self.heads:
            raise ValueError('Invalid nodes/hidden/heads')
        if self.layers < 1 or not 0 <= self.dropout < 1:
            raise ValueError('Invalid layers/dropout')
        if self.response not in ('structured', 'mlp', 'none'):
            raise ValueError('Unknown response type')
        if self.response_curve not in ('piecewise', 'single'):
            raise ValueError('Unknown response curve')
        if self.response_library not in ('manual', 'knowledge', 'knowledge_calibrated', 'knowledge_control',
                                         'knowledge_latest', 'knowledge_recurrent', 'knowledge_duration',
                                         'knowledge_latest_duration', 'knowledge_blend_duration', 'knowledge_hybrid'):
            raise ValueError('Unknown response library')
        if self.response_library.startswith('knowledge') and (self.response != 'structured' or not self.contextual_response):
            raise ValueError('Knowledge library requires structured contextual response')
        if self.context_fusion not in ('original', 'gate', 'cross_attention', 'gate_cross_attention'):
            raise ValueError('Unknown evidence fusion')
        if self.context_fusion != 'original' and not self.contextual_response:
            raise ValueError('Evidence fusion requires contextual response')
        if not 0 < self.anticipation_bound <= 1:
            raise ValueError('Anticipation bound must be in (0,1]')
        if self.backbone not in ('legacy', 'staeformer', 'stae_gru', 'fist_periodic', 'dynamic_gcru', 'dynamic_original'):
            raise ValueError('Unknown backbone')
        if self.dynamic_mode not in ('flow', 'weather', 'structured', 'local') or self.cross_region_topk < 0:
            raise ValueError('Invalid dynamic mode or spatial support')
        if not 1 <= self.short_steps <= self.history or self.gru_hidden < 1 or self.stae_layers < 1:
            raise ValueError('Invalid hybrid backbone configuration')
        if (self.fist_dim < 4 or self.fist_dim % 4 or self.fist_heads < 1
                or self.fist_dim % self.fist_heads or self.fist_layers < 1
                or self.fist_ff < 4 or self.fist_ff % 4 or self.fist_edges < 1):
            raise ValueError('Invalid FIST periodic configuration')


class AxialBlock(nn.Module):
    """Temporal attention per node, then spatial attention with a graph prior."""
    def __init__(self, hidden, heads, dropout):
        super().__init__()
        self.time = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.space = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(3)])
        self.ff = nn.Sequential(nn.Linear(hidden, hidden*2), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(hidden*2, hidden))

    def forward(self, x, graph_bias=None, spatial=True):
        b, t, n, d = x.shape
        q = self.norms[0](x).transpose(1, 2).reshape(b*n, t, d)
        x = x + self.time(q, q, q, need_weights=False)[0].reshape(b, n, t, d).transpose(1, 2)
        if spatial:
            q = self.norms[1](x).reshape(b*t, n, d)
            x = x + self.space(q, q, q, attn_mask=graph_bias,
                               need_weights=False)[0].reshape(b, t, n, d)
        return x + self.ff(self.norms[2](x))


class DisasterContext(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden
        self.input = nn.Linear(12 if cfg.contextual_response else 10, d)
        self.node = nn.Parameter(torch.randn(cfg.nodes, d)*.02)
        self.position = nn.Parameter(torch.randn(cfg.history, d)*.02)
        self.blocks = nn.ModuleList([AxialBlock(d, cfg.heads, cfg.dropout)
                                     for _ in range(cfg.layers)])
        self.query, self.key = nn.Linear(d, d), nn.Linear(d, d)
        self.b_left = nn.Parameter(torch.randn(cfg.nodes, 8)*.02)
        self.b_right = nn.Parameter(torch.randn(cfg.nodes, 8)*.02)
        self.source_weather = nn.Linear(2, d, bias=False)
        self.fuse = nn.Sequential(nn.Linear(d*2, d), nn.GELU(), nn.LayerNorm(d))
        self.register_buffer('weather_scale', torch.tensor([17.2, 50/6]))

    def modulate_features(self, x):
        return x

    def attention_queries_keys(self, h, inputs, node_scale):
        return self.query(h), self.key(h)

    def message_values(self, h, inputs):
        return h + self.source_weather(inputs['weather'][:, -1]/self.weather_scale)

    def forward(self, inputs, graph_bias, node_scale):
        n = self.cfg.nodes
        social = inputs['social'] if self.cfg.social else torch.zeros_like(inputs['social'])
        cal = inputs['history_calendar'][:, :, None].expand(-1, -1, n, -1)
        x = torch.cat(((inputs['flow']/node_scale)[..., None],
                       inputs['anomaly'][..., None], inputs['weather']/self.weather_scale,
                       social, cal), -1)
        if self.cfg.contextual_response:
            flow = inputs['flow']/node_scale
            trend = torch.cat((torch.zeros_like(flow[:, :1]), flow[:, 1:]-flow[:, :-1]), 1)
            deviation = (inputs['flow']-inputs['history_reference'])/node_scale
            x = torch.cat((x, trend[..., None], deviation[..., None]), -1)
        x = self.input(self.modulate_features(x)) + self.node[None, None] + self.position[None, :, None]
        for block in self.blocks:
            x = block(x, graph_bias, spatial=self.cfg.cross_region)
        h = x[:, -1]
        if self.cfg.cross_region:
            # B: source-target relevance; rows are receivers, columns are sources.
            queries, keys = self.attention_queries_keys(h, inputs, node_scale)
            logits = queries @ keys.transpose(-1, -2)/math.sqrt(h.shape[-1])
            logits = logits + graph_bias + self.b_left @ self.b_right.T/math.sqrt(8)
            logits = logits.masked_fill(torch.eye(n, device=h.device, dtype=torch.bool), -torch.inf)
            if self.cfg.cross_region_topk:
                index = logits.topk(min(self.cfg.cross_region_topk, n-1), dim=-1).indices
                keep = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, index, True)
                logits = logits.masked_fill(~keep, -torch.inf)
            weights = logits.softmax(-1)
            # P: source weather travels in the message, separately from raw exposure.
            messages = weights @ self.message_values(h, inputs)
        else:
            weights = h.new_zeros(h.shape[0], n, n)
            messages = torch.zeros_like(h)
        context = self.fuse(torch.cat((h, messages), -1))
        indirect_history = torch.einsum('bij,btjk->btik', weights, inputs['weather'])
        return context, weights, indirect_history


class ResponseBank(nn.Module):
    def __init__(self, cfg, threshold):
        super().__init__()
        self.cfg = cfg
        self.raw_slope = nn.Parameter(torch.zeros(4))
        self.raw_threshold = nn.Parameter(torch.zeros(4))
        self.raw_amplitude = nn.Parameter(torch.tensor(-1.))
        self.raw_retention = nn.Parameter(torch.tensor(1.))
        self.condition = nn.Linear(cfg.hidden, 11)
        nn.init.zeros_(self.condition.weight)
        nn.init.zeros_(self.condition.bias)
        self.anticipation = nn.Linear(cfg.hidden, 1)
        self.register_buffer('threshold_origin', torch.as_tensor(threshold, dtype=torch.float32).repeat(2))
        self.register_buffer('weather_scale', torch.tensor([17.2, 50/6, 17.2, 50/6]))

    def parameters_for(self, context):
        # Bounded local adaptation around globally shared curve parameters.
        offset = .25*torch.tanh(self.condition(context))
        threshold = self.threshold_origin * torch.exp(self.raw_threshold + offset[..., :4])
        slope = F.softplus(self.raw_slope + offset[..., 4:8]) + 1e-4
        amplitude = torch.sigmoid(self.raw_amplitude + offset[..., 8])
        retention = .99*torch.sigmoid(self.raw_retention + offset[..., 9])
        anticipation = self.cfg.anticipation_bound*torch.tanh(self.anticipation(context).squeeze(-1)+offset[..., 10])
        return dict(threshold=threshold, slope=slope, amplitude=amplitude,
                    retention=retention, anticipation=anticipation, offset=offset)

    def instantaneous(self, weather, parameters):
        """Continuous bounded monotone inhibition, ZERO when below all thresholds.

        weather [B,T,N,4]; parameters held fixed during intensity interventions.
        Raw wind/rain are scaled by physical units, never clipped at training max.
        """
        excess = F.relu((weather-parameters['threshold'][:, None])/self.weather_scale)
        dose = (excess*parameters['slope'][:, None]).sum(-1)
        return parameters['amplitude'][:, None] * (-torch.expm1(-dose))

    @staticmethod
    def advance(instant, state, retention):
        return 1-(1-instant)*(1-retention*state)

    def forward(self, context, history_weather, future_weather, social):
        p = self.parameters_for(context)
        u_hist = self.instantaneous(history_weather, p)
        u_future = self.instantaneous(future_weather, p)
        state = torch.zeros_like(u_hist[:, 0])
        rho = p['retention'] if self.cfg.memory else torch.zeros_like(p['retention'])
        for u in u_hist.unbind(1):
            state = self.advance(u, state, rho)
        initial_state = state
        states = []
        for u in u_future.unbind(1):
            state = self.advance(u, state, rho)
            states.append(state)
        suppression = torch.stack(states, 1)
        # Signed anticipatory adjustment; no future observed tweets are needed.
        decay = torch.exp(-torch.arange(1, 7, device=context.device, dtype=context.dtype)/3)
        social_strength = -torch.expm1(-social[:, -1, :, 1].clamp_min(0))
        anticipation = p['anticipation'][:, None]*social_strength[:, None]*decay[None, :, None]
        response = anticipation-suppression
        details = dict(instantaneous=u_future, suppression=suppression, anticipation=anticipation,
                       initial_state=initial_state, retention=rho, parameter_offset=p['offset'])
        return response, details


class SelectiveGate(nn.Module):
    """Small applicability gate; cannot read base flow or response amplitude."""
    def __init__(self, nodes):
        super().__init__()
        self.raw_weights = nn.Parameter(torch.zeros(4))
        self.node_bias = nn.Parameter(torch.full((nodes,), -1.))
        self.horizon_bias = nn.Parameter(torch.zeros(6))

    def forward(self, local, indirect, anomaly, social):
        features = torch.stack((local, indirect, anomaly, social), -1).clamp_min(0)
        evidence = -torch.expm1(-features)
        applicability = torch.sigmoid((evidence*F.softplus(self.raw_weights)).sum(-1)
                                     + self.node_bias[None, None] + self.horizon_bias[None, :, None])
        # Exactly zero when there is no evidence, even with arbitrary gate biases.
        return applicability*(-torch.expm1(-features.sum(-1)))


class StructuredDisasterModel(nn.Module):
    def __init__(self, config, preprocessing):
        super().__init__()
        self.config = config if isinstance(config, ModelConfig) else ModelConfig(**config)
        c = self.config
        self.register_buffer('node_scale', torch.as_tensor(preprocessing['node_scale'], dtype=torch.float32))
        self.register_buffer('graph_bias', torch.as_tensor(preprocessing['graph'], dtype=torch.float32).clamp_min(1e-4).log())
        self.register_buffer('weather_threshold', torch.as_tensor(preprocessing['weather_threshold'], dtype=torch.float32))
        self.register_buffer('weather_scale', torch.tensor([17.2, 50/6]))
        from .hybrid_backbone import STAEGRUBackbone
        self.backbone = STAEGRUBackbone(c, preprocessing['mean'], preprocessing['std'])
        self.context = DisasterContext(c)
        self.bank = ResponseBank(c, preprocessing['weather_threshold'])
        self.gate = SelectiveGate(c.nodes)
        if c.contextual_response:
            from .contextual_response import ContextualBank, ContextualGate
            self.bank = ContextualBank(c, preprocessing['weather_threshold'])
            self.gate = ContextualGate(c)
        if c.response_library == 'knowledge_hybrid':
            from .knowledge_response import LearnedHybridBank
            self.bank = LearnedHybridBank(c.hidden, preprocessing['weather_threshold'], horizon=c.horizon)
        if hasattr(self.bank, 'condition_response'):
            self.bank.condition_response = c.condition_response
            self.bank.shared_parameters = c.shared_response_parameters
        self.free_response = (nn.Sequential(nn.Linear(c.hidden+7, c.hidden), nn.GELU(), nn.Linear(c.hidden, 1))
                              if c.response == 'mlp' else None)
    def configuration(self):
        return asdict(self.config)

    def free_response_features(self, context, future_weather, social, base, inputs):
        b, _, n = base.shape
        horizon = torch.arange(1, 7, device=base.device, dtype=base.dtype)[None, :, None, None].expand(b, -1, n, -1)/6
        return torch.cat((context[:, None].expand(-1, 6, -1, -1),
                          future_weather/self.bank.weather_scale,
                          social[:, -1:, :, :].expand(-1, 6, -1, -1), horizon), -1)

    def forward(self, inputs, *, unclipped_backbone=False, training_labels=None, batches_seen=None):
        if training_labels is not None and (not self.training or self.config.backbone != 'dynamic_original'):
            raise ValueError('Teacher-forcing labels are allowed only during original Dynamic training')
        required = {'flow', 'weather', 'social', 'excess', 'anomaly', 'history_reference',
                    'future_reference', 'history_calendar', 'future_calendar', 'future_weather'}
        if set(inputs) != required:
            raise ValueError(f'Model expects only causal inputs: {sorted(required)}')
        b, t, n = inputs['flow'].shape
        if (t, n) != (self.config.history, self.config.nodes):
            raise ValueError(f'Expected [B,{self.config.history},N] flow')
        h = self.config.horizon
        expected = {k: (b, t, n) for k in ('flow', 'anomaly', 'history_reference')}
        expected['future_reference'] = (b, h, n)
        expected.update({k: (b, t, n, 2) for k in ('weather', 'social', 'excess')})
        expected['future_weather'] = (b, h, n, 2)
        expected.update(history_calendar=(b, t, 4), future_calendar=(b, h, 4))
        for k, shape in expected.items():
            if inputs[k].shape != shape or not torch.isfinite(inputs[k]).all():
                raise ValueError(f'Invalid {k}')
        if (inputs['weather'] < 0).any() or (inputs['future_weather'] < 0).any():
            raise ValueError('Negative weather')
        # From-scratch end-to-end fitting needs gradients even when a random
        # linear readout is negative. This explicit training-only option leaves
        # the existing nonnegative inference contract unchanged.
        if self.config.backbone == 'dynamic_original':
            base = self.backbone(inputs, self.graph_bias, labels=training_labels,
                                 batches_seen=batches_seen, unclipped=unclipped_backbone)
        elif unclipped_backbone:
            if not self.training:
                raise ValueError('Unclipped predictions are only for training')
            base = self.backbone.forward_unclipped(inputs, self.graph_bias)
        else:
            base = self.backbone(inputs, self.graph_bias)
        if self.config.response == 'none':
            details = {'base': base, 'gate': torch.zeros_like(base), 'response': torch.zeros_like(base)}
            if self.config.backbone == 'dynamic_gcru':
                if self.backbone.last_adjustments is not None:
                    details['parameter_adjustment'] = self.backbone.last_adjustments
                    details['parameter_support'] = self.backbone.last_support
            return base[..., None], details
        context, weights, indirect_hist = self.context(inputs, self.graph_bias, self.node_scale)
        future = inputs['future_weather']
        indirect_future = torch.einsum('bij,btjk->btik', weights, future)
        hist_weather = torch.cat((inputs['weather'], indirect_hist), -1)
        future_weather = torch.cat((future, indirect_future), -1)
        social = inputs['social'] if self.config.social else torch.zeros_like(inputs['social'])
        if self.config.response == 'structured':
            if self.config.response_library.startswith('knowledge'):
                response, details = self.bank(context, hist_weather, future_weather, social,
                                              inputs, self.node_scale, base)
            else:
                response, details = self.bank(context, hist_weather, future_weather, social)
        else:
            features = self.free_response_features(context, future_weather, social, base, inputs)
            # A matched bounded, signed free MLP ablation, without shape assumptions.
            response = self.config.mlp_response_bound*torch.tanh(self.free_response(features).squeeze(-1))
            details = {}
        local = F.relu((future-self.weather_threshold)/self.weather_scale).sum(-1)
        indirect = F.relu((indirect_future-self.weather_threshold)/self.weather_scale).sum(-1)
        # Preserve applicability during recovery even if future weather is calm.
        local = torch.maximum(local, inputs['excess'].sum(-1).amax(1, keepdim=True))
        indirect = torch.maximum(indirect, F.relu((indirect_hist-self.weather_threshold)/self.weather_scale).sum(-1).amax(1, keepdim=True))
        anomaly = F.relu(inputs['anomaly'].abs().mean(1, keepdim=True)-2).expand(-1, self.config.horizon, -1)
        social_gate = social[:, -1:, :, 1].expand(-1, self.config.horizon, -1)
        if self.config.gate:
            gate = (self.gate(local, indirect, anomaly, social_gate, context)
                    if self.config.contextual_response else self.gate(local, indirect, anomaly, social_gate))
        else:
            gate = torch.ones_like(base)
        if self.config.impact_support != 'none':
            support, probability = self.support(inputs)
            details.update(original_gate=gate,impact_support=support,impact_probability=probability)
            gate = gate*support
        prediction = base*(1+gate*response)
        if not unclipped_backbone and self.config.nonnegative_output:
            prediction = prediction.clamp_min(0)
        details.update(base=base, gate=gate, response=response, context=context, spatial_weights=weights,
                       indirect_weather=indirect_future, local_weather=future)
        return prediction[..., None], details

    def shape_penalty(self, context):
        """Only regularize instantaneous inhibition at FIXED context, never total flow."""
        p = self.bank.parameters_for(context)
        level = torch.linspace(0, 3, 16, device=context.device, dtype=context.dtype)
        weather = level[None, :, None, None]*self.bank.weather_scale[None, None, None]
        weather = weather.expand(context.shape[0], -1, self.config.nodes, -1)
        curve = self.bank.instantaneous(weather, p)
        curvature = curve[:, 2:]-2*curve[:, 1:-1]+curve[:, :-2]
        penalty = curvature.square().mean() + .01*p['offset'].square().mean()
        if hasattr(self.bank, 'learned_smoothness'):
            penalty = penalty + self.bank.learned_smoothness()
        return penalty
