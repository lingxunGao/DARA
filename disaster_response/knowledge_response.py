"""The reviewed v1 candidate library, integrated with the original main model.

Knowledge extraction: scarcity_prototype/artifacts/llama_extraction_v1/.
Time scales and numerical functions are researcher choices, not LLM discoveries.
"""
import torch
from torch import nn
from torch.nn import functional as F


class KnowledgeResponseBank(nn.Module):
    def __init__(self, hidden, threshold, horizon=6):
        super().__init__()
        # Set by StructuredDisasterModel from ModelConfig.  Keeping this as a
        # plain flag avoids changing checkpoints or parameter counts.
        self.condition_response = True
        self.shared_parameters = False
        self.head = nn.Sequential(nn.Linear(hidden+4, hidden), nn.GELU(), nn.Linear(hidden, 7))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.head[-1].bias.data[:2] = -3.
        self.register_buffer('threshold', torch.as_tensor(threshold).float().repeat(2))
        self.register_buffer('weather_scale', torch.tensor([17.2, 50/6, 17.2, 50/6]))
        self.register_buffer('steps', torch.arange(1, horizon+1).float())
        self.register_buffer('tau', torch.tensor([2., 6., 24.]))

    def _condition_context(self, context):
        """Return the context used by conditional response parameters.

        ``condition_response=False`` is the context-blind control used by the
        contextual ablation.  Keeping the tensor shape while zeroing its
        values preserves the same response-bank parameterization and number
        of parameters, but removes disaster-context conditioning.
        """
        if self.condition_response and not self.shared_parameters:
            return context
        return torch.zeros_like(context)

    def parameters_for(self, context):
        # Existing conditional-curve diagnostics hold the additional observation
        # features at zero. This plots the instantaneous candidate, not total flow.
        context_for_parameters = self._condition_context(context)
        p = self.head(torch.cat((context_for_parameters,
                                 context.new_zeros(*context.shape[:-1], 4)), -1))
        return dict(threshold=self.threshold.expand(*context.shape[:-1], 4),
                    slope=torch.ones_like(p[..., :4]), amplitude=p[..., 0].sigmoid(),
                    offset=p*0)

    def instantaneous(self, weather, parameters):
        dose = F.relu((weather-parameters['threshold'][:, None])/self.weather_scale).sum(-1)
        return parameters['amplitude'][:, None]*(1-torch.exp(-dose))

    def coefficients(self, context, evidence, inputs, node_scale, base=None):
        # Unlike context-only removal, shared coefficients cannot depend on
        # local observed deviations, exposure, or social evidence either.
        if self.shared_parameters:
            evidence = torch.zeros_like(evidence)
        return self.head(torch.cat((self._condition_context(context), evidence), -1))

    def observed_state(self, deviation):
        return deviation[:, -3:].median(1).values

    def persistence(self, context, mixture):
        decay = torch.exp(-self.steps[:, None]/self.tau[None])
        shapes = torch.cat((torch.ones(len(self.steps), 1, device=context.device), decay), -1)
        return torch.einsum('bnk,hk->bhn', mixture, shapes)

    def compose_response(self, shock, carry, anticipation, p, mixture, recent):
        return shock+carry+anticipation, {}

    def forward(self, context, history_weather, future_weather, social, inputs, node_scale, base=None):
        denominator = torch.maximum(inputs['history_reference'], .1*node_scale)
        deviation = ((inputs['flow']-inputs['history_reference'])/denominator).clamp(-1., 1.)
        recent = self.observed_state(deviation)
        trend = (deviation[:, -1]-deviation[:, -3])/2
        historical = F.relu((history_weather-self.threshold)/self.weather_scale).sum(-1)
        future = F.relu((future_weather-self.threshold)/self.weather_scale).sum(-1)
        evidence = torch.stack((recent, trend, historical.amax(1), social[:, -1, :, 1]), -1)
        p = self.coefficients(context, evidence, inputs, node_scale, base)
        mixture = p[..., 2:6].softmax(-1)
        persistence = self.persistence(context, mixture)
        shock = -p[..., 0].sigmoid()[:, None]*(1-torch.exp(-future))
        carry = p[..., 1].sigmoid()[:, None]*recent[:, None]*persistence
        anticipation = .3*torch.tanh(p[..., 6])[:, None]*(1-torch.exp(-social[:, -1:, :, 1].clamp_min(0)))
        anticipation = anticipation*torch.exp(-self.steps[None, :, None]/3)
        response, extra = self.compose_response(shock, carry, anticipation, p, mixture, recent)
        response = response.clamp(-.95, .95)
        details = dict(instantaneous=-shock, observed_state=recent, persistent=carry,
                       anticipation=anticipation, mixture=mixture, response=response)
        details.update(extra)
        return response, details


class LatestObservationBank(KnowledgeResponseBank):
    """Avoid median lag at an abrupt change; trade off robustness to one noisy hour."""
    def observed_state(self, deviation):
        return deviation[:, -1]


class ForcedStateBank(KnowledgeResponseBank):
    """Replace additive instantaneous+decay terms with a bounded driven state.

    The observed deviation initializes the state. Each future step relaxes it
    toward a weather-driven suppression target. Recovery is the same update
    with zero forcing. The convex update cannot overshoot its state/forcing.
    """
    def compose_response(self, shock, carry, anticipation, p, mixture, recent):
        candidates = torch.cat((self.tau.new_ones(1), torch.exp(-1/self.tau)))
        retention = (mixture*candidates).sum(-1)
        initial = p[..., 1].sigmoid()*recent
        state = initial
        trajectory, inherited = [], []
        for forcing in shock.unbind(1):
            inherited.append(retention*state)
            state = retention*state+(1-retention)*forcing
            trajectory.append(state)
        trajectory = torch.stack(trajectory, 1)
        return trajectory+anticipation, dict(initial_state=initial, retention=retention,
            dynamic_state=trajectory, persistent=torch.stack(inherited, 1))


class AdaptiveDurationBank(KnowledgeResponseBank):
    """Context-dependent bounded recovery rates; exact old-bank initialization."""
    def __init__(self, hidden, threshold, horizon=6):
        super().__init__(hidden, threshold, horizon=horizon)
        # Extra zero-initialized parameters must not shift the training RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.duration = nn.Linear(hidden, 3)
        nn.init.zeros_(self.duration.weight)
        nn.init.zeros_(self.duration.bias)

    def persistence(self, context, mixture):
        tau = self.tau*torch.exp(1.3862943611198906*torch.tanh(
            self.duration(self._condition_context(context))))
        decay = torch.exp(-self.steps[None, :, None, None]/tau[:, None])
        shapes = torch.cat((torch.ones_like(decay[..., :1]), decay), -1)
        return (mixture[:, None]*shapes).sum(-1)


class LatestAdaptiveDurationBank(AdaptiveDurationBank):
    """Latest observed state with context-dependent persistence, no extra head."""
    def observed_state(self, deviation):
        return deviation[:, -1]


class LearnedHybridBank(LatestAdaptiveDurationBank):
    """Two shared learned four-knot curves supplement the preset candidates.

    Context selects signed amplitudes; total extra response is bounded by .3.
    A zero amplitude head preserves the preset model's initial predictions.
    These curves learn from training targets only, with no test-time fitting.
    """
    def __init__(self, hidden, threshold, horizon=6):
        super().__init__(hidden, threshold, horizon=horizon)
        with torch.random.fork_rng(devices=[]):
            self.learned_amplitude = nn.Linear(hidden, 2)
        nn.init.zeros_(self.learned_amplitude.weight)
        nn.init.zeros_(self.learned_amplitude.bias)
        self.curve_knots = nn.Parameter(torch.tensor([
            [-.6, -.2, .2, .6], [.6, -.3, -.3, .6]]))

    def learned_shapes(self):
        return F.interpolate(self.curve_knots.tanh()[None], size=len(self.steps),
                             mode='linear', align_corners=True)[0]

    def forward(self, context, history_weather, future_weather, social, inputs, node_scale, base=None):
        _, details = super().forward(context, history_weather, future_weather,
                                     social, inputs, node_scale, base)
        amplitude = .15*self.learned_amplitude(self._condition_context(context)).tanh()
        extra = torch.einsum('bnk,kh->bhn', amplitude, self.learned_shapes())
        response = (-details['instantaneous']+details['persistent']+
                    details['anticipation']+extra).clamp(-.95, .95)
        details.update(response=response, learned_response=extra, learned_amplitude=amplitude)
        return response, details

    def learned_smoothness(self):
        shapes = self.learned_shapes()
        return (shapes[:, 2:]-2*shapes[:, 1:-1]+shapes[:, :-2]).square().mean()


class BlendedAdaptiveDurationBank(AdaptiveDurationBank):
    """Learn a convex latest/median state blend from observed local history.

    This state estimator adds six shared parameters, not another output gate.
    It starts halfway between latest and median; no future observations enter.
    """
    def __init__(self, hidden, threshold, horizon=6):
        super().__init__(hidden, threshold, horizon=horizon)
        with torch.random.fork_rng(devices=[]):
            self.state_gate = nn.Linear(5, 1)
        nn.init.zeros_(self.state_gate.weight)
        nn.init.zeros_(self.state_gate.bias)

    def state_weight(self, deviation):
        recent = deviation[:, -3:].transpose(1, 2)
        features = torch.cat((recent, recent[..., 1:]-recent[..., :-1]), -1)
        return self.state_gate(features).squeeze(-1).sigmoid()

    def observed_state(self, deviation):
        weight = self.state_weight(deviation)
        median = deviation[:, -3:].median(1).values
        return weight*deviation[:, -1]+(1-weight)*median


class ForecastCalibratedBank(KnowledgeResponseBank):
    """Condition candidate coefficients on the backbone's own forecast trajectory.

    The forecast is an issue-time model output, never the future target. Detaching
    it avoids a gradient shortcut through this conditioning input; the backbone
    still trains end-to-end through the final prediction. The matched control
    has identical parameters/initialization but zeros these six input features.
    This allows learning to avoid duplicate corrections, but does not enforce it.
    """
    def __init__(self, hidden, threshold, use_forecast=True, horizon=6):
        super().__init__(hidden, threshold, horizon=horizon)
        self.use_forecast = use_forecast
        self.calibration = nn.Sequential(nn.Linear(hidden+4+6, hidden), nn.GELU(), nn.Linear(hidden, 7))
        nn.init.zeros_(self.calibration[-1].weight)
        nn.init.zeros_(self.calibration[-1].bias)

    def coefficients(self, context, evidence, inputs, node_scale, base=None):
        if base is None:
            raise ValueError('Calibration requires the backbone forecast')
        reference = inputs['future_reference']
        denominator = torch.maximum(reference, .1*node_scale)
        forecast = ((base.detach().clamp_min(0)-reference)/denominator).clamp(-1., 1.)
        forecast = forecast.transpose(1, 2)
        if not self.use_forecast:
            forecast = torch.zeros_like(forecast)
        return super().coefficients(context, evidence, inputs, node_scale, base) + self.calibration(
            torch.cat((context, evidence, forecast), -1))

    def parameters_for(self, context):
        # Conditional diagnostics hold observation AND forecast features at zero.
        p = self.head(torch.cat((context, context.new_zeros(*context.shape[:-1], 4)), -1))
        p = p + self.calibration(torch.cat((context, context.new_zeros(*context.shape[:-1], 10)), -1))
        return dict(threshold=self.threshold.expand(*context.shape[:-1], 4),
                    slope=torch.ones_like(p[..., :4]), amplitude=p[..., 0].sigmoid(), offset=p*0)
