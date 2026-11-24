"""Author's STAEformer + local GRU, fused before one multi-step readout.

The default uses all six observed hours vs the last three observed hours.
These are two relative scales, not a claim of 24h/weekly observed context.
This is our combination, not an architecture attributed to the STAEformer paper.
"""
import torch
from torch import nn

from .staeformer import STAEformer


class STAEGRUBackbone(nn.Module):
    def __init__(self, cfg, mean, std):
        super().__init__()
        self.cfg = cfg
        self.register_buffer('mean', torch.tensor(float(mean)))
        self.register_buffer('std', torch.tensor(float(std)))
        self.transformer = STAEformer(num_nodes=cfg.nodes, in_steps=cfg.history, out_steps=cfg.horizon,
                                      steps_per_day=24, input_dim=3, output_dim=1,
                                      input_embedding_dim=24, tod_embedding_dim=24,
                                      dow_embedding_dim=24, adaptive_embedding_dim=80,
                                      feed_forward_dim=256, num_heads=4,
                                      num_layers=cfg.stae_layers, dropout=cfg.dropout)
        if cfg.backbone == 'stae_gru':
            self.short_gru = nn.GRU(4, cfg.gru_hidden, batch_first=True)
            self.short_projection = nn.Linear(cfg.gru_hidden, cfg.history*self.transformer.model_dim)
            self.fusion_gate = nn.Linear(self.transformer.model_dim+cfg.gru_hidden, 1)
            # Loading STAEformer starts with exactly its learned forecast; the GRU
            # then learns feature updates, rather than overwriting a strong model.
            nn.init.zeros_(self.short_projection.weight)
            nn.init.zeros_(self.short_projection.bias)
            nn.init.constant_(self.fusion_gate.bias, -1.)

    @staticmethod
    def calendar_indices(calendar):
        hour = torch.round(torch.remainder(torch.atan2(calendar[..., 0], calendar[..., 1]),
                                           2*torch.pi)*24/(2*torch.pi)).long() % 24
        weekday = torch.round(torch.remainder(torch.atan2(calendar[..., 2], calendar[..., 3]),
                                              2*torch.pi)*7/(2*torch.pi)).long() % 7
        return hour, weekday

    def model_inputs(self, inputs):
        raw = inputs['flow']
        if self.cfg.isolate_history:
            # Exposure-only isolation: a normal flow deviation by itself is not
            # evidence of a disaster. Normal pretraining inputs remain intact.
            reliability = torch.exp(-inputs['excess'].sum(-1))
            raw = reliability*raw + (1-reliability)*inputs['history_reference']
        hour, weekday = self.calendar_indices(inputs['history_calendar'])
        calendar = torch.stack((hour.to(raw.dtype)/24, weekday.to(raw.dtype)), -1)
        calendar = calendar[:, :, None].expand(-1, -1, self.cfg.nodes, -1)
        return torch.cat((((raw-self.mean)/self.std)[..., None], calendar), -1)

    def encode(self, inputs):
        observed = self.model_inputs(inputs)
        features = self.transformer.encode(observed)
        if self.cfg.backbone == 'stae_gru':
            b, _, n, d = features.shape
            recent = observed[:, -self.cfg.short_steps:]
            difference = torch.cat((torch.zeros_like(recent[:, :1, :, :1]),
                                    recent[:, 1:, :, :1]-recent[:, :-1, :, :1]), 1)
            recent = torch.cat((recent, difference), -1)
            sequence = recent.transpose(1, 2).reshape(b*n, self.cfg.short_steps, 4)
            _, hidden = self.short_gru(sequence)
            hidden = hidden[-1].reshape(b, n, self.cfg.gru_hidden)
            gate = self.fusion_gate(torch.cat((features[:, -1], hidden), -1)).sigmoid()
            short_update = self.short_projection(hidden).reshape(b, n, self.cfg.history, d).transpose(1, 2)
            features = features + gate[:, None]*short_update
        return features

    def forward_unclipped(self, inputs, graph_bias=None):
        features = self.encode(inputs)
        # The author's mixed projection maps all six encoded hours to six
        # independent horizon outputs. No mean pooling or profile multiplier.
        prediction = self.transformer.decode(features)[..., 0]*self.std+self.mean
        return prediction

    def forward(self, inputs, graph_bias=None):
        return self.forward_unclipped(inputs, graph_bias).clamp_min(0)
