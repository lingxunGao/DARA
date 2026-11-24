"""Causal adaptation of the STAE-GRU readout, upstream of frozen response/gate.

Shared mode learns a delta to the existing 6x912 projection and its bias.
Local-bias mode adds a node-specific bias to that normalized readout (no new
encoder). Frozen checkpoint weights are never overwritten; effective weights
at deployment are W0+delta_W. Stored issue-time encoder features are reused.
"""
from collections import deque

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def cache_readout(model, split, limit=None):
    model.eval()
    captured, values = {}, {}
    def hook(module, inputs, output):
        captured['features'] = inputs[0]
        captured['raw'] = output.transpose(1,2)
    handle = model.backbone.transformer.output_proj.register_forward_hook(hook)
    count = 0
    try:
        for batch in split.sequential(16):
            prediction, details = model(batch['inputs'])
            row = dict(features=captured['features'],raw=captured['raw'],
                base=details['base'],factor=1+details['gate']*details['response'],
                original=prediction[...,0])
            remaining = len(prediction) if limit is None else min(len(prediction),limit-count)
            for k,v in row.items():
                values.setdefault(k,[]).append(v[:remaining].detach().cpu())
            count += remaining
            if limit is not None and count>=limit:
                break
    finally:
        handle.remove()
    return {k:torch.cat(v).numpy() for k,v in values.items()}


class OnlineBackboneReadout(nn.Module):
    def __init__(self, nodes, width, mean, std, mode):
        super().__init__()
        if mode not in ('shared_readout','local_bias'):
            raise ValueError(mode)
        self.mode = mode
        self.mean, self.std = float(mean), float(std)
        if mode=='shared_readout':
            self.weight = nn.Parameter(torch.zeros(6,width))
            self.bias = nn.Parameter(torch.zeros(6))
        else:
            self.bias = nn.Parameter(torch.zeros(6,nodes))

    def delta(self, cache, origins, leads=None):
        if self.mode=='shared_readout':
            features = cache['features'][origins]
            if leads is None:
                return F.linear(features,self.weight,self.bias).transpose(1,2)
            return (features*self.weight[leads,None]).sum(-1)+self.bias[leads,None]
        if leads is None:
            return self.bias[None].expand(len(origins),-1,-1)
        return self.bias[leads]

    def forward(self, cache, origins, leads=None):
        index = origins if leads is None else (origins,leads)
        raw = cache['raw'][index]+self.delta(cache,origins,leads)
        new_base = (raw*self.std+self.mean).clamp_min(0)
        # Preserves original float32 outputs at initialization; this is the
        # change in backbone output passed THROUGH the original response factor.
        return (cache['original'][index]+(new_base-cache['base'][index])*cache['factor'][index]).clamp_min(0)


def run_backbone_stream(cache, starts, reveal, scale, mean, std, config, device):
    starts = np.asarray(starts,dtype=np.int64)
    if not len(starts) or not np.all(np.diff(starts)==1):
        raise ValueError('Expected consecutive origins')
    scale = torch.as_tensor(scale,dtype=torch.float32,device=device)
    adapter = OnlineBackboneReadout(len(scale),cache['features'].shape[-1],mean,std,config['mode']).to(device)
    optimizer = torch.optim.Adam(adapter.parameters(),lr=config['lr'])
    records, predictions, audits, traces = deque(), [], [], []
    for index,start in enumerate(starts):
        clock = int(start)+5
        observed = torch.as_tensor(reveal(clock),dtype=torch.float32,device=device).detach().clone()
        if observed.shape!=scale.shape or not torch.isfinite(observed).all() or (observed<0).any():
            raise ValueError('Invalid revealed observation')
        for lead in range(6):
            origin = index-1-lead
            if origin>=0:
                assert int(starts[origin])+6+lead==clock
                records.append((clock,origin,lead,observed))
        while records and records[0][0]<clock-config['window']+1:
            records.popleft()
        loss_value = None
        if records:
            times, origins, leads, labels = zip(*records)
            assert max(times)<=clock and max(origins)<index
            origins = torch.tensor(origins,device=device)
            leads = torch.tensor(leads,device=device)
            labels = torch.stack(labels)
            valid = labels!=0
            optimizer.zero_grad(set_to_none=True)
            pred = adapter(cache,origins,leads)
            error = (pred-labels)/scale
            objective = F.smooth_l1_loss(error,torch.zeros_like(error),beta=.1,reduction='none')
            # Functional anchoring on the SAME revealed examples. No targets
            # outside the feedback buffer enter either term.
            drift = ((pred-cache['original'][origins,leads])/scale).square()
            per_node = ((objective+config['anchor']*drift)*valid).sum(0)/valid.sum(0).clamp_min(1)
            loss = per_node.mean() if config['mode']=='shared_readout' else per_node.sum()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite feedback objective')
            loss.backward()
            if config['mode']=='shared_readout':
                torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.,error_if_nonfinite=True)
            else:
                grad = adapter.bias.grad
                if not torch.isfinite(grad).all():
                    raise FloatingPointError('Nonfinite local gradient')
                grad.mul_(1/grad.norm(dim=0,keepdim=True).clamp_min(1))
            optimizer.step()
            loss_value = float(loss.detach())
        with torch.no_grad():
            p = adapter(cache,[index])[0]
            predictions.append(p.cpu().numpy().copy())
            traces.append(dict(issue_hour=clock,loss=loss_value,
                mean_absolute_adjustment=float((p-cache['original'][index]).abs().mean()),
                parameter_delta_norm=float(sum(v.square().sum() for v in adapter.parameters()).sqrt())))
        audits.append(dict(issue_hour=clock,first_target_hour=clock+1,last_target_hour=clock+6,
            latest_feedback_hour=clock if records else None,feedback_pairs=len(records),
            feedback_unique_hours=len({r[0] for r in records})))
    state = dict(state_dict={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()},
        optimizer=optimizer.state_dict(),config=config,mean=mean,std=std,
        note='Final delta only; W_effective=W_checkpoint+delta_W. Earlier forecasts are immutable.')
    return np.stack(predictions)[...,None],audits,traces,state
