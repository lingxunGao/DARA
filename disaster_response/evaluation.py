"""Five retrospective evaluation scopes; causal readout feedback replay."""
import numpy as np
import torch
from . import protocol
from .backbone_online_adaptation import cache_readout, run_backbone_stream
from .training import CachedSplit

SCOPES=dict(zip((*protocol.SCOPES,'disaster_time_unaffected_nodes'),(
    '全部测试时间、全部测试地点','固定灾害时间、全部测试地点','固定灾害时间、选定受灾地点',
    '全部测试时间、其余未受灾地点','灾害时段、未受影响地区')))


def scopes(pred,truth,data):
    event,affected=data['event_windows'],data['affected']
    all_w=np.ones(len(pred),bool);all_n=np.ones(len(affected),bool)
    masks=[(all_w,all_n),(event,all_n),(event,affected),(all_w,~affected),(event,~affected)]
    return {scope:dict(window_count=int(w.sum()),node_count=int(n.sum()),
                overall=protocol.score_metrics(pred[w][:,:,n],truth[w][:,:,n]))
            for scope,(w,n) in zip(SCOPES,masks)}


def evaluate(model,data,online,device):
    model.eval()
    values=cache_readout(model,CachedSplit(data,'test',device))
    starts=data['splits']['test']
    truth=data['flow'][starts[:,None]+6+np.arange(6)][...,None]
    frozen=values['original'][...,None]
    cache={k:torch.from_numpy(v).to(device) for k,v in values.items()}
    pred,audit,traces,state=run_backbone_stream(cache,starts,lambda hour:data['flow'][hour],
                model.node_scale,float(model.backbone.mean),float(model.backbone.std),online,device)
    assert all(r['latest_feedback_hour'] is None or r['latest_feedback_hour']<r['first_target_hour'] for r in audit)
    return dict(frozen=frozen,online=pred,target=truth,starts=starts,audit=audit,traces=traces,state=state,
                frozen_scores=scopes(frozen,truth,data),online_scores=scopes(pred,truth,data))
