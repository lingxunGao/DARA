"""Portable fixed-configuration training and frozen/online evaluation."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from . import protocol
from .datasets import load,check_checkpoint_data
from .training import CachedSplit,fit_trial,atomic_save,write_csv
from .inference import load_model
from .evaluation import evaluate,SCOPES

CHANGES={'full':{},'backbone':dict(response='none',response_library='manual'),
         'shared':dict(shared_response_parameters=True),
         'mlp':dict(response='mlp',response_library='manual',mlp_response_bound=.95)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['train','evaluate'])
    p.add_argument('--config',type=Path,required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--data',type=Path,help='Portable NPZ archive')
    source.add_argument('--raw-root',type=Path,help='Original dataset directory structure')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--device',default='cpu')
    p.add_argument('--split',choices=['original','purged'],help='Override recipe policy; use a new output and retrain')
    p.add_argument('--variant',choices=list(CHANGES),default='full')
    p.add_argument('--epochs',type=int,help='Override epoch budget for quick checks')
    args=p.parse_args();torch.set_num_threads(1)
    spec=json.loads(args.config.read_text());hp=dict(spec['training'])
    hp.update(CHANGES[args.variant]);hp['name']=args.variant
    if args.epochs is not None:
        if args.epochs<1:p.error('epochs must be positive')
        hp.update(epochs=args.epochs,min_epochs=min(hp['min_epochs'],args.epochs))
    if not hp.get('normal_aux_free') or hp.get('reference_weight',0) or hp.get('denoising_weight',0):
        p.error('Release recipes must use joint training without auxiliary warm-up objectives')
    if spec['online']['mode']!='shared_readout':p.error('Only shared readout online adaptation is exposed')
    policy=args.split or spec.get('split','purged')
    protocol.seed_everything(hp['seed'])
    data=load(spec['dataset'],archive=args.data,raw_root=args.raw_root,split=policy)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.command=='train':
        plan=dict(training=hp,online=spec['online'],split=policy,split_sha256=data['manifest']['split_sha256'],
                  data_manifest=data['manifest'],device=args.device)
        path=args.output/'run_config.json'
        if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Run configuration changed; use a new output')
        protocol.write_json(path,plan)
        train,val=[CachedSplit(data,s,args.device) for s in ('train','val')]
        result=fit_trial(spec['dataset'],data,train,val,hp,SimpleNamespace(output=args.output,device=args.device,smoke=False))
        print('Checkpoint:',result['model_path'])
        return
    if args.checkpoint is None:p.error('evaluate requires --checkpoint')
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    check_checkpoint_data(saved,data)
    model,_=load_model(args.checkpoint,args.device)
    result=evaluate(model,data,spec['online'],args.device)
    np.savez_compressed(args.output/'predictions.npz',prediction=result['online'],frozen_prediction=result['frozen'],
            target=result['target'],starts=result['starts'],node_names=np.asarray(data['preprocessing']['node_names']))
    protocol.write_json(args.output/'feedback_audit.json',result['audit'])
    atomic_save(result['state'],args.output/'final_readout_delta.pt')
    protocol.write_json(args.output/'results.json',dict(seed=hp['seed'],checkpoint_sha256=protocol.sha256(args.checkpoint),
          online=spec['online'],split=policy,split_sha256=data['manifest']['split_sha256'],
          frozen=result['frozen_scores'],online_scores=result['online_scores']))
    rows=[]
    for mode in ('frozen','online'):
        row={'版本':mode,'种子':hp['seed']}
        for scope,title in SCOPES.items():
            for metric in ('mae','rmse'):row[title+'_'+metric.upper()]=result[mode+'_scores'][scope]['overall'][metric]
        rows.append(row)
    write_csv(args.output/'results_wide_cn.csv',rows)
    print(args.output/'results_wide_cn.csv')


if __name__=='__main__':main()
