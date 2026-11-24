"""Generate synthetic hourly observations and a small runnable recipe."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('demo_data'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(17);t,n=480,4
    times=np.datetime64('2020-01-06T00','h')+np.arange(t)*np.timedelta64(1,'h')
    flow=(100+20*np.sin(np.arange(t)[:,None]*2*np.pi/24)+rng.normal(0,4,(t,n))).astype(np.float32)
    weather=rng.uniform(0,5,(t,n,2)).astype(np.float32)
    weather[405:441,:2,0]+=20;flow[405:441,:2]*=.7
    np.savez_compressed(args.output/'observations.npz',flow=flow,weather=weather,
        tweet_counts=rng.poisson(3,(t,n)).astype(np.float32),times=times.astype(str),
        tweet_times=(times-np.timedelta64(1,'h')).astype(str),node_names=np.array(['A','B','C','D']),
        affected=np.array([True,True,False,False]),event_start=np.array(str(times[405])),event_stop=np.array(str(times[441])))
    hp=dict(lr=.001,weight_decay=.0003,batch_size=16,dropout=.05,stae_layers=1,hidden=16,layers=1,
       gru_hidden=16,short_steps=3,isolate_history=False,balanced_weight=.05,mixed_sampling=False,
       regularization=.0001,epochs=2,min_epochs=1,patience=2,schedule='cosine',seed=0,
       contextual_response=True,normal_aux_free=True,reference_weight=0.,detach_reference=False,
       denoising_weight=0.,mse_weight=0.,selection_rmse_weight=.1,name='full',backbone='stae_gru',
       response='structured',response_library='knowledge_hybrid',gate=False,condition_response=True,
       shared_response_parameters=False,mlp_response_bound=.95)
    spec=dict(dataset='demo',split='purged',training=hp,online=dict(mode='shared_readout',window=12,lr=.0003,anchor=.1))
    (args.output/'config.json').write_text(json.dumps(spec,indent=2)+'\n')
    print(args.output/'observations.npz')


if __name__=='__main__':main()
