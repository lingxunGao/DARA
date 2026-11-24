"""Portable hourly NPZ loader and adapter for the original three datasets."""
from pathlib import Path
import hashlib
import numpy as np
from . import protocol
from .data import prepare_data, fit_preprocessing, intensity


def checked_splits(splits, frames, policy):
    splits={k:np.asarray(splits[k],dtype=np.int64).copy() for k in ('train','val','test')}
    for key,values in splits.items():
        if values.ndim!=1 or not len(values) or (np.diff(values)!=1).any() or values[0]<0 or values[-1]+11>=frames:
            raise ValueError('Each split must have nonempty, consecutive, valid window starts: '+key)
    if not (splits['train'][-1]<splits['val'][0] and splits['val'][-1]<splits['test'][0]):
        raise ValueError('Splits must be chronologically ordered')
    if policy=='purged':
        for left,right in [('train','val'),('val','test')]:
            splits[left]=splits[left][splits[left]+11<splits[right][0]+6]
            if not len(splits[left]):raise ValueError('Purging removed a complete split')
    overlap={}
    for left,right in [('train','val'),('val','test'),('train','test')]:
        a=np.unique(splits[left][:,None]+np.arange(6,12));b=np.unique(splits[right][:,None]+np.arange(6,12))
        overlap[left+'_'+right]=len(np.intersect1d(a,b))
    if policy=='purged' and any(overlap.values()):raise ValueError('Purging failed')
    return splits,overlap


def refit(data):
    prep,normal=fit_preprocessing(data['flow'],data['weather'],data['tweets'],data['times'],
        data['splits']['train'],data['manifest']['node_names'],1,all_training_reference=True)
    hours=np.unique(data['splits']['train'][:,None]+np.arange(6))
    severity=intensity(data['weather'])
    cutoffs=np.quantile(severity[hours],[.75,.9])
    groups=np.searchsorted(cutoffs,severity,side='right').astype(np.int64)
    data.update(preprocessing=prep,normal=normal,groups=groups,severity=severity,forecasts=None)
    data['manifest'].update(preprocessing=prep,scaling=dict(mean=prep['mean'],std=prep['std']),
                            exposure_group_cutoffs=cutoffs.tolist())
    return data


def load(dataset, *, archive=None, raw_root=None, split='purged'):
    if split not in ('original','purged'):raise ValueError('Unknown split policy')
    if (archive is None)==(raw_root is None):raise ValueError('Supply exactly one of archive or raw_root')
    if raw_root is not None:
        data=prepare_data(dataset,Path(raw_root),all_training_reference=True,history=6,horizon=6)
        if split=='purged':
            data['splits'],overlap=checked_splits(data['splits'],len(data['flow']),split)
            refit(data)
            for key in ('split_audit','retained_split_audit','original_window_counts'):
                if key in data['manifest']:
                    data['manifest']['legacy_'+key]=data['manifest'].pop(key)
        else:
            data['splits'],overlap=checked_splits(data['splits'],len(data['flow']),split)
    else:
        with np.load(archive,allow_pickle=False) as z:
            required={'flow','weather','tweet_counts','times','tweet_times','node_names','affected','event_start','event_stop'}
            if not required.issubset(z.files):raise ValueError('Missing NPZ keys: '+str(required-set(z.files)))
            flow=np.asarray(z['flow'],np.float32);weather=np.asarray(z['weather'],np.float32)
            tweets=np.asarray(z['tweet_counts'],np.float32)
            raw_times=np.asarray(z['times'],dtype='datetime64[s]')
            raw_tt=np.asarray(z['tweet_times'],dtype='datetime64[s]')
            times=raw_times.astype('datetime64[h]');tt=raw_tt.astype('datetime64[h]')
            if not np.array_equal(raw_times,times) or not np.array_equal(raw_tt,tt):
                raise ValueError('Timestamps must lie on exact hourly boundaries')
            names=z['node_names'].tolist();affected=np.asarray(z['affected'],bool)
            if flow.ndim!=2 or flow.shape[1]<2 or len(flow)<72:raise ValueError('Expected flow[T,N], T>=72 and N>=2')
            t,n=flow.shape
            if weather.shape!=(t,n,2) or tweets.shape!=flow.shape:raise ValueError('Input shape mismatch')
            if times.shape!=(t,) or np.isnat(times).any() or (np.diff(times)!=np.timedelta64(1,'h')).any():raise ValueError('Consecutive hourly timestamps required')
            if tt.shape!=times.shape or not np.array_equal(tt,times-np.timedelta64(1,'h')):raise ValueError('tweet_counts must refer to the preceding hourly bin')
            for a in (flow,weather,tweets):
                if not np.isfinite(a).all() or (a<0).any():raise ValueError('Inputs must be finite and nonnegative')
            if len(names)!=n or len(set(names))!=n or affected.shape!=(n,):raise ValueError('Invalid node metadata')
            keys=['train_starts','val_starts','test_starts']
            if any(k in z.files for k in keys) and not all(k in z.files for k in keys):raise ValueError('Provide all three split arrays')
            if all(k in z.files for k in keys):
                if any(not np.issubdtype(z[k].dtype,np.integer) for k in keys):raise ValueError('Split starts must be integers')
                splits={k:z[k+'_starts'].copy() for k in ('train','val','test')}
            else:splits=protocol.make_splits(t)
            event_start=np.datetime64(str(z['event_start'].item()),'h');event_stop=np.datetime64(str(z['event_stop'].item()),'h')
            if np.isnat(event_start) or np.isnat(event_stop) or event_stop<=event_start:raise ValueError('Invalid event interval')
        splits,overlap=checked_splits(splits,t,split)
        origins=splits['test']
        event=(times[origins]>=event_start)&(times[origins+11]<event_stop)
        data=dict(flow=flow,weather=weather,tweets=tweets,times=times,splits=splits,affected=affected,
             event_windows=event,manifest=dict(dataset=dataset,node_names=names,node_count=n,history=6,horizon=6,
                study_start=str(times[0]),study_stop_exclusive=str(times[-1]+np.timedelta64(1,'h')),
                source_sha256=protocol.sha256(Path(archive)),fixed_disaster=dict(start=str(event_start),stop_exclusive=str(event_stop))))
        refit(data)
    digest=hashlib.sha256()
    for key in ('train','val','test'):digest.update(data['splits'][key].astype('<i8').tobytes())
    data['manifest'].update(release_split_policy=split,split_sha256=digest.hexdigest(),
        target_overlap=overlap,boundary_target_overlap_hours=max(overlap.values()),
        window_counts={k:len(v) for k,v in data['splits'].items()})
    return data


def check_checkpoint_data(saved,data):
    prep=saved['preprocessing']
    if set(prep)!=set(data['preprocessing']):raise ValueError('Preprocessing fields differ')
    for key,value in prep.items():
        actual=data['preprocessing'][key]
        if isinstance(value,(int,float,list)) and key!='node_names':
            np.testing.assert_allclose(value,actual,rtol=1e-6,atol=1e-6,err_msg=key)
        elif value!=actual:raise ValueError('Checkpoint preprocessing mismatch: '+key)
    original=saved.get('manifest',{}).get('data',{})
    if 'split_sha256' in original and original['split_sha256']!=data['manifest']['split_sha256']:
        raise ValueError('Checkpoint and evaluation split differ')
    # Preserve training-time float32 graph coefficients after numerical comparison.
    data['preprocessing']=prep;data['manifest']['preprocessing']=prep
