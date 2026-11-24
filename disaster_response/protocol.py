"""Data/evaluation snapshot from Ours_OOD/reference/disaster_protocol.py.

Preserves raw_datas/Description.md study periods, window-first splits and
regional scoring. The external-input policy is replaced by data.prepare_data.
This package does not import or modify the previous model implementations.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import random

import numpy as np
import torch

from .regions import OFFICIAL_CONFIG, select_regions

CONFIGS = {
    'japan': dict(folder='Japan-Typhoon-inflow', flow='inflow_hour20180101_20210228.npy',
                  adj='adjacency_matrix.npy', names='Japan_prefectures.csv',
                  source_start='2018-01-01T00', frames=2928, nodes=47,
                  disaster_start='2019-10-10T00', disaster_stop='2019-10-16T00',
                  event='Hagibis', scenario='prior_events_to_hagibis',
                  expected_counts=[1750, 583, 584], expected_disaster=133),
    'us': dict(folder='US-Hurricane', flow='poi_hour20190601_20191030.npy',
               adj='FL_county_adj_mtx.npy', names='Florida_counties.csv',
               source_start='2019-06-01T00', frames=1728, nodes=67,
               disaster_start='2019-08-31T00', disaster_stop='2019-09-05T00',
               event='Dorian', scenario='no_curated_event_to_dorian',
               expected_counts=[1030, 343, 344], expected_disaster=109),
}
CONFIGS['japan_outflow'] = dict(CONFIGS['japan'], folder='Japan-Typhoon-outflow',
    flow='outflow_hour20180101_20210228.npy', context_folder='Japan-Typhoon-inflow',
    region_dataset='japan')
START = np.datetime64('2019-07-01T00', 'h')
HISTORY = HORIZON = 6
SCOPES = ('all_time_all_nodes', 'disaster_time_all_nodes',
          'disaster_time_affected_nodes', 'all_time_unaffected_nodes')


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def make_splits(frames, history=HISTORY, horizon=HORIZON):
    starts = np.arange(frames - history - horizon + 1, dtype=np.int64)
    nt, nv = round(len(starts) * .6), round(len(starts) * .2)
    return dict(train=starts[:nt], val=starts[nt:nt+nv], test=starts[nt+nv:])


def disaster_mask(starts, start_hour, stop_hour, history=HISTORY, horizon=HORIZON):
    """All 12 input/target hours must be inside the right-open event interval."""
    return (starts >= start_hour) & (starts + history + horizon <= stop_hour)


def score_metrics(pred, target):
    """Aggregate valid prediction instances; never average horizon RMSEs."""
    if pred.shape != target.shape or not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError('Invalid prediction/target shape or non-finite values')
    valid = target != 0
    count = int(valid.sum())
    if not count:
        return dict(mae=None, rmse=None, valid_count=0, status='empty')
    error = pred.astype(np.float64)[valid] - target.astype(np.float64)[valid]
    return dict(mae=float(np.abs(error).mean()), rmse=float(np.sqrt(np.square(error).mean())),
                valid_count=count, status='ok')


def scope_metrics(pred, target, starts, event_windows, affected):
    selections = ((np.ones(len(starts), bool), np.ones(len(affected), bool)),
                  (event_windows, np.ones(len(affected), bool)),
                  (event_windows, affected),
                  (np.ones(len(starts), bool), ~affected))
    result = {}
    for scope, (windows, nodes) in zip(SCOPES, selections):
        p, y = pred[windows][:, :, nodes], target[windows][:, :, nodes]
        result[scope] = dict(window_count=int(windows.sum()), node_count=int(nodes.sum()),
                             overall=score_metrics(p, y),
                             horizons={str(h+1): score_metrics(p[:, h], y[:, h]) for h in range(target.shape[1])})
    return result


def prepare_data(dataset, raw_root, threshold, us_rule='official', *, history=6, horizon=6):
    if history < 1 or horizon < 1:
        raise ValueError('History and horizon must be positive')
    HISTORY, HORIZON = history, horizon
    cfg = CONFIGS[dataset]
    root = raw_root / cfg['folder']
    context_root = raw_root / cfg.get('context_folder', cfg['folder']) / 'disaster_context'
    region_dataset = cfg.get('region_dataset', dataset)
    offset = int((START - np.datetime64(cfg['source_start'], 'h')) / np.timedelta64(1, 'h'))
    flow = np.load(root / cfg['flow'], mmap_mode='r')[offset:offset+cfg['frames']].astype(np.float32)
    if flow.shape != (cfg['frames'], cfg['nodes']) or not np.isfinite(flow).all():
        raise ValueError(f'{dataset}: unexpected flow shape or non-finite flow')
    with (root / cfg['names']).open(encoding='utf-8-sig') as f:
        names = [r['prefecture_en'] for r in csv.DictReader(f)]
    # Japan's CSV has a trailing 48th 'Unknown'; the flow has only 47 prefectures.
    if region_dataset == 'japan' and names == names[:cfg['nodes']] + ['Unknown']:
        names = names[:cfg['nodes']]
    with (context_root / 'locations.csv').open(encoding='utf-8-sig') as f:
        locations = list(csv.DictReader(f))
    if names != [r['node_name'] for r in locations] or [int(r['node_id']) for r in locations] != list(range(cfg['nodes'])):
        raise ValueError(f'{dataset}: node order mismatch')
    with (context_root / 'timestamps_local.csv').open(encoding='utf-8') as f:
        time_rows = list(csv.DictReader(f))
    timestamps = START + np.arange(len(flow)).astype('timedelta64[h]')
    if (len(time_rows) != len(flow) or
            [int(r['time_index']) for r in time_rows] != list(range(len(flow))) or
            not np.array_equal(np.array([r['time_local'] for r in time_rows], dtype='datetime64[h]'), timestamps)):
        raise ValueError(f'{dataset}: hourly context time alignment mismatch')
    fields = json.loads((context_root / 'feature_names.json').read_text())
    context = np.load(context_root / 'hourly_disaster_context.npy', mmap_mode='r')
    if context.shape != (len(flow), len(names), len(fields)):
        raise ValueError(f'{dataset}: context shape mismatch')
    event_flag = context[:, :, fields.index('curated_event_flag')]
    if not np.isfinite(event_flag).all() or not np.all(event_flag == event_flag[:, :1]):
        raise ValueError(f'{dataset}: inconsistent event flags')
    event_names = [''] * len(flow)
    seen = set()
    with gzip.open(context_root / 'hourly_disaster_context.csv.gz', 'rt') as f:
        for row in csv.DictReader(f):
            if int(row['node_id']) == 0:
                i = int((np.datetime64(row['time_local'], 'h') - START) / np.timedelta64(1, 'h'))
                if i < 0 or i >= len(flow) or i in seen:
                    raise ValueError('Invalid event table time index')
                event_names[i] = row['curated_event_name']
                seen.add(i)
    if len(seen) != len(flow) or not np.array_equal(np.array([bool(e) for e in event_names]), event_flag[:, 0] > 0):
        raise ValueError('Event names/flags are inconsistent')
    event_names = np.asarray(event_names)
    splits = make_splits(len(flow), history, horizon)
    if (history, horizon) == (6, 6) and [len(splits[k]) for k in ('train', 'val', 'test')] != cfg['expected_counts']:
        raise ValueError('Split counts mismatch')
    train_x = flow[splits['train'][:, None] + np.arange(HISTORY)[None]]
    mean, std = float(train_x.mean()), float(train_x.std())
    if std < 1e-8:
        std = 1.
    hours = np.arange(len(flow))
    # Both datasets' study periods begin at Monday 00:00 local time.
    features = np.stack(((flow-mean)/std,
                         np.broadcast_to((hours % 24 / 24.)[:, None], flow.shape),
                         np.broadcast_to((hours // 24 % 7)[:, None], flow.shape)), axis=-1).astype(np.float32)
    lo = int((np.datetime64(cfg['disaster_start']) - START) / np.timedelta64(1, 'h'))
    hi = int((np.datetime64(cfg['disaster_stop']) - START) / np.timedelta64(1, 'h'))
    hazard = np.asarray(context[:, :, fields.index('meteorological_hazard_proxy_v1')])
    if not np.isfinite(hazard[lo:hi]).all():
        raise ValueError('Missing hazard proxy inside the fixed disaster window')
    peak = hazard[lo:hi].max(axis=0)
    affected, affected_rule = select_regions(region_dataset, names, peak, threshold, us_rule)
    event_windows = disaster_mask(splits['test'], lo, hi, history, horizon)
    if int(event_windows.sum()) != hi-lo-history-horizon+1:
        raise ValueError('Fixed disaster-window count mismatch')
    audit = {}
    for split, origins in splits.items():
        # Include BOTH input and target frames; deduplicate overlapping hours.
        covered = np.unique(origins[:, None] + np.arange(HISTORY + HORIZON)[None])
        labels, counts = np.unique(event_names[covered], return_counts=True)
        audit[split] = dict(window_count=len(origins), first_window_start=int(origins[0]),
                            last_window_start=int(origins[-1]),
                            curated_events={str(e): int(c) for e, c in zip(labels, counts) if e},
                            event_hour_count=int((event_flag[covered, 0] > 0).sum()),
                            meteorological_proxy_peak=float(np.nanmax(hazard[covered])))
    if cfg['event'] in audit['train']['curated_events'] or cfg['event'] in audit['val']['curated_events']:
        raise ValueError('The held-out major event appears in training/validation')
    if dataset == 'us' and (audit['train']['event_hour_count'] or audit['val']['event_hour_count']):
        raise ValueError('US train/validation unexpectedly contains curated disaster hours')
    if region_dataset == 'japan' and not audit['train']['event_hour_count']:
        raise ValueError('Japan training unexpectedly contains no prior events')
    used = [root / cfg['flow'], root / cfg['names'],
            context_root / 'hourly_disaster_context.npy', context_root / 'feature_names.json',
            context_root / 'locations.csv', context_root / 'timestamps_local.csv',
            context_root / 'hourly_disaster_context.csv.gz']
    if dataset == 'us' and us_rule == 'official':
        used.append(OFFICIAL_CONFIG)
    manifest = dict(dataset=dataset, scenario=cfg['scenario'], study_start=str(START),
                    study_stop_exclusive=str(timestamps[-1] + np.timedelta64(1, 'h')),
                    history=HISTORY, horizon=HORIZON, node_count=len(names),
                    split_policy='all windows first, chronological round(60%)/round(20%)/remainder',
                    boundary_target_overlap_hours=horizon-1, scaling=dict(mean=mean, std=std),
                    fixed_disaster=dict(event=cfg['event'], start=cfg['disaster_start'],
                                        stop_exclusive=cfg['disaster_stop'], window_count=int(event_windows.sum()),
                                        selection=f'entire {history}-hour input AND {horizon}-hour target inside interval'),
                    affected_rule=affected_rule,
                    affected_nodes=[dict(node_id=int(i), name=names[i], peak=float(peak[i])) for i in np.flatnonzero(affected)],
                    unaffected_node_ids=np.flatnonzero(~affected).tolist(),
                    node_names=names, split_audit=audit,
                    scope_counts={k: dict(window_count=v['window_count'], node_count=v['node_count'])
                                  for k, v in scope_metrics(np.zeros((len(splits['test']), horizon, len(names), 1)),
                                                           flow[splits['test'][:, None]+history+np.arange(horizon)[None], :, None],
                                                           splits['test'], event_windows, affected).items()},
                    caveats=['Earlier Japan events have no authoritative small-disaster severity label.',
                             'US no-disaster means no curated-event labels, not absence of all hazardous weather.',
                             'US official designation and Japan meteorological exposure are different evidence definitions.',
                             'Unaffected means the complement of the chosen region definition, not proof of no impact.',
                             'Weather and event labels are used only for retrospective evaluation; model uses historical calendars only; no future weather/event/flow inputs.'],
                    source_files={str(p.resolve()): sha256(p) for p in used})
    return dict(flow=flow, features=features, splits=splits, mean=mean, std=std, event_names=event_names,
                affected=affected, event_windows=event_windows, manifest=manifest)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

