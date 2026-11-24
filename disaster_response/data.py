"""Description.md-compatible windows, train-only fitting, causal input builder."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from . import protocol

WEATHER_SCALE = np.array([17.2, 50/6], np.float32)


def intensity(weather):
    """Instantaneous physical proxy, NOT the retrospective 24h regional proxy."""
    return (np.asarray(weather)/WEATHER_SCALE).max(-1)


def calendar(times):
    hours = times.astype('datetime64[h]').astype(np.int64)
    days = times.astype('datetime64[D]').astype(np.int64)
    hour, weekday = hours % 24, (days+3) % 7
    features = np.stack((np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
                         np.sin(2*np.pi*weekday/7), np.cos(2*np.pi*weekday/7)), -1)
    return features.astype(np.float32), weekday*24+hour


def checked_times(times, shape):
    values = np.asarray(times, dtype='datetime64[s]')
    if values.shape != shape or np.isnat(values).any() or not np.array_equal(values, values.astype('datetime64[h]')):
        raise ValueError('Expected finite exact hourly timestamps')
    if (np.diff(values, axis=1) != np.timedelta64(1, 'h')).any():
        raise ValueError('History timestamps must be consecutive')
    return values.astype('datetime64[h]')


def build_inputs(flow, weather, times, preprocessing, *, tweet_counts, tweet_times,
                 tweet_available_at=None, forecast=None):
    """Raw history -> tensors. No target, future observation, or event label argument.

    Counts are [B,6,N], individually time-stamped exactly lag hours before
    each corresponding history bin. forecast is an archived issue-time object:
    weather [B,6,N,2], issued_at [B], valid_times [B,6]. Caller must supply
    an actual forecast product, not observations relabelled as forecasts.
    """
    p = preprocessing
    history, horizon = p.get('history', 6), p.get('horizon', 6)
    flow, weather = np.asarray(flow, np.float32), np.asarray(weather, np.float32)
    if flow.ndim != 3 or flow.shape[1:] != (history, len(p['node_names'])):
        raise ValueError(f'Expected flow [B,{history},trained nodes]')
    b, _, n = flow.shape
    times = checked_times(times, (b, history))
    if weather.shape != (b, history, n, 2) or not np.isfinite(weather).all() or (weather < 0).any():
        raise ValueError('Invalid historical weather')
    if not np.isfinite(flow).all() or (flow < 0).any():
        raise ValueError('Flow must be finite and nonnegative')
    counts = np.asarray(tweet_counts, np.float32)
    tt = checked_times(tweet_times, (b, history))
    if not np.array_equal(tt, times-np.timedelta64(p['tweet_lag'], 'h')):
        raise ValueError('Tweet timestamps do not match the saved availability lag')
    if counts.shape != flow.shape or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError('Invalid tweet counts')
    if tweet_available_at is not None:
        released = np.asarray(tweet_available_at, dtype='datetime64[s]')
        if (released.shape != tt.shape or np.isnat(released).any()
                or (released > times[:, -1:]).any() or (released < tt).any()):
            raise ValueError('Tweet availability violates issue time')
    hist_cal, slots = calendar(times)
    future_times = times[:, -1:] + np.arange(1, horizon+1)*np.timedelta64(1, 'h')
    future_cal, future_slots = calendar(future_times)
    profile = np.asarray(p['profile'], np.float32)
    reference, future_reference = profile[slots], profile[future_slots]
    z = ((np.log1p(counts)-np.asarray(p['social_mean'])[slots % 24])
         / np.asarray(p['social_std'])[slots % 24])
    social = np.stack((z, np.maximum(z-1, 0)), -1)
    if forecast is None:
        future_weather = np.repeat(weather[:, -1:], horizon, axis=1)
    else:
        future_weather = np.asarray(forecast['weather'], np.float32)
        issued = np.asarray(forecast['issued_at'], dtype='datetime64[s]')
        valid = np.asarray(forecast['valid_times'], dtype='datetime64[s]')
        if (issued.shape != (b,) or np.isnat(issued).any() or (issued > times[:, -1]).any()
                or valid.shape != (b, horizon) or not np.array_equal(valid, future_times)):
            raise ValueError('Forecast issue/valid times violate prediction-time availability')
        if future_weather.shape != (b, horizon, n, 2) or not np.isfinite(future_weather).all() or (future_weather < 0).any():
            raise ValueError('Invalid forecast weather')
    values = dict(flow=flow, weather=weather, social=social,
                  excess=np.maximum(weather-np.asarray(p['weather_threshold']), 0)/WEATHER_SCALE,
                  anomaly=(flow-reference)/np.asarray(p['mad']),
                  history_reference=reference, future_reference=future_reference,
                  history_calendar=hist_cal, future_calendar=future_cal, future_weather=future_weather)
    return {k: torch.from_numpy(np.asarray(v, dtype=np.float32).copy()) for k, v in values.items()}


def load_tweets(path, names, times):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        if any(header.count(n) != 1 for n in names):
            raise ValueError('Missing or duplicated tweet node')
        columns = [header.index(n) for n in names]
        rows = {}
        for row in reader:
            stamp = np.datetime64(row[0], 's')
            if stamp in rows:
                raise ValueError('Duplicate tweet timestamp')
            rows[stamp] = [float(row[c]) for c in columns]
    try:
        values = np.array([rows[t.astype('datetime64[s]')] for t in times], np.float32)
    except KeyError as exc:
        raise ValueError(f'Missing lagged tweet bin: {exc}') from exc
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Tweets must be finite and nonnegative')
    return values


def fit_preprocessing(flow, weather, tweets, times, starts, names, tweet_lag, *, all_training_reference=False, history=6, horizon=6):
    inputs = starts[:, None]+np.arange(history)
    hours = np.unique(inputs)
    # Preserve the description's overlapping training-input-window mean/std.
    mean, std = float(flow[inputs].mean()), max(float(flow[inputs].std()), 1e-6)
    scale = np.maximum(flow[hours].mean(0), 1.).astype(np.float32)
    threshold = np.maximum(np.quantile(weather[hours].reshape(-1, 2), .9, axis=0), [.1, .01])
    _, slots = calendar(times)
    log_counts = np.log1p(tweets)
    social_mean, social_std = [], []
    for h in range(24):
        selected = hours[slots[hours] % 24 == h]
        if not len(selected):
            raise ValueError('Retained training inputs must cover all 24 hours')
        social_mean.append(log_counts[selected].mean(0))
        social_std.append(np.maximum(log_counts[selected].std(0), .25))
    social_mean, social_std = np.asarray(social_mean), np.asarray(social_std)
    social_z = (log_counts-social_mean[slots % 24])/social_std[slots % 24]
    normal = (weather <= threshold).all(-1) & (social_z <= 1)
    reference_mask = np.ones_like(normal) if all_training_reference else normal
    # Each node is fitted only on its normal retained training input observations.
    profile = np.empty((168, flow.shape[1]), np.float32)
    fallbacks = 0
    for node in range(flow.shape[1]):
        valid_hours = hours[reference_mask[hours, node]]
        if not len(valid_hours):
            raise ValueError(f'No normal reference observations for node {names[node]}')
        fallback = np.median(flow[valid_hours, node])
        for slot in range(168):
            subset = valid_hours[slots[valid_hours] == slot]
            profile[slot, node] = np.median(flow[subset, node]) if len(subset) else fallback
            fallbacks += int(not len(subset))
    # A tiny positive floor prevents a zero profile from blocking multiplicative learning.
    profile = np.maximum(profile, .001*scale)
    mad = np.empty(flow.shape[1], np.float32)
    for node in range(flow.shape[1]):
        selected = hours[reference_mask[hours, node]]
        mad[node] = max(1.4826*np.median(np.abs(flow[selected, node]-profile[slots[selected], node])), .05*scale[node])
    cleaned = flow[hours] if all_training_reference else np.where(normal[hours], flow[hours], profile[slots[hours]])
    centered = cleaned-cleaned.mean(0)
    denominator = np.linalg.norm(centered, axis=0)
    corr = (centered.T@centered)/np.maximum(denominator[:, None]*denominator[None], 1e-8)
    graph = np.maximum(corr, 0)+.01
    np.fill_diagonal(graph, 1.)
    prep = dict(history=history, horizon=horizon, node_names=list(names), mean=mean, std=std, node_scale=scale.tolist(),
                weather_threshold=threshold.tolist(), profile=profile.tolist(), mad=mad.tolist(),
                social_mean=social_mean.tolist(), social_std=social_std.tolist(), tweet_lag=tweet_lag,
                graph=graph.tolist(), graph_policy='positive correlation of normal/replaced training input flow',
                normal_policy='wind and rain <= training input q90; lagged social z<=1',
                profile_fallback_slots=fallbacks, series_start=str(times[0]))
    if all_training_reference:
        prep.update(graph_policy='positive correlation of all training input flow',
                    reference_policy='all training input observations; descriptive median, no normal filtering')
    return prep, normal


def prepare_data(dataset, raw_root, *, tweet_lag=1, train_max_intensity=None, forecast_file=None,
                 all_training_reference=False, history=6, horizon=6):
    if tweet_lag < 1:
        raise ValueError('At least one-hour lag is required for complete tweet bins')
    if train_max_intensity is not None and (not np.isfinite(train_max_intensity) or train_max_intensity <= 0):
        raise ValueError('Training intensity cap must be finite and positive')
    d = protocol.prepare_data(dataset, Path(raw_root), threshold=1., history=history, horizon=horizon)
    cfg = protocol.CONFIGS[dataset]
    folder = Path(raw_root)/cfg['folder']
    context_folder = Path(raw_root)/cfg.get('context_folder', cfg['folder'])/'disaster_context'
    fields = json.loads((context_folder/'feature_names.json').read_text())
    context = np.load(context_folder/'hourly_disaster_context.npy', mmap_mode='r')
    weather = np.asarray(context[..., [fields.index('wind_speed_10m_m_s'), fields.index('precip_mm_h')]], np.float32)
    if not np.isfinite(weather).all() or (weather < 0).any():
        raise ValueError('Invalid weather observations')
    times = protocol.START + np.arange(len(weather))*np.timedelta64(1, 'h')
    tweet_path = folder/('Japan_2019Hurricane_Total_tweet_count.csv' if cfg.get('region_dataset', dataset) == 'japan'
                         else 'US_2019Hurricane_FL_all_tweet_count_by_county.csv')
    # Read actual pre-period tweets when available; never silently zero-pad them.
    tweets = load_tweets(tweet_path, d['manifest']['node_names'], times-np.timedelta64(tweet_lag, 'h'))
    severity = intensity(weather)
    original_counts = {k: len(v) for k, v in d['splits'].items()}
    if train_max_intensity is not None:
        for split in ('train', 'val'):
            starts = d['splits'][split]
            support = starts[:, None]+np.arange(history+horizon)
            d['splits'][split] = starts[severity[support].max(axis=(1, 2)) <= train_max_intensity]
            if not len(d['splits'][split]):
                raise ValueError(f'Intensity cap removes all {split} windows')
    prep, normal = fit_preprocessing(d['flow'], weather, tweets, times, d['splits']['train'],
                                     d['manifest']['node_names'], tweet_lag,
                                     all_training_reference=all_training_reference, history=history, horizon=horizon)
    inputs = d['splits']['train'][:, None]+np.arange(history)
    support = np.unique(d['splits']['train'][:, None]+np.arange(history+horizon))
    weak_cutoff = max(float(np.quantile(severity[np.unique(inputs)], .9)), 1e-6)
    groups = np.where(normal, 0, np.where(severity < weak_cutoff, 1, 2)).astype(np.int64)
    if all_training_reference:
        cutoffs = np.quantile(severity[np.unique(inputs)], [.75, .9])
        groups = np.searchsorted(cutoffs, severity, side='right').astype(np.int64)
        d['manifest']['exposure_group_cutoffs'] = cutoffs.tolist()
    forecasts = None
    if forecast_file is not None:
        with np.load(forecast_file, allow_pickle=False) as f:
            if not np.array_equal(f['node_names'], prep['node_names']):
                raise ValueError('Forecast node order mismatch')
            origins = f['starts']
            if origins.ndim != 1 or not np.issubdtype(origins.dtype, np.integer) or len(np.unique(origins)) != len(origins):
                raise ValueError('Forecast starts must be unique integers')
            if f['weather'].shape != (len(origins), horizon, len(prep['node_names']), 2) or f['issued_at'].shape != (len(origins),) or f['valid_times'].shape != (len(origins), horizon):
                raise ValueError('Invalid forecast archive dimensions')
            issued = f['issued_at'].astype('datetime64[s]')
            valid = f['valid_times'].astype('datetime64[s]')
            expected = protocol.START+(origins[:, None]+history+np.arange(horizon))*np.timedelta64(1, 'h')
            issue_limit = protocol.START+(origins+history-1)*np.timedelta64(1, 'h')
            if (np.isnat(issued).any() or (issued > issue_limit).any() or not np.array_equal(valid, expected)
                    or not np.isfinite(f['weather']).all() or (f['weather'] < 0).any()):
                raise ValueError('Forecast archive violates weather or issue/valid time contract')
            forecasts = {int(s): {key: f[key][i:i+1].copy() for key in ('weather', 'issued_at', 'valid_times')}
                         for i, s in enumerate(origins)}
        if any(int(s) not in forecasts for s in np.concatenate(list(d['splits'].values()))):
            raise ValueError('Forecast archive does not cover the selected splits')
    audit = {}
    for split, starts in d['splits'].items():
        covered = np.unique(starts[:, None]+np.arange(history+horizon))
        labels, counts = np.unique(d['event_names'][covered], return_counts=True)
        audit[split] = dict(windows=len(starts), intensity_max=float(severity[covered].max()),
                            unique_hours=len(covered), curated_events={str(k): int(v) for k, v in zip(labels, counts) if k})
    d['manifest'].update(model='StructuredDisasterResponse', preprocessing=prep,
                         original_window_counts=original_counts, retained_split_audit=audit,
                         train_max_intensity=train_max_intensity,
                         intensity_proxy='max(wind_m_s/17.2, rain_mm_h/(50/6)); no upper clipping',
                         training_intensity_max=float(severity[support].max()), group_intensity_cutoff=weak_cutoff,
                         scaling=dict(mean=prep['mean'], std=prep['std']),
                         weather_mode='archived_forecast' if forecasts is not None else 'persistence',
                         social=dict(source=str(tweet_path.resolve()), lag_hours=tweet_lag,
                                     timezone='CSV labels aligned to local hours; original collection timezone unverified'))
    d['manifest']['source_files'][str(tweet_path.resolve())] = protocol.sha256(tweet_path)
    if forecast_file is not None:
        d['manifest']['source_files'][str(Path(forecast_file).resolve())] = protocol.sha256(Path(forecast_file))
    d['manifest']['caveats'] = [
        f'Original window protocol has {horizon-1} overlapping target hours at split boundaries.',
        'Historical NASA POWER weather is retrospective; operational release latency is unverified.',
        'Tweet timezone/collection latency is unverified; counts are discussion proxies, not warning labels.',
        'Persistence cannot anticipate new future weather shocks; no future observed weather is an input.',
        'Normal/exposure groups are proxies; fixed affected-node labels are evaluation-only.',
        'An intensity cap selects windows, not independent weak events. Inspect event and intensity audits.',
        'US training has no curated disaster events. Earlier Japan events are not necessarily weaker.',
        'Overlapping event windows are dependent; diagnostics do not establish independent-event generalization.',
        f'Only {history} history hours initialize suppression; longer pre-existing impacts are not directly observed.',
        'Weak-event data do not uniquely identify strong-event response shapes.']
    d.update(weather=weather, tweets=tweets, times=times, preprocessing=prep, normal=normal,
             groups=groups, severity=severity, forecasts=forecasts)
    return d


class Windows(Dataset):
    def __init__(self, data, split):
        self.data = data
        self.starts = data['splits'][split]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        d, s = self.data, int(self.starts[index])
        history, horizon = d['preprocessing'].get('history', 6), d['preprocessing'].get('horizon', 6)
        sl, target = slice(s, s+history), slice(s+history, s+history+horizon)
        times = d['times'][None, sl]
        inputs = build_inputs(d['flow'][None, sl], d['weather'][None, sl], times, d['preprocessing'],
                              tweet_counts=d['tweets'][None, sl],
                              tweet_times=times-np.timedelta64(d['preprocessing']['tweet_lag'], 'h'),
                              forecast=None if d['forecasts'] is None else d['forecasts'][s])
        # A normal target must also have a normal six-hour local input history.
        normal = d['normal'][target] & d['normal'][sl].all(0)[None]
        return dict(inputs={k: v[0] for k, v in inputs.items()},
                    target=torch.from_numpy(d['flow'][target, :, None].copy()),
                    normal=torch.from_numpy(normal.copy()),
                    group=torch.from_numpy(d['groups'][target].copy()), start=s)


class MixedSampler(Sampler):
    """Half upper-quartile exposure / half remaining windows, fitted on train only.

    Using ANY exposed node makes nearly every 12h window positive with 47/67
    nodes. Rank whole-window physical exposure to avoid oversampling a handful
    of completely quiet windows thousands of times.
    """
    def __init__(self, dataset, batch_size, seed):
        if batch_size < 2:
            raise ValueError('Mixed sampling needs batch_size >= 2')
        if not np.array_equal(dataset.starts, dataset.data['splits']['train']):
            raise ValueError('Mixed sampling is fitted only on training windows')
        self.size, self.batch_size = len(dataset), batch_size
        support = dataset.starts[:, None]+np.arange(dataset.data['preprocessing'].get('history', 6)+dataset.data['preprocessing'].get('horizon', 6))
        strength = dataset.data['severity'][support].max(axis=(1, 2))
        self.cutoff = float(np.quantile(strength, .75))
        has_high = strength >= self.cutoff
        self.high, self.ordinary = np.flatnonzero(has_high), np.flatnonzero(~has_high)
        self.seed, self.epoch = seed, 0
        self.fallback = not len(self.high) or not len(self.ordinary)

    def __len__(self):
        return (self.size+self.batch_size-1)//self.batch_size

    def __iter__(self):
        rng = np.random.default_rng(self.seed+self.epoch)
        self.epoch += 1
        if self.fallback:
            order = rng.permutation(self.size)
            for i in range(0, self.size, self.batch_size):
                yield order[i:i+self.batch_size].tolist()
        else:
            for _ in range(len(self)):
                batch = np.r_[rng.choice(self.high, self.batch_size//2),
                              rng.choice(self.ordinary, self.batch_size-self.batch_size//2)]
                rng.shuffle(batch)
                yield batch.tolist()
