# Data format

## Portable NPZ archive

Use `numpy.savez_compressed`, numeric/string arrays only (no pickled objects).

| Key | Shape/type | Meaning |
|---|---|---|
| `flow` | `[T,N]`, float32 | Finite nonnegative hourly flow |
| `weather` | `[T,N,2]`, float32 | Observed wind in m/s, rainfall in mm/hour |
| `tweet_counts` | `[T,N]`, float32 | Counts from the lagged hourly bins specified by `tweet_times` |
| `times` | `[T]`, timestamp strings | Consecutive exact hourly flow/weather timestamps |
| `tweet_times` | `[T]`, timestamp strings | Exactly `times - 1 hour`; actual collection lag must support availability |
| `node_names` | `[N]`, strings | Stable, unique region order |
| `affected` | `[N]`, bool | Retrospective evaluation subset, never model input |
| `event_start` | scalar timestamp string | Inclusive fixed disaster interval start |
| `event_stop` | scalar timestamp string | Exclusive fixed disaster interval stop |
| `train_starts`, `val_starts`, `test_starts` | optional 1D integer arrays | Supply all three or none; zero-based window input starts |

A start `s` corresponds to observations `s..s+5` and targets `s+6..s+11`.
Each split must be chronological and consecutive, and splits must not interleave.
Absent split arrays, the original 60/20/20 window scheme is constructed before
applying the requested `original` or `purged` policy. Training inputs must cover
all 24 hours of day. If a retrospective evaluation subset is empty, its metrics
are reported as empty rather than silently imputed. At least two regions and
72 hourly observations are required by the archive loader.

`tweet_counts[t]` represents the bin at `tweet_times[t]`, not the bin at `times[t]`.
The model input builder also supports explicit social release times. The default
dataset loader assumes the one-hour lag is sufficient; it cannot independently
verify a provider's release latency.

Data conversion must not normalize using the full series. The loader fits all
statistics on the retained training input windows and saves them in the checkpoint.

## Original dataset folders

The original adapter expects the research data layout under `--raw-root`:

```text
raw_datas/
  Japan-Typhoon-inflow/
    inflow_hour20180101_20210228.npy
    Japan_prefectures.csv
    Japan_2019Hurricane_Total_tweet_count.csv
    disaster_context/...
  Japan-Typhoon-outflow/
    outflow_hour20180101_20210228.npy
    Japan_prefectures.csv
    Japan_2019Hurricane_Total_tweet_count.csv
  US-Hurricane/
    poi_hour20190601_20191030.npy
    Florida_counties.csv
    US_2019Hurricane_FL_all_tweet_count_by_county.csv
    disaster_context/...
```

Japan outflow reuses the inflow `disaster_context` folder. Each context folder
contains `hourly_disaster_context.npy`, `feature_names.json`, `locations.csv`,
`timestamps_local.csv`, and `hourly_disaster_context.csv.gz`.
The original adapter verifies node order, timestamps, event flags and source files.
It uses these feature names: `wind_speed_10m_m_s`, `precip_mm_h`,
`meteorological_hazard_proxy_v1`, and `curated_event_flag`.
The compressed table supplies `time_local`, `node_id`, `curated_event_name`.
Location tables use `node_id` and `node_name`; the original region name CSVs use
the existing `prefecture_en` column even for Florida. Timestamp CSVs use
`time_index` and `time_local`. Social CSVs have timestamps in the first column and
one count column per region name, including lagged observations before the period.

Study windows start on 2019-07-01. Japan uses 2,928 hours and 47 prefectures;
Florida uses 1,728 hours and 67 counties. Japan's fixed Hagibis interval is
2019-10-10 through 2019-10-16 exclusive; Florida's fixed Dorian interval is
2019-08-31 through 2019-09-05 exclusive.
The original scope has 133 disaster windows for Japan and 109 for Florida.

The model's graph is fitted from training flow correlations; the separate raw
geographical adjacency matrices used by some baselines are not required here.
The affected-region definition for Florida is included as a documented JSON
configuration. It is retrospective and includes a later amendment; it is not
an operational predictor input. Dataset redistribution rights are not granted
by this code archive. There is no automatic private-data download.

## One-shot inference input

`python -m disaster_response.inference --checkpoint ... --input history.npz --output forecast.npz`
requires: `flow[B,6,N]`, `weather[B,6,N,2]`, `times[B,6]`, `node_names[N]`,
`tweet_counts[B,6,N]`, and `tweet_times[B,6]`. It performs a frozen checkpoint
forecast. An optional archived weather forecast uses `forecast_weather[B,6,N,2]`,
`forecast_issued_at[B]`, and `forecast_valid_times[B,6]`; issue times must not be
later than the current observation time. Online replay and delta export are
provided by `cli evaluate`; they are not implicitly applied by one-shot inference.
