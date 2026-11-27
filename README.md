# DARA

Disaster-aware flow forecasting with an STAEformer–GRU backbone,
context-conditioned responses, and online readout adaptation.
The model uses six observed hours to predict the next six hours.

## Install

```bash
pip install -r requirements.txt
```

## 1. Offline training

Jointly train the backbone and response modules on historical observations:

```bash
python main.py train --data data/train.npz --output runs/offline --device cpu
```

The final 20% of observations are reserved for chronological validation,
with no overlapping training/validation targets. Preprocessing is fitted only
on training inputs. The checkpoint with the lowest validation loss is saved
as `runs/offline/model.pt`. Use `--epochs` to set the training budget.

## 2. Online prediction and update

Load the offline checkpoint and replay subsequent hourly observations:

```bash
python main.py online --data data/online.npz --checkpoint runs/offline/model.pt --output runs/online --device cpu
```

The backbone encoder and response modules stay frozen. As true observations
arrive, only the shared prediction layer's weight and bias increments are
updated, before issuing the next forecast. Future labels are never used.
Each invocation starts a new replay from the offline checkpoint; it is not a
live data service. Outputs are `predictions.npz` and `readout_delta.pt`.

## Data

Supply your own NPZ files with these fields (`T`: hours, `N`: nodes):

| Field | Shape / meaning |
| --- | --- |
| `flow` | `[T, N]`, nonnegative observed flow |
| `weather` | `[T, N, 2]`, wind (m/s) and rainfall (mm/hour) |
| `tweet_counts` | `[T, N]`, social counts from the preceding hour |
| `times` | `[T]`, consecutive hourly timestamp strings |
| `tweet_times` | `[T]`, exactly one hour before `times` |
| `node_names` | `[N]`, unique strings in the same order in both files |

Use at least two nodes and 48 hours for offline training with the default split.
Online data needs at least 12 hours; the first six are initial history, and
all forecast targets must follow the offline training/validation period.
Future weather defaults to persistence of the latest observation.
Use `--device cuda:0` for GPU execution and `python main.py train --help` or
`python main.py online --help` for runtime options.

## Paper datasets and disaster definitions

This section documents the datasets, event labels, sampling rules, and regional
evaluation groups used in the paper. The generic NPZ training example above has
its own default 80:20 split; it does not automatically reproduce the paper's
60:20:20 window split or limited-disaster sampling protocol below.

### Study periods and evaluation periods

All observations are hourly. Dates below include both endpoints, from 00:00 on
the first day through 23:00 on the last day. Japan uses JST (UTC+9); Florida uses
EDT (UTC-4) throughout its study period. The two Japanese tasks share the same
locations, event labels, time splits, and regional evaluation groups.

| Dataset | Forecast target | Locations | Study period (2019) | Fixed disaster evaluation period (2019) | Affected / other locations |
| --- | --- | ---: | --- | --- | ---: |
| Typhoon-JP inflow | Regional inflow | 47 prefectures | Jul 1–Oct 30 | Oct 10–15, Hagibis | 22 / 25 |
| Typhoon-JP outflow | Regional outflow | 47 prefectures | Jul 1–Oct 30 | Oct 10–15, Hagibis | 22 / 25 |
| Hurricane-US | Regional POI visits | 67 Florida counties | Jul 1–Sep 10 | Aug 31–Sep 4, Dorian | 14 / 53 |

Official reports establish the events and their documented impacts. The fixed
evaluation periods above are study-defined intervals around these events, not
verbatim official incident periods or complete storm lifetimes. In particular,
the [JMA Hagibis bulletin](https://www.jma.go.jp/jma/press/1910/15a/20191015_saimota.html)
covers October 10–13, whereas our evaluation includes October 14–15 as well.
The [FEMA Dorian situation report](https://content.govdelivery.com/attachments/USDHSFEMA/2020/01/03/file_attachments/1352101/SITREP3-FL-4468-01032020.pdf)
records an August 28–September 9 incident period, from which our evaluation uses
the narrower interval above.

Each sample contains six input hours and six target hours, with a one-hour
stride. All windows are generated first and then split chronologically:
`train = round(0.6 * M)`, `validation = round(0.2 * M)`, and test receives the
remainder, where `M = T - 11` and `T` is the number of hourly observations.
Window starts and node IDs are zero-based.

| Dataset | Total windows | Train starts (count) | Validation starts (count) | Test starts (count) |
| --- | ---: | --- | --- | --- |
| Typhoon-JP, each task | 2,917 | 0–1749 (1,750) | 1750–2332 (583) | 2333–2916 (584) |
| Hurricane-US | 1,717 | 0–1029 (1,030) | 1030–1372 (343) | 1373–1716 (344) |

The original window-first paper split shares five target hours across each
adjacent split boundary. It differs from the non-overlapping-target split in
the generic training example above.

For disaster-period **evaluation**, all twelve input/target hours must lie
inside the fixed evaluation interval. This yields 133 Hagibis windows and
109 Dorian windows. The training-window labeling rule below instead uses any
overlap; these are two distinct filters.

### Historical training events and window labels

Typhoon-JP has four named typhoons and one heavy-rain event in its curated
training-event set. The following labeling intervals are selected for the study,
including surrounding observations; they are not official storm lifetimes.
They do not constitute an exhaustive catalog of every hazardous weather event
in Japan during the study period.

| Event | Type / JMA typhoon number | Labeling interval (2019, JST; inclusive) | Disaster-related training windows |
| --- | --- | --- | ---: |
| Nari | Typhoon T1906 | Jul 25–30 | 155 |
| Francisco | Typhoon T1908 | Aug 4–9 | 155 |
| Krosa | Typhoon T1910 | Aug 13–20 | 203 |
| Northern Kyushu heavy rain | Frontal heavy rain | Aug 25–31 | 179 |
| Faxai | Typhoon T1915 | Sep 7–15 | 129 |
| **Total** | | | **821** |

Event identities can be located in the [JMA 2019 typhoon table, Table 1.3-1](https://www.data.jma.go.jp/cpdinfo/monitor/2019/pdf/ccmr2019_all.pdf)
and the [JMA Northern Kyushu heavy-rain report](https://www.data.jma.go.jp/stats/data/bosai/report/2019/20190826/jyun_sokuji20190826-0829.pdf).
The latter documents August 26–29; our August 25–31 label is a study-defined
extension. The exact intervals in the table, rather than a presumed uniform
extension of official dates, define the labels.

Only training windows are eligible for sampling. For Faxai, the last training
window starts on September 11 at 21:00 and its final target is September 12 at
08:00 JST. Thus its full September 7–15 labeling interval is not fully available
for training; later Faxai observations occur in the fixed validation split.

A training window is disaster-related if **any hour in its six-hour input or
six-hour prediction target overlaps a curated event-labeling interval**.
For a window starting at hour index `s`, check hours `s, ..., s+11`. Otherwise
it is a non-event window. The label is temporal and applies to the whole
multi-region sample, not separately to the 22 affected prefectures. Labels are
used for sample selection, not supplied as model inputs.

The Japanese training split has 821 disaster-related and 929 non-event windows.
Hurricane-US has 1,030 non-event training windows and no curated disaster
training windows, so it is excluded from the limited-experience experiment.
"Non-event" means outside the curated intervals, not proven absence of all
severe weather.

For completeness, additional curated labels in the study periods are:

| Dataset | Event | Labeling interval (2019; local time) | Training windows |
| --- | --- | --- | ---: |
| Typhoon-JP | Hagibis | Oct 9–20 | 0 |
| Typhoon-JP | Oct 24–26 heavy rain | Oct 23–30 | 0 |
| Hurricane-US | Dorian | Aug 29–Sep 10 | 0 |

These wider labels must not be substituted for the fixed disaster evaluation
periods given above. Hagibis and Dorian are absent from both training and
validation event labels.

### Retaining 25%, 50%, and 100% of disaster training windows

The full (100%) set comprises all 821 disaster-related training windows across
the five historical events. Lower fractions are sampled proportionally across
events and their temporal strata, with nested subsets: 25% is contained in 50%,
which is contained in 100%. Both models use identical subsets for a given seed;
inflow and outflow share the same sampled window starts.

| Retained fraction | Disaster windows | Fixed non-event windows | Total training windows |
| --- | ---: | ---: | ---: |
| 25% | 205 | 929 | 1,134 |
| 50% | 411 | 929 | 1,340 |
| 100% | 821 | 929 | 1,750 |

For exact sampling, sort the training-window starts for each event and divide
them into four consecutive, approximately equal-sized groups (`numpy.array_split`).
Within each group, draw a random anchor and grow a contiguous block one start at
a time: choose left or right with equal probability when both are available,
and use the remaining side at a boundary. For fraction `p`, allocate `p` times
the group size, take the floor, then distribute remaining slots by descending
fractional remainder (random tie-breaking) to reach `floor(821*p + 0.5)` windows.
Use prefixes of the same block-growth order at each fraction, checking nesting.
Events are processed chronologically and groups in temporal order, with
`numpy.random.default_rng(20261118 + seed)`. The four groups are chronological
strata of the available training portion, not independently identified physical
warning/impact/recovery stages.

All non-event training windows and the validation/test splits stay fixed.
Shared preprocessing is fitted only on the non-event training inputs. Models
are retrained at each fraction, with online updates disabled for this comparison.
The ratio measures supervised-window retention, not independent event count or
unique-hour coverage; adjacent windows overlap. The fixed validation set still
contains later Faxai observations.

### Rules for affected regions

The two datasets use different operational definitions. Neither list is selected
using mobility values or model prediction errors. These fixed lists are used
only for retrospective regional evaluation. "Other regions" means the complement
of the selected list, not confirmed absence of storm impacts.

**Hurricane-US: official assistance designation (14 counties).** Select counties
designated for Public Assistance categories A–G under FEMA-4468-DR-FL. The
October 21, 2019 declaration names 12 counties; Amendment No. 1, issued December 9,
adds Broward and Volusia. Both entries are recorded on page 1 of the
[FEMA Dorian report](https://content.govdelivery.com/attachments/USDHSFEMA/2020/01/03/file_attachments/1352101/SITREP3-FL-4468-01032020.pdf).
The statewide emergency declaration and statewide hazard-mitigation eligibility
are not the selection criterion. This is an official response/recovery designation,
not an assertion of equal physical damage in all selected counties.

**Typhoon-JP: meteorological exposure proxy (22 prefectures).** For each prefecture,
compute the maximum hourly exposure score during October 10–15, 2019 (JST):

$$
A_i = \mathbf{1}\!\left[
\max_{t\in I_{\mathrm{Hagibis}}}
\max\left(\frac{P^{6h}_{i,t}}{50},
          \frac{P^{24h}_{i,t}}{100},
          \frac{W^{10m}_{i,t}}{17.2}\right) \ge 1
\right].
$$

Here `I_Hagibis = [2019-10-10 00:00, 2019-10-16 00:00)`; `P6h` and `P24h`
are trailing six-hour and 24-hour accumulated precipitation (mm), and `W10m`
is wind speed at 10 m (m/s). Reaching any one threshold at any hour selects the
prefecture. The proxy field is named `meteorological_hazard_proxy_v1`.

Weather is sampled from NASA POWER at prefectural-government representative
locations, not averaged over prefectural areas. In the archived POWER responses,
`PRECTOTCORR` is an hourly sampled rate with units `mm/day`; divide by 24 to obtain
the hourly-equivalent amount before accumulating precipitation. Time stamps are
converted from UTC to JST before interval selection. Representative-point data
and these thresholds define an exposure proxy; JMA does not certify this exact
22-prefecture list, and the thresholds are study-defined rather than an official
damage classification.

### Complete location lists and affected-set membership

Node IDs below follow the original flow-matrix column order and start at zero.
"Yes" denotes membership in the event-specific affected set defined above.
Japan's metadata contains an additional `Unknown` row, which is excluded because
the flow arrays contain only the 47 prefectures.

#### Typhoon-JP inflow and outflow: all 47 prefectures

| Node ID | Prefecture | Japanese name | Hagibis affected set |
| ---: | --- | --- | --- |
| 0 | Hokkaido | 北海道 | No |
| 1 | Aomori | 青森県 | No |
| 2 | Iwate | 岩手県 | No |
| 3 | Miyagi | 宮城県 | Yes |
| 4 | Akita | 秋田県 | No |
| 5 | Yamagata | 山形県 | Yes |
| 6 | Fukushima | 福島県 | Yes |
| 7 | Ibaraki | 茨城県 | Yes |
| 8 | Tochigi | 栃木県 | No |
| 9 | Gunma | 群馬県 | Yes |
| 10 | Saitama | 埼玉県 | Yes |
| 11 | Chiba | 千葉県 | Yes |
| 12 | Tokyo | 東京都 | Yes |
| 13 | Kanagawa | 神奈川県 | Yes |
| 14 | Niigata | 新潟県 | Yes |
| 15 | Toyama | 富山県 | No |
| 16 | Ishikawa | 石川県 | Yes |
| 17 | Fukui | 福井県 | Yes |
| 18 | Yamanashi | 山梨県 | Yes |
| 19 | Nagano | 長野県 | No |
| 20 | Gifu | 岐阜県 | No |
| 21 | Shizuoka | 静岡県 | Yes |
| 22 | Aichi | 愛知県 | Yes |
| 23 | Mie | 三重県 | Yes |
| 24 | Shiga | 滋賀県 | Yes |
| 25 | Kyoto | 京都府 | Yes |
| 26 | Osaka | 大阪府 | Yes |
| 27 | Hyogo | 兵庫県 | No |
| 28 | Nara | 奈良県 | Yes |
| 29 | Wakayama | 和歌山県 | No |
| 30 | Tottori | 鳥取県 | No |
| 31 | Shimane | 島根県 | Yes |
| 32 | Okayama | 岡山県 | No |
| 33 | Hiroshima | 広島県 | No |
| 34 | Yamaguchi | 山口県 | No |
| 35 | Tokushima | 徳島県 | No |
| 36 | Kagawa | 香川県 | No |
| 37 | Ehime | 愛媛県 | No |
| 38 | Kochi | 高知県 | Yes |
| 39 | Fukuoka | 福岡県 | No |
| 40 | Saga | 佐賀県 | No |
| 41 | Nagasaki | 長崎県 | No |
| 42 | Kumamoto | 熊本県 | No |
| 43 | Oita | 大分県 | No |
| 44 | Miyazaki | 宮崎県 | No |
| 45 | Kagoshima | 鹿児島県 | No |
| 46 | Okinawa | 沖縄県 | No |

Affected prefectures (22): Miyagi, Yamagata, Fukushima, Ibaraki, Gunma, Saitama, Chiba, Tokyo, Kanagawa, Niigata, Ishikawa, Fukui, Yamanashi, Shizuoka, Aichi, Mie, Shiga, Kyoto, Osaka, Nara, Shimane, Kochi.

#### Hurricane-US: all 67 Florida counties

| Node ID | County | Dorian affected set | Designation |
| ---: | --- | --- | --- |
| 0 | Alachua | No | — |
| 1 | Baker | No | — |
| 2 | Bay | No | — |
| 3 | Bradford | No | — |
| 4 | Brevard | Yes | October 21 declaration |
| 5 | Broward | Yes | December 9 amendment |
| 6 | Calhoun | No | — |
| 7 | Charlotte | No | — |
| 8 | Citrus | No | — |
| 9 | Clay | No | — |
| 10 | Collier | No | — |
| 11 | Columbia | No | — |
| 12 | DeSoto | No | — |
| 13 | Dixie | No | — |
| 14 | Duval | Yes | October 21 declaration |
| 15 | Escambia | No | — |
| 16 | Flagler | Yes | October 21 declaration |
| 17 | Franklin | No | — |
| 18 | Gadsden | No | — |
| 19 | Gilchrist | No | — |
| 20 | Glades | No | — |
| 21 | Gulf | No | — |
| 22 | Hamilton | No | — |
| 23 | Hardee | No | — |
| 24 | Hendry | No | — |
| 25 | Hernando | No | — |
| 26 | Highlands | No | — |
| 27 | Hillsborough | No | — |
| 28 | Holmes | No | — |
| 29 | Indian River | Yes | October 21 declaration |
| 30 | Jackson | No | — |
| 31 | Jefferson | No | — |
| 32 | Lafayette | No | — |
| 33 | Lake | No | — |
| 34 | Lee | No | — |
| 35 | Leon | No | — |
| 36 | Levy | No | — |
| 37 | Liberty | No | — |
| 38 | Madison | No | — |
| 39 | Manatee | No | — |
| 40 | Marion | No | — |
| 41 | Martin | Yes | October 21 declaration |
| 42 | Miami-Dade | No | — |
| 43 | Monroe | No | — |
| 44 | Nassau | Yes | October 21 declaration |
| 45 | Okaloosa | No | — |
| 46 | Okeechobee | No | — |
| 47 | Orange | No | — |
| 48 | Osceola | Yes | October 21 declaration |
| 49 | Palm Beach | Yes | October 21 declaration |
| 50 | Pasco | No | — |
| 51 | Pinellas | No | — |
| 52 | Polk | No | — |
| 53 | Putnam | Yes | October 21 declaration |
| 54 | Santa Rosa | No | — |
| 55 | Sarasota | No | — |
| 56 | Seminole | Yes | October 21 declaration |
| 57 | St. Johns | Yes | October 21 declaration |
| 58 | St. Lucie | Yes | October 21 declaration |
| 59 | Sumter | No | — |
| 60 | Suwannee | No | — |
| 61 | Taylor | No | — |
| 62 | Union | No | — |
| 63 | Volusia | Yes | December 9 amendment |
| 64 | Wakulla | No | — |
| 65 | Walton | No | — |
| 66 | Washington | No | — |

Affected counties (14): Brevard, Broward, Duval, Flagler, Indian River, Martin, Nassau, Osceola, Palm Beach, Putnam, Seminole, St. Johns, St. Lucie, Volusia.

### Source roles and links

- [FEMA Dorian situation report, January 3, 2020, page 1](https://content.govdelivery.com/attachments/USDHSFEMA/2020/01/03/file_attachments/1352101/SITREP3-FL-4468-01032020.pdf): official incident dates, original 12-county PA A–G designation, and two-county amendment.
- [JMA Hagibis bulletin, October 15, 2019](https://www.jma.go.jp/jma/press/1910/15a/20191015_saimota.html): documented October 10–13 heavy rain and wind event; does not define our meteorological threshold or 22-region set.
- [JMA 2019 climate monitoring report, Table 1.3-1](https://www.data.jma.go.jp/cpdinfo/monitor/2019/pdf/ccmr2019_all.pdf): identities and official typhoon lifetimes for the named historical storms.
- [JMA Northern Kyushu heavy-rain report](https://www.data.jma.go.jp/stats/data/bosai/report/2019/20190826/jyun_sokuji20190826-0829.pdf): August 26–29 frontal heavy-rain event.
- [JMA historical disaster-event archive](https://www.data.jma.go.jp/stats/data/bosai/report/index_1989.html): additional event reports, including Faxai and October 2019 events.
- [NASA POWER hourly point API](https://power.larc.nasa.gov/api/temporal/hourly/point): meteorological data used to construct exposure scores.

The exact event-label intervals, evaluation intervals, window filters, exposure
formula, and node lists in this README define the study protocol; official
reports provide supporting event evidence rather than replacing these choices.

## Attribution

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[LICENSE_STATUS.md](LICENSE_STATUS.md) for source attribution and licensing status.
