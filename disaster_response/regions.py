"""Auditable region selection independent of model predictions."""
import json
from pathlib import Path

import numpy as np

OFFICIAL_CONFIG = Path(__file__).resolve().parent / 'configs/dorian_affected_counties.json'


def select_regions(dataset, names, peak, threshold=1., us_rule='official'):
    if len(names) != len(set(names)):
        raise ValueError('Repeated node names')
    if us_rule not in ('official', 'proxy'):
        raise ValueError('Unknown US affected-region rule')
    if dataset == 'us' and us_rule == 'official':
        rule = json.loads(OFFICIAL_CONFIG.read_text(encoding='utf-8'))
        counties = rule['counties']
        if len(counties) != rule['county_count'] or len(set(counties)) != len(counties):
            raise ValueError('Invalid official county configuration')
        if set(counties) != set(rule['initial_counties']) | set(rule['added_counties']):
            raise ValueError('County list differs from initial declaration plus amendment')
        missing = set(counties) - set(names)
        if missing:
            raise ValueError(f'Official counties absent from the dataset: {sorted(missing)}')
        return np.array([name in counties for name in names]), dict(rule, method='official',
                                                                    config_file=str(OFFICIAL_CONFIG))
    peak = np.asarray(peak)
    if peak.shape != (len(names),) or not np.isfinite(peak).all():
        raise ValueError('Invalid per-node exposure peaks')
    return peak >= threshold, dict(method='proxy', feature='meteorological_hazard_proxy_v1',
                                   threshold=threshold, aggregation='maximum within fixed disaster interval',
                                   interpretation='retrospective exposure proxy, not confirmed damage',
                                   complement_interpretation='below exposure threshold, not proof of no impact')
