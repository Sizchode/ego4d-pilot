"""Audit complete baseline scores without changing predictions or scoring rules."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from scripts.recovery.summarize_main_matrix import build_matrix, read


CAMERA_METRICS = (
    'relative_rotation_mean_deg',
    'relative_translation_direction_mean_deg',
    'camera_ate_rmse_m_using_point_alignment',
)


def audit(plan, specification):
    matrix = build_matrix(plan, specification)
    rows = []
    for column in specification['methods']:
        available = matrix['methods'][column['label']]
        unavailable = {r['repeat'] for r in available['unavailable']}
        for repeat, path in enumerate(column['reports']):
            if repeat in unavailable:
                continue
            cases = read(path)['cases']
            for dataset in matrix['dataset_sequence_counts']:
                group = [c for c in cases if c['dataset'] == dataset]
                scored = [c['result'] for c in group if c['state'] == 'scored']
                failures = sum(c['state'] == 'model_failed' for c in group)
                row = {
                    'method': column['label'], 'repeat': repeat, 'dataset': dataset,
                    'sequences': len(group), 'model_execution_failures': failures,
                    'alignment_failures': sum(c['alignment_failure'] is not None for c in scored),
                    'mean_missing_prediction_fraction': (
                        sum(c['missing_prediction_pixels'] / c['gt_valid_pixels'] for c in scored)
                        + failures) / len(group),
                }
                for metric in CAMERA_METRICS:
                    values = [c.get('cameras', {}).get(metric) for c in scored]
                    defined = [v for v in values if v is not None and math.isfinite(v)]
                    row[metric] = statistics.mean(defined) if defined else None
                    row[metric + '_defined_sequences'] = len(defined)
                rows.append(row)
    return {'complete': matrix['complete'], 'rows': rows,
            'camera_aggregation': 'Equal weight per sequence among defined values; each metric reports its denominator. Missing camera estimates are not zero error.',
            'point_aggregation': 'All planned sequences retained. Model execution failures count as fully missing predictions.',
            'sources': specification['methods']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--specification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(read(args.plan), read(args.specification))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'score-audit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    if result['rows']:
        with (args.output / 'camera-and-failure-audit.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result['rows'][0]))
            writer.writeheader()
            writer.writerows(result['rows'])


if __name__ == '__main__':
    main()
