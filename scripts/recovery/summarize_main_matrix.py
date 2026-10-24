"""Build the main matrix only from complete, identity-checked score populations."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def validate_report(report, cases, method):
    if report['method'] != method:
        raise ValueError('Unexpected model identity')
    if not report['completed']:
        return None
    rows = report['cases']
    expected = {r['case_id']: r for r in cases}
    if (len(expected) != len(cases) or len(rows) != len(expected)
            or report['expected_cases'] != len(expected)
            or len({r['case_id'] for r in rows}) != len(rows)
            or {r['case_id'] for r in rows} != set(expected)):
        raise ValueError('Complete score report does not cover the exact population')
    result = []
    for row in rows:
        case = expected[row['case_id']]
        if any(row[key] != case[key] for key in ['sequence_key', 'dataset']):
            raise ValueError('Score sequence identity differs from evaluation plan')
        if row['state'] == 'scored':
            values = row['result']
        elif row['state'] == 'model_failed':
            values = row
            if values['auc_5cm'] != 0 or values['auc_10cm'] != 0:
                raise ValueError('Model failure must remain a zero in the full denominator')
        else:
            raise ValueError('Unresolved evaluation error cannot be summarized')
        if (values['gt_valid_pixels'] <= 0 or any(not math.isfinite(values[k]) or not 0 <= values[k] <= 1
                                                for k in ['auc_5cm', 'auc_10cm'])):
            raise ValueError('Invalid score or GT domain')
        result.append({key: row[key] for key in ['case_id', 'sequence_key', 'dataset', 'state']} |
                      {key: values[key] for key in ['auc_5cm', 'auc_10cm', 'gt_valid_pixels']})
    return result


def build_matrix(plan, specification):
    cases = plan['cases']
    if plan['stage'] != 'baseline_evaluation':
        raise ValueError('Expected main baseline evaluation plan')
    datasets = sorted({r['dataset'] for r in cases})
    counts = {d: sum(r['dataset'] == d for r in cases) for d in datasets}
    output = {'complete': True, 'dataset_sequence_counts': counts, 'methods': {}, 'per_case': []}
    gt_domains = {}
    for column in specification['methods']:
        label = column['label']
        if label in output['methods'] or not column['reports']:
            raise ValueError('Duplicate column or empty repeat list')
        runs, unavailable = [], []
        for repeat, path in enumerate(column['reports']):
            if not Path(path).is_file():
                unavailable.append({'repeat': repeat, 'state': 'not_available'})
                continue
            rows = validate_report(read(path), cases, column['method'])
            if rows is None:
                unavailable.append({'repeat': repeat, 'state': 'incomplete'})
                continue
            for row in rows:
                previous = gt_domains.setdefault(row['case_id'], row['gt_valid_pixels'])
                if previous != row['gt_valid_pixels']:
                    raise ValueError('Methods were scored on different GT domains')
                output['per_case'].append({'method': label, 'repeat': repeat, **row})
            runs.append({d: {
                **{key: statistics.mean(r[key] for r in rows if r['dataset'] == d)
                   for key in ['auc_5cm', 'auc_10cm']},
                'model_failures': sum(r['state'] == 'model_failed' for r in rows if r['dataset'] == d)}
                         for d in datasets})
        entry = {'expected_repeats': len(column['reports']), 'completed_repeats': len(runs),
                 'sources': column['reports'], 'unavailable': unavailable, 'datasets': {}}
        output['complete'] &= not unavailable
        for d in datasets:
            if runs:
                metrics = {key: {'mean': statistics.mean(r[d][key] for r in runs),
                                 'std_across_runs': statistics.stdev(r[d][key] for r in runs) if len(runs) > 1 else None}
                           for key in ['auc_5cm', 'auc_10cm']}
                entry['datasets'][d] = {'sequences': counts[d], **metrics,
                                        'model_failures_per_run': [r[d]['model_failures'] for r in runs]}
        output['methods'][label] = entry
    return output


def save(summary, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output/'matrix.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    columns = list(summary['methods'])
    lines = ['# Main evaluation baseline matrix', '',
             '每个格子只使用该方法完整覆盖87个序列的评分。AUC5为1–5cm阈值内正确点比例的平均值，越高越好；表中显示百分数。',
             '数据来自GGPT发布的16-view子集；序列数不等于独立物理场景数。', '',
             '| 数据集（序列数） | '+' | '.join(columns)+' |',
             '|---|'+'---:|'*len(columns)]
    for dataset, count in summary['dataset_sequence_counts'].items():
        cells = []
        for col in summary['methods'].values():
            if dataset not in col['datasets']:
                cells.append('待评分')
            else:
                metric = col['datasets'][dataset]['auc_5cm']
                text = f"{metric['mean']*100:.2f}"
                if metric['std_across_runs'] is not None:
                    text += f" ± {metric['std_across_runs']*100:.2f}"
                if col['expected_repeats'] > 1:
                    text += f"（{col['completed_repeats']}/{col['expected_repeats']}次）"
                cells.append(text)
        lines.append(f'| {dataset} ({count}) | '+' | '.join(cells)+' |')
    lines += ['', 'AUC10、逐次执行失败数及源文件见matrix.json；逐序列分数见per-sequence.csv。',
              '本表使用预先固定的全图评分域与最小二乘Sim(3)对齐；GGPT原评估包含鲁棒对齐步骤，因此这些重跑数值不能直接与论文原表排名混用。',
              '±表示重复运行的样本标准差，不是场景置信区间。尚未完成的重复不会当作零分。', '']
    (output/'matrix.md').write_text('\n'.join(lines))
    if summary['per_case']:
        with (output/'per-sequence.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary['per_case'][0]))
            writer.writeheader(); writer.writerows(summary['per_case'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--specification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    save(build_matrix(read(args.plan), read(args.specification)), args.output)


if __name__ == '__main__':
    main()
