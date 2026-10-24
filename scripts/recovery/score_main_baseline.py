"""Score one fixed main-evaluation baseline against evaluator-only GT."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


FAILED = {'execution_failed', 'upstream_prediction_failed', 'output_validation_failed'}
SUCCESS = {'completed', 'reused_verified_prediction'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--gt-preparation', type=Path, required=True)
    parser.add_argument('--expected-method', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan, batch, gt = read(args.plan), read(args.batch), read(args.gt_preparation)
    cases, predictions, truths = plan['cases'], batch['cases'], gt['cases']
    if plan['schema'] != 'rgb-population-plan-v1' or plan['stage'] != 'baseline_evaluation':
        raise ValueError('Scoring needs the frozen baseline evaluation plan')
    if (batch['method'] != args.expected_method or not batch['completed']
            or not gt['completed'] or gt['prediction_access'] or gt['scores_computed']
            or len(cases) != len(predictions) or len(cases) != len(truths) or len(cases) != 87):
        raise ValueError('Prediction/GT population or method differs from main evaluation')
    report = {'completed': False, 'method': batch['method'], 'expected_cases': len(cases),
              'cases': [], 'prediction_batch': str(args.batch), 'GT_preparation': str(args.gt_preparation)}
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'batch.json', report)
    for case, prediction, truth in zip(cases, predictions, truths):
        identifier = prediction.get('case_id', prediction.get('scene'))
        if (identifier != case['case_id'] or truth['case_id'] != identifier
                or truth['sequence_key'] != case['sequence_key'] or truth['dataset'] != case['dataset']
                or Path(prediction['input']).resolve() != Path(case['input']).resolve()
                or prediction['views'] != case['views']):
            raise ValueError('Score inputs do not match the exact frozen case order and identity')
        record = {'case_id': identifier, 'sequence_key': case['sequence_key'],
                  'dataset': case['dataset'], 'prediction_state': prediction['state']}
        if prediction['state'] in FAILED:
            record.update(state='model_failed', auc_5cm=0.0, auc_10cm=0.0,
                          gt_valid_pixels=truth['gt_valid_pixels'])
        elif prediction['state'] in SUCCESS:
            public = Path(prediction['public_prediction'])
            if read(public/'input_manifest.json') != read(Path(case['input'])/'input_manifest.json'):
                raise ValueError('Prediction input identity changed')
            scored = args.output/'cases'/identifier
            log = args.output/(identifier+'.log')
            with log.open('w') as stream:
                subprocess.run([sys.executable, str(Path(__file__).with_name('score_archives.py')),
                                '--prediction', str(public), '--ground-truth', truth['ground_truth'],
                                '--output', str(scored)], check=True, stdout=stream,
                               stderr=subprocess.STDOUT)
            result = read(scored/'result.json')
            record.update(state='scored', prediction=str(public), result=result)
        else:
            raise ValueError('Unresolved prediction state')
        report['cases'].append(record)
        write(args.output/'batch.json', report)
        print(f"Scored {batch['method']} {len(report['cases'])}/{len(cases)}", flush=True)
    report['completed'] = True
    report['model_failed_cases'] = sum(r['state'] == 'model_failed' for r in report['cases'])
    for metric in ['auc_5cm', 'auc_10cm']:
        report['mean_'+metric] = sum((r['result'][metric] if r['state'] == 'scored' else r[metric])
                                      for r in report['cases'])/len(cases)
        report['by_dataset_'+metric] = {
            dataset: sum((r['result'][metric] if r['state'] == 'scored' else r[metric])
                         for r in report['cases'] if r['dataset'] == dataset)
            /sum(r['dataset'] == dataset for r in report['cases'])
            for dataset in sorted({r['dataset'] for r in report['cases']})}
    write(args.output/'batch.json', report)


if __name__ == '__main__':
    main()
