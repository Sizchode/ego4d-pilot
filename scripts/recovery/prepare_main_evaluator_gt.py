"""Stage published pinhole GT for the fixed main baseline population, evaluator only."""
import argparse
import json
from pathlib import Path

from stage_pinhole_gt import stage


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan['schema'] != 'rgb-population-plan-v1' or plan['stage'] != 'baseline_evaluation':
        raise ValueError('Expected fixed main baseline population')
    cases = plan['cases']
    if len(cases) != 87 or len({r['sequence_key'] for r in cases}) != len(cases):
        raise ValueError('Main evaluation inventory differs from the frozen 87 sequences')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'completed': False, 'prediction_access': False, 'scores_computed': False,
              'expected_cases': len(cases), 'cases': []}
    write(args.output/'preparation.json', report)
    for index, case in enumerate(cases):
        if case['case_id'] != f'{index:06d}':
            raise ValueError('Main case order changed')
        key = Path(case['sequence_key'])
        if key.is_absolute() or '..' in key.parts or not key.parts or key.parts[0] != 'ind_data':
            raise ValueError('Unexpected published sequence key')
        source = args.source_root/key
        if source.resolve().parent != (args.source_root/key.parent).resolve():
            raise ValueError('GT source path escaped the published inventory')
        out = args.output/'cases'/case['case_id']
        result = stage(source, Path(case['input'])/'input_manifest.json', out)
        if result['staged_views'] != case['views'] or result['gt_valid_pixels'] <= 0:
            raise ValueError('Invalid staged GT coverage')
        report['cases'].append({'case_id': case['case_id'], 'sequence_key': case['sequence_key'],
                                'dataset': case['dataset'], 'ground_truth': str(out), **result})
        write(args.output/'preparation.json', report)
        print(f'Staged evaluator GT {index + 1}/{len(cases)}', flush=True)
    report['completed'] = True
    write(args.output/'preparation.json', report)


if __name__ == '__main__':
    main()
