"""Offline paired evaluation audit; never changes predictions or trains a model.

Calls the official GGPT alignment/metric functions from their installed source.
Only those unchanged function definitions are loaded, avoiding visualization
imports unrelated to scoring. Missing predictions remain failures in the full
GT denominator; official scores on the available domain are reported separately.
"""
import argparse
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import pycolmap


def official_functions(repository):
    path = Path(repository) / 'evaluation.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    names = {'umeyama_alignment', 'rmse', 'eval_points'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError('Official evaluation function inventory changed')
    namespace = {'np': np, 'torch': torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def metrics(errors, domain):
    selected = errors[domain]
    fractions = [(selected < cm / 100).mean().item() for cm in range(1, 11)]
    finite = selected[np.isfinite(selected)]
    return {'auc_5cm': float(np.mean(fractions[:5])), 'auc_10cm': float(np.mean(fractions)),
            'fractions_below_threshold': fractions,
            'finite_error_quantiles_m': np.quantile(finite, [.1, .5, .9, .99]).tolist() if len(finite) else None,
            'gt_pixels': int(domain.sum()), 'missing_pixels': int((~np.isfinite(selected)).sum())}


def robust_score(p, pv, g, gv, official, seed):
    usable = gv & pv & np.isfinite(p).all(-1)
    if not np.isfinite(g[gv]).all() or usable.sum() < 3:
        raise ValueError('Invalid GT or insufficient prediction correspondences')
    # Compact tensors retain every available correspondence, in original order.
    a, b = torch.from_numpy(p[usable].astype(np.float32)), torch.from_numpy(g[usable].astype(np.float32))
    trace = []
    robust_original, plain_original = pycolmap.estimate_sim3d_robust, pycolmap.estimate_sim3d
    def record_robust(*args, **kwargs):
        start = time.monotonic()
        value = robust_original(*args, **kwargs)
        trace.append({'kind': 'robust', 'options': kwargs['estimation_options'].todict(),
                      'success': value is not None, 'seconds': time.monotonic() - start})
        return value
    def record_plain(*args, **kwargs):
        value = plain_original(*args, **kwargs)
        trace.append({'kind': 'official_least_squares_fallback', 'success': value is not None})
        return value
    pycolmap.set_random_seed(seed)
    pycolmap.estimate_sim3d_robust, pycolmap.estimate_sim3d = record_robust, record_plain
    try:
        aligned, transform = official['umeyama_alignment'](B=b, A=a, max_error=.03)
    finally:
        pycolmap.estimate_sim3d_robust, pycolmap.estimate_sim3d = robust_original, plain_original
    errors = np.full(gv.shape, np.inf, np.float64)
    errors[usable] = torch.linalg.vector_norm(aligned - b, dim=-1).numpy()
    summary = metrics(errors, gv)
    # Check independent AUC computation against the author's unchanged metric.
    author = official['rmse'](b, aligned)
    for cm in (5, 10):
        expected = round(metrics(errors, usable)[f'auc_{cm}cm'] * 100, 2)
        observed = author[f'points/auc@{cm:02d}cm(%)']
        if abs(expected - observed) > .011:
            raise AssertionError(f'Official AUC parity failed: {expected} != {observed}')
    summary.update(transform=transform.numpy().tolist(), alignment_calls=trace,
                   available_domain_official_auc5_percent=author['points/auc@05cm(%)'],
                   available_domain_official_auc10_percent=author['points/auc@10cm(%)'])
    return summary


def summarize_benchmark(directory, inventory_path, dataset, methods):
    """只汇总完整的同一协议结果，禁止把漏跑序列从分母中删除。"""
    import csv
    directory = Path(directory)
    inventory = json.loads(Path(inventory_path).read_text())
    population = {row['case_id'] for row in inventory if row['dataset'] == dataset}
    if not population or not methods or len(methods) != len(set(methods)):
        raise ValueError('Expected a nonempty benchmark and unique methods')
    summaries = []
    for method in methods:
        expected = {row['case_id']: row['sequence_key'] for row in inventory
                    if row['dataset'] == dataset and row['method'] == method}
        if set(expected) != population:
            raise ValueError('Method inventory does not cover the full population')
        output = directory / method
        complete = json.loads((output / 'complete.json').read_text())
        if complete != {'completed': len(population), 'population': len(population)}:
            raise ValueError('Scoring is incomplete; do not publish a partial average')
        files = set(output.glob('*.json')) - {output / 'configuration.json', output / 'complete.json'}
        if {path.stem for path in files} != population:
            raise ValueError('Score files differ from the fixed benchmark population')
        results = [json.loads(path.read_text()) for path in sorted(files)]
        if {row['case_id'] for row in results} != population:
            raise ValueError('Result IDs do not cover the benchmark')
        for row in results:
            if (row['method'] != method or row['dataset'] != dataset
                    or row['sequence_key'] != expected[row['case_id']]
                    or row['canvas'] != 'official518'):
                raise ValueError('Result identity or scoring canvas changed')
            score = row['official_robust']
            if not all(np.isfinite(score[key]) and 0 <= score[key] <= 1
                       for key in ('auc_5cm', 'auc_10cm')):
                raise ValueError('Invalid AUC; do not silently skip a failed sequence')
        summaries.append(dict(dataset=dataset, method=method, sequences=len(population),
            auc_5cm=float(np.mean([r['official_robust']['auc_5cm'] for r in results])),
            auc_10cm=float(np.mean([r['official_robust']['auc_10cm'] for r in results])),
            gt_pixels=sum(r['official_robust']['gt_pixels'] for r in results),
            missing_pixels=sum(r['official_robust']['missing_pixels'] for r in results),
            scoring_seconds=sum(r['seconds'] for r in results)))
    with (directory / 'benchmark-summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--ggpt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--dataset', default='tandt')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--canvas', choices=['public504','official518'], default='public504')
    parser.add_argument('--root', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.runner / 'scripts/recovery'))
    from score_archives import read_views
    from pointmap_score import score_pointmaps
    official = official_functions(args.ggpt)
    inventory = json.loads(args.inventory.read_text())
    population = {r['case_id'] for r in inventory if r['dataset'] == args.dataset}
    rows = [r for r in inventory if r['dataset'] == args.dataset and r['method'] == args.method]
    if not population or len(rows) != len(population) or {r['case_id'] for r in rows} != population:
        raise ValueError('Method does not cover the complete fixed benchmark population')
    rows.sort(key=lambda r:r['case_id'])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'configuration.json').write_text(json.dumps({**vars(args), 'torch': torch.__version__,
        'pycolmap': pycolmap.__version__, 'scope':'scoring only; unchanged authoritative predictions',
        'missing_policy':'failure in full GT denominator; separate available-domain official score'},default=str,indent=2))
    for row in rows[:args.limit]:
        started = time.monotonic()
        print('START', args.method, row['case_id'], flush=True)
        previous = row['result']
        pred, pm = read_views(previous['prediction_archive'], 'public-world-point-map-v1')
        gt, gm = read_views(previous['gt_archive'], 'public-ground-truth-v1')
        if pm != gm:
            raise ValueError('Prediction and GT image ordering differ')
        flatten = lambda xs, k: np.concatenate([x[k].reshape(-1,3) if k=='points' else x[k].reshape(-1) for x in xs])
        p,pv,g,gv = flatten(pred,'points'),flatten(pred,'valid'),flatten(gt,'points'),flatten(gt,'valid')
        old_ls,_,_ = score_pointmaps(p,pv,g,gv)
        if abs(old_ls['auc_5cm']-previous['auc_5cm']) > 1e-10 or old_ls['gt_valid_pixels']!=previous['gt_valid_pixels']:
            raise AssertionError('Existing score cannot be reproduced from retained raw results')
        print('LS_REPRODUCED',old_ls['auc_5cm'], flush=True)
        canvas_audit = None
        if args.canvas == 'official518':
            if args.root is None:
                raise ValueError('Explicit dataset root required for official GT canvas')
            from ggpt_evaluation_canvas import read_official_canvas
            p,pv,g,gv,canvas_audit = read_official_canvas(row,args.root,args.ggpt)
        ls,_,_ = score_pointmaps(p,pv,g,gv)
        robust = robust_score(p,pv,g,gv,official,args.seed)
        result = {'method':args.method,'dataset':args.dataset,'case_id':row['case_id'],'sequence_key':row['sequence_key'],
                  'old_public504_least_squares':old_ls,'canvas':args.canvas,'canvas_audit':canvas_audit,
                  'least_squares':ls,'official_robust':robust,
                  'seconds':time.monotonic()-started}
        (args.output/f"{row['case_id']}.json").write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print('DONE', row['case_id'], 'LS', ls['auc_5cm'], 'ROBUST',robust['auc_5cm'], 'seconds',result['seconds'],flush=True)
    (args.output/'complete.json').write_text(json.dumps({'completed':len(rows[:args.limit]),'population':len(population)}))


if __name__ == '__main__':
    main()
