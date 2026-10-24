"""Check official PGSR rendered-view scores against the heldout camera manifest."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image


def verify(dataset, model):
    dataset, model = Path(dataset), Path(model)
    meta = json.loads((dataset/'transforms_test.json').read_text())
    expected = {Path(f['file_path']).stem+'.png' for f in meta['frames']}
    if len(expected) != len(meta['frames']) or not expected:
        raise ValueError('Empty or duplicate heldout image names')
    summary = json.loads((model/'results.json').read_text())
    scores = json.loads((model/'per_view.json').read_text())
    if set(summary) != set(scores) or not summary:
        raise ValueError('Missing or mismatched PGSR metric iterations')
    report = {}
    for iteration, aggregate in summary.items():
        rendered = model/'test'/iteration
        for kind in ['renders', 'gt']:
            paths = {p.name:p for p in (rendered/kind).glob('*.png')}
            if set(paths) != expected:
                raise ValueError(f'{kind} does not contain exactly the heldout views')
            for path in paths.values():
                with Image.open(path) as image:
                    image.load()
                    if image.size != (int(meta['w']), int(meta['h'])):
                        raise ValueError('Scored image resolution differs from camera manifest')
        # Verify PGSR's scoring targets are the actual heldout input pixels.
        for frame in meta['frames']:
            name=Path(frame['file_path']).stem+'.png'
            source=dataset/(frame['file_path']+'.png')
            target=rendered/'gt'/name
            with Image.open(source) as a, Image.open(target) as b:
                if not np.array_equal(np.asarray(a.convert('RGB')),np.asarray(b.convert('RGB'))):
                    raise ValueError('Scored target differs from heldout RGB')
        for metric in ['PSNR','SSIM','LPIPS']:
            values=scores[iteration][metric]
            if set(values)!=expected or not all(math.isfinite(v) for v in values.values()):
                raise ValueError(f'{metric} missing/nonfinite heldout scores')
            mean=sum(values.values())/len(values)
            if not math.isclose(mean,aggregate[metric],rel_tol=1e-5,abs_tol=1e-6):
                raise ValueError(f'{metric} aggregate does not match per-view scores')
        report[iteration]={'views':len(expected),'metrics':aggregate}
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    args=p.parse_args()
    report=verify(args.dataset,args.model)
    (args.model/'score-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
