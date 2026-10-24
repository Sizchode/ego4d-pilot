"""Run released AREA3D with an observational record of actual local VLM calls.

The model, prompts, parsing, uncertainty and selection algorithms are unchanged.
The upstream uniform-weight fallback is not accepted as a successful VLM run.
"""
import argparse
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    args, upstream_args = parser.parse_known_args()
    sys.path.insert(0, str(args.source.resolve()))
    import active_recon
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    records = []
    def emit(row):
        with args.trace.open('a') as f:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
            f.flush()
    if args.trace.exists():
        raise FileExistsError('Refusing to mix separate VLM runs in one trace')
    class AuditedEvaluator(active_recon.InternVLEvaluator):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            original = self.model.chat
            def chat(*a, **kw):
                row = {'call_index':len(records), 'started_at':time.time()}
                records.append(row)
                try:
                    result = original(*a, **kw)
                    row.update({'success':True,'response':result})
                    return result
                except Exception as exc:
                    row.update({'success':False,'error_type':type(exc).__name__,'error':str(exc)})
                    raise
                finally:
                    row['elapsed_seconds'] = time.time()-row['started_at']
                    emit(row)
            self.model.chat = chat
        def _parse_regions(self, response):
            result = super()._parse_regions(response)
            emit({'parsed_regions':result,'call_index':len(records)-1})
            return result
    active_recon.InternVLEvaluator = AuditedEvaluator
    sys.argv = [str(args.source/'active_recon.py'), *upstream_args]
    active_recon.main()
    if not records or any(not r['success'] for r in records):
        raise RuntimeError('AREA3D completed with missing/failed VLM calls; uniform fallback is not accepted')
    emit({'run_complete':True,'successful_chat_calls':len(records)})


if __name__ == '__main__':
    main()
