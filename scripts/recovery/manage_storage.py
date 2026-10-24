"""Apply the explicit artifact retention policy and record the actual quota report."""
import argparse
from datetime import datetime, timezone
import getpass
import json
from pathlib import Path
import subprocess

from prune_matching_audits import prune as prune_matching
from prune_derived_exports import prune as prune_exports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text())
    root = Path(policy['root']).resolve()
    matching = [Path(p) for p in policy['matching_batches'] if Path(p).is_file()]
    exports = [Path(p) for p in policy['derived_export_batches'] if Path(p).is_file()]
    # Validate both policies before either mutates files.
    prune_matching(root, matching, input_captures=True)
    prune_exports(root, exports)
    results = [prune_matching(root, matching, args.apply, input_captures=True),
               prune_exports(root, exports, args.apply)]
    text = subprocess.check_output(['checkquota'], text=True, timeout=60)
    quota_rows = [line.split() for line in text.splitlines()
                  if line.split()[:2] == [getpass.getuser(), '/oscar/scratch']]
    if len(quota_rows) != 1:
        raise ValueError('Cannot identify the actual user scratch quota')
    row = quota_rows[0]
    quota = {'used_GB': float(row[2]), 'soft_limit_GB': float(row[4]),
             'hard_limit_GB': float(row[5]), 'state': row[9],
             'grace_remaining': ' '.join(row[10:])}
    quota['headroom_to_soft_limit_GB'] = quota['soft_limit_GB'] - quota['used_GB']
    quota['maximum_usage_GB'] = policy.get('maximum_usage_GB', quota['soft_limit_GB'])
    quota['headroom_to_usage_budget_GB'] = quota['maximum_usage_GB'] - quota['used_GB']
    quota['below_requested_headroom'] = quota['headroom_to_usage_budget_GB'] < policy['reserve_GB']
    record = {'checked_at': datetime.now(timezone.utc).isoformat(), 'apply': args.apply,
              'quota': quota, 'cleanup': results,
              'removed_logical_bytes_this_pass': sum(r['bytes'] for r in results),
              'note': 'Quota is the checkquota report; deleted logical bytes are not a substitute for actual released quota.'}
    destination = Path(policy['status_file'])
    if root not in destination.resolve().parents:
        raise ValueError('Status file must stay in the project')
    destination.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'checked_at': record['checked_at'], 'quota': quota,
                      'removed_logical_GiB': record['removed_logical_bytes_this_pass'] / 1024**3}))


if __name__ == '__main__':
    main()
