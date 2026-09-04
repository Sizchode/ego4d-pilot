"""Run and archive an official EgoTempo/V-STaR evaluation command verbatim."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("egotempo", "vstar"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Use -- followed by the official evaluator command")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("an official evaluator command is required; this wrapper does not define benchmark metrics")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"dataset": args.dataset, "command": command, "started_at": datetime.now(timezone.utc).isoformat()}
    (output_dir / "command.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    (output_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    metadata.update({"finished_at": datetime.now(timezone.utc).isoformat(), "returncode": completed.returncode})
    (output_dir / "command.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command, completed.stdout, completed.stderr)


if __name__ == "__main__":
    main()
