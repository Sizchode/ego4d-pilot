"""Plot ordered conditional F-score gains from GT geometry interventions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FACTORS = (
    (
        "extrinsics",
        "Extrinsics",
        "extrinsics_mean_conditional_gain",
        "extrinsics_unique_best_positive_cases",
        "#4C78A8",
    ),
    (
        "intrinsics_after_extrinsics",
        "Intrinsics\nafter extrinsics",
        "intrinsics_after_extrinsics_mean_conditional_gain",
        "intrinsics_after_extrinsics_unique_best_positive_cases",
        "#F58518",
    ),
    (
        "depth_after_extrinsics_intrinsics",
        "Depth after cameras",
        "depth_after_extrinsics_intrinsics_mean_conditional_gain",
        "depth_after_extrinsics_intrinsics_unique_best_positive_cases",
        "#54A24B",
    ),
)


def read_fscore_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["metric"] == "fscore"]
    if not rows:
        raise ValueError("No F-score rows")
    if len({row["model"] for row in rows}) != len(rows):
        raise ValueError("Duplicate F-score model rows")
    return rows


def plot(rows: list[dict[str, str]], output: Path) -> None:
    models = [row["model"] for row in rows]
    x = np.arange(len(models))
    width = 0.24
    fig, (gain_axis, count_axis) = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.4),
        gridspec_kw={"width_ratios": [1.35, 1]},
        constrained_layout=True,
    )

    for index, (_key, label, mean_field, count_field, color) in enumerate(FACTORS):
        offset = (index - 1) * width
        gains = [float(row[mean_field]) for row in rows]
        counts = [int(row[count_field]) for row in rows]
        gain_axis.bar(x + offset, gains, width, label=label.replace("\n", " "), color=color)
        count_axis.bar(x + offset, counts, width, label=label.replace("\n", " "), color=color)

    gain_axis.axhline(0, color="#333333", linewidth=0.8)
    gain_axis.set_ylabel("Mean conditional F-score gain")
    gain_axis.set_title("Ordered GT intervention gain")
    gain_axis.set_xticks(x, [model.upper() if model != "mapanything" else "MapAnything" for model in models])
    gain_axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    gain_axis.set_axisbelow(True)

    count_axis.set_ylabel("Scenes where factor is unique best")
    count_axis.set_title("Largest positive conditional gain")
    count_axis.set_xticks(x, [model.upper() if model != "mapanything" else "MapAnything" for model in models])
    count_axis.set_ylim(0, max(int(row["cases"]) for row in rows) + 1)
    count_axis.set_yticks(range(0, max(int(row["cases"]) for row in rows) + 1, 5))
    count_axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    count_axis.set_axisbelow(True)

    handles, labels = gain_axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(
        "ScanNet++: nested oracle factors on 20 scenes\n"
        "Order: native → GT extrinsics → +GT intrinsics → +GT depth",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Refusing to overwrite output: {args.output}")
    plot(read_fscore_rows(args.input), args.output)


if __name__ == "__main__":
    main()
