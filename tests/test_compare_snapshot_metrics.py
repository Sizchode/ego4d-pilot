from scripts.failure_mining.compare_snapshot_metrics import compare


def test_compare_snapshot_uses_only_common_scene_population() -> None:
    models = ["a", "b"]
    rows = []
    values = {"a": [0.2, 0.8], "b": [0.4, 0.7]}
    for model in models:
        for index, scene in enumerate(("one", "two")):
            rows.append(
                {
                    "model": model,
                    "dataset": "set",
                    "scene": scene,
                    "metrics": {"pose": {"auc30": values[model][index]}},
                }
            )
    rows.append(
        {"model": "a", "dataset": "set", "scene": "three", "metrics": {"pose": {"auc30": 0.1}}}
    )
    output = compare(rows, models)
    assert output["common_populations"] == [{"dataset": "set", "common_scenes": 2}]
    assert len(output["model_metric_rank_rows"]) == 4
    assert len(output["pairwise_rank_agreement"]) == 1
    assert len(output["oracle_scene_headroom"]) == 2
    assert len(output["oracle_summary"]) == 1


def test_higher_is_worse_metric_assigns_worst_rank_to_largest_value() -> None:
    values = {"a": (("good", 0.1), ("bad", 0.9)), "b": (("good", 0.3), ("bad", 0.8))}
    rows = [
        {"model": model, "dataset": "set", "scene": scene, "metrics": {"recon_unposed": {"overall": value}}}
        for model in ("a", "b")
        for scene, value in values[model]
    ]
    output = compare(rows, ["a", "b"])
    ranks = {(row["model"], row["scene"]): row["failure_rank"] for row in output["model_metric_rank_rows"]}
    assert ranks[("a", "bad")] == 1
    assert ranks[("a", "good")] == 2
