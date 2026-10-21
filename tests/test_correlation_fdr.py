from scripts.failure_mining.correlation_fdr import analyze, benjamini_hochberg


def test_benjamini_hochberg_is_monotone_in_sorted_p_values() -> None:
    p_values = [0.04, 0.001, 0.03, 0.2]
    q_values = benjamini_hochberg(p_values)
    ordered = sorted(zip(p_values, q_values))
    assert [q for _p, q in ordered] == sorted(q for _p, q in ordered)
    assert all(p <= q <= 1 for p, q in zip(p_values, q_values, strict=True))


def test_analyze_corrects_each_endpoint_family_separately() -> None:
    evidence = [
        {"dataset": "set", "scene": f"s{i}", "image.signal": str(i), "image.constant": "1"}
        for i in range(6)
    ]
    targets = [
        {"dataset": "set", "scene": f"s{i}", "metric": metric, "target": str(i if metric == "a" else 5 - i)}
        for metric in ("a", "b")
        for i in range(6)
    ]
    rows = analyze(
        evidence,
        targets,
        evidence_prefix="image.",
        excluded_suffixes=(),
        target_value_field="target",
        target_metric_field="metric",
        datasets={"set"},
        metrics={"a", "b"},
    )
    assert len(rows) == 2
    assert {row["target_metric"] for row in rows} == {"a", "b"}
    assert {row["spearman_rho"] for row in rows} == {-1.0, 1.0}
    assert all(row["tests_in_endpoint_family"] == 1 for row in rows)


def test_analyze_accepts_an_explicit_cross_prefix_evidence_family() -> None:
    evidence = [
        {
            "dataset": "set",
            "scene": f"s{i}",
            "fusion.ratio": str(i),
            "diagnostic.error": str(5 - i),
            "ignored": str(i * 2),
        }
        for i in range(6)
    ]
    targets = [
        {"dataset": "set", "scene": f"s{i}", "metric": "gain", "target": str(i)}
        for i in range(6)
    ]
    rows = analyze(
        evidence,
        targets,
        evidence_prefix="",
        excluded_suffixes=(),
        target_value_field="target",
        target_metric_field="metric",
        datasets={"set"},
        metrics={"gain"},
        evidence_fields=["fusion.ratio", "diagnostic.error"],
    )
    assert [row["evidence_field"] for row in rows] == [
        "fusion.ratio",
        "diagnostic.error",
    ]
    assert {row["spearman_rho"] for row in rows} == {-1.0, 1.0}
    assert all(row["tests_in_endpoint_family"] == 2 for row in rows)
