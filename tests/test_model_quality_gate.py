from __future__ import annotations

import math

import pytest

from scripts.model_quality_gate import (
    compare_artifacts,
    evaluate_artifact,
    validate_artifact,
    validate_dataset,
)


FEATURES = ["ram", "vram"]
X = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
Y = [1.0, 2.0, 3.0]


def artifact(value: float, *, feature_order=FEATURES, tree=None) -> dict:
    return {
        "model_version": 1,
        "feature_order": feature_order,
        "trees": [tree or {"leaf": True, "value": value}],
    }


def test_compare_rejects_missing_selection_evidence():
    baseline = artifact(2.0)
    candidate = artifact(0.0, tree={"feature": 0, "threshold": 1.5, "left": {"leaf": True, "value": 1.5}, "right": {"leaf": True, "value": 3.0}})

    report = compare_artifacts(candidate, baseline, X, Y, min_selection_groups=3)

    assert report["passed"] is False
    assert any("selection groups" in failure for failure in report["failures"])
    assert report["candidate"]["rmsle"] < report["baseline"]["rmsle"]


def test_compare_blocks_rmsle_regression():
    report = compare_artifacts(artifact(8.0), artifact(2.0), X, Y)

    assert report["passed"] is False
    assert any("rmsle" in failure for failure in report["failures"])


def test_compare_blocks_p90_ape_regression():
    report = compare_artifacts(
        artifact(2.5), artifact(2.0), X, Y, max_rmsle_regression=10.0
    )

    assert report["passed"] is False
    assert any("p90_absolute_percentage_error" in failure for failure in report["failures"])


def test_validate_dataset_rejects_insufficient_or_over_rejected_data():
    with pytest.raises(ValueError, match="too few"):
        validate_dataset({"raw_rows": 10, "rejected_rows": 0, "unique_configurations": 19, "direct_v5_unique_configurations": 19})
    with pytest.raises(ValueError, match="rejection"):
        validate_dataset({"raw_rows": 10, "rejected_rows": 3, "unique_configurations": 20, "direct_v5_unique_configurations": 20})
    with pytest.raises(ValueError, match="too few"):
        validate_dataset({"raw_rows": 0, "rejected_rows": 0, "unique_configurations": 0, "direct_v5_unique_configurations": 0})


@pytest.mark.parametrize(
    ("bad_y", "bad_artifact", "match"),
    [
        ([math.nan], None, "finite"),
        ([-1.0], None, "non-negative"),
        ([1.0], artifact(1.0, tree={"feature": 2, "threshold": 0.0, "left": {"leaf": True, "value": 1.0}, "right": {"leaf": True, "value": 1.0}}), "outside"),
    ],
)
def test_evaluation_rejects_invalid_data_or_artifact(bad_y, bad_artifact, match):
    with pytest.raises(ValueError, match=match):
        evaluate_artifact(bad_artifact or artifact(1.0), [[0.0, 0.0]], bad_y)


def test_validate_artifact_rejects_non_finite_threshold_and_leaf():
    with pytest.raises(ValueError, match="feature_order"):
        validate_artifact(artifact(1.0, feature_order=["vram", "ram"]), FEATURES)
    with pytest.raises(ValueError, match="threshold"):
        validate_artifact(
            artifact(1.0, tree={"feature": 0, "threshold": math.nan, "left": {"leaf": True, "value": 1.0}, "right": {"leaf": True, "value": 1.0}}),
            FEATURES,
        )
    with pytest.raises(ValueError, match="value"):
        validate_artifact(artifact(math.inf), FEATURES)


def test_evaluation_requires_exact_feature_vector_length():
    with pytest.raises(ValueError, match="expected 2"):
        evaluate_artifact(artifact(1.0), [[0.0]], [1.0])


def test_evaluation_uses_one_tok_per_second_floor_for_zero_target_ape():
    metrics = evaluate_artifact(artifact(4.0), [[0.0, 0.0]], [0.0])

    assert math.isfinite(metrics["p90_absolute_percentage_error"])
    assert metrics["p90_absolute_percentage_error"] == 4.0


def test_compare_requires_matching_feature_contracts():
    with pytest.raises(ValueError, match="feature_order must match"):
        compare_artifacts(artifact(1.0), artifact(1.0, feature_order=["ram"]), X, Y)


def test_dataset_rejection_rate_must_be_a_fraction():
    with pytest.raises(ValueError, match="at most 1"):
        validate_dataset(
            {"raw_rows": 20, "rejected_rows": 0, "unique_configurations": 20, "direct_v5_unique_configurations": 20},
            max_rejection_rate=1.1,
        )


SELECTION_FEATURES = [
    "quant_bits",
    "ram_gb",
    "param_count_b",
    "model_size_gb",
    "active_param_count_b",
]
SELECTION_X = [
    [4.0, 16.0, 3.0, 2.0, 3.0],
    [8.0, 16.0, 7.0, 5.0, 7.0],
]
SELECTION_Y = [0.0, 2.0]


def selection_artifact(*, reversed_prediction: bool = False) -> dict:
    # Uses model feature names, not positions: quant_bits is deliberately index 0.
    return artifact(
        0.0,
        feature_order=SELECTION_FEATURES,
        tree={
            "feature": 0,
            "threshold": 6.0,
            "left": {"leaf": True, "value": 3.0 if reversed_prediction else 0.0},
            "right": {"leaf": True, "value": 0.0 if reversed_prediction else 3.0},
        },
    )


def test_selection_and_fit_metrics_are_finite_and_use_feature_names():
    metrics = evaluate_artifact(selection_artifact(), SELECTION_X, SELECTION_Y)

    assert metrics["selection_group_count"] == 1
    assert metrics["top1_selection_accuracy"] == 1.0
    assert metrics["mean_normalized_regret"] == 0.0
    assert metrics["p90_normalized_regret"] == 0.0
    assert metrics["fit_balanced_accuracy"] == 1.0
    assert metrics["fit_false_positive_rate"] == 0.0
    assert all(math.isfinite(value) for value in metrics.values() if isinstance(value, float))


def test_selection_mistake_has_expected_normalized_regret():
    metrics = evaluate_artifact(selection_artifact(reversed_prediction=True), SELECTION_X, SELECTION_Y)

    assert metrics["top1_selection_accuracy"] == 0.0
    assert metrics["mean_normalized_regret"] == 1.0
    assert metrics["p90_normalized_regret"] == 1.0
    assert metrics["fit_balanced_accuracy"] == 0.0
    assert metrics["fit_false_positive_rate"] == 1.0


def test_selection_without_model_candidates_and_single_fit_class_is_null():
    metrics = evaluate_artifact(artifact(2.0), X, Y)

    assert metrics["selection_group_count"] == 0
    assert metrics["top1_selection_accuracy"] is None
    assert metrics["mean_normalized_regret"] is None
    assert metrics["p90_normalized_regret"] is None
    assert metrics["fit_balanced_accuracy"] is None
    assert metrics["fit_false_positive_rate"] is None


def test_selection_regression_blocks_and_missing_evidence_rejects():
    report = compare_artifacts(
        selection_artifact(reversed_prediction=True),
        selection_artifact(),
        SELECTION_X,
        SELECTION_Y,
        max_rmsle_regression=10.0,
        max_p90_ape_regression=10.0, min_selection_groups=1,
    )
    assert report["passed"] is False
    assert any("top1_selection_accuracy" in failure for failure in report["failures"])
    assert any("mean_normalized_regret" in failure for failure in report["failures"])

    no_evidence = compare_artifacts(artifact(2.0), artifact(2.0), X, Y)
    assert no_evidence["passed"] is False
    assert any("missing evidence" in failure for failure in no_evidence["failures"])
    assert no_evidence["thresholds"]["max_selection_metric_regression"] == 0.05


def test_compare_allows_sufficient_selection_evidence():
    X = [
        [quant, ram, parameters, parameters * 0.7, parameters]
        for ram in (16.0, 32.0, 64.0)
        for quant, parameters in ((4.0, 3.0), (8.0, 7.0))
    ]
    y = [0.0, 2.0] * 3

    report = compare_artifacts(selection_artifact(), selection_artifact(), X, y)

    assert report["passed"] is True
    assert report["failures"] == []
