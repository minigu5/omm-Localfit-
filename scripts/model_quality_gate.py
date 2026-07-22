"""Dependency-light quality checks for serialized Localfit tree artifacts."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

from omm.mltree import predict_ensemble


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_node(node: Any, feature_count: int, path: str) -> None:
    if not isinstance(node, dict):
        raise ValueError(f"{path} must be an object")
    if node.get("leaf") is True:
        _finite_number(node.get("value"), f"{path}.value")
        return

    feature = node.get("feature")
    if isinstance(feature, bool) or not isinstance(feature, int):
        raise ValueError(f"{path}.feature must be an integer")
    if not 0 <= feature < feature_count:
        raise ValueError(f"{path}.feature is outside feature_order")
    _finite_number(node.get("threshold"), f"{path}.threshold")
    if "left" not in node or "right" not in node:
        raise ValueError(f"{path} must contain left and right children")
    _validate_node(node["left"], feature_count, f"{path}.left")
    _validate_node(node["right"], feature_count, f"{path}.right")


def validate_artifact(artifact: dict, expected_feature_order: list[str]) -> None:
    """Reject artifacts that cannot safely be evaluated by ``predict_ensemble``."""
    if not isinstance(artifact, dict):
        raise ValueError("artifact must be an object")
    version = artifact.get("model_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("model_version must be an integer >= 1")
    if artifact.get("feature_order") != expected_feature_order:
        raise ValueError("artifact feature_order does not match the expected order")
    if not isinstance(expected_feature_order, list) or not expected_feature_order:
        raise ValueError("expected_feature_order must be a non-empty list")
    trees = artifact.get("trees")
    if not isinstance(trees, list) or not trees:
        raise ValueError("artifact must contain non-empty trees")
    for index, tree in enumerate(trees):
        _validate_node(tree, len(expected_feature_order), f"trees[{index}]")


def _validate_examples(
    X: list[list[float]], y: list[float], *, feature_count: int
) -> None:
    if not isinstance(X, list) or not isinstance(y, list) or not X:
        raise ValueError("X and y must be non-empty lists")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")
    for row_index, (features, actual) in enumerate(zip(X, y)):
        if not isinstance(features, list):
            raise ValueError(f"X[{row_index}] must be a list")
        if len(features) != feature_count:
            raise ValueError(
                f"X[{row_index}] has {len(features)} features; expected {feature_count}"
            )
        for feature_index, value in enumerate(features):
            _finite_number(value, f"X[{row_index}][{feature_index}]")
        actual_value = _finite_number(actual, f"y[{row_index}]")
        if actual_value < 0:
            raise ValueError(f"y[{row_index}] must be non-negative")


def _percentile_90(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(0.9 * len(ordered)) - 1]


_MODEL_FEATURES = {
    "param_count_b",
    "quant_bits",
    "model_size_gb",
    "active_param_count_b",
}
_RUNTIME_TUNING_FEATURES = {"gpu_offload_ratio", "cpu_threads", "num_batch"}


def selection_context_key(feature_order: list[str], features: list[float]) -> tuple[float, ...]:
    """Return the hardware/request context shared by competing model choices."""
    excluded = _MODEL_FEATURES | _RUNTIME_TUNING_FEATURES
    return tuple(
        value for name, value in zip(feature_order, features) if name not in excluded
    )


def _selection_and_fit_metrics(
    feature_order: list[str],
    X: list[list[float]],
    y: list[float],
    predictions: list[float],
) -> dict:
    """Compute recommendation and fit metrics without assuming feature positions."""
    model_indices = {
        index for index, name in enumerate(feature_order) if name in _MODEL_FEATURES
    }
    selection_groups: dict[tuple[float, ...], list[int]] = {}
    # Selection is meaningful only with the complete, model-varying contract.
    if len(model_indices) == len(_MODEL_FEATURES):
        for index, features in enumerate(X):
            group_key = selection_context_key(feature_order, features)
            selection_groups.setdefault(group_key, []).append(index)

    selection_accuracy: list[float] = []
    normalized_regrets: list[float] = []
    for row_indices in selection_groups.values():
        signatures = {
            tuple(X[row_index][feature_index] for feature_index in sorted(model_indices))
            for row_index in row_indices
        }
        actual_best = max(y[row_index] for row_index in row_indices)
        if len(signatures) < 2 or actual_best < 1.0:
            continue
        predicted_best = max(predictions[row_index] for row_index in row_indices)
        predicted_top = [
            row_index for row_index in row_indices
            if predictions[row_index] == predicted_best
        ]
        actual_top = {
            row_index for row_index in row_indices if y[row_index] == actual_best
        }
        selection_accuracy.append(
            sum(row_index in actual_top for row_index in predicted_top) / len(predicted_top)
        )
        selected_actual = sum(y[row_index] for row_index in predicted_top) / len(predicted_top)
        normalized_regrets.append(
            (actual_best - selected_actual) / max(actual_best, 1.0)
        )

    positives = negatives = true_positives = true_negatives = false_positives = 0
    for actual, prediction in zip(y, predictions):
        actual_fit = actual >= 1.0
        predicted_fit = prediction >= 1.0
        if actual_fit:
            positives += 1
            true_positives += int(predicted_fit)
        else:
            negatives += 1
            true_negatives += int(not predicted_fit)
            false_positives += int(predicted_fit)

    balanced_accuracy = None
    if positives and negatives:
        balanced_accuracy = ((true_positives / positives) + (true_negatives / negatives)) / 2
    false_positive_rate = false_positives / negatives if negatives else None
    if not selection_accuracy:
        return {
            "selection_group_count": 0,
            "top1_selection_accuracy": None,
            "mean_normalized_regret": None,
            "p90_normalized_regret": None,
            "fit_balanced_accuracy": balanced_accuracy,
            "fit_false_positive_rate": false_positive_rate,
        }
    return {
        "selection_group_count": len(selection_accuracy),
        "top1_selection_accuracy": sum(selection_accuracy) / len(selection_accuracy),
        "mean_normalized_regret": sum(normalized_regrets) / len(normalized_regrets),
        "p90_normalized_regret": _percentile_90(normalized_regrets),
        "fit_balanced_accuracy": balanced_accuracy,
        "fit_false_positive_rate": false_positive_rate,
    }


def evaluate_artifact(artifact: dict, X: list[list[float]], y: list[float]) -> dict:
    """Evaluate a JSON tree ensemble on held-out, non-negative targets."""
    feature_order = artifact.get("feature_order") if isinstance(artifact, dict) else None
    validate_artifact(artifact, feature_order)
    _validate_examples(X, y, feature_count=len(feature_order))

    absolute_errors: list[float] = []
    squared_log_errors: list[float] = []
    absolute_percentage_errors: list[float] = []
    predictions: list[float] = []
    for row_index, (features, actual) in enumerate(zip(X, y)):
        prediction = _finite_number(
            predict_ensemble(artifact["trees"], features), f"prediction[{row_index}]"
        )
        if prediction < 0:
            raise ValueError(f"prediction[{row_index}] must be non-negative")
        predictions.append(prediction)
        actual_value = float(actual)
        absolute_errors.append(abs(prediction - actual_value))
        squared_log_errors.append((math.log1p(prediction) - math.log1p(actual_value)) ** 2)
        # Zero throughput is valid.  Floor at 1 tok/s: this keeps APE finite
        # without letting a diagnostic zero target dominate every percentile.
        absolute_percentage_errors.append(abs(prediction - actual_value) / max(actual_value, 1.0))

    rows = len(y)
    metrics = {
        "rows": rows,
        "mae": sum(absolute_errors) / rows,
        "rmsle": math.sqrt(sum(squared_log_errors) / rows),
        "p90_absolute_percentage_error": _percentile_90(absolute_percentage_errors),
    }
    metrics.update(_selection_and_fit_metrics(feature_order, X, y, predictions))
    return metrics


def _threshold(value: Any, name: str) -> float:
    threshold = _finite_number(value, name)
    if threshold < 0:
        raise ValueError(f"{name} must be non-negative")
    return threshold


def compare_artifacts(
    candidate: dict,
    baseline: dict,
    X: list[list[float]],
    y: list[float],
    *,
    max_rmsle_regression: float = 0.02,
    max_p90_ape_regression: float = 0.05,
    max_selection_metric_regression: float = 0.05,
    min_selection_groups: int = 3,
) -> dict:
    """Return a JSON-safe regression-gate report for two model artifacts."""
    rmsle_threshold = _threshold(max_rmsle_regression, "max_rmsle_regression")
    p90_threshold = _threshold(max_p90_ape_regression, "max_p90_ape_regression")
    selection_threshold = _threshold(
        max_selection_metric_regression, "max_selection_metric_regression"
    )
    if isinstance(min_selection_groups, bool) or not isinstance(min_selection_groups, int):
        raise ValueError("min_selection_groups must be an integer")
    if min_selection_groups < 1:
        raise ValueError("min_selection_groups must be at least 1")
    candidate_order = candidate.get("feature_order") if isinstance(candidate, dict) else None
    baseline_order = baseline.get("feature_order") if isinstance(baseline, dict) else None
    if candidate_order != baseline_order:
        raise ValueError("candidate and baseline feature_order must match")
    candidate_metrics = evaluate_artifact(candidate, X, y)
    baseline_metrics = evaluate_artifact(baseline, X, y)
    limits = {
        "rmsle": rmsle_threshold,
        "p90_absolute_percentage_error": p90_threshold,
    }
    failures = []
    if candidate_metrics["selection_group_count"] < min_selection_groups:
        failures.append(
            "insufficient selection groups: "
            f"candidate={candidate_metrics['selection_group_count']}, required={min_selection_groups}"
        )
    for metric, threshold in limits.items():
        baseline_value = baseline_metrics[metric]
        candidate_value = candidate_metrics[metric]
        allowed = 0.0 if baseline_value == 0.0 else baseline_value * (1.0 + threshold)
        if candidate_value > allowed:
            failures.append(
                f"{metric} regressed: candidate={candidate_value}, allowed={allowed}"
            )
    for metric, higher_is_better in (
        ("top1_selection_accuracy", True),
        ("fit_balanced_accuracy", True),
        ("mean_normalized_regret", False),
        ("p90_normalized_regret", False),
        ("fit_false_positive_rate", False),
    ):
        candidate_value = candidate_metrics[metric]
        baseline_value = baseline_metrics[metric]
        if candidate_value is None or baseline_value is None:
            failures.append(f"{metric} is missing evidence")
            continue
        if (higher_is_better and candidate_value < baseline_value - selection_threshold) or (
            not higher_is_better and candidate_value > baseline_value + selection_threshold
        ):
            failures.append(
                f"{metric} regressed: candidate={candidate_value}, baseline={baseline_value}, "
                f"tolerance={selection_threshold}"
            )
    return {
        "passed": not failures,
        "failures": failures,
        "candidate": candidate_metrics,
        "baseline": baseline_metrics,
        "thresholds": {
            "max_rmsle_regression": rmsle_threshold,
            "max_p90_ape_regression": p90_threshold,
            "max_selection_metric_regression": selection_threshold,
            "min_selection_groups": min_selection_groups,
        },
    }


def validate_dataset(
    audit: dict,
    *,
    min_unique_configurations: int = 20,
    max_rejection_rate: float = 0.25,
) -> None:
    """Require enough direct-v5 telemetry configurations with bounded rejection."""
    if not isinstance(audit, dict):
        raise ValueError("audit must be an object")
    if isinstance(min_unique_configurations, bool) or not isinstance(min_unique_configurations, int):
        raise ValueError("min_unique_configurations must be an integer")
    if min_unique_configurations < 0:
        raise ValueError("min_unique_configurations must be non-negative")
    maximum = _threshold(max_rejection_rate, "max_rejection_rate")
    if maximum > 1.0:
        raise ValueError("max_rejection_rate must be at most 1")
    raw_rows = audit.get("raw_rows")
    rejected_rows = audit.get("rejected_rows")
    unique = audit.get("unique_configurations")
    direct_v5_unique = audit.get("direct_v5_unique_configurations")
    for name, value in (
        ("raw_rows", raw_rows),
        ("rejected_rows", rejected_rows),
        ("unique_configurations", unique),
        ("direct_v5_unique_configurations", direct_v5_unique),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"audit.{name} must be a non-negative integer")
    if rejected_rows > raw_rows:
        raise ValueError("audit.rejected_rows cannot exceed audit.raw_rows")
    if direct_v5_unique > unique:
        raise ValueError(
            "audit.direct_v5_unique_configurations cannot exceed audit.unique_configurations"
        )
    if direct_v5_unique < min_unique_configurations:
        raise ValueError("dataset has too few unique direct-v5 configurations")
    rejection_rate = 0.0 if raw_rows == 0 else rejected_rows / raw_rows
    if rejection_rate > maximum:
        raise ValueError("dataset rejection rate exceeds limit")
