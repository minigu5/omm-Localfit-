"""CI-only training script (run by .github/workflows/train.yml).

Fetches benchmark telemetry from a self-hosted export endpoint or legacy
Firebase-compatible JSON, validates it, and
trains a small random-forest regressor predicting tokens/sec. Sparse data is
bootstrapped with synthetic rows derived from the bundled default rules. The
forest is exported as plain JSON (see omm.mltree for why: no pickle and no
runtime scikit-learn dependency for end users).

Not part of the omm package itself - requires scikit-learn, which is a
CI-only dependency (see requirements-train.txt), never shipped to users.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import _tree

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from omm.featurize import (  # noqa: E402
    FEATURE_ORDER,
    build_features,
    candidate_active_parameter_count_billions,
    candidate_parameter_count_billions,
    candidate_quant_bits,
    estimate_model_size_gb,
)
from omm.atomic import atomic_write_text, locked  # noqa: E402
from scripts.model_quality_gate import (  # noqa: E402
    compare_artifacts,
    selection_context_key,
    validate_artifact,
    validate_dataset,
)

TELEMETRY_URL = "http://127.0.0.1:8000/v1/benchmarks/export"
MIN_REAL_ROWS = 10
MAX_REAL_ROWS = 5000
REAL_BOOTSTRAP_WEIGHT = 12.0
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "published" / "recommend-model.json"

RAM_GRID = [4, 8, 16, 24, 32, 64, 128]
VRAM_GRID = [0, 4, 6, 8, 12, 16, 24, 32, 48, 64]
PARAMETER_GRID_B = [0.35, 0.5, 0.6, 1.0, 1.1, 1.5, 2.0, 3.0, 4.0, 7.0, 8.0, 13.0, 20.0, 27.0, 32.0, 70.0]
MOE_PARAMETER_GRID_B = [(8.0, 1.0), (14.0, 3.0), (30.0, 3.0), (35.0, 3.0)]
QUANT_GRID_BITS = [3.0, 4.0, 5.0, 6.0, 8.0, 16.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Localfit recommendation artifact.")
    parser.add_argument(
        "--telemetry-file",
        type=Path,
        action="append",
        default=[],
        help="Append a local JSON or JSONL benchmark file; repeatable.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Do not read Firebase; train only from supplied files and bootstrap rows.",
    )
    parser.add_argument(
        "--telemetry-url",
        default=os.environ.get("LOCALFIT_TELEMETRY_URL", TELEMETRY_URL),
        help="Firebase JSON URL or self-hosted /v1/benchmarks/export endpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Artifact destination (defaults to published/recommend-model.json).",
    )
    parser.add_argument("--baseline", type=Path, help="Incumbent artifact used by the quality gate.")
    parser.add_argument(
        "--quality-gate",
        action="store_true",
        help="Validate telemetry and reject candidate regressions before publishing.",
    )
    parser.add_argument("--minimum-real-configurations", type=int, default=20)
    parser.add_argument("--maximum-rejection-rate", type=float, default=0.25)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--quality-report", type=Path, help="Optional JSON quality-gate report.")
    return parser.parse_args()


def is_firebase_realtime_database_json_url(url: str) -> bool:
    """Return whether *url* is an official Firebase RTDB JSON endpoint."""
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    is_firebase_host = hostname.endswith(".firebaseio.com") or hostname.endswith(
        ".firebasedatabase.app"
    )
    return is_firebase_host and parsed.path.endswith(".json")


def fetch_real_rows(url: str = TELEMETRY_URL) -> list[dict]:
    token = os.environ.get("LOCALFIT_ADMIN_TOKEN")
    # Firebase RTDB JSON endpoints are public-read in this workflow.  Avoid
    # sending a configured admin credential to that third-party URL.
    headers = (
        {"Authorization": f"Bearer {token}"}
        if token and not is_firebase_realtime_database_json_url(url)
        else {}
    )
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"Warning: couldn't fetch telemetry ({e}), treating as 0 real rows.")
        return []
    if isinstance(data, dict) and isinstance(data.get("benchmarks"), list):
        return [row for row in data["benchmarks"][-MAX_REAL_ROWS:] if isinstance(row, dict)]
    if isinstance(data, dict):
        return [row for row in list(data.values())[-MAX_REAL_ROWS:] if isinstance(row, dict)]
    if isinstance(data, list):
        return [row for row in data[-MAX_REAL_ROWS:] if isinstance(row, dict)]
    return []


def load_telemetry_file(path: Path) -> list[dict]:
    """Load Firebase-shaped JSON, a JSON list, or local benchmark JSONL."""
    try:
        raw = path.read_text()
    except OSError as error:
        raise ValueError(f"could not read {path}: {error}") from error
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        rows = []
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from error
            if isinstance(row, dict):
                rows.append(row)
        return rows
    if isinstance(payload, dict):
        if isinstance(payload.get("benchmarks"), list):
            return [row for row in payload["benchmarks"] if isinstance(row, dict)]
        # A single benchmark event is distinguished from Firebase's push-ID
        # mapping by the required measurement field.
        if "tokens_per_sec" in payload:
            return [payload]
        return [row for row in payload.values() if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    raise ValueError(f"{path} must contain a JSON object, list, or JSONL records")


def _bounded_number(value, minimum: float, maximum: float) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not minimum <= number <= maximum:
        return None
    return number


def _direct_bounded_number(value, minimum: float, maximum: float) -> float | None:
    """Accept only JSON numeric values, never stringified model metadata."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return _bounded_number(value, minimum, maximum)


def _real_row_to_sample(row: dict) -> tuple[tuple[list[float], float] | None, str | None]:
    if not isinstance(row, dict):
        return None, "not_an_object"
    engine = row.get("engine") or "ollama"
    if engine not in ("ollama", "llama.cpp", "lmstudio"):
        return None, "unsupported_engine"
    benchmark_version = row.get("benchmark_version")
    if benchmark_version not in (None, 1, 2, 3, 4, 5):
        return None, "unsupported_schema"

    tokens_per_sec = _bounded_number(row.get("tokens_per_sec"), 0.0, 10_000.0)
    ram_gb = _bounded_number(row.get("ram_gb"), 1.0, 1024.0)
    vram_gb = _bounded_number(row.get("vram_gb"), 0.0, 512.0)
    gpu_tflops = _bounded_number(row.get("gpu_tflops"), 0.0, 1000.0)
    is_v5 = benchmark_version == 5
    if is_v5:
        required_runtime = (
            "runtime_profile", "context_length", "gpu_offload_percent", "cpu_threads", "num_batch",
        )
        if any(row.get(field) is None for field in required_runtime):
            return None, "missing_runtime_metadata"
        context_length = _direct_bounded_number(row.get("context_length"), 256.0, 131072.0)
        gpu_offload_percent = _direct_bounded_number(row.get("gpu_offload_percent"), 0.0, 100.0)
        cpu_threads = _direct_bounded_number(row.get("cpu_threads"), 1.0, 1024.0)
        num_batch = _direct_bounded_number(row.get("num_batch"), 1.0, 65536.0)
    else:
        context_length = _bounded_number(row.get("context_length", 2048), 256.0, 131072.0)
        gpu_offload_percent = _bounded_number(
            row.get("gpu_offload_percent", 100 if (vram_gb or 0) > 0 else 0), 0.0, 100.0
        )
        cpu_threads = _bounded_number(row.get("cpu_threads", 0), 0.0, 1024.0)
        num_batch = _bounded_number(row.get("num_batch", 0), 0.0, 65536.0)
    if tokens_per_sec is None or ram_gb is None:
        return None, "invalid_measurement"
    if None in (context_length, gpu_offload_percent, cpu_threads, num_batch):
        return None, "invalid_runtime"
    if benchmark_version in (4, 5):
        if is_v5 and any(
            row.get(field) is None
            for field in ("sample_count", "tokens_per_sec_min", "tokens_per_sec_max")
        ):
            return None, "missing_sample_summary"
        number_parser = _direct_bounded_number if is_v5 else _bounded_number
        sample_count = number_parser(row.get("sample_count", 1), 1.0, 10.0)
        sample_min = number_parser(
            row.get("tokens_per_sec_min", tokens_per_sec), 0.0, 10_000.0
        )
        sample_max = number_parser(
            row.get("tokens_per_sec_max", tokens_per_sec), 0.0, 10_000.0
        )
        if (
            sample_count is None
            or not sample_count.is_integer()
            or sample_min is None
            or sample_max is None
            or not sample_min <= tokens_per_sec <= sample_max
            or (is_v5 and sample_count < 3)
        ):
            return None, "invalid_samples"

    if is_v5:
        required_model_metadata = (
            "parameter_count_b", "active_parameter_count_b", "quant_bits", "engine_version", "client_version",
        )
        if any(row.get(field) is None for field in required_model_metadata):
            return None, "missing_model_metadata"
        param_count_b = _direct_bounded_number(row.get("parameter_count_b"), 0.0, 10_000.0)
        active_param_count_b = _direct_bounded_number(
            row.get("active_parameter_count_b"), 0.0, 10_000.0
        )
        quant_bits = _direct_bounded_number(row.get("quant_bits"), 0.5, 32.0)
        if (
            param_count_b is None
            or active_param_count_b is None
            or quant_bits is None
            or param_count_b <= 0
            or active_param_count_b <= 0
            or active_param_count_b > param_count_b
            or not isinstance(row.get("runtime_profile"), str)
            or not row["runtime_profile"].strip()
            or not isinstance(row.get("engine_version"), str)
            or not row["engine_version"].strip()
            or not isinstance(row.get("client_version"), str)
            or not row["client_version"].strip()
            or len(row["engine_version"]) > 100
            or len(row["client_version"]) > 100
        ):
            return None, "invalid_model_metadata"
        size_bytes = _bounded_number(row.get("model_size_bytes"), 1.0, 10**15)
        model_size_gb = (
            size_bytes / (1024**3)
            if size_bytes is not None
            else param_count_b * quant_bits / 8.0 * 1.1
        )
    else:
        installed_name = str(row.get("model_installed") or "")
        repo_name = str(row.get("model_repo_id") or "").rsplit("/", 1)[-1]
        text = f"{installed_name} {repo_name}"
        candidate = {
            "name": installed_name,
            "filename": installed_name,
            "repo_id": row.get("model_repo_id"),
            "size_bytes": row.get("model_size_bytes"),
        }
        param_count_b = candidate_parameter_count_billions(candidate)
        active_param_count_b = candidate_active_parameter_count_billions(candidate)
        quant_bits = candidate_quant_bits(candidate)
        model_size_gb = estimate_model_size_gb(text, row.get("model_size_bytes"))
        if param_count_b is None or quant_bits is None or model_size_gb is None:
            return None, "unparseable_model"

    features = build_features(
        ram_gb=ram_gb,
        vram_gb=vram_gb,
        unified_memory=bool(row.get("unified_memory")),
        param_count_b=param_count_b,
        quant_bits=quant_bits,
        model_size_gb=model_size_gb,
        gpu_tflops=gpu_tflops or 0.0,
        context_length=context_length,
        gpu_offload_ratio=gpu_offload_percent / 100.0,
        cpu_threads=cpu_threads,
        num_batch=num_batch,
        active_param_count_b=active_param_count_b,
        engine=engine,
    )
    return (features, tokens_per_sec), None


def real_rows_to_training_data_with_audit(
    rows: list[dict],
) -> tuple[list[list[float]], list[float], dict]:
    groups: dict[tuple[float, ...], list[float]] = {}
    rejections: dict[str, int] = {}
    valid_rows = 0
    samples_used = 0
    samples_capped = 0
    direct_v5_groups: set[tuple[float, ...]] = set()
    for row in rows:
        sample, reason = _real_row_to_sample(row)
        if sample is None:
            rejections[reason or "unknown"] = rejections.get(reason or "unknown", 0) + 1
            continue
        valid_rows += 1
        features, tokens_per_sec = sample
        # Collapse repeated measurements of the same configuration to a
        # median. This makes community retraining useful without allowing one
        # noisy client or a burst of duplicate uploads to dominate the fit.
        group_key = tuple(round(value, 3) for value in features)
        if row.get("benchmark_version") == 5:
            # Reaching this point means _real_row_to_sample accepted the
            # explicit v5 model, runtime, and sample metadata.
            direct_v5_groups.add(group_key)
        samples = groups.setdefault(group_key, [])
        if len(samples) < 50:
            samples.append(tokens_per_sec)
            samples_used += 1
        else:
            samples_capped += 1

    X = [list(features) for features in groups]
    y = [statistics.median(samples) for samples in groups.values()]
    audit = {
        "raw_rows": len(rows),
        "valid_rows": valid_rows,
        "rejected_rows": len(rows) - valid_rows,
        "samples_used": samples_used,
        "samples_capped": samples_capped,
        "unique_configurations": len(groups),
        "direct_v5_unique_configurations": len(direct_v5_groups),
        "duplicates_collapsed": samples_used - len(groups),
        "rejections": dict(sorted(rejections.items())),
    }
    return X, y, audit


def real_rows_to_training_data(rows: list[dict]) -> tuple[list[list[float]], list[float]]:
    X, y, _audit = real_rows_to_training_data_with_audit(rows)
    return X, y


def stable_holdout_split(
    X: list[list[float]], y: list[float], holdout_fraction: float
) -> tuple[list[list[float]], list[float], list[list[float]], list[float]]:
    """Split complete selection contexts without leaking sibling candidates."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")
    groups: dict[tuple[float, ...], list[tuple[list[float], float]]] = {}
    for features, target in zip(X, y):
        groups.setdefault(selection_context_key(FEATURE_ORDER, features), []).append((features, target))
    if len(groups) < 2:
        raise ValueError("quality gate requires at least two selection contexts")
    ordered_groups = sorted(
        (
            sorted(
                group,
                key=lambda sample: json.dumps(sample, separators=(",", ":"), ensure_ascii=True),
            )
            for group in groups.values()
        ),
        key=lambda group: hashlib.sha256(
            json.dumps(selection_context_key(FEATURE_ORDER, group[0][0]), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest(),
    )
    holdout_target = max(1, int(math.ceil(len(X) * holdout_fraction)))
    holdout = []
    holdout_group_count = 0
    for group in ordered_groups[:-1]:
        if len(holdout) >= holdout_target:
            break
        holdout.extend(group)
        holdout_group_count += 1
    training = [sample for group in ordered_groups[holdout_group_count:] for sample in group]
    return (
        [features for features, _target in training],
        [target for _features, target in training],
        [features for features, _target in holdout],
        [target for _features, target in holdout],
    )


def synthetic_rows_from_rules() -> tuple[list[list[float]], list[float]]:
    """Build a broad, deterministic bootstrap grid.

    The first version only sampled the three bundled model rules (1.1B, 7B,
    and 8B). A tree trained from those points assigned the same throughput to
    nearly every sub-2B model. The denser parameter/quantization grid keeps the
    cold-start estimator honest about size differences while real telemetry
    remains the higher-weight source of truth.
    """
    X, y = [], []
    architectures = [(parameters, parameters) for parameters in PARAMETER_GRID_B]
    architectures.extend(MOE_PARAMETER_GRID_B)
    for param_count_b, active_param_count_b in architectures:
        for quant_bits in QUANT_GRID_BITS:
            model_size_gb = param_count_b * quant_bits / 8.0 * 1.1
            for ram_gb in RAM_GRID:
                for vram_gb in VRAM_GRID:
                    unified = vram_gb == ram_gb and vram_gb > 0
                    required_gb = model_size_gb * 1.2
                    available_gb = max(ram_gb * 0.8, vram_gb * 0.9)
                    meets = available_gb >= required_gb
                    if unified:
                        gpu_offload_ratio = 1.0
                    elif vram_gb <= 0:
                        gpu_offload_ratio = 0.0
                    elif model_size_gb <= vram_gb * 0.85:
                        gpu_offload_ratio = 1.0
                    else:
                        gpu_offload_ratio = max(0.1, min(0.9, vram_gb * 0.8 / model_size_gb))
                    accelerator_factor = 0.35 + 0.65 * gpu_offload_ratio
                    baseline_speed = 100.0 / (
                        active_param_count_b * math.sqrt(quant_bits / 4.0)
                    )
                    speed = baseline_speed * accelerator_factor if meets else 0.0
                    for context_length, context_factor in ((2048, 1.0), (4096, 0.97), (8192, 0.92)):
                        X.append(
                            build_features(
                                ram_gb=ram_gb,
                                vram_gb=vram_gb,
                                unified_memory=unified,
                                param_count_b=param_count_b,
                                quant_bits=quant_bits,
                                model_size_gb=model_size_gb,
                                context_length=context_length,
                                gpu_offload_ratio=gpu_offload_ratio,
                                cpu_threads=8,
                                num_batch=512 if gpu_offload_ratio >= 0.8 else 256 if gpu_offload_ratio > 0 else 128,
                                active_param_count_b=active_param_count_b,
                            )
                        )
                        y.append(speed * context_factor)
    return X, y


def export_node(tree, node_id: int) -> dict:
    if tree.children_left[node_id] == _tree.TREE_LEAF:
        return {"leaf": True, "value": float(tree.value[node_id][0][0])}
    return {
        "feature": int(tree.feature[node_id]),
        "threshold": float(tree.threshold[node_id]),
        "left": export_node(tree, tree.children_left[node_id]),
        "right": export_node(tree, tree.children_right[node_id]),
    }


def train_artifact(
    X: list[list[float]],
    y: list[float],
    *,
    sample_weight: list[float] | None,
    training_mode: str,
    bootstrap_method: str | None,
    real_rows: list[dict],
    telemetry_audit: dict,
    input_sources: list[str],
    evaluation: dict | None,
) -> dict:
    model = RandomForestRegressor(
        n_estimators=64,
        max_depth=8,
        min_samples_leaf=2,
        random_state=0,
        n_jobs=-1,
    )
    model.fit(X, y, sample_weight=sample_weight)
    candidates = load_candidates()
    return {
        "model_version": 4,
        "feature_schema_version": 1,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_order": FEATURE_ORDER,
        "training_mode": training_mode,
        "bootstrap_method": bootstrap_method,
        "raw_real_row_count": len(real_rows),
        "real_row_count": telemetry_audit["unique_configurations"],
        "training_row_count": len(X),
        "telemetry_audit": telemetry_audit,
        "telemetry_sources": input_sources,
        "evaluation": evaluation,
        "trees": [export_node(estimator.tree_, 0) for estimator in model.estimators_],
        "candidates": candidates,
    }


def load_candidates() -> list[dict]:
    candidates_path = Path(__file__).resolve().parent.parent / "published" / "candidates.json"
    if candidates_path.exists():
        return json.loads(candidates_path.read_text())
    print("Warning: no published/candidates.json found, falling back to curated index only.")
    from omm.hub import CURATED_INDEX

    return [
        {"name": name, "repo_id": repo_id, "filename": filename, "description": ""}
        for name, (repo_id, filename) in CURATED_INDEX.items()
    ]


def main() -> None:
    args = parse_args()
    real_rows = [] if args.offline else fetch_real_rows(args.telemetry_url)
    input_sources = [] if args.offline else [
        "firebase_legacy"
        if is_firebase_realtime_database_json_url(args.telemetry_url)
        else "self_hosted"
    ]
    for telemetry_path in args.telemetry_file:
        file_rows = load_telemetry_file(telemetry_path)
        real_rows.extend(file_rows)
        input_sources.append("local_file")
        print(f"Loaded {len(file_rows)} local telemetry row(s) from {telemetry_path}.")
    real_rows = real_rows[-MAX_REAL_ROWS:]
    real_X, real_y, telemetry_audit = real_rows_to_training_data_with_audit(real_rows)
    print(
        f"Fetched {telemetry_audit['valid_rows']} valid telemetry rows "
        f"({telemetry_audit['raw_rows']} raw, {len(real_X)} unique configurations, "
        f"{telemetry_audit['rejected_rows']} rejected)."
    )

    if args.quality_gate:
        # Validate before constructing a candidate or touching the destination.
        validate_dataset(
            telemetry_audit,
            min_unique_configurations=args.minimum_real_configurations,
            max_rejection_rate=args.maximum_rejection_rate,
        )
        if args.baseline is None:
            raise ValueError("--baseline is required with --quality-gate")
        try:
            baseline = json.loads(args.baseline.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"could not read baseline artifact {args.baseline}: {error}") from error
        validate_artifact(baseline, FEATURE_ORDER)
        train_X, train_y, holdout_X, holdout_y = stable_holdout_split(
            real_X, real_y, args.holdout_fraction
        )
        candidate = train_artifact(
            train_X,
            train_y,
            sample_weight=None,
            training_mode="telemetry",
            bootstrap_method=None,
            real_rows=real_rows,
            telemetry_audit=telemetry_audit,
            input_sources=input_sources,
            evaluation=None,
        )
        evaluation = compare_artifacts(candidate, baseline, holdout_X, holdout_y)
        evaluation.update(
            {
                "holdout_fraction": args.holdout_fraction,
                "training_rows": len(train_X),
                "holdout_rows": len(holdout_X),
            }
        )
        if args.quality_report:
            args.quality_report.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(args.quality_report, json.dumps(evaluation, indent=2) + "\n")
        if not evaluation["passed"]:
            raise SystemExit("quality gate rejected candidate: " + "; ".join(evaluation["failures"]))
        artifact = train_artifact(
            real_X,
            real_y,
            sample_weight=None,
            training_mode="telemetry",
            bootstrap_method=None,
            real_rows=real_rows,
            telemetry_audit=telemetry_audit,
            input_sources=input_sources,
            evaluation=evaluation,
        )
    elif len(real_X) < MIN_REAL_ROWS:
        synth_X, synth_y = synthetic_rows_from_rules()
        print(f"Below {MIN_REAL_ROWS}-row threshold, adding {len(synth_X)} synthetic rows.")
        X, y = synth_X + real_X, synth_y + real_y
        sample_weight = [1.0] * len(synth_X) + [REAL_BOOTSTRAP_WEIGHT] * len(real_X)
        training_mode = "hybrid_bootstrap"
        artifact = train_artifact(
            X, y, sample_weight=sample_weight, training_mode=training_mode,
            bootstrap_method="dense_moe_parameter_quantization_grid_v2", real_rows=real_rows,
            telemetry_audit=telemetry_audit, input_sources=input_sources, evaluation=None,
        )
    else:
        X, y = real_X, real_y
        training_mode = "telemetry"
        artifact = train_artifact(
            X, y, sample_weight=None, training_mode=training_mode, bootstrap_method=None,
            real_rows=real_rows, telemetry_audit=telemetry_audit, input_sources=input_sources,
            evaluation=None,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with locked(args.output):
        atomic_write_text(args.output, json.dumps(artifact, indent=2) + "\n")
    print(
        f"Wrote {args.output} ({len(artifact['candidates'])} candidates, "
        f"{artifact['training_row_count']} training rows)"
    )


if __name__ == "__main__":
    main()
