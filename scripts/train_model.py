"""CI-only training script (run by .github/workflows/train.yml).

Fetches real install telemetry from the Firebase RTDB, bootstraps with
synthetic rows derived from the bundled default rules when real data is
sparse, trains a small DecisionTreeRegressor predicting tokens/sec, and
exports it as plain JSON (see omm.mltree for why: no pickle, no runtime
scikit-learn dependency for end users).

Not part of the omm package itself - requires scikit-learn, which is a
CI-only dependency (see requirements-train.txt), never shipped to users.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from sklearn.tree import DecisionTreeRegressor, _tree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from omm.featurize import (  # noqa: E402
    FEATURE_ORDER,
    build_features,
    parse_chip_score,
    parse_param_count_billions,
    parse_quant_bits,
)
from omm.rules import DEFAULT_RULES  # noqa: E402

TELEMETRY_URL = "https://localfit-8ab57-default-rtdb.firebaseio.com/telemetry.json"
MIN_REAL_ROWS = 10
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "published" / "recommend-model.json"

RAM_GRID = [2, 4, 8, 16, 32, 64]
VRAM_GRID = [0, 4, 6, 8, 12, 24]


def fetch_real_rows() -> list[dict]:
    try:
        resp = requests.get(TELEMETRY_URL, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"Warning: couldn't fetch telemetry ({e}), treating as 0 real rows.")
        return []
    if not data:
        return []
    return list(data.values())


def real_rows_to_training_data(rows: list[dict]) -> tuple[list[list[float]], list[float]]:
    X, y = [], []
    for row in rows:
        if "tokens_per_sec" not in row or "ram_gb" not in row:
            continue
        text = f"{row.get('model_installed', '')} {row.get('model_repo_id', '')}"
        cpu_score, cpu_tier = parse_chip_score(row.get("cpu") or "")
        gpu_score, gpu_tier = parse_chip_score(row.get("gpu") or "")
        features = build_features(
            ram_gb=row["ram_gb"],
            vram_gb=row.get("vram_gb"),
            unified_memory=bool(row.get("unified_memory")),
            param_count_b=parse_param_count_billions(text),
            quant_bits=parse_quant_bits(text),
            cpu_score=cpu_score,
            cpu_tier=cpu_tier,
            gpu_score=gpu_score,
            gpu_tier=gpu_tier,
        )
        X.append(features)
        y.append(float(row["tokens_per_sec"]))
    return X, y


def synthetic_rows_from_rules() -> tuple[list[list[float]], list[float]]:
    """Bootstrap training data from the hand-written heuristic rules so the
    first model (before any real telemetry exists) is still sensible."""
    X, y = [], []
    for rule in DEFAULT_RULES:
        param_count_b = parse_param_count_billions(rule["name"]) or 7.0
        quant_bits = parse_quant_bits(rule["name"]) or 4.0
        for ram_gb in RAM_GRID:
            for vram_gb in VRAM_GRID:
                unified = vram_gb == ram_gb  # crude stand-in for an Apple Silicon row
                meets = ram_gb >= rule["min_ram_gb"] and vram_gb >= rule["min_vram_gb"]
                speed = (60.0 / param_count_b) if meets else 0.0
                X.append(
                    build_features(
                        ram_gb=ram_gb,
                        vram_gb=vram_gb,
                        unified_memory=unified,
                        param_count_b=param_count_b,
                        quant_bits=quant_bits,
                    )
                )
                y.append(speed)
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


def load_candidates() -> list[dict]:
    candidates_path = Path(__file__).resolve().parent.parent / "published" / "candidates.json"
    if candidates_path.exists():
        return json.loads(candidates_path.read_text())
    print("Warning: no dist/candidates.json found, falling back to curated index only.")
    from omm.hub import CURATED_INDEX

    return [
        {"name": name, "repo_id": repo_id, "filename": filename, "description": ""}
        for name, (repo_id, filename) in CURATED_INDEX.items()
    ]


def main() -> None:
    real_rows = fetch_real_rows()
    X, y = real_rows_to_training_data(real_rows)
    print(f"Fetched {len(X)} usable real telemetry rows.")

    if len(X) < MIN_REAL_ROWS:
        synth_X, synth_y = synthetic_rows_from_rules()
        print(f"Below {MIN_REAL_ROWS}-row threshold, adding {len(synth_X)} synthetic rows.")
        X, y = X + synth_X, y + synth_y

    model = DecisionTreeRegressor(max_depth=6, min_samples_leaf=2, random_state=0)
    model.fit(X, y)

    candidates = load_candidates()
    artifact = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_order": FEATURE_ORDER,
        "real_row_count": len(real_rows),
        "trees": [export_node(model.tree_, 0)],
        "candidates": candidates,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2))
    print(f"Wrote {OUTPUT_PATH} ({len(candidates)} candidates, {len(X)} training rows)")


if __name__ == "__main__":
    main()
