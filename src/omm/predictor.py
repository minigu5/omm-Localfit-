"""Fetches the CI-trained recommendation model and ranks candidate GGUF
models against local hardware. Falls back through: live fetch -> last
cached copy -> the legacy heuristic rules (see rules.py) if neither is
available, so `omm recommend` always has something to show.
"""

from __future__ import annotations

import json

import requests

from omm.config import RECOMMEND_MODEL_PATH
from omm.featurize import build_features, parse_param_count_billions, parse_quant_bits
from omm.hardware import HardwareInfo
from omm.mltree import predict_ensemble


class ModelFetchError(Exception):
    pass


def fetch_and_cache_model(url: str) -> dict:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    artifact = resp.json()
    RECOMMEND_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECOMMEND_MODEL_PATH.write_text(json.dumps(artifact))
    return artifact


def load_cached_model() -> dict | None:
    if not RECOMMEND_MODEL_PATH.exists():
        return None
    return json.loads(RECOMMEND_MODEL_PATH.read_text())


def load_model(url: str | None) -> dict | None:
    """Live fetch first, falling back to the last cached copy."""
    if url:
        try:
            return fetch_and_cache_model(url)
        except (requests.RequestException, ValueError):
            pass
    return load_cached_model()


def rank_candidates(artifact: dict, hw: HardwareInfo) -> list[tuple[dict, float]]:
    trees = artifact["trees"]
    has_gpu = hw.vram_total_gb is not None

    ranked = []
    for candidate in artifact["candidates"]:
        text = f"{candidate.get('name', '')} {candidate.get('filename', '')}"
        features = build_features(
            ram_gb=hw.ram_total_gb,
            vram_gb=hw.vram_total_gb if has_gpu else 0.0,
            unified_memory=hw.unified_memory,
            param_count_b=parse_param_count_billions(text),
            quant_bits=parse_quant_bits(text),
        )
        predicted_tokens_per_sec = predict_ensemble(trees, features)
        ranked.append((candidate, predicted_tokens_per_sec))

    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked
