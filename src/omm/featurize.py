"""Shared feature extraction for the recommendation model - used both when
building training rows (scripts/train_model.py) and when scoring candidate
models at `omm recommend` time, so both sides encode features identically.
"""

from __future__ import annotations

import re

# Order matters: this is the exact input vector the trained tree expects.
FEATURE_ORDER = ["ram_gb", "vram_gb", "unified_memory", "param_count_b", "quant_bits"]

_QUANT_BITS = {
    "Q2": 2.0,
    "Q3": 3.0,
    "Q4": 4.0,
    "Q5": 5.0,
    "Q6": 6.0,
    "Q8": 8.0,
    "F16": 16.0,
    "FP16": 16.0,
    "F32": 32.0,
    "FP32": 32.0,
}


def parse_param_count_billions(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)[Bb](?:[-_.]|$)", text)
    return float(m.group(1)) if m else None


def parse_quant_bits(text: str) -> float | None:
    m = re.search(r"(FP?32|FP?16|Q\d)", text.upper().replace("_", ""))
    if not m:
        return None
    return _QUANT_BITS.get(m.group(1))


def build_features(
    ram_gb: float,
    vram_gb: float | None,
    unified_memory: bool,
    param_count_b: float | None,
    quant_bits: float | None,
) -> list[float]:
    """Fixed-order numeric feature vector matching FEATURE_ORDER, with
    missing values defaulted to 0.0 (the tree learns around this)."""
    return [
        ram_gb,
        vram_gb if vram_gb is not None else 0.0,
        1.0 if unified_memory else 0.0,
        param_count_b if param_count_b is not None else 0.0,
        quant_bits if quant_bits is not None else 0.0,
    ]
