"""Shared feature extraction for the recommendation model - used both when
building training rows (scripts/train_model.py) and when scoring candidate
models at `omm recommend` time, so both sides encode features identically.
"""

from __future__ import annotations

import re

# Order matters: this is the exact input vector the trained tree expects.
# cpu/gpu_score+tier were appended after the original 5 - appending (not
# inserting) keeps old cached model.json artifacts scorable, since their
# trees only ever reference feature indices 0-4.
FEATURE_ORDER = [
    "ram_gb",
    "vram_gb",
    "unified_memory",
    "param_count_b",
    "quant_bits",
    "cpu_score",
    "cpu_tier",
    "gpu_score",
    "gpu_tier",
]

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

# Ordinal tier bump within a chip generation - e.g. "M2 Pro" and "M2 Max"
# share the same generation number but aren't the same chip, so the
# generation number alone would collapse them. Highest matching word wins.
_TIER_WORDS = {
    "ultra": 3.0,
    "max": 2.0,
    "pro": 1.0,
    "ti": 1.0,
    "x3d": 1.0,
    "super": 0.5,
}

_CHIP_MODEL_RE = re.compile(
    r"\bM(\d+)\b|\b(?:i[3579]-?|Ryzen\s*\d\s*|RTX\s?|GTX\s?)(\d{3,5})",
    re.IGNORECASE,
)


def parse_param_count_billions(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)[Bb](?:[-_.]|$)", text)
    return float(m.group(1)) if m else None


def parse_quant_bits(text: str) -> float | None:
    m = re.search(r"(FP?32|FP?16|Q\d)", text.upper().replace("_", ""))
    if not m:
        return None
    return _QUANT_BITS.get(m.group(1))


def parse_chip_score(text: str) -> tuple[float, float]:
    """Best-effort (model_number, tier_ordinal) parsed from a raw CPU/GPU
    name string, e.g. "Apple M2 Pro" -> (2.0, 1.0), "RTX 4090" -> (4090.0, 0.0).
    Returns (0.0, 0.0) for unrecognized or empty input."""
    m = _CHIP_MODEL_RE.search(text)
    score = float(m.group(1) or m.group(2)) if m else 0.0
    lowered = text.lower()
    tier = max((v for w, v in _TIER_WORDS.items() if w in lowered), default=0.0)
    return score, tier


def build_features(
    ram_gb: float,
    vram_gb: float | None,
    unified_memory: bool,
    param_count_b: float | None,
    quant_bits: float | None,
    cpu_score: float = 0.0,
    cpu_tier: float = 0.0,
    gpu_score: float = 0.0,
    gpu_tier: float = 0.0,
) -> list[float]:
    """Fixed-order numeric feature vector matching FEATURE_ORDER, with
    missing values defaulted to 0.0 (the tree learns around this)."""
    return [
        ram_gb,
        vram_gb if vram_gb is not None else 0.0,
        1.0 if unified_memory else 0.0,
        param_count_b if param_count_b is not None else 0.0,
        quant_bits if quant_bits is not None else 0.0,
        cpu_score,
        cpu_tier,
        gpu_score,
        gpu_tier,
    ]
