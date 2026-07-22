"""Small, reproducible quality-and-speed evidence runs through Ollama.

This deliberately stays narrower than a leaderboard suite. It provides a
versioned local smoke evaluation while preserving enough metadata to repeat a
run and compare it with Localfit's speed recommendations.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from importlib.resources import files
from pathlib import Path

import requests

from omm.hardware import HardwareInfo
from omm import tuning

OLLAMA_HOST = "http://localhost:11434"
MAX_PACK_BYTES = 1_000_000
MAX_ITEMS = 100
MAX_PROMPT_CHARS = 10_000
_FINAL_NUMBER_RE = re.compile(r"FINAL\s*[:=]\s*([-+]?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)
_ANY_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


class QualityEvaluationError(RuntimeError):
    """Raised when the pack or local Ollama response cannot be trusted."""


@dataclass(frozen=True)
class SpeedSummary:
    median_tokens_per_sec: float
    samples: tuple[float, ...]


def default_pack_path() -> Path:
    return Path(str(files("omm").joinpath("data/quality-pack-v1.json")))


def _canonical_pack_bytes(pack: dict) -> bytes:
    return json.dumps(pack, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _bounded_int(value, minimum: int, maximum: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise QualityEvaluationError(f"{field} must be an integer from {minimum} to {maximum}")
    return value


def load_pack(path: Path | None = None) -> tuple[dict, str]:
    pack_path = path or default_pack_path()
    try:
        raw = pack_path.read_bytes()
    except OSError as error:
        raise QualityEvaluationError(f"could not read quality pack {pack_path}: {error}") from error
    if len(raw) > MAX_PACK_BYTES:
        raise QualityEvaluationError("quality pack exceeds the 1 MB safety limit")
    try:
        pack = json.loads(raw)
    except json.JSONDecodeError as error:
        raise QualityEvaluationError(f"quality pack is not valid JSON: {error}") from error
    if not isinstance(pack, dict) or pack.get("schema_version") != 1:
        raise QualityEvaluationError("quality pack must be a schema-version 1 object")
    if not isinstance(pack.get("pack_id"), str) or not pack["pack_id"]:
        raise QualityEvaluationError("quality pack requires a non-empty pack_id")
    template = pack.get("prompt_template")
    if not isinstance(template, str) or template.count("{question}") != 1:
        raise QualityEvaluationError("prompt_template must contain {question} exactly once")
    if len(template) > MAX_PROMPT_CHARS:
        raise QualityEvaluationError("prompt_template is too long")
    generation = pack.get("generation")
    if not isinstance(generation, dict):
        raise QualityEvaluationError("quality pack requires generation settings")
    _bounded_int(generation.get("seed"), 0, 2**31 - 1, "generation.seed")
    _bounded_int(generation.get("num_ctx"), 256, 131_072, "generation.num_ctx")
    _bounded_int(generation.get("num_predict"), 1, 512, "generation.num_predict")
    temperature = generation.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise QualityEvaluationError("generation.temperature must be numeric")
    if not math.isfinite(float(temperature)) or not 0 <= float(temperature) <= 2:
        raise QualityEvaluationError("generation.temperature must be from 0 to 2")
    if not isinstance(generation.get("think"), bool):
        raise QualityEvaluationError("generation.think must be boolean")

    items = pack.get("items")
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_ITEMS:
        raise QualityEvaluationError(f"quality pack must contain 1 to {MAX_ITEMS} items")
    seen_ids: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise QualityEvaluationError("every quality item must be an object")
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id or item_id in seen_ids:
            raise QualityEvaluationError("quality item IDs must be unique non-empty strings")
        seen_ids.add(item_id)
        if item.get("answer_type") != "number":
            raise QualityEvaluationError(f"{item_id}: only number answers are supported")
        for field in ("category", "language", "question", "expected"):
            value = item.get(field)
            if not isinstance(value, str) or not value:
                raise QualityEvaluationError(f"{item_id}: {field} must be a non-empty string")
        if len(item["question"]) > MAX_PROMPT_CHARS:
            raise QualityEvaluationError(f"{item_id}: question is too long")
        if _normalize_number(item["expected"]) is None:
            raise QualityEvaluationError(f"{item_id}: expected is not a valid number")
    return pack, hashlib.sha256(_canonical_pack_bytes(pack)).hexdigest()


def _normalize_number(value: str) -> str | None:
    try:
        number = Decimal(value.replace(",", ""))
    except (InvalidOperation, AttributeError):
        return None
    if not number.is_finite():
        return None
    normalized = format(number.normalize(), "f")
    return "0" if normalized in ("-0", "+0") else normalized


def parse_numeric_answer(response: str) -> str | None:
    match = _FINAL_NUMBER_RE.search(response)
    if match:
        return _normalize_number(match.group(1))
    matches = _ANY_NUMBER_RE.findall(response)
    return _normalize_number(matches[-1]) if matches else None


def _request_json(method: str, path: str, payload: dict | None = None, timeout: int = 180) -> dict:
    try:
        response = requests.request(
            method,
            f"{OLLAMA_HOST}{path}",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as error:
        raise QualityEvaluationError(f"Ollama {path} request failed: {error}") from error
    if not isinstance(data, dict):
        raise QualityEvaluationError(f"Ollama {path} returned a non-object response")
    return data


def ollama_version() -> str | None:
    try:
        value = _request_json("GET", "/api/version", timeout=5).get("version")
    except QualityEvaluationError:
        return None
    return value if isinstance(value, str) and len(value) <= 64 else None


def _model_metadata(tag: str) -> dict:
    tags = _request_json("GET", "/api/tags", timeout=10).get("models")
    if not isinstance(tags, list):
        raise QualityEvaluationError("Ollama model list is missing")
    listed = next((item for item in tags if isinstance(item, dict) and item.get("name") == tag), None)
    if listed is None:
        raise QualityEvaluationError(f"Ollama model '{tag}' is not installed")
    shown = _request_json("POST", "/api/show", {"model": tag}, timeout=30)
    details = shown.get("details") if isinstance(shown.get("details"), dict) else {}
    model_info = shown.get("model_info") if isinstance(shown.get("model_info"), dict) else {}
    capabilities = shown.get("capabilities")
    if not isinstance(capabilities, list):
        capabilities = []
    size_bytes = listed.get("size")
    return {
        "tag": tag,
        "digest": listed.get("digest") if isinstance(listed.get("digest"), str) else None,
        "size_bytes": size_bytes if isinstance(size_bytes, int) and size_bytes > 0 else None,
        "format": details.get("format"),
        "family": details.get("family"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
        "license": model_info.get("general.license"),
        "license_link": model_info.get("general.license.link"),
        "capabilities": [
            value
            for value in capabilities
            if isinstance(value, str) and len(value) <= 64
        ][:32],
    }


def _normalized_digest(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.removeprefix("sha256:").lower()
    return value if re.fullmatch(r"[0-9a-f]{64}", value) else None


def runtime_snapshot(tag: str, digest: str | None, options: dict) -> dict | None:
    """Return measured Ollama residency values, never inferred GPU use."""
    try:
        models = _request_json("GET", "/api/ps", timeout=10).get("models")
    except QualityEvaluationError:
        return None
    if not isinstance(models, list):
        return None
    expected_digest = _normalized_digest(digest)
    if expected_digest is not None:
        row = next(
            (
                item
                for item in models
                if isinstance(item, dict)
                and _normalized_digest(item.get("digest")) == expected_digest
            ),
            None,
        )
    else:
        row = next(
            (
                item
                for item in models
                if isinstance(item, dict) and item.get("name") == tag
            ),
            None,
        )
    if row is None:
        return None
    context, size, size_vram = row.get("context_length"), row.get("size"), row.get("size_vram")
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in (context, size, size_vram)):
        return None
    if context <= 0 or size <= 0 or size_vram < 0:
        return None
    threads, batch = options.get("num_thread"), options.get("num_batch")
    if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in (threads, batch)):
        return None
    return {
        "context_length": context,
        "gpu_offload_percent": max(0, min(100, round(100 * size_vram / size))),
        "cpu_threads": threads,
        "num_batch": batch,
        "runtime_profile": "explicit_ollama_options",
    }


def unload_model(tag: str) -> bool:
    """Best-effort isolation between models; never deletes model files."""
    try:
        _request_json(
            "POST",
            "/api/generate",
            {"model": tag, "stream": False, "keep_alive": 0},
            timeout=30,
        )
    except QualityEvaluationError:
        return False
    return True


def _generate(tag: str, prompt: str, generation: dict, num_predict: int | None = None,
              runtime_options: dict | None = None) -> dict:
    options = {
        "temperature": generation["temperature"],
        "seed": generation["seed"],
        "num_ctx": generation["num_ctx"],
        "num_predict": num_predict or generation["num_predict"],
    }
    options.update(runtime_options or {})
    data = _request_json(
        "POST",
        "/api/generate",
        {
            "model": tag,
            "prompt": prompt,
            "stream": False,
            "think": generation["think"],
            "options": options,
        },
    )
    if not isinstance(data.get("response"), str):
        raise QualityEvaluationError(f"Ollama returned no text response for '{tag}'")
    return data


def _generate_with_runtime(
    tag: str, prompt: str, generation: dict, num_predict: int | None, runtime_options: dict | None
) -> dict:
    """Avoid changing the call shape used by older test/integration fakes."""
    if runtime_options is None:
        return _generate(tag, prompt, generation, num_predict=num_predict)
    return _generate(tag, prompt, generation, num_predict=num_predict, runtime_options=runtime_options)


def _tokens_per_second(response: dict) -> float | None:
    count = response.get("eval_count")
    duration = response.get("eval_duration")
    if (
        isinstance(count, int)
        and not isinstance(count, bool)
        and count > 0
        and isinstance(duration, int)
        and not isinstance(duration, bool)
        and duration > 0
    ):
        return count / (duration / 1_000_000_000)
    return None


def _speed_probe(tag: str, generation: dict, runs: int, runtime_options: dict | None = None) -> SpeedSummary:
    _bounded_int(runs, 1, 10, "speed runs")
    prompt = "Explain what an operating system is in a concise paragraph."
    _generate_with_runtime(tag, prompt, generation, 8, runtime_options)
    samples = []
    for _ in range(runs):
        speed = _tokens_per_second(_generate_with_runtime(tag, prompt, generation, 64, runtime_options))
        if speed is None:
            raise QualityEvaluationError(f"Ollama returned no timing metrics for '{tag}'")
        samples.append(speed)
    return SpeedSummary(statistics.median(samples), tuple(samples))


def evaluate_model(tag: str, pack: dict, speed_runs: int = 3, runtime_options: dict | None = None,
                   model_metadata: dict | None = None) -> dict:
    metadata = model_metadata or _model_metadata(tag)
    template = pack["prompt_template"]
    generation = pack["generation"]
    item_results = []
    quality_speeds = []
    for item in pack["items"]:
        response = _generate_with_runtime(
            tag, template.format(question=item["question"]), generation, None, runtime_options
        )
        predicted = parse_numeric_answer(response["response"])
        expected = _normalize_number(item["expected"])
        correct = predicted is not None and predicted == expected
        speed = _tokens_per_second(response)
        if speed is not None:
            quality_speeds.append(speed)
        item_results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "language": item["language"],
                "expected": expected,
                "predicted": predicted,
                "correct": correct,
            }
        )

    correct_count = sum(1 for item in item_results if item["correct"])
    category_total = Counter(item["category"] for item in item_results)
    category_correct = Counter(item["category"] for item in item_results if item["correct"])
    speed = _speed_probe(tag, generation, speed_runs, runtime_options=runtime_options)
    return {
        **metadata,
        "quality": {
            "correct": correct_count,
            "total": len(item_results),
            "accuracy": round(correct_count / len(item_results), 4),
            "by_category": {
                category: {
                    "correct": category_correct[category],
                    "total": total,
                    "accuracy": round(category_correct[category] / total, 4),
                }
                for category, total in sorted(category_total.items())
            },
            "items": item_results,
            "raw_responses_stored": False,
        },
        "speed": {
            "median_tokens_per_sec": round(speed.median_tokens_per_sec, 2),
            "samples_tokens_per_sec": [round(value, 2) for value in speed.samples],
            "runs": len(speed.samples),
            "probe_num_predict": 64,
            "quality_generation_median_tokens_per_sec": (
                round(statistics.median(quality_speeds), 2) if quality_speeds else None
            ),
        },
        "runtime": runtime_snapshot(tag, metadata.get("digest"), runtime_options or {}),
    }


def collect_evidence(
    tags: list[str],
    hardware: HardwareInfo,
    pack_path: Path | None = None,
    speed_runs: int = 3,
) -> dict:
    if not tags:
        raise QualityEvaluationError("at least one Ollama model tag is required")
    if len(tags) > 20:
        raise QualityEvaluationError("at most 20 Ollama models may be evaluated at once")
    if len(set(tags)) != len(tags) or any(not tag or len(tag) > 256 for tag in tags):
        raise QualityEvaluationError("model tags must be unique non-empty strings")
    pack, pack_sha256 = load_pack(pack_path)
    models = []
    for tag in tags:
        try:
            metadata = _model_metadata(tag)
            profile = tuning.recommend_runtime_settings(hardware, metadata)
            options = profile.ollama_options
            try:
                result = evaluate_model(tag, pack, speed_runs=speed_runs,
                                        runtime_options=options, model_metadata=metadata)
            except TypeError:  # third-party/legacy monkeypatch compatibility
                result = evaluate_model(tag, pack, speed_runs=speed_runs)
        except QualityEvaluationError:
            # Preserve the public evaluator hook for callers which provide
            # their own metadata/evaluator; it simply cannot emit v5 runtime.
            result = evaluate_model(tag, pack, speed_runs=speed_runs)
        finally:
            unloaded = unload_model(tag)
        result["measurement_isolation"] = {
            "unloaded_after_run": unloaded,
            "model_files_deleted": False,
        }
        result.setdefault("runtime", None)
        models.append(result)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "small reproducible smoke evaluation; not a leaderboard",
        "pack": {
            "id": pack["pack_id"],
            "version": pack.get("pack_version"),
            "sha256": pack_sha256,
            "item_count": len(pack["items"]),
            "sources": pack.get("sources", []),
        },
        "environment": {
            "engine": "ollama",
            "engine_version": ollama_version(),
            "os": hardware.os_name,
            "architecture": platform.machine(),
            "ram_gb": round(hardware.ram_total_gb, 1),
            "vram_gb": (
                round(hardware.vram_total_gb, 1) if hardware.vram_total_gb is not None else None
            ),
            "unified_memory": hardware.unified_memory,
            "raw_hardware_names_stored": False,
        },
        "generation": pack["generation"],
        "models": models,
    }


def write_evidence(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)
