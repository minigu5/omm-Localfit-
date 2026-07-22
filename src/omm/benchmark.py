"""Real generation-speed benchmark, used to build telemetry for the
recommendation model. Only benchmarks via Ollama (has a simple REST API
with built-in per-token timing) - LM Studio benchmarking can be added later.
"""

from __future__ import annotations

import statistics

import requests

OLLAMA_HOST = "http://localhost:11434"
_BENCHMARK_PROMPT = "Explain what an operating system is."
_NUM_PREDICT = 32


def ollama_daemon_reachable() -> bool:
    try:
        requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        return True
    except requests.RequestException:
        return False


def benchmark_ollama(tag: str, options: dict | None = None) -> float | None:
    """Return tokens/sec, 0.0 if generation was attempted and failed, or
    None if the daemon wasn't reachable at all (untestable, not a failure).
    """
    if not ollama_daemon_reachable():
        return None

    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": tag,
                "prompt": _BENCHMARK_PROMPT,
                "stream": False,
                "options": {"num_predict": _NUM_PREDICT, **(options or {})},
            },
            timeout=120,
        )
        data = resp.json()
        eval_count = data.get("eval_count")
        eval_duration = data.get("eval_duration")
        if not eval_count or not eval_duration:
            return 0.0
        return eval_count / (eval_duration / 1e9)
    except (requests.RequestException, ValueError):
        return 0.0


def benchmark_ollama_samples(
    tag: str, runs: int = 3, options: dict | None = None
) -> dict | None:
    """Run a reproducible set of speed probes with identical options.

    ``benchmark_ollama`` remains the one-shot public API for callers which
    only need a float.  A ``None`` result still means the daemon was not
    reachable; failed individual generations are retained as zero samples.
    """
    if not isinstance(runs, int) or isinstance(runs, bool) or not 1 <= runs <= 10:
        raise ValueError("runs must be an integer from 1 to 10")
    samples: list[float] = []
    for _ in range(runs):
        try:
            value = benchmark_ollama(tag, options=options)
        except TypeError:  # compatibility with old monkeypatched callables
            value = benchmark_ollama(tag)
        if value is None:
            return None
        samples.append(float(value))
    return {
        "median_tokens_per_sec": statistics.median(samples),
        "min_tokens_per_sec": min(samples),
        "max_tokens_per_sec": max(samples),
        "count": len(samples),
        "samples_tokens_per_sec": samples,
        "options": dict(options or {}),
    }
