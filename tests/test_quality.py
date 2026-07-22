from __future__ import annotations

import json

import pytest

from omm import benchmark, quality
from omm.hardware import HardwareInfo


def _hardware() -> HardwareInfo:
    return HardwareInfo(
        os_name="macOS",
        os_version="",
        cpu="private CPU name",
        ram_total_gb=24,
        ram_available_gb=18,
        unified_memory=True,
        gpu_name="private GPU name",
        vram_total_gb=24,
        vram_free_gb=18,
    )


def test_bundled_quality_pack_is_versioned_bounded_and_attributed():
    pack, digest = quality.load_pack()

    assert pack["pack_id"] == "localfit-gsm8k-bilingual-smoke"
    assert pack["pack_version"] == "1.1.0"
    assert len(pack["items"]) == 8
    assert {item["language"] for item in pack["items"]} == {"en", "ko"}
    assert pack["sources"][0]["license"] == "MIT"
    assert len(digest) == 64


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("FINAL: 18", "18"),
        ("work here\nFINAL = 70,000", "70000"),
        ("The result is 3.0", "3"),
        ("no numeric answer", None),
    ],
)
def test_parse_numeric_answer(response, expected):
    assert quality.parse_numeric_answer(response) == expected


def test_quality_pack_rejects_duplicate_ids(tmp_path):
    pack, _digest = quality.load_pack()
    pack["items"][1]["id"] = pack["items"][0]["id"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(pack))

    with pytest.raises(quality.QualityEvaluationError, match="unique"):
        quality.load_pack(path)


def test_evaluate_model_stores_parsed_answers_not_raw_text(monkeypatch):
    pack, _digest = quality.load_pack()
    monkeypatch.setattr(
        quality,
        "_model_metadata",
        lambda tag: {
            "tag": tag,
            "digest": "sha256:abc",
            "size_bytes": 123,
            "format": "gguf",
            "family": "test",
            "parameter_size": "1B",
            "quantization_level": "Q4_K_M",
            "license": "apache-2.0",
            "license_link": None,
            "capabilities": ["completion"],
        },
    )
    answers = iter(item["expected"] for item in pack["items"])

    def fake_generate(tag, prompt, generation, num_predict=None):
        answer = next(answers) if num_predict is None else "1"
        return {
            "response": f"private reasoning must not persist\nFINAL: {answer}",
            "eval_count": 10,
            "eval_duration": 100_000_000,
        }

    monkeypatch.setattr(quality, "_generate", fake_generate)
    result = quality.evaluate_model("model:latest", pack, speed_runs=2)

    assert result["quality"]["accuracy"] == 1.0
    assert result["quality"]["raw_responses_stored"] is False
    assert all("response" not in item for item in result["quality"]["items"])
    assert result["speed"]["samples_tokens_per_sec"] == [100.0, 100.0]


def test_collect_evidence_redacts_hardware_names(monkeypatch):
    monkeypatch.setattr(quality, "ollama_version", lambda: "0.30.10")
    monkeypatch.setattr(
        quality,
        "evaluate_model",
        lambda tag, pack, speed_runs=3: {"tag": tag, "quality": {}, "speed": {}},
    )
    unloaded = []
    monkeypatch.setattr(quality, "unload_model", lambda tag: unloaded.append(tag) or True)

    report = quality.collect_evidence(["model:one"], _hardware())

    assert report["environment"]["ram_gb"] == 24
    assert report["environment"]["raw_hardware_names_stored"] is False
    assert "private CPU name" not in json.dumps(report)
    assert "private GPU name" not in json.dumps(report)
    assert unloaded == ["model:one"]
    assert report["models"][0]["measurement_isolation"]["unloaded_after_run"] is True


def test_unload_model_uses_keep_alive_zero_without_deleting(monkeypatch):
    calls = []
    monkeypatch.setattr(
        quality,
        "_request_json",
        lambda method, path, payload=None, timeout=180: calls.append(
            (method, path, payload, timeout)
        )
        or {},
    )

    assert quality.unload_model("model:latest") is True
    assert calls == [
        (
            "POST",
            "/api/generate",
            {"model": "model:latest", "stream": False, "keep_alive": 0},
            30,
        )
    ]


def test_write_evidence_replaces_atomically(tmp_path):
    path = tmp_path / "nested" / "evidence.json"
    quality.write_evidence({"schema_version": 1}, path)

    assert json.loads(path.read_text()) == {"schema_version": 1}
    assert not path.with_suffix(".json.tmp").exists()


def test_runtime_snapshot_prefers_digest_and_reports_actual_offload(monkeypatch):
    digest = "a" * 64
    monkeypatch.setattr(
        quality,
        "_request_json",
        lambda *args, **kwargs: {
            "models": [
                {
                    "name": "model:latest",
                    "digest": "b" * 64,
                    "context_length": 2048,
                    "size": 100,
                    "size_vram": 0,
                },
                {
                    "name": "other:latest",
                    "digest": digest,
                    "context_length": 4096,
                    "size": 100,
                    "size_vram": 75,
                },
            ]
        },
    )

    snapshot = quality.runtime_snapshot(
        "model:latest",
        digest,
        {"num_ctx": 4096, "num_thread": 8, "num_batch": 512},
    )

    assert snapshot == {
        "context_length": 4096,
        "gpu_offload_percent": 75,
        "cpu_threads": 8,
        "num_batch": 512,
        "runtime_profile": "explicit_ollama_options",
    }


def test_multi_sample_benchmark_reuses_identical_options(monkeypatch):
    calls = []
    monkeypatch.setattr(
        benchmark,
        "benchmark_ollama",
        lambda tag, options=None: calls.append((tag, dict(options or {}))) or 10.0,
    )

    result = benchmark.benchmark_ollama_samples(
        "model:latest", runs=3, options={"num_ctx": 4096, "num_thread": 8}
    )

    assert result["count"] == 3
    assert calls == [("model:latest", {"num_ctx": 4096, "num_thread": 8})] * 3
