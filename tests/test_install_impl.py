import threading
import time
from types import SimpleNamespace

import pytest

from omm import cli
from omm.downloader import DownloadCancelled
from omm.hub import ResolvedModel


def _resolved(filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"):
    return ResolvedModel(url="https://example.com/x.gguf", filename=filename, repo_id="org/repo")


def _stub_common(monkeypatch, ollama=True, lmstudio=False):
    monkeypatch.setattr(cli, "sha256_file", lambda dest: "deadbeef")
    monkeypatch.setattr(cli.linker, "is_lmstudio_installed", lambda: lmstudio)
    monkeypatch.setattr(cli.linker, "is_ollama_installed", lambda: ollama)
    monkeypatch.setattr(cli.linker, "link_ollama", lambda dest, tag: ollama)
    monkeypatch.setattr(cli.linker, "sanitize_ollama_tag", lambda filename: "tinyllama")


def test_skip_unfit_returns_outcome_without_prompting_or_downloading(isolated_omm_home, monkeypatch):
    monkeypatch.setattr(cli.predictor, "load_cached_model", lambda: {"trees": [{}]})
    monkeypatch.setattr(cli, "scan_hardware", lambda: object())
    monkeypatch.setattr(cli.predictor, "predict_speed", lambda trees, hw, candidate: 0.0)
    monkeypatch.setattr(
        cli, "_ask_confirm", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no prompt"))
    )
    download_calls = []
    monkeypatch.setattr(cli, "download_file", lambda *a, **k: download_calls.append(a))

    outcome = cli._install_impl(_resolved(), skip_unfit=True)

    assert outcome.skipped_unfit is True
    assert outcome.linked == {}
    assert download_calls == []


def test_auto_upload_skips_confirm_prompt_and_sends_telemetry(isolated_omm_home, monkeypatch):
    monkeypatch.setattr(cli.predictor, "load_cached_model", lambda: None)
    monkeypatch.setattr(cli, "download_file", lambda url, dest: dest.write_bytes(b"x"))
    _stub_common(monkeypatch)
    monkeypatch.setattr(
        cli, "_ask_confirm", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no prompt"))
    )
    monkeypatch.setattr(cli.benchmark, "benchmark_ollama", lambda tag: 55.0)
    monkeypatch.setattr(cli.telemetry, "send_event", lambda event, force=False: True)

    outcome = cli._install_impl(_resolved(), auto_upload=True)

    assert outcome.tokens_per_sec == 55.0
    assert outcome.telemetry_sent is True


def test_stop_event_set_before_download_raises_contribution_stopped(isolated_omm_home, monkeypatch):
    monkeypatch.setattr(cli.predictor, "load_cached_model", lambda: None)

    def fake_download(url, dest, stop_check=None):
        assert stop_check is not None
        raise DownloadCancelled("interrupted")

    monkeypatch.setattr(cli, "download_file", fake_download)
    stop_event = threading.Event()
    stop_event.set()

    with pytest.raises(cli.ContributionStopped) as exc_info:
        cli._install_impl(_resolved(), stop_event=stop_event)

    assert exc_info.value.filename == "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"


def test_stop_event_set_during_benchmark_raises_contribution_stopped(isolated_omm_home, monkeypatch):
    monkeypatch.setattr(cli.predictor, "load_cached_model", lambda: None)
    monkeypatch.setattr(
        cli, "download_file", lambda url, dest, stop_check=None: dest.write_bytes(b"x")
    )
    _stub_common(monkeypatch)

    def slow_benchmark(tag):
        time.sleep(2)
        return 10.0

    monkeypatch.setattr(cli.benchmark, "benchmark_ollama", slow_benchmark)
    stop_event = threading.Event()
    threading.Timer(0.05, stop_event.set).start()

    with pytest.raises(cli.ContributionStopped):
        cli._install_impl(_resolved(), stop_event=stop_event)


def test_plain_install_path_unaffected_by_stop_event_none(isolated_omm_home, monkeypatch):
    monkeypatch.setattr(cli.predictor, "load_cached_model", lambda: None)
    calls = []

    def fake_download(url, dest):
        calls.append("no-kwargs")
        dest.write_bytes(b"x")

    monkeypatch.setattr(cli, "download_file", fake_download)
    _stub_common(monkeypatch)
    monkeypatch.setattr(cli, "_ask_confirm", lambda *a, **k: True)
    monkeypatch.setattr(cli.benchmark, "benchmark_ollama", lambda tag: 10.0)
    monkeypatch.setattr(cli.telemetry, "send_event", lambda event, force=False: True)

    outcome = cli._install_impl(_resolved())

    assert calls == ["no-kwargs"]
    assert outcome.tokens_per_sec == 10.0


def test_benchmark_always_runs_but_upload_needs_confirm(isolated_omm_home, monkeypatch):
    monkeypatch.setattr(cli.predictor, "load_cached_model", lambda: None)
    monkeypatch.setattr(cli, "download_file", lambda url, dest: dest.write_bytes(b"x"))
    _stub_common(monkeypatch)
    monkeypatch.setattr(cli, "_ask_confirm", lambda *a, **k: False)
    bench_calls = []
    monkeypatch.setattr(
        cli.benchmark, "benchmark_ollama", lambda tag: bench_calls.append(tag) or 42.0
    )
    sent = []
    monkeypatch.setattr(
        cli.telemetry, "send_event", lambda event, force=False: sent.append((event, force))
    )

    outcome = cli._install_impl(_resolved())

    assert bench_calls == ["tinyllama"]
    assert outcome.tokens_per_sec == 42.0
    assert sent == []
    assert outcome.telemetry_sent is False


def test_auto_calibrate_runs_silently_when_cached_model_available(isolated_omm_home, monkeypatch):
    monkeypatch.setattr(
        cli.predictor, "load_cached_model", lambda: {"trees": [{"leaf": True, "value": 20.0}]}
    )
    hw_stub = SimpleNamespace(
        ram_total_gb=16.0, vram_total_gb=None, unified_memory=False, gpu_tflops=None
    )
    monkeypatch.setattr(cli, "scan_hardware", lambda: hw_stub)
    monkeypatch.setattr(
        cli.predictor,
        "predict_speed_interval",
        lambda *args, **kwargs: (20.0, 20.0, 20.0),
    )
    monkeypatch.setattr(cli, "download_file", lambda url, dest: dest.write_bytes(b"x"))
    _stub_common(monkeypatch)
    monkeypatch.setattr(cli, "_ask_confirm", lambda *a, **k: True)
    monkeypatch.setattr(cli.benchmark, "benchmark_ollama", lambda tag: 30.0)
    monkeypatch.setattr(cli.telemetry, "send_event", lambda event, force=False: True)
    recorded = {}
    monkeypatch.setattr(
        cli.calibration,
        "record_calibration",
        lambda hardware, **kwargs: recorded.update(kwargs) or 1.5,
    )

    cli._install_impl(_resolved())

    assert recorded["measured_tokens_per_sec"] == 30.0
    assert recorded["predicted_tokens_per_sec"] == 20.0


def test_resolve_upload_decision_always_skips_prompt(isolated_omm_home):
    cli.config_mod.update_config(telemetry_send_policy="always")

    assert cli._resolve_upload_decision("prompt") is True


def test_resolve_upload_decision_never_skips_prompt(isolated_omm_home, monkeypatch):
    cli.config_mod.update_config(telemetry_send_policy="never")
    monkeypatch.setattr(
        cli, "_ask_confirm", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no prompt"))
    )

    assert cli._resolve_upload_decision("prompt") is False


def test_resolve_upload_decision_ask_falls_back_to_confirm(isolated_omm_home, monkeypatch):
    cli.config_mod.update_config(telemetry_send_policy="ask")
    monkeypatch.setattr(cli, "_ask_confirm", lambda message, **k: message == "prompt")

    assert cli._resolve_upload_decision("prompt") is True
    assert cli._resolve_upload_decision("other") is False


def test_install_auto_uploads_without_confirm_when_policy_always(isolated_omm_home, monkeypatch):
    cli.config_mod.update_config(telemetry_send_policy="always")
    monkeypatch.setattr(cli.predictor, "load_cached_model", lambda: None)
    monkeypatch.setattr(cli, "download_file", lambda url, dest: dest.write_bytes(b"x"))
    _stub_common(monkeypatch)
    monkeypatch.setattr(
        cli, "_ask_confirm", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no prompt"))
    )
    monkeypatch.setattr(cli.benchmark, "benchmark_ollama", lambda tag: 42.0)
    sent = []
    monkeypatch.setattr(
        cli.telemetry, "send_event", lambda event, force=False: sent.append(event) or True
    )

    outcome = cli._install_impl(_resolved())

    assert outcome.telemetry_sent is True
    assert sent


def test_install_never_uploads_without_confirm_when_policy_never(isolated_omm_home, monkeypatch):
    cli.config_mod.update_config(telemetry_send_policy="never")
    monkeypatch.setattr(cli.predictor, "load_cached_model", lambda: None)
    monkeypatch.setattr(cli, "download_file", lambda url, dest: dest.write_bytes(b"x"))
    _stub_common(monkeypatch)
    monkeypatch.setattr(
        cli, "_ask_confirm", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no prompt"))
    )
    monkeypatch.setattr(cli.benchmark, "benchmark_ollama", lambda tag: 42.0)
    monkeypatch.setattr(
        cli.telemetry, "send_event", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no send"))
    )

    outcome = cli._install_impl(_resolved())

    assert outcome.telemetry_sent is False


def test_report_telemetry_includes_quality_fields_when_provided(isolated_omm_home, monkeypatch):
    monkeypatch.setattr(
        cli, "scan_hardware",
        lambda: SimpleNamespace(ram_total_gb=16.0, vram_total_gb=None, unified_memory=False, gpu_tflops=None),
    )
    sent = []
    monkeypatch.setattr(
        cli.telemetry, "send_event", lambda event, force=False: sent.append(event) or True
    )

    cli._report_telemetry(
        "small:latest",
        "org/small",
        42.5,
        size_bytes=123,
        sample_count=3,
        speed_min=40.0,
        speed_max=45.0,
        quality={"pack_id": "pack-1", "pack_version": "1.1.0", "correct": 6, "total": 8, "accuracy": 0.75},
    )

    event = sent[0]
    assert event["model_size_bytes"] == 123
    assert event["sample_count"] == 3
    assert event["tokens_per_sec_min"] == 40.0
    assert event["tokens_per_sec_max"] == 45.0
    assert event["quality_pack_id"] == "pack-1"
    assert event["quality_correct"] == 6
    assert event["quality_total"] == 8
    assert event["quality_accuracy"] == 0.75


def test_report_telemetry_omits_quality_fields_by_default(isolated_omm_home, monkeypatch):
    monkeypatch.setattr(
        cli, "scan_hardware",
        lambda: SimpleNamespace(ram_total_gb=16.0, vram_total_gb=None, unified_memory=False, gpu_tflops=None),
    )
    sent = []
    monkeypatch.setattr(
        cli.telemetry, "send_event", lambda event, force=False: sent.append(event) or True
    )

    cli._report_telemetry("model.gguf", "org/repo", 10.0)

    assert "quality_pack_id" not in sent[0]
    assert sent[0]["sample_count"] == 1
