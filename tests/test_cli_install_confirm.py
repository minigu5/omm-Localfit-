from typer.testing import CliRunner

from omm import cli, linker
from omm.hub import ResolvedModel

runner = CliRunner()


def _stub_successful_install(monkeypatch, isolated_omm_home):
    filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    monkeypatch.setattr(
        cli,
        "resolve_model",
        lambda name: ResolvedModel(url="https://example.com/x.gguf", filename=filename, repo_id="org/repo"),
    )

    def fake_download(url, dest):
        dest.write_bytes(b"fake-gguf")

    monkeypatch.setattr(cli, "download_file", fake_download)
    monkeypatch.setattr(cli, "sha256_file", lambda dest: "deadbeef")
    monkeypatch.setattr(linker, "is_lmstudio_installed", lambda: False)
    monkeypatch.setattr(linker, "is_ollama_installed", lambda: True)
    monkeypatch.setattr(linker, "link_ollama", lambda dest, tag: True)
    monkeypatch.setattr(linker, "sanitize_ollama_tag", lambda filename: "tinyllama")
    return filename


def test_install_runs_benchmark_and_telemetry_on_yes(isolated_omm_home, monkeypatch):
    filename = _stub_successful_install(monkeypatch, isolated_omm_home)
    monkeypatch.setattr(cli, "_ask_confirm", lambda message, default=False: True)
    monkeypatch.setattr(cli.benchmark, "benchmark_ollama", lambda tag: 42.0)
    sent = []
    monkeypatch.setattr(cli.telemetry, "send_event", lambda event, force=False: sent.append((event, force)))

    result = runner.invoke(cli.app, ["install", "tinyllama-1.1b-q4"])

    assert result.exit_code == 0, result.stdout
    assert "42.0" in result.stdout or "42" in result.stdout
    assert len(sent) == 1
    assert sent[0][1] is True


def test_install_runs_benchmark_but_skips_upload_on_no(isolated_omm_home, monkeypatch):
    _stub_successful_install(monkeypatch, isolated_omm_home)
    monkeypatch.setattr(cli, "_ask_confirm", lambda message, default=False: False)
    bench_calls = []
    monkeypatch.setattr(cli.benchmark, "benchmark_ollama", lambda tag: bench_calls.append(tag) or 42.0)
    sent = []
    monkeypatch.setattr(cli.telemetry, "send_event", lambda event, force=False: sent.append((event, force)))

    result = runner.invoke(cli.app, ["install", "tinyllama-1.1b-q4"])

    assert result.exit_code == 0, result.stdout
    assert bench_calls == ["tinyllama"] * 3
    assert sent == []


def test_ask_confirm_uses_questionary_with_auto_enter(monkeypatch):
    captured = {}

    class FakeQuestion:
        def ask(self):
            return True

    def fake_confirm(message, default=False, auto_enter=True):
        captured["message"] = message
        captured["default"] = default
        captured["auto_enter"] = auto_enter
        return FakeQuestion()

    monkeypatch.setattr(cli.questionary, "confirm", fake_confirm)

    result = cli._ask_confirm("질문?", default=False)

    assert result is True
    assert captured == {"message": "질문?", "default": False, "auto_enter": True}
