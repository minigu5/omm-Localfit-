from typer.testing import CliRunner

from omm import cli, linker

runner = CliRunner()


def test_autoremove_reports_zero_when_nothing_broken(monkeypatch):
    monkeypatch.setattr(linker, "is_lmstudio_installed", lambda: True)
    monkeypatch.setattr(linker, "is_ollama_installed", lambda: True)
    monkeypatch.setattr(linker, "autoremove_lmstudio", lambda: 0)
    monkeypatch.setattr(linker, "autoremove_ollama", lambda: (0, 0))

    result = runner.invoke(cli.app, ["autoremove"])

    assert result.exit_code == 0, result.stdout
    assert "No broken symlinks found" in result.stdout


def test_autoremove_reports_counts_from_both_engines(monkeypatch):
    monkeypatch.setattr(linker, "is_lmstudio_installed", lambda: True)
    monkeypatch.setattr(linker, "is_ollama_installed", lambda: True)
    monkeypatch.setattr(linker, "autoremove_lmstudio", lambda: 2)
    monkeypatch.setattr(linker, "autoremove_ollama", lambda: (1, 1))

    result = runner.invoke(cli.app, ["autoremove"])

    assert result.exit_code == 0, result.stdout
    assert "Removed 2 broken LM Studio symlink(s)" in result.stdout
    assert "1 broken Ollama blob(s)" in result.stdout


def test_autoremove_skips_uninstalled_engines(monkeypatch):
    monkeypatch.setattr(linker, "is_lmstudio_installed", lambda: False)
    monkeypatch.setattr(linker, "is_ollama_installed", lambda: False)
    lmstudio_calls = []
    ollama_calls = []
    monkeypatch.setattr(linker, "autoremove_lmstudio", lambda: lmstudio_calls.append(1) or 0)
    monkeypatch.setattr(linker, "autoremove_ollama", lambda: ollama_calls.append(1) or (0, 0))

    result = runner.invoke(cli.app, ["autoremove"])

    assert result.exit_code == 0, result.stdout
    assert lmstudio_calls == []
    assert ollama_calls == []
