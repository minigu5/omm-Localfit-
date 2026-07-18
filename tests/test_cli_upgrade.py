import subprocess

from typer.testing import CliRunner

from omm import cli

runner = CliRunner()


def test_install_spec_uses_bare_repo_url_on_darwin(monkeypatch):
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")

    assert cli._install_spec() == cli.REPO_URL


def test_install_spec_adds_nvidia_extra_on_non_darwin(monkeypatch):
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")

    assert cli._install_spec() == f"omm[nvidia] @ {cli.REPO_URL}"


def test_upgrade_reinstalls_via_pipx_then_refreshes_data(monkeypatch):
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    calls = []
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda args, **kwargs: calls.append(args)
        or subprocess.CompletedProcess(args, returncode=0, stdout="", stderr=""),
    )
    refresh_calls = []
    monkeypatch.setattr(cli, "_refresh_data", lambda: refresh_calls.append(1))

    result = runner.invoke(cli.app, ["upgrade"])

    assert result.exit_code == 0, result.stdout
    assert calls == [["pipx", "install", "--force", cli.REPO_URL]]
    assert refresh_calls == [1]
    assert "reinstalled" in result.stdout.lower()


def test_upgrade_reports_error_when_pipx_missing(monkeypatch):
    def _raise(*args, **kwargs):
        raise FileNotFoundError("pipx")

    monkeypatch.setattr(cli.subprocess, "run", _raise)
    refresh_calls = []
    monkeypatch.setattr(cli, "_refresh_data", lambda: refresh_calls.append(1))

    result = runner.invoke(cli.app, ["upgrade"])

    assert result.exit_code == 1
    assert "pipx not found" in result.stdout
    assert refresh_calls == []


def test_upgrade_reports_error_and_skips_data_refresh_on_pipx_failure(monkeypatch):
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args, returncode=1, stdout="", stderr="boom"
        ),
    )
    refresh_calls = []
    monkeypatch.setattr(cli, "_refresh_data", lambda: refresh_calls.append(1))

    result = runner.invoke(cli.app, ["upgrade"])

    assert result.exit_code == 1
    assert "boom" in result.stdout
    assert refresh_calls == []
