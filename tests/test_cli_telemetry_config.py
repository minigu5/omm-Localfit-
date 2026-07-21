from typer.testing import CliRunner

from omm import cli, config

runner = CliRunner()


def test_telemetry_requires_explicit_endpoint_before_opt_in(isolated_omm_home):
    result = runner.invoke(cli.app, ["setting", "telemetry", "--enable"])

    assert result.exit_code == 1
    assert config.load_config()["telemetry_send_policy"] == "ask"


def test_telemetry_accepts_local_self_hosted_endpoint(isolated_omm_home):
    result = runner.invoke(
        cli.app,
        [
            "setting",
            "telemetry",
            "--endpoint",
            "http://127.0.0.1:8000/v1/benchmarks",
            "--enable",
        ],
    )

    assert result.exit_code == 0, result.stdout
    saved = config.load_config()
    assert saved["telemetry_backend"] == "self_hosted"
    assert saved["telemetry_send_policy"] == "always"


def test_telemetry_ask_resets_policy_to_ask(isolated_omm_home):
    config.update_config(telemetry_send_policy="always", telemetry_endpoint="https://example.com")

    result = runner.invoke(cli.app, ["setting", "telemetry", "--ask"])

    assert result.exit_code == 0, result.stdout
    assert config.load_config()["telemetry_send_policy"] == "ask"


def test_telemetry_disable_sets_never_policy(isolated_omm_home):
    result = runner.invoke(cli.app, ["setting", "telemetry", "--disable"])

    assert result.exit_code == 0, result.stdout
    assert config.load_config()["telemetry_send_policy"] == "never"


def test_telemetry_rejects_multiple_policy_flags(isolated_omm_home):
    result = runner.invoke(cli.app, ["setting", "telemetry", "--enable", "--disable"])

    assert result.exit_code == 1
    assert "only one" in result.stdout.lower()
