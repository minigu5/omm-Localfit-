from typer.testing import CliRunner

from omm import cli, registry

runner = CliRunner()


def test_uninstall_all_removes_every_registered_model(isolated_omm_home, monkeypatch):
    for filename in ("a.gguf", "b.gguf"):
        (cli.MODELS_DIR / filename).write_bytes(b"fake-gguf")
    registry.save_registry(
        {
            "a.gguf": {"linked": {"lmstudio": False, "ollama": False}},
            "b.gguf": {"linked": {"lmstudio": False, "ollama": False}},
        }
    )
    monkeypatch.setattr(cli, "_ask_confirm", lambda message, default=False: True)

    result = runner.invoke(cli.app, ["uninstall", "all"])

    assert result.exit_code == 0, result.stdout
    assert registry.load_registry() == {}
    assert not (cli.MODELS_DIR / "a.gguf").exists()
    assert not (cli.MODELS_DIR / "b.gguf").exists()


def test_uninstall_all_cancelled_leaves_registry_untouched(isolated_omm_home, monkeypatch):
    registry.save_registry({"a.gguf": {"linked": {"lmstudio": False, "ollama": False}}})
    monkeypatch.setattr(cli, "_ask_confirm", lambda message, default=False: False)

    result = runner.invoke(cli.app, ["uninstall", "all"])

    assert result.exit_code == 0, result.stdout
    assert registry.load_registry() != {}


def test_uninstall_all_with_empty_registry_reports_nothing_to_do(isolated_omm_home):
    result = runner.invoke(cli.app, ["uninstall", "all"])

    assert result.exit_code == 0, result.stdout
    assert "No models installed" in result.stdout


def test_remove_accepts_filename_without_gguf_suffix(isolated_omm_home):
    filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    dest = cli.MODELS_DIR / filename
    dest.write_bytes(b"fake-gguf")
    registry.save_registry({filename: {"linked": {"lmstudio": False, "ollama": False}}})

    result = runner.invoke(cli.app, ["uninstall", "tinyllama-1.1b-chat-v1.0.Q4_K_M"])

    assert result.exit_code == 0, result.stdout
    assert f"Removed {filename}" in result.stdout
    assert registry.load_registry() == {}
    assert not dest.exists()


def test_remove_cleans_up_orphaned_part_file(isolated_omm_home):
    part = cli.MODELS_DIR / "orphan.gguf.part"
    cli.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    part.write_bytes(b"partial")

    result = runner.invoke(cli.app, ["uninstall", "orphan.gguf"])

    assert result.exit_code == 0, result.stdout
    assert "orphan.gguf" in result.stdout
    assert not part.exists()


def test_remove_cleans_up_unregistered_complete_download(isolated_omm_home):
    dest = cli.MODELS_DIR / "orphan.gguf"
    cli.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"complete-but-unregistered")

    result = runner.invoke(cli.app, ["uninstall", "orphan.gguf"])

    assert result.exit_code == 0, result.stdout
    assert not dest.exists()


def test_remove_still_errors_when_nothing_on_disk(isolated_omm_home):
    result = runner.invoke(cli.app, ["uninstall", "nothing-here.gguf"])

    assert result.exit_code == 1
    assert "is not installed via omm" in result.stdout
