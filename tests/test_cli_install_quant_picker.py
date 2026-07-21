from typer.testing import CliRunner

from omm import cli, linker
from omm.hardware import HardwareInfo
from omm.hub import AmbiguousModelError, ResolvedModel

runner = CliRunner()

_HARDWARE = HardwareInfo(
    os_name="Linux",
    os_version="",
    cpu="",
    ram_total_gb=6.0,
    ram_available_gb=6.0,
    unified_memory=False,
    gpu_name=None,
    vram_total_gb=None,
    vram_free_gb=None,
)


def _stub_download_and_links(monkeypatch):
    def fake_download(url, dest):
        dest.write_bytes(b"fake-gguf")

    monkeypatch.setattr(cli, "download_file", fake_download)
    monkeypatch.setattr(cli, "sha256_file", lambda dest: "deadbeef")
    monkeypatch.setattr(linker, "is_lmstudio_installed", lambda: False)
    monkeypatch.setattr(linker, "is_ollama_installed", lambda: False)


def test_install_with_repo_only_prompts_quant_and_recurses_with_choice(isolated_omm_home, monkeypatch):
    _stub_download_and_links(monkeypatch)
    repo_id = "TheBloke/Llama-2-7B-GGUF"
    chosen_filename = "llama-2-7b.Q4_K_M.gguf"
    candidates = ["llama-2-7b.Q2_K.gguf", chosen_filename, "llama-2-7b.Q8_0.gguf"]

    calls = []

    def fake_resolve(name):
        calls.append(name)
        if name == repo_id:
            raise AmbiguousModelError(repo_id, candidates)
        return ResolvedModel(url="https://example.com/x.gguf", filename=chosen_filename, repo_id=repo_id)

    monkeypatch.setattr(cli, "resolve_model", fake_resolve)
    monkeypatch.setattr(cli, "scan_hardware", lambda: _HARDWARE)
    # questionary.select(...) is evaluated eagerly as an argument to
    # _ask_select, so it must be stubbed too - constructing a real Question
    # tries to open a console, which CI runners (esp. Windows) don't have.
    monkeypatch.setattr(cli.questionary, "select", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_ask_select", lambda question: chosen_filename)

    result = runner.invoke(cli.app, ["install", repo_id])

    assert result.exit_code == 0, result.stdout
    assert calls == [repo_id, f"{repo_id}:{chosen_filename}"]
    assert "Installed" in result.stdout


def test_install_cancels_cleanly_when_quant_prompt_is_escaped(isolated_omm_home, monkeypatch):
    repo_id = "TheBloke/Llama-2-7B-GGUF"
    candidates = ["llama-2-7b.Q4_K_M.gguf"]

    def fake_resolve(name):
        raise AmbiguousModelError(repo_id, candidates)

    monkeypatch.setattr(cli, "resolve_model", fake_resolve)
    monkeypatch.setattr(cli, "scan_hardware", lambda: _HARDWARE)
    monkeypatch.setattr(cli.questionary, "select", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_ask_select", lambda question: None)

    result = runner.invoke(cli.app, ["install", repo_id])

    assert result.exit_code == 0
    assert "Cancelled" in result.stdout
