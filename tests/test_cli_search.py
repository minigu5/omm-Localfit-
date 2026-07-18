from typer.testing import CliRunner

from omm import cli, search as search_mod

runner = CliRunner()


def test_search_groups_results_by_family(monkeypatch):
    monkeypatch.setattr(cli, "load_config", lambda: {"model_url": None})
    monkeypatch.setattr(
        search_mod,
        "local_candidate_pool",
        lambda model_url: [
            {
                "name": "tinyllama-1.1b-q4",
                "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                "description": "Curated default",
            },
            {
                "name": "mistral-7b-instruct-q4",
                "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                "description": "Curated default",
            },
        ],
    )
    monkeypatch.setattr(search_mod, "search_huggingface", lambda query, **kwargs: [])

    result = runner.invoke(cli.app, ["search", "q4"])

    assert result.exit_code == 0, result.stdout
    assert "==> TinyLlama" in result.stdout
    assert "==> Mistral" in result.stdout
    assert "tinyllama-1.1b-q4" in result.stdout
    assert "mistral-7b-instruct-q4" in result.stdout


def test_search_exits_nonzero_when_nothing_matches(monkeypatch):
    monkeypatch.setattr(cli, "load_config", lambda: {"model_url": None})
    monkeypatch.setattr(search_mod, "local_candidate_pool", lambda model_url: [])
    monkeypatch.setattr(search_mod, "search_huggingface", lambda query, **kwargs: [])

    result = runner.invoke(cli.app, ["search", "nonexistent-xyz"])

    assert result.exit_code == 1
    assert "No models found" in result.stdout
