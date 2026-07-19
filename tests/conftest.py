"""Shared pytest fixtures for the omm test suite."""

from __future__ import annotations

import pytest

from omm import cli, config, predictor, registry


@pytest.fixture
def isolated_omm_home(tmp_path, monkeypatch):
    """Redirect all of omm's ~/.omm paths into a throwaway tmp_path so
    tests never touch (or depend on) the real user home directory."""
    home = tmp_path / ".omm"
    models_dir = home / "models"

    monkeypatch.setattr(config, "OMM_HOME", home)
    monkeypatch.setattr(config, "MODELS_DIR", models_dir)
    monkeypatch.setattr(config, "CONFIG_PATH", home / "config.json")
    monkeypatch.setattr(config, "REGISTRY_PATH", home / "models.json")
    monkeypatch.setattr(config, "RULES_PATH", home / "rules.json")
    monkeypatch.setattr(config, "RECOMMEND_MODEL_PATH", home / "recommend-model.json")

    monkeypatch.setattr(registry, "REGISTRY_PATH", config.REGISTRY_PATH)
    monkeypatch.setattr(cli, "MODELS_DIR", models_dir)
    monkeypatch.setattr(predictor, "RECOMMEND_MODEL_PATH", config.RECOMMEND_MODEL_PATH)

    config.ensure_omm_home()
    return home
