"""Central paths and user config (~/.omm)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OMM_HOME = Path.home() / ".omm"
MODELS_DIR = OMM_HOME / "models"
CONFIG_PATH = OMM_HOME / "config.json"
REGISTRY_PATH = OMM_HOME / "models.json"
RULES_PATH = OMM_HOME / "rules.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "telemetry_opt_in": False,
    "telemetry_endpoint": "https://localfit-8ab57-default-rtdb.firebaseio.com/telemetry.json",
    "rules_url": None,
    "default_engine": None,
}


def ensure_omm_home() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any]:
    ensure_omm_home()
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)
    data = json.loads(CONFIG_PATH.read_text())
    return {**DEFAULT_CONFIG, **data}


def save_config(config: dict[str, Any]) -> None:
    ensure_omm_home()
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
