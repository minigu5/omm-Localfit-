"""Central registry of models installed via omm (~/.omm/models.json)."""

from __future__ import annotations

import json
from typing import Any

from omm.config import REGISTRY_PATH, ensure_omm_home


def load_registry() -> dict[str, Any]:
    ensure_omm_home()
    if not REGISTRY_PATH.exists():
        return {}
    return json.loads(REGISTRY_PATH.read_text())


def save_registry(registry: dict[str, Any]) -> None:
    ensure_omm_home()
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))


def upsert_entry(filename: str, **fields: Any) -> None:
    registry = load_registry()
    entry = registry.setdefault(filename, {"linked": {}})
    entry.update({k: v for k, v in fields.items() if k != "linked"})
    if "linked" in fields:
        entry["linked"].update(fields["linked"])
    save_registry(registry)


def remove_entry(filename: str) -> None:
    registry = load_registry()
    registry.pop(filename, None)
    save_registry(registry)
