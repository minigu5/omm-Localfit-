"""Opt-in, best-effort telemetry. Never raises, never blocks the CLI."""

from __future__ import annotations

from typing import Any

import requests

from omm.config import load_config


def send_event(event: dict[str, Any]) -> None:
    config = load_config()
    if not config.get("telemetry_opt_in"):
        return
    endpoint = config.get("telemetry_endpoint")
    if not endpoint:
        return
    try:
        requests.post(endpoint, json=event, timeout=5)
    except requests.RequestException:
        pass
