"""Resolve a model name into a downloadable URL + filename.

Accepts three forms for `omm install <model_name>`:
  1. A curated short name (see CURATED_INDEX below), e.g. "tinyllama-1.1b-q4"
  2. An explicit HuggingFace ref "org/repo:filename.gguf"
  3. A direct https:// URL to a .gguf file
"""

from __future__ import annotations

import requests

HF_API = "https://huggingface.co/api/models/{repo_id}"
HF_DOWNLOAD = "https://huggingface.co/{repo_id}/resolve/main/{filename}"

# Small curated index of popular GGUF models. Not exhaustive - `omm update`
# is meant to replace/extend this from a hosted index later.
CURATED_INDEX: dict[str, tuple[str, str]] = {
    "tinyllama-1.1b-q4": (
        "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    ),
    "llama3.1-8b-instruct-q4": (
        "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    ),
    "mistral-7b-instruct-q4": (
        "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    ),
}


class ModelResolutionError(Exception):
    pass


def _list_gguf_files(repo_id: str) -> list[str]:
    resp = requests.get(HF_API.format(repo_id=repo_id), timeout=15)
    resp.raise_for_status()
    siblings = resp.json().get("siblings", [])
    return [s["rfilename"] for s in siblings if s["rfilename"].endswith(".gguf")]


def resolve_model(model_name: str) -> tuple[str, str]:
    """Return (download_url, filename)."""
    if model_name in CURATED_INDEX:
        repo_id, filename = CURATED_INDEX[model_name]
        return HF_DOWNLOAD.format(repo_id=repo_id, filename=filename), filename

    if model_name.startswith("http://") or model_name.startswith("https://"):
        filename = model_name.rsplit("/", 1)[-1].split("?", 1)[0]
        return model_name, filename

    if "/" in model_name:
        if ":" in model_name:
            repo_id, filename = model_name.split(":", 1)
        else:
            repo_id, filename = model_name, None
            candidates = _list_gguf_files(repo_id)
            if not candidates:
                raise ModelResolutionError(f"No .gguf files found in HF repo '{repo_id}'.")
            if len(candidates) > 1:
                raise ModelResolutionError(
                    f"Repo '{repo_id}' has multiple .gguf files, specify one: "
                    f"{repo_id}:<filename>\nOptions: {', '.join(candidates)}"
                )
            filename = candidates[0]
        return HF_DOWNLOAD.format(repo_id=repo_id, filename=filename), filename

    raise ModelResolutionError(
        f"Unknown model '{model_name}'. Use a curated name "
        f"({', '.join(CURATED_INDEX)}), an 'org/repo:file.gguf' ref, or a direct URL."
    )
