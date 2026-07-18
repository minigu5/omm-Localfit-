"""Zero-duplication linker: symlink central .gguf files into LM Studio and
Ollama without copying bytes.

Ollama's on-disk manifest format is not officially documented. The shape
used here (schemaVersion 2, OCI-style config+layers) was captured by
running `ollama create` on a bare GGUF file and inspecting the resulting
manifest/config/blobs directly, and may need updates if Ollama changes it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from omm.gguf import read_gguf_metadata
from omm.hashutil import sha256_file

LMSTUDIO_MODELS_DIR = Path.home() / ".cache" / "lm-studio" / "models"


def ollama_models_dir() -> Path:
    env_dir = os.environ.get("OLLAMA_MODELS")
    if env_dir:
        return Path(env_dir).expanduser()
    return Path.home() / ".ollama" / "models"


def is_lmstudio_installed() -> bool:
    return (Path.home() / ".cache" / "lm-studio").exists()


def is_ollama_installed() -> bool:
    return (Path.home() / ".ollama").exists()


class LinkError(Exception):
    pass


def _symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError as e:
        raise LinkError(
            f"Could not create symlink at {dst}: {e}. "
            "On Windows, enable Developer Mode or run as Administrator."
        ) from e


# --- LM Studio -------------------------------------------------------------
#
# LM Studio only recognizes models laid out as models/<publisher>/<repo>/
# <file>.gguf (mirrors the HuggingFace repo layout) - a flat models/<file>.gguf
# is silently ignored by its scanner. Confirmed against a real LM Studio 0.4.19
# install via its bundled `lms ls` CLI.


def _lmstudio_publisher_repo(repo_id: str | None, filename: str) -> tuple[str, str]:
    if repo_id and "/" in repo_id:
        publisher, repo = repo_id.split("/", 1)
        return publisher, repo
    return "local", Path(filename).stem


def link_lmstudio(gguf_path: Path, repo_id: str | None) -> Path:
    publisher, repo = _lmstudio_publisher_repo(repo_id, gguf_path.name)
    dst = LMSTUDIO_MODELS_DIR / publisher / repo / gguf_path.name
    _symlink(gguf_path, dst)
    return dst


def unlink_lmstudio(filename: str, repo_id: str | None) -> None:
    publisher, repo = _lmstudio_publisher_repo(repo_id, filename)
    dst = LMSTUDIO_MODELS_DIR / publisher / repo / filename
    if dst.is_symlink():
        dst.unlink()
        for parent in (dst.parent, dst.parent.parent):
            try:
                parent.rmdir()
            except OSError:
                break


# --- Ollama ------------------------------------------------------------


def sanitize_ollama_tag(filename: str) -> str:
    """Ollama model names must be lowercase [a-z0-9._-]."""
    name = filename
    if name.lower().endswith(".gguf"):
        name = name[: -len(".gguf")]
    name = name.lower()
    return re.sub(r"[^a-z0-9._-]+", "-", name).strip("-")


def _guess_param_size(filename: str) -> str:
    m = re.search(r"(\d+(?:\.\d+)?)[Bb](?:[-_.]|$)", filename)
    return f"{m.group(1)}B" if m else "unknown"


def _guess_quant(filename: str) -> str:
    m = re.search(r"(Q\d(?:_[A-Z0-9]+)*)", filename, re.IGNORECASE)
    return m.group(1).upper() if m else "unknown"


def link_ollama(gguf_path: Path, model_name: str) -> bool:
    """Link into Ollama. Returns True if the source GGUF has an embedded
    chat template (Ollama reads it from the model blob at runtime), False
    if none was found and the caller should warn the user about it.
    """
    model_sha256 = sha256_file(gguf_path)
    model_digest = f"sha256:{model_sha256}"

    gguf_meta = read_gguf_metadata(gguf_path, {"general.architecture", "tokenizer.chat_template"})
    architecture = gguf_meta.get("general.architecture", "unknown")
    has_chat_template = "tokenizer.chat_template" in gguf_meta

    blobs_dir = ollama_models_dir() / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    model_blob = blobs_dir / f"sha256-{model_sha256}"
    _symlink(gguf_path, model_blob)

    # Mirrors the config produced by `ollama create` for a bare GGUF (no
    # Modelfile TEMPLATE override): a single model layer, config mediaType
    # "application/vnd.docker.container.image.v1+json", and model_family
    # taken from the GGUF's own general.architecture field. With this shape,
    # Ollama reads tokenizer.chat_template straight from the GGUF at
    # inference time - no separate template layer is needed or created by
    # `ollama create` itself in this case.
    config = {
        "model_format": "gguf",
        "model_family": architecture,
        "model_families": [architecture],
        "model_type": _guess_param_size(gguf_path.name),
        "file_type": _guess_quant(gguf_path.name),
        "architecture": "amd64",
        "os": "linux",
        "rootfs": {"type": "layers", "diff_ids": [model_digest]},
    }
    config_bytes = json.dumps(config).encode()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    config_blob = blobs_dir / f"sha256-{config_sha256}"
    config_blob.write_bytes(config_bytes)

    manifest = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
        "config": {
            "mediaType": "application/vnd.docker.container.image.v1+json",
            "digest": f"sha256:{config_sha256}",
            "size": len(config_bytes),
        },
        "layers": [
            {
                "mediaType": "application/vnd.ollama.image.model",
                "digest": model_digest,
                "size": gguf_path.stat().st_size,
            }
        ],
    }

    manifest_dir = (
        ollama_models_dir() / "manifests" / "registry.ollama.ai" / "library" / model_name
    )
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "latest").write_text(json.dumps(manifest, indent=2))

    return has_chat_template


def unlink_ollama(model_name: str) -> None:
    manifest_path = (
        ollama_models_dir()
        / "manifests"
        / "registry.ollama.ai"
        / "library"
        / model_name
        / "latest"
    )
    if not manifest_path.exists():
        return

    manifest = json.loads(manifest_path.read_text())
    blobs_dir = ollama_models_dir() / "blobs"

    config_digest = manifest["config"]["digest"].replace(":", "-")
    config_blob = blobs_dir / config_digest
    if config_blob.exists():
        config_blob.unlink()

    for layer in manifest["layers"]:
        layer_digest = layer["digest"].replace(":", "-")
        layer_blob = blobs_dir / layer_digest
        if layer_blob.is_symlink():
            layer_blob.unlink()

    manifest_path.unlink()
    try:
        manifest_path.parent.rmdir()
    except OSError:
        pass
