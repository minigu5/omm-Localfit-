"""CI-only script: pull a fresh pool of candidate GGUF models from the
HuggingFace Hub so `omm recommend` reflects newly published models without
an omm release. Output feeds into scripts/train_model.py's artifact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from omm.featurize import parse_param_count_billions  # noqa: E402
from omm.hub import CURATED_INDEX  # noqa: E402
from omm.linker import sanitize_ollama_tag  # noqa: E402
from omm.search import _claims_fake_provenance, pick_gguf_file  # noqa: E402

HF_SEARCH_URL = "https://huggingface.co/api/models"
CANDIDATE_LIMIT = 30
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "published" / "candidates.json"


def fetch_trending_candidates() -> list[dict]:
    resp = requests.get(
        HF_SEARCH_URL,
        params={
            "filter": "gguf",
            "pipeline_tag": "text-generation",
            "sort": "downloads",
            "direction": -1,
            "limit": CANDIDATE_LIMIT,
            "full": "true",
        },
        timeout=30,
    )
    resp.raise_for_status()

    candidates = []
    for model in resp.json():
        if _claims_fake_provenance(model["id"]):
            continue
        filename = pick_gguf_file(model.get("siblings", []))
        if filename is None:
            continue
        # Skip repos whose param count we can't parse from id/filename -
        # they'd otherwise fall back to 0 and get mis-ranked as tiny/fast.
        if parse_param_count_billions(f"{model['id']} {filename}") is None:
            continue
        candidates.append(
            {
                "name": sanitize_ollama_tag(model["id"]),
                "repo_id": model["id"],
                "filename": filename,
                "description": f"{model.get('downloads', 0):,} downloads on HuggingFace",
            }
        )
    return candidates


def curated_candidates() -> list[dict]:
    return [
        {"name": name, "repo_id": repo_id, "filename": filename, "description": "Curated default"}
        for name, (repo_id, filename) in CURATED_INDEX.items()
    ]


def main() -> None:
    try:
        trending = fetch_trending_candidates()
    except requests.RequestException as e:
        print(f"Warning: HF fetch failed ({e}), using curated candidates only.")
        trending = []

    seen_repo_ids = set()
    candidates = []
    for c in curated_candidates() + trending:
        if c["repo_id"] in seen_repo_ids:
            continue
        seen_repo_ids.add(c["repo_id"])
        candidates.append(c)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(candidates, indent=2))
    print(f"Wrote {OUTPUT_PATH} ({len(candidates)} candidates, {len(trending)} from HF trending)")


if __name__ == "__main__":
    main()
