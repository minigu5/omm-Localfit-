"""Resumable file downloads with a Rich progress bar."""

from __future__ import annotations

from pathlib import Path

import requests
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

_CHUNK_SIZE = 1024 * 1024


class DownloadError(Exception):
    pass


def download_file(url: str, dest: Path) -> None:
    """Download `url` to `dest`, resuming from a partial .part file if present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest.with_suffix(dest.suffix + ".part")
    resume_pos = part_path.stat().st_size if part_path.exists() else 0

    headers = {"Range": f"bytes={resume_pos}-"} if resume_pos else {}
    resp = requests.get(url, headers=headers, stream=True, timeout=30)

    if resume_pos and resp.status_code == 200:
        # Server ignored the Range request; restart from scratch.
        resume_pos = 0
        mode = "wb"
    elif resp.status_code == 416:
        # Already fully downloaded.
        part_path.rename(dest)
        return
    elif resp.status_code in (200, 206):
        resp.raise_for_status()
        mode = "ab" if resume_pos and resp.status_code == 206 else "wb"
    else:
        raise DownloadError(f"Download failed: HTTP {resp.status_code} for {url}")

    total = int(resp.headers.get("Content-Length", 0)) + resume_pos

    with Progress(
        TextColumn("[cyan]{task.fields[filename]}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("download", total=total or None, completed=resume_pos, filename=dest.name)
        with part_path.open(mode) as f:
            for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))

    part_path.rename(dest)
