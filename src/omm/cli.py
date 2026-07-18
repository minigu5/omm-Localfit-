"""omm CLI entry point (apt/brew-style command routing)."""

import platform
import subprocess
from datetime import datetime, timezone

import questionary
import requests
import typer
from prompt_toolkit.keys import Keys
from rich.console import Console
from rich.table import Table

from omm import benchmark, linker, predictor, registry, rules as rules_mod, search as search_mod, telemetry
from omm.completion import complete_install_name, complete_remove_filename
from omm.config import MODELS_DIR, load_config
from omm.downloader import DownloadError, download_file
from omm.hardware import scan_hardware
from omm.hashutil import sha256_file
from omm.hub import ModelResolutionError, resolve_model

app = typer.Typer(
    name="omm",
    help="Open source Model Manager - package manager for local LLMs (GGUF).",
    no_args_is_help=True,
)
console = Console()

REPO_URL = "git+https://github.com/minigu5/Localfit.git"


def _install_spec() -> str:
    """NVIDIA VRAM detection is dead weight on Mac (no NVIDIA GPUs since
    2016) - only pull that extra in on other platforms, mirroring
    install.sh."""
    if platform.system() == "Darwin":
        return REPO_URL
    return f"omm[nvidia] @ {REPO_URL}"


@app.command()
def scan() -> None:
    """Scan current PC hardware (RAM, VRAM, OS) and print a summary table."""
    info = scan_hardware()

    table = Table(title="omm hardware scan")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("OS", f"{info.os_name} {info.os_version}")
    table.add_row("CPU", info.cpu)
    table.add_row("RAM (total)", f"{info.ram_total_gb:.1f} GB")
    table.add_row("RAM (available)", f"{info.ram_available_gb:.1f} GB")

    if info.unified_memory:
        table.add_row("Memory type", "Unified (Apple Silicon)")
        table.add_row("GPU", info.gpu_name or "Unknown")
    elif info.gpu_name:
        table.add_row("GPU", info.gpu_name)
        table.add_row("VRAM (total)", f"{info.vram_total_gb:.1f} GB")
        table.add_row("VRAM (free)", f"{info.vram_free_gb:.1f} GB")
    else:
        table.add_row("GPU", "None detected (no NVIDIA GPU found)")

    console.print(table)


def _refresh_data() -> None:
    """Unconditionally re-fetch rules.json and recommend-model.json from
    their configured URLs (used by `omm upgrade` for a full data sync)."""
    config = load_config()

    rules_url = config.get("rules_url")
    if rules_url:
        try:
            fetched = rules_mod.fetch_rules(rules_url)
            console.print(f"[green]Updated rules.json ({len(fetched)} entries) from {rules_url}[/green]")
        except requests.RequestException as e:
            console.print(f"[red]Failed to fetch rules from {rules_url}: {e}[/red]")
    else:
        console.print("[dim]No rules_url configured - using bundled defaults.[/dim]")

    model_url = config.get("model_url")
    if model_url:
        try:
            artifact = predictor.fetch_and_cache_model(model_url)
            console.print(
                f"[green]Updated recommend-model.json "
                f"({len(artifact.get('candidates', []))} candidates) from {model_url}[/green]"
            )
        except (requests.RequestException, ValueError) as e:
            console.print(f"[red]Failed to fetch trained model from {model_url}: {e}[/red]")


@app.command()
def upgrade() -> None:
    """Reinstall omm from the latest source via pipx, then refresh rules/model data."""
    console.print(f"Upgrading omm from {REPO_URL} ...")
    try:
        result = subprocess.run(
            ["pipx", "install", "--force", _install_spec()],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        console.print(
            "[red]pipx not found. Install it first, or rerun the installer:[/red]\n"
            "  curl -fsSL https://raw.githubusercontent.com/minigu5/Localfit/main/install.sh | sh"
        )
        raise typer.Exit(1)

    if result.returncode != 0:
        console.print(f"[red]pipx install failed:[/red]\n{result.stderr}")
        raise typer.Exit(1)

    console.print("[green]omm reinstalled from the latest source.[/green]")
    _refresh_data()


def _add_escape_to_cancel(question: questionary.Question) -> questionary.Question:
    """questionary only aborts on Ctrl+C/Ctrl+Q by default; make Escape do
    the same so `.ask()` returns None instead of requiring Ctrl+C."""

    def _abort(event) -> None:
        event.app.exit(exception=KeyboardInterrupt, style="class:aborting")

    question.application.key_bindings.add(Keys.Escape, eager=True)(_abort)
    return question


def _ask_select(question: questionary.Question):
    return _add_escape_to_cancel(question).ask()


@app.command()
def recommend() -> None:
    """Scan hardware and suggest a model to install, ranked by a model
    trained on real install telemetry (falls back to static rules if the
    trained model can't be fetched)."""
    info = scan_hardware()
    config = load_config()

    artifact, changed = predictor.load_model_with_change_note(config.get("model_url"))
    if changed:
        console.print("[dim]Fetched updated recommendation data from GitHub.[/dim]")
    if artifact and artifact.get("candidates"):
        ranked = predictor.rank_candidates(artifact, info)
        viable = [(c, speed) for c, speed in ranked if speed > 0][:10]
        if not viable:
            console.print("[red]No model is predicted to run on this hardware.[/red]")
            raise typer.Exit(1)

        choices = [
            questionary.Choice(
                title=f"{c['name']} (~{speed:.0f} tok/s predicted) - {c.get('description', '')}",
                value=f"{c['repo_id']}:{c['filename']}",
            )
            for c, speed in viable
        ]
        selected = _ask_select(questionary.select("Pick a model to install:", choices=choices))
        if selected is None:
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)
        install(selected)
        return

    console.print("[dim]No trained model available, falling back to static rules.[/dim]")
    rules_url = config.get("rules_url")
    if rules_url:
        try:
            _, rules_changed = rules_mod.refresh_rules_with_change_note(rules_url)
            if rules_changed:
                console.print("[dim]Fetched updated rules from GitHub.[/dim]")
        except requests.RequestException:
            pass

    has_gpu = info.vram_total_gb is not None
    available_gb = info.vram_total_gb if has_gpu else info.ram_total_gb

    rule_list = rules_mod.load_rules()
    matches = rules_mod.matching_rules(rule_list, available_gb, has_gpu=has_gpu)

    if not matches:
        console.print("[red]No model in the current rules fits this hardware.[/red]")
        raise typer.Exit(1)

    choices = [
        questionary.Choice(title=f"{r['name']} - {r['description']}", value=r["name"])
        for r in matches
    ]
    selected = _ask_select(questionary.select("Pick a model to install:", choices=choices))
    if selected is None:
        console.print("[yellow]Cancelled.[/yellow]")
        raise typer.Exit(0)

    install(selected)


@app.command()
def install(
    model_name: str = typer.Argument(..., autocompletion=complete_install_name),
) -> None:
    """Download a model into the central hub and link it into installed engines."""
    try:
        resolved = resolve_model(model_name)
    except ModelResolutionError as e:
        console.print(f"[red]{e}[/red]")
        _print_install_suggestions(model_name)
        raise typer.Exit(1) from e

    url, filename, repo_id = resolved.url, resolved.filename, resolved.repo_id
    dest = MODELS_DIR / filename
    if dest.exists():
        console.print(f"[yellow]{filename} already downloaded, skipping fetch.[/yellow]")
    else:
        try:
            download_file(url, dest)
        except DownloadError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from e

    console.print("Verifying checksum...")
    sha256 = sha256_file(dest)

    linked = {"lmstudio": False, "ollama": False}
    ollama_tag = linker.sanitize_ollama_tag(filename)

    if linker.is_lmstudio_installed():
        try:
            linker.link_lmstudio(dest, repo_id)
            linked["lmstudio"] = True
        except linker.LinkError as e:
            console.print(f"[yellow]LM Studio link skipped: {e}[/yellow]")
    else:
        console.print("[dim]LM Studio not detected, skipping link.[/dim]")

    if linker.is_ollama_installed():
        try:
            has_chat_template = linker.link_ollama(dest, ollama_tag)
            linked["ollama"] = True
            if not has_chat_template:
                console.print(
                    "[yellow]This GGUF has no embedded chat template - "
                    "Ollama will fall back to raw completion (no chat formatting).[/yellow]"
                )
        except linker.LinkError as e:
            console.print(f"[yellow]Ollama link skipped: {e}[/yellow]")
    else:
        console.print("[dim]Ollama not detected, skipping link.[/dim]")

    registry.upsert_entry(
        filename,
        sha256=sha256,
        source=url,
        size_bytes=dest.stat().st_size,
        installed_at=datetime.now(timezone.utc).isoformat(),
        ollama_name=ollama_tag,
        repo_id=repo_id,
        linked=linked,
    )

    if linked["ollama"]:
        console.print("Benchmarking...")
        tokens_per_sec = benchmark.benchmark_ollama(ollama_tag)
        _report_telemetry(filename, repo_id, tokens_per_sec)

    console.print(f"[green]Installed {filename}[/green]")
    if linked["ollama"]:
        console.print(f"  Ollama: [green]ollama run {ollama_tag}[/green]")
    if linked["lmstudio"]:
        console.print("  LM Studio: visible in your local models list")
    console.print(f"  Uninstall with: [cyan]omm remove {filename}[/cyan]")


@app.command()
def remove(
    filename: str = typer.Argument(..., autocompletion=complete_remove_filename),
) -> None:
    """Remove a model and clean up all symlinks/manifests."""
    reg = registry.load_registry()
    entry = reg.get(filename)
    if entry is None and not filename.lower().endswith(".gguf"):
        filename = f"{filename}.gguf"
        entry = reg.get(filename)
    if entry is None:
        console.print(f"[red]{filename} is not installed via omm. See `omm list`.[/red]")
        raise typer.Exit(1)

    linked = entry.get("linked", {})
    if linked.get("lmstudio"):
        linker.unlink_lmstudio(filename, entry.get("repo_id"))
    if linked.get("ollama"):
        linker.unlink_ollama(entry.get("ollama_name", linker.sanitize_ollama_tag(filename)))

    dest = MODELS_DIR / filename
    dest.unlink(missing_ok=True)
    dest.with_suffix(dest.suffix + ".part").unlink(missing_ok=True)

    registry.remove_entry(filename)
    console.print(f"[green]Removed {filename}[/green]")


@app.command(name="list")
def list_models() -> None:
    """Show models installed via omm and their linked status."""
    reg = registry.load_registry()
    if not reg:
        console.print("No models installed via omm yet. Try `omm recommend` or `omm install`.")
        raise typer.Exit(0)

    table = Table(title="omm models")
    table.add_column("Filename", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("LM Studio")
    table.add_column("Ollama")

    for filename, entry in reg.items():
        size_gb = entry.get("size_bytes", 0) / (1024**3)
        linked = entry.get("linked", {})
        table.add_row(
            filename,
            f"{size_gb:.2f} GB",
            "[green]yes[/green]" if linked.get("lmstudio") else "no",
            "[green]yes[/green]" if linked.get("ollama") else "no",
        )
    console.print(table)


@app.command()
def search(query: str) -> None:
    """Search curated models, cached candidates, and HuggingFace by name."""
    config = load_config()
    pool = search_mod.local_candidate_pool(config.get("model_url"))
    local_matches = search_mod.match_candidates(pool, query)

    local_repo_ids = {c.get("repo_id") for c in local_matches if c.get("repo_id")}
    hf_matches = [
        c
        for c in search_mod.search_huggingface(query)
        if c.get("repo_id") not in local_repo_ids
    ]

    combined = local_matches + hf_matches
    if not combined:
        console.print(f"[yellow]No models found matching '{query}'.[/yellow]")
        raise typer.Exit(1)

    groups = search_mod.group_by_family(combined)
    for family in sorted(groups):
        console.print(f"[bold cyan]==> {family}[/bold cyan]")
        for c in groups[family]:
            label = c.get("name") or c.get("repo_id")
            desc = c.get("description") or ""
            console.print(f"  {label}  [dim]{desc}[/dim]")
        console.print()


def _print_install_suggestions(query: str) -> None:
    config = load_config()
    pool = search_mod.local_candidate_pool(config.get("model_url"))
    suggestions = search_mod.suggest_similar(query, pool, limit=3)

    existing_labels = {s.get("name") or s.get("repo_id") for s in suggestions}
    if len(suggestions) < 3:
        for hit in search_mod.search_huggingface(query, limit=5):
            if len(suggestions) >= 3:
                break
            label = hit.get("name") or hit.get("repo_id")
            if label in existing_labels:
                continue
            suggestions.append(hit)
            existing_labels.add(label)

    if not suggestions:
        return

    console.print("[yellow]이런 모델을 찾으셨나요?[/yellow]")
    for s in suggestions:
        console.print(f"  - {s.get('name') or s.get('repo_id')}")


@app.command()
def apply() -> None:
    """Retry linking any installed models that couldn't be linked before
    (e.g. LM Studio or Ollama was installed after `omm install` ran)."""
    reg = registry.load_registry()
    if not reg:
        console.print("No models installed via omm yet.")
        raise typer.Exit(0)

    linked_count = 0
    skipped_missing = 0
    already_ok = 0

    for filename, entry in reg.items():
        dest = MODELS_DIR / filename
        if not dest.exists():
            skipped_missing += 1
            continue

        linked = entry.get("linked", {})
        new_linked: dict[str, bool] = {}
        update_fields: dict[str, str] = {}
        changed = False

        if not linked.get("lmstudio") and linker.is_lmstudio_installed():
            try:
                linker.link_lmstudio(dest, entry.get("repo_id"))
                new_linked["lmstudio"] = True
                changed = True
            except linker.LinkError as e:
                console.print(f"[yellow]{filename}: LM Studio link skipped: {e}[/yellow]")

        if not linked.get("ollama") and linker.is_ollama_installed():
            ollama_tag = entry.get("ollama_name") or linker.sanitize_ollama_tag(filename)
            try:
                linker.link_ollama(dest, ollama_tag)
                new_linked["ollama"] = True
                update_fields["ollama_name"] = ollama_tag
                changed = True
            except linker.LinkError as e:
                console.print(f"[yellow]{filename}: Ollama link skipped: {e}[/yellow]")

        if changed:
            registry.upsert_entry(filename, linked=new_linked, **update_fields)
            linked_count += 1
        else:
            already_ok += 1

    console.print(
        f"[green]{linked_count} model(s) newly linked.[/green] "
        f"{already_ok} already up to date, {skipped_missing} skipped (file missing)."
    )


@app.command()
def autoremove() -> None:
    """Remove broken symlinks left behind when a model's source .gguf was
    deleted without going through `omm remove`."""
    lmstudio_removed = linker.autoremove_lmstudio() if linker.is_lmstudio_installed() else 0
    ollama_blobs_removed, ollama_manifests_removed = (
        linker.autoremove_ollama() if linker.is_ollama_installed() else (0, 0)
    )

    if lmstudio_removed == 0 and ollama_blobs_removed == 0:
        console.print("[green]No broken symlinks found.[/green]")
        return

    console.print(
        f"[green]Removed {lmstudio_removed} broken LM Studio symlink(s) and "
        f"{ollama_blobs_removed} broken Ollama blob(s) "
        f"({ollama_manifests_removed} manifest(s) cleaned up).[/green]"
    )


def _report_telemetry(filename: str, repo_id: str | None, tokens_per_sec: float | None) -> None:
    if tokens_per_sec is None:
        # Ollama daemon wasn't reachable - not a real "it doesn't run" signal,
        # so skip rather than polluting the speed-regression training data.
        return
    info = scan_hardware()
    telemetry.send_event(
        {
            "os": info.os_name,
            "ram_gb": round(info.ram_total_gb, 1),
            "vram_gb": round(info.vram_total_gb, 1) if info.vram_total_gb is not None else None,
            "unified_memory": info.unified_memory,
            "model_installed": filename,
            "model_repo_id": repo_id,
            "engine": "ollama",
            "tokens_per_sec": round(tokens_per_sec, 2),
        }
    )


if __name__ == "__main__":
    app()
