"""omm CLI entry point (apt/brew-style command routing)."""

from datetime import datetime, timezone

import questionary
import requests
import typer
from rich.console import Console
from rich.table import Table

from omm import linker, registry, rules as rules_mod, telemetry
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


@app.command()
def update() -> None:
    """Fetch the latest recommendation rules.json from the configured URL."""
    config = load_config()
    url = config.get("rules_url")
    if not url:
        console.print(
            "[yellow]No rules_url configured in ~/.omm/config.json - using bundled defaults.[/yellow]"
        )
        raise typer.Exit(0)
    try:
        fetched = rules_mod.fetch_rules(url)
    except requests.RequestException as e:
        console.print(f"[red]Failed to fetch rules from {url}: {e}[/red]")
        raise typer.Exit(1) from e
    console.print(f"[green]Updated rules.json ({len(fetched)} entries) from {url}[/green]")


@app.command()
def recommend() -> None:
    """Scan hardware and suggest a model to install."""
    info = scan_hardware()
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
    selected = questionary.select("Pick a model to install:", choices=choices).ask()
    if selected is None:
        console.print("[yellow]Cancelled.[/yellow]")
        raise typer.Exit(0)

    install(selected)


@app.command()
def install(model_name: str) -> None:
    """Download a model into the central hub and link it into installed engines."""
    try:
        url, filename = resolve_model(model_name)
    except ModelResolutionError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    dest = MODELS_DIR / filename
    if dest.exists():
        console.print(f"[yellow]{filename} already downloaded, skipping fetch.[/yellow]")
    else:
        try:
            download_file(url, dest)
        except DownloadError as e:
            console.print(f"[red]{e}[/red]")
            _report_telemetry(filename, success=False)
            raise typer.Exit(1) from e

    console.print("Verifying checksum...")
    sha256 = sha256_file(dest)

    linked = {"lmstudio": False, "ollama": False}
    ollama_tag = linker.sanitize_ollama_tag(filename)

    if linker.is_lmstudio_installed():
        try:
            linker.link_lmstudio(dest)
            linked["lmstudio"] = True
        except linker.LinkError as e:
            console.print(f"[yellow]LM Studio link skipped: {e}[/yellow]")
    else:
        console.print("[dim]LM Studio not detected, skipping link.[/dim]")

    if linker.is_ollama_installed():
        try:
            linker.link_ollama(dest, ollama_tag)
            linked["ollama"] = True
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
        linked=linked,
    )

    _report_telemetry(filename, success=True)

    console.print(f"[green]Installed {filename}[/green]")
    if linked["ollama"]:
        console.print(f"  Ollama: [green]ollama run {ollama_tag}[/green]")
    if linked["lmstudio"]:
        console.print("  LM Studio: visible in your local models list")
    console.print(f"  Uninstall with: [cyan]omm remove {filename}[/cyan]")


@app.command()
def remove(filename: str) -> None:
    """Remove a model and clean up all symlinks/manifests."""
    entry = registry.load_registry().get(filename)
    if entry is None:
        console.print(f"[red]{filename} is not installed via omm. See `omm list`.[/red]")
        raise typer.Exit(1)

    linked = entry.get("linked", {})
    if linked.get("lmstudio"):
        linker.unlink_lmstudio(filename)
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


def _report_telemetry(filename: str, success: bool) -> None:
    info = scan_hardware()
    telemetry.send_event(
        {
            "os": info.os_name,
            "ram_gb": round(info.ram_total_gb, 1),
            "vram_gb": round(info.vram_total_gb, 1) if info.vram_total_gb is not None else None,
            "model_installed": filename,
            "success": success,
        }
    )


if __name__ == "__main__":
    app()
