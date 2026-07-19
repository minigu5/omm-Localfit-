"""omm CLI entry point (apt/brew-style command routing)."""

import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone

import questionary
import requests
import typer
from prompt_toolkit.keys import Keys
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from omm import benchmark, linker, predictor, registry, rules as rules_mod, search as search_mod, session_cache, telemetry
from omm.completion import complete_install_name, complete_remove_filename
from omm.config import MODELS_DIR, load_config
from omm.downloader import DownloadError, download_file
from omm.hardware import scan_hardware
from omm.hashutil import sha256_file
from omm.hub import AmbiguousModelError, ModelResolutionError, rank_quant_variants, resolve_model

app = typer.Typer(
    name="omm",
    help="Open source Model Manager - package manager for local LLMs (GGUF).",
)
console = Console()

REPO_URL = "git+https://github.com/minigu5/Localfit.git"


def _omm_version() -> str:
    try:
        return importlib.metadata.version("omm")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        console.print(f"omm {_omm_version()}")
        raise typer.Exit(0)


@app.command(name="help")
def help_cmd(
    ctx: typer.Context,
    command: str = typer.Argument(None, help="Show help for a specific subcommand."),
) -> None:
    """Show help, same as --help."""
    root_ctx = ctx.find_root()
    if command is None:
        console.print(root_ctx.get_help())
        raise typer.Exit(0)

    cmd_obj = root_ctx.command.get_command(root_ctx, command)
    if cmd_obj is None:
        console.print(f"[red]No such command '{command}'. See `omm help`.[/red]")
        raise typer.Exit(1)

    sub_ctx = cmd_obj.make_context(command, [], parent=root_ctx, resilient_parsing=True)
    console.print(cmd_obj.get_help(sub_ctx))


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


_BARE_REPO_URL = REPO_URL.removeprefix("git+")


def _installed_commit() -> str | None:
    """The commit `omm` was actually installed from, read from pip's PEP 610
    `direct_url.json` - present whenever pip installed from a VCS URL (i.e.
    every real `pipx install`). None for editable/local-path dev installs,
    which carry no vcs_info to compare against."""
    try:
        raw = importlib.metadata.distribution("omm").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    if not raw:
        return None
    return json.loads(raw).get("vcs_info", {}).get("commit_id")


def _remote_head_commit(ref: str = "main") -> str | None:
    """Latest commit on the given ref of the omm repo, via `git ls-remote`
    (no GitHub API rate limit, no auth needed for a public repo)."""
    try:
        result = subprocess.run(
            ["git", "ls-remote", _BARE_REPO_URL, ref],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout.split()[0]


# pipx gives no byte-level install progress, but it does print a fixed,
# ordered sequence of stage lines to stdout - use those as real (if coarse)
# progress checkpoints instead of an indeterminate animation that never
# actually reflects how far along the install is.
_PIPX_INSTALL_STAGES = [
    "creating virtual environment",
    "determining package name",
    "installing omm from spec",
    "done!",
    "installed package",
]


def _run_pipx_install(args: list[str], progress: Progress, task_id) -> subprocess.CompletedProcess:
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output_lines: list[str] = []
    stage = 0
    for line in proc.stdout:
        output_lines.append(line)
        lowered = line.lower()
        for i in range(stage, len(_PIPX_INSTALL_STAGES)):
            if _PIPX_INSTALL_STAGES[i] in lowered:
                stage = i + 1
                progress.update(task_id, completed=stage)
                break
    returncode = proc.wait()
    output = "".join(output_lines)
    return subprocess.CompletedProcess(args, returncode, stdout=output, stderr=output)


@app.command()
def upgrade() -> None:
    """Reinstall omm from the latest source via pipx, then refresh rules/model data."""
    installed = _installed_commit()
    latest = _remote_head_commit() if installed else None
    if installed and latest and installed == latest:
        console.print(f"[green]omm is already up to date ({installed[:7]}).[/green]")
        _refresh_data()
        return

    console.print(f"Upgrading omm from {REPO_URL} ...")
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[cyan]Reinstalling omm via pipx...[/cyan]"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("upgrade", total=len(_PIPX_INSTALL_STAGES))
            result = _run_pipx_install(
                ["pipx", "install", "--force", _install_spec()], progress, task_id
            )
            progress.update(task_id, completed=len(_PIPX_INSTALL_STAGES))
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


def _ask_confirm(message: str, default: bool = False) -> bool:
    """Yes/no prompt that answers on the y/n keypress itself (no Enter
    needed) via questionary's auto_enter. questionary.confirm's internal
    key bindings are already merged by the time we get the Question object,
    so (unlike _ask_select) we can't bolt an Escape binding on here -
    Ctrl+C/Ctrl+Q still cancel via questionary's own bindings."""
    answer = questionary.confirm(message, default=default, auto_enter=True).ask()
    return bool(answer)


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

        refs = [f"{c['repo_id']}:{c['filename']}" for c, speed in viable]
        session_cache.record_seen(refs)
        choices = [
            questionary.Choice(
                title=f"{c['name']} (~{speed:.0f} tok/s predicted) - {c.get('description', '')}",
                value=ref,
            )
            for (c, speed), ref in zip(viable, refs)
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

    session_cache.record_seen([r["name"] for r in matches])
    choices = [
        questionary.Choice(title=f"{r['name']} - {r['description']}", value=r["name"])
        for r in matches
    ]
    selected = _ask_select(questionary.select("Pick a model to install:", choices=choices))
    if selected is None:
        console.print("[yellow]Cancelled.[/yellow]")
        raise typer.Exit(0)

    install(selected)


def _resolve_ref(arg: str) -> str:
    """If `arg` is a bare integer, treat it as a 1-based index into the last
    `omm search`/`omm list` results shown in this terminal. Any non-numeric
    arg passes through unchanged."""
    if not arg.isdigit():
        return arg

    results = session_cache.load_last_results()
    if not results:
        console.print(
            "[red]Run `omm search` or `omm list` first to install/uninstall by number.[/red]"
        )
        raise typer.Exit(1)

    idx = int(arg)
    if idx < 1 or idx > len(results):
        console.print(f"[red]No result #{idx} (1-{len(results)}).[/red]")
        raise typer.Exit(1)

    return results[idx - 1]


def _pick_quant_variant(error: AmbiguousModelError) -> str | None:
    """Rank the ambiguous repo's .gguf files by fit against this PC's RAM/VRAM
    and let the user pick one, cursor defaulted to the best-fitting, highest
    quality option."""
    info = scan_hardware()
    has_gpu = info.vram_total_gb is not None
    available_gb = info.vram_total_gb if has_gpu else info.ram_total_gb

    variants = rank_quant_variants(error.candidates, available_gb)
    choices = []
    for v in variants:
        if v.fits is True:
            note = f"fits, ~{v.required_gb:.1f}GB needed"
        elif v.fits is False:
            note = f"may not fit, ~{v.required_gb:.1f}GB needed (you have {available_gb:.1f}GB)"
        else:
            note = "fit unknown"
        choices.append(questionary.Choice(title=f"{v.filename}  ({note})", value=v.filename))

    return _ask_select(
        questionary.select(f"Select a quantization variant for '{error.repo_id}':", choices=choices)
    )


@app.command()
def install(
    model_name: str = typer.Argument(..., autocompletion=complete_install_name),
) -> None:
    """Download a model into the central hub and link it into installed engines."""
    model_name = _resolve_ref(model_name)
    try:
        resolved = resolve_model(model_name)
    except AmbiguousModelError as e:
        chosen = _pick_quant_variant(e)
        if chosen is None:
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)
        install(f"{e.repo_id}:{chosen}")
        return
    except ModelResolutionError as e:
        console.print(f"[red]{e}[/red]")
        _print_install_suggestions(model_name)
        raise typer.Exit(1) from e

    url, filename, repo_id = resolved.url, resolved.filename, resolved.repo_id

    artifact = predictor.load_cached_model()
    trees = artifact.get("trees") if artifact else None
    if trees is not None:
        hw = scan_hardware()
        speed = predictor.predict_speed(trees, hw, {"repo_id": repo_id, "filename": filename})
        if speed <= 0:
            console.print(
                f"[red]Warning: this hardware is predicted not to run {filename}.[/red]"
            )
            if not _ask_confirm("Install anyway?"):
                console.print("[yellow]Cancelled.[/yellow]")
                raise typer.Exit(0)

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
        if _ask_confirm("Benchmark this model's speed and send the result to the server?"):
            console.print("Benchmarking...")
            tokens_per_sec = benchmark.benchmark_ollama(ollama_tag)
            if tokens_per_sec:
                console.print(f"[cyan]{tokens_per_sec:.1f} tok/s[/cyan]")
            _report_telemetry(filename, repo_id, tokens_per_sec)

    console.print(f"[green]Installed {filename}[/green]")
    if linked["ollama"]:
        console.print(f"  Ollama: [green]ollama run {ollama_tag}[/green]")
    if linked["lmstudio"]:
        console.print("  LM Studio: visible in your local models list")
    console.print(f"  Uninstall with: [cyan]omm uninstall {filename}[/cyan]")


def _cleanup_incomplete_install(filename: str) -> bool:
    dest = MODELS_DIR / filename
    part = dest.with_suffix(dest.suffix + ".part")
    cleaned = False
    if part.exists():
        part.unlink()
        cleaned = True
    if dest.exists():
        dest.unlink()
        cleaned = True
    return cleaned


def _remove_one(filename: str, entry: dict) -> None:
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


@app.command(name="uninstall")
def remove(
    filename: str = typer.Argument(..., autocompletion=complete_remove_filename),
) -> None:
    """Uninstall a model and clean up all symlinks/manifests. Pass `all` to
    uninstall every model installed via omm."""
    if filename.lower() == "all":
        reg = registry.load_registry()
        if not reg:
            console.print("No models installed via omm yet.")
            raise typer.Exit(0)
        if not _ask_confirm(f"Uninstall all {len(reg)} model(s)?"):
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)
        for name, entry in list(reg.items()):
            _remove_one(name, entry)
        return

    filename = _resolve_ref(filename)
    reg = registry.load_registry()
    entry = reg.get(filename)
    if entry is None and not filename.lower().endswith(".gguf"):
        filename = f"{filename}.gguf"
        entry = reg.get(filename)
    if entry is None:
        if _cleanup_incomplete_install(filename):
            console.print(f"[green]Cleaned up incomplete install of {filename}[/green]")
            raise typer.Exit(0)
        console.print(f"[red]{filename} is not installed via omm. See `omm list`.[/red]")
        raise typer.Exit(1)

    _remove_one(filename, entry)


@app.command(name="list")
def list_models() -> None:
    """Show models installed via omm and their linked status."""
    reg = registry.load_registry()
    if not reg:
        console.print("No models installed via omm yet. Try `omm recommend` or `omm install`.")
        raise typer.Exit(0)

    table = Table(title="omm models")
    table.add_column("#", justify="right")
    table.add_column("Filename", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("LM Studio")
    table.add_column("Ollama")

    for idx, (filename, entry) in enumerate(reg.items(), start=1):
        size_gb = entry.get("size_bytes", 0) / (1024**3)
        linked = entry.get("linked", {})
        table.add_row(
            str(idx),
            filename,
            f"{size_gb:.2f} GB",
            "[green]yes[/green]" if linked.get("lmstudio") else "no",
            "[green]yes[/green]" if linked.get("ollama") else "no",
        )
    console.print(table)
    session_cache.record_results(list(reg.keys()))


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

    # No network call here - only score against whatever's already cached
    # locally, same as install completion.
    artifact = predictor.load_cached_model()
    trees = artifact.get("trees") if artifact else None
    hw = scan_hardware() if trees else None

    groups = search_mod.group_by_family(combined)
    refs: list[str] = []
    for family in sorted(groups):
        console.print(f"[bold cyan]==> {family}[/bold cyan]")
        for c in groups[family]:
            ref = search_mod.install_ref(c)
            refs.append(ref)
            desc = c.get("description") or ""
            if trees is not None and predictor.predict_speed(trees, hw, c) <= 0:
                console.print(f"  [{len(refs)}] [red]{ref}  (predicted not to run on this hardware)[/red]")
            else:
                console.print(f"  [{len(refs)}] {ref}  [dim]{desc}[/dim]")
        console.print()

    session_cache.record_results(refs)


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

    console.print("[yellow]Did you mean one of these?[/yellow]")
    for s in suggestions:
        console.print(f"  - {search_mod.install_ref(s)}")


@app.command()
def relink() -> None:
    """Re-verify every installed model's LM Studio/Ollama links and repair
    them. Covers models that were never linked *and* ones whose link is now
    broken, missing, or stale - link_lmstudio/link_ollama always replace the
    existing symlink/manifest, so this always re-links rather than trusting
    the registry's stored `linked` flag."""
    reg = registry.load_registry()
    if not reg:
        console.print("No models installed via omm yet.")
        raise typer.Exit(0)

    lmstudio_installed = linker.is_lmstudio_installed()
    ollama_installed = linker.is_ollama_installed()

    relinked_count = 0
    skipped_missing = 0

    for filename, entry in reg.items():
        dest = MODELS_DIR / filename
        if not dest.exists():
            skipped_missing += 1
            continue

        new_linked: dict[str, bool] = {}
        update_fields: dict[str, str] = {}
        changed = False

        if lmstudio_installed:
            try:
                linker.link_lmstudio(dest, entry.get("repo_id"))
                new_linked["lmstudio"] = True
                changed = True
            except linker.LinkError as e:
                console.print(f"[yellow]{filename}: LM Studio link skipped: {e}[/yellow]")

        if ollama_installed:
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
            relinked_count += 1

    console.print(
        f"[green]{relinked_count} model(s) relinked/verified.[/green] "
        f"{skipped_missing} skipped (file missing)."
    )


def _autoremove_incomplete_installs() -> int:
    if not MODELS_DIR.exists():
        return 0

    reg = registry.load_registry()
    removed = 0
    for path in MODELS_DIR.iterdir():
        if not path.is_file():
            continue
        if path.suffix == ".part":
            if path.with_suffix("").name not in reg:
                path.unlink()
                removed += 1
        elif path.suffix == ".gguf" and path.name not in reg:
            path.unlink()
            removed += 1
    return removed


@app.command()
def autoremove() -> None:
    """Remove broken symlinks left behind when a model's source .gguf was
    deleted without going through `omm uninstall`, plus any orphaned partial or
    unregistered downloads in the models directory."""
    lmstudio_removed = linker.autoremove_lmstudio() if linker.is_lmstudio_installed() else 0
    ollama_blobs_removed, ollama_manifests_removed = (
        linker.autoremove_ollama() if linker.is_ollama_installed() else (0, 0)
    )
    incomplete_removed = _autoremove_incomplete_installs()

    if lmstudio_removed == 0 and ollama_blobs_removed == 0 and incomplete_removed == 0:
        console.print("[green]No broken symlinks found.[/green]")
        return

    console.print(
        f"[green]Removed {lmstudio_removed} broken LM Studio symlink(s) and "
        f"{ollama_blobs_removed} broken Ollama blob(s) "
        f"({ollama_manifests_removed} manifest(s) cleaned up), "
        f"{incomplete_removed} incomplete install file(s) cleaned up.[/green]"
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
            "cpu": info.cpu,
            "gpu": info.gpu_name,
            "ram_gb": round(info.ram_total_gb, 1),
            "vram_gb": round(info.vram_total_gb, 1) if info.vram_total_gb is not None else None,
            "unified_memory": info.unified_memory,
            "model_installed": filename,
            "model_repo_id": repo_id,
            "engine": "ollama",
            "tokens_per_sec": round(tokens_per_sec, 2),
        },
        force=True,
    )


if __name__ == "__main__":
    app()
