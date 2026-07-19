# omm — Open source Model Manager

`omm` is an apt/brew-style package manager for local LLMs (GGUF). It installs models into a central hub, links them into LM Studio and Ollama automatically, and can recommend a model that fits your hardware.

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/minigu5/Localfit/main/install.sh | sh
```

This bootstraps `python3`, `git`, and `pipx` if missing (Debian/Ubuntu via `apt`, or Homebrew on macOS), then installs `omm` as an isolated CLI via `pipx`. Open a new shell afterward so your `PATH` picks up `omm`.

Requirements: Python 3.10+. GPU detection extras (`omm[nvidia]`) are installed automatically on non-macOS platforms.

## Usage

```sh
omm scan             # Print a hardware summary (RAM, VRAM, OS)
omm recommend        # Suggest a model that fits this machine, then offer to install it
omm search <query>   # Search curated models, cached candidates, and HuggingFace
omm install <name>   # Download a model and link it into LM Studio / Ollama
omm uninstall <name> # Uninstall a model and clean up its symlinks/manifests
omm uninstall all    # Uninstall every model installed via omm
omm list             # Show models installed via omm and their linked status
omm relink           # Re-verify and repair every installed model's LM Studio/Ollama links
omm autoremove       # Clean up broken symlinks and orphaned partial downloads
omm upgrade          # Reinstall omm from the latest source, then refresh rules/model data
omm help [command]   # Show help, same as --help
```

`install` and `uninstall` accept either a model name/reference or the numeric index shown by the last `omm search` or `omm list` run in that terminal. `search`/`install` mark models predicted not to run on this machine's hardware in red.

## Development

```sh
pip install -e ".[dev]"
pytest
```
