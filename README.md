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
omm tune <name>      # Recommend context, GPU offload, threads, and batch size
omm benchmark qwen3:4b exaone3.5:2.4b  # Local quality + speed smoke evidence
omm search <query>   # Search curated models, cached candidates, and HuggingFace
omm install <name>   # Download a model and link it into LM Studio / Ollama
omm uninstall <name> # Uninstall a model and clean up its symlinks/manifests
omm uninstall all    # Uninstall every model installed via omm
omm list             # Show models installed via omm and their linked status
omm info <name>      # Show a model's name, version, size, and linked-program run commands
omm upgrade <name>   # Refresh a model against its source if it has changed since install
omm upgrade          # Check every installed model for updates
omm link             # Re-verify and repair every installed model's LM Studio/Ollama links
omm link <directory> # Reuse central GGUF files in another app without copying them
omm calibrate        # Locally correct predicted speed with an installed Ollama model
omm setting          # Interactive menu for UI mode, telemetry, and catalog trust
omm setting ui compact       # Use short everyday tables (`detailed` for diagnostics)
omm setting catalog-status   # Show signed recommendation data and rollback snapshots
omm autoremove       # Clean up broken symlinks and orphaned partial downloads
omm update           # Reinstall omm from the latest source, then refresh rules/model data
omm help [command]   # Show help, same as --help
```

`install`, `uninstall`, `info`, and `upgrade` accept either a model name/reference or the numeric index shown by the last `omm search` or `omm list` run in that terminal. `search`/`install` mark models predicted not to run on this machine's hardware in red.

Localfit does not assume all installed memory belongs to the model. A live
scan subtracts memory currently used by other applications, keeps at least
2 GB (or 10% of RAM) for the OS and newly opened apps, and applies total-memory
caps. Recommendation fit and `omm tune` use this safe budget, so rerunning a
command adapts after memory-heavy applications are opened or closed.

`omm benchmark` runs a versioned eight-item bilingual arithmetic smoke pack
against models already installed in Ollama. It stores parsed answers,
correctness, pinned model metadata, and fixed-length timings under
`~/.omm/evaluations/`; it stores no generated text or raw hardware names.
Results are uploaded only after explicit opt-in. The pack is intentionally
small and is not a leaderboard.

## Self-hosted benchmark data

Benchmark uploads are disabled and have no server endpoint by default. To run
the bundled FastAPI + SQLite collector locally:

```sh
pip install -e ".[server]"
export LOCALFIT_DB_PATH="$PWD/localfit.db"
export LOCALFIT_ADMIN_TOKEN="replace-with-a-long-random-token"
localfit-server
```

Explicitly configure the endpoint and opt in before uploading:

```sh
omm setting telemetry --endpoint http://127.0.0.1:8000/v1/benchmarks --enable
```

Training can consume the authenticated export directly:

```sh
export LOCALFIT_ADMIN_TOKEN="replace-with-a-long-random-token"
python scripts/train_model.py \
  --telemetry-url http://127.0.0.1:8000/v1/benchmarks/export
```

The old Firebase Realtime Database JSON endpoint remains supported only when
explicitly configured. Its official `*.firebaseio.com` or
`*.firebasedatabase.app` `.json` URL can be read without an admin token;
self-hosted raw export requires `LOCALFIT_ADMIN_TOKEN`. Exact duplicate events
are ignored.

Automated retraining is fail-closed. Configure
`LOCALFIT_TELEMETRY_EXPORT_URL`; configure `LOCALFIT_ADMIN_TOKEN` as well for a
self-hosted export (it is optional for an official Firebase JSON URL). The
scheduled job otherwise stops without changing the published artifact. It
requires at least 100 distinct valid v5 configurations with explicit runtime
metadata (legacy rows do not satisfy this minimum), rejects datasets with more
than 25% invalid rows, and reserves a deterministic 20% holdout. A 64-tree v4
candidate replaces the incumbent only when both holdout RMSLE and P90 absolute
percentage error stay within the configured regression limits. Selection is
evaluated on whole hardware/request contexts, so sibling model variants never
leak across training and holdout sets. Publishing also requires at least three
multi-model selection groups plus complete top-1, regret, balanced-fit, and
false-positive evidence. Missing evidence fails the gate. The artifact records
the complete candidate/baseline evaluation report.

The same gate can validate an exported local dataset without contacting the
collector:

```sh
python scripts/train_model.py --offline \
  --telemetry-file benchmarks.jsonl \
  --quality-gate --minimum-real-configurations 100 \
  --baseline published/recommend-model.json \
  --output candidate.json --quality-report quality-report.json
```

Synthetic bootstrap training remains available for local development, but the
scheduled publishing workflow never uses it as a substitute for missing real
benchmark data.

## Signed recommendation data

`omm setting catalog-trust --manifest-url <https-url> --public-key <base64-key>`
enables Ed25519 verification for future recommendation downloads. Existing
artifacts are snapshotted before replacement and `omm setting catalog-rollback`
restores the most recent different snapshot.

## Development

```sh
pip install -e ".[dev]"
pytest
```
