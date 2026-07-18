#!/usr/bin/env sh
# Installs omm (Open source Model Manager) as an isolated CLI command via pipx.
# Usage: curl -fsSL https://raw.githubusercontent.com/minigu5/Localfit/main/install.sh | sh
set -eu

REPO_URL="git+https://github.com/minigu5/Localfit.git"

# NVIDIA VRAM detection is dead weight on Mac (no NVIDIA GPUs since 2016) -
# only pull that extra in on other platforms.
if [ "$(uname -s)" = "Darwin" ]; then
    INSTALL_SPEC="$REPO_URL"
else
    INSTALL_SPEC="omm[nvidia] @ $REPO_URL"
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found. Install Python 3.10+ first: https://www.python.org/downloads/" >&2
    exit 1
fi

PY_OK=$(python3 -c 'import sys; print(1 if sys.version_info >= (3, 10) else 0)')
if [ "$PY_OK" != "1" ]; then
    echo "omm requires Python 3.10+, found: $(python3 --version)" >&2
    exit 1
fi

# Run pipx either as a direct command (brew install, or once PATH catches
# up) or as `python3 -m pipx` (works right after a pip --user install,
# before PATH is refreshed in this shell).
run_pipx() {
    if command -v pipx >/dev/null 2>&1; then
        pipx "$@"
    else
        python3 -m pipx "$@"
    fi
}

if ! command -v pipx >/dev/null 2>&1 && ! python3 -m pipx --version >/dev/null 2>&1; then
    echo "pipx not found, installing it..."
    if command -v brew >/dev/null 2>&1; then
        # Skip brew's implicit `brew update` before installing - it can add
        # tens of seconds to minutes if the formula index is stale, and
        # pipx's version rarely matters enough to need the latest index.
        HOMEBREW_NO_AUTO_UPDATE=1 brew install pipx
    elif python3 -m pip install --user --quiet pipx 2>/dev/null; then
        :
    else
        # Homebrew/PEP-668 "externally-managed-environment" Pythons refuse
        # plain --user installs; pipx itself is safe to force here since it
        # only manages its own isolated venvs afterward.
        python3 -m pip install --user --quiet --break-system-packages pipx
    fi
    run_pipx ensurepath
fi

echo "Installing omm from $REPO_URL ..."
run_pipx install --force "$INSTALL_SPEC"

echo
echo "Done. If 'omm' isn't found, open a new shell (pipx just updated your PATH)."
echo "Try:  omm scan"
echo "Tip: run 'omm --install-completion' once (then restart your shell) to enable Tab completion for install/remove."
