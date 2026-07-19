# OMM CLI 세션 UX / 정리 / 텔레메트리 동의 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `omm` CLI에 (1) 세션 스코프 Tab 자동완성 확장, (2) 검색/목록 결과 번호 참조, (3) 미완료 설치 정리(`remove`/`autoremove`), (4) install 끝의 벤치마크+텔레메트리 y/n 동의를 추가한다.

**Architecture:** 새 모듈 `session_cache.py`가 터미널(TTY)별로 `~/.omm/session/<sha1(tty)>.json`에 최근 결과를 저장한다 — Tab 자동완성이 키 입력마다 새 프로세스로 뜨기 때문에 파일 기반 상태가 필요하다. `search`/`list`/`recommend`가 이 캐시에 기록하고, `completion.py`와 `install`/`remove`의 신규 숫자 인자 처리가 이를 읽는다. `remove`/`autoremove`는 registry에 없는 `.part`/미등록 `.gguf` 파일을 정리하는 로직을 추가한다. `telemetry.send_event()`에 `force` 파라미터를 추가하고, `install()`의 벤치마크+전송 블록을 `typer.confirm()` 하나로 감싼다.

**Tech Stack:** Python 3.10+, Typer/Click, pytest 8 (`typer.testing.CliRunner`), 기존 `isolated_omm_home` fixture 재사용.

## Global Constraints

- 세션 캐시는 어떤 이유로든(TTY 없음, 파일 I/O 실패, 손상된 JSON) 예외를 밖으로 던지지 않는다 — 항상 빈 상태로 폴백.
- `config.OMM_HOME`은 함수 안에서 `from omm import config` 후 `config.OMM_HOME`으로 매번 동적 조회한다 (모듈 최상단에서 이름만 임포트하면 `isolated_omm_home` fixture의 monkeypatch가 적용 안 됨 — 기존 레지스트리 코드에서 이미 겪은 문제, [[project-omm-cli-convenience]] 참고).
- 기존 43+ 테스트는 전부 그대로 통과해야 한다. 커밋마다 `pytest`를 돌린다.
- 새 텍스트는 기존 한국어 사용자 메시지 톤(`console.print(f"[green]...[/green]")` 등)을 따른다.
- 커밋은 태스크 단위로 자주. 이 저장소는 direct-to-main, PR 없음.

---

## 파일 구조

- **신규** `src/omm/session_cache.py` — TTY 스코프 세션 캐시. `search`/`list`가 쓰고 `completion.py`/`cli.py`가 읽음.
- **신규** `tests/test_session_cache.py`
- **수정** `src/omm/completion.py` — `complete_install_name`에 세션 캐시 병합.
- **수정** `tests/test_completion.py` — 세션 캐시 병합 테스트 추가.
- **수정** `src/omm/cli.py` — `search`/`list_models`/`recommend`에 번호 출력 + 세션 기록, `install`/`remove`에 숫자 인자 해석, `remove`/`autoremove`에 미완료 설치 정리, `install` 끝의 confirm 게이팅.
- **수정** `tests/test_cli_search.py`, 신규 `tests/test_cli_list.py`, 신규 `tests/test_cli_index_ref.py`, `tests/test_cli_remove.py`(추가 케이스), `tests/test_cli_autoremove.py`(추가 케이스), 신규 `tests/test_cli_install_confirm.py`.
- **수정** `src/omm/telemetry.py` — `send_event(event, force=False)`.
- **신규** `tests/test_telemetry.py`.

---

## Task 1: `session_cache` 모듈

**Files:**
- Create: `src/omm/session_cache.py`
- Test: `tests/test_session_cache.py`

**Interfaces:**
- Produces:
  - `record_seen(refs: list[str]) -> None`
  - `record_results(refs: list[str]) -> None` (last_results 덮어쓰기 + seen에도 병합)
  - `load_seen() -> list[str]`
  - `load_last_results() -> list[str]`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_session_cache.py`:

```python
from omm import session_cache


def _fake_tty(monkeypatch, name="/dev/faketty0"):
    monkeypatch.setattr(session_cache.os, "ttyname", lambda fd: name)


def _no_tty(monkeypatch):
    def _raise(fd):
        raise OSError("not a tty")

    monkeypatch.setattr(session_cache.os, "ttyname", _raise)


def test_record_and_load_seen_roundtrips(isolated_omm_home, monkeypatch):
    _fake_tty(monkeypatch)

    session_cache.record_seen(["a", "b"])

    assert session_cache.load_seen() == ["a", "b"]


def test_record_seen_dedupes_and_moves_to_front(isolated_omm_home, monkeypatch):
    _fake_tty(monkeypatch)

    session_cache.record_seen(["a", "b"])
    session_cache.record_seen(["b", "c"])

    assert session_cache.load_seen() == ["b", "c", "a"]


def test_record_seen_caps_at_50(isolated_omm_home, monkeypatch):
    _fake_tty(monkeypatch)

    for i in range(60):
        session_cache.record_seen([f"model-{i}"])

    assert len(session_cache.load_seen()) == 50
    assert "model-59" in session_cache.load_seen()


def test_record_results_overwrites_last_results(isolated_omm_home, monkeypatch):
    _fake_tty(monkeypatch)

    session_cache.record_results(["x", "y"])
    assert session_cache.load_last_results() == ["x", "y"]

    session_cache.record_results(["z"])
    assert session_cache.load_last_results() == ["z"]


def test_record_results_also_updates_seen(isolated_omm_home, monkeypatch):
    _fake_tty(monkeypatch)

    session_cache.record_results(["x", "y"])

    assert session_cache.load_seen() == ["x", "y"]


def test_no_tty_is_a_silent_noop(isolated_omm_home, monkeypatch):
    _no_tty(monkeypatch)

    session_cache.record_seen(["a"])
    session_cache.record_results(["b"])

    assert session_cache.load_seen() == []
    assert session_cache.load_last_results() == []


def test_different_ttys_do_not_share_state(isolated_omm_home, monkeypatch):
    _fake_tty(monkeypatch, "/dev/tty-one")
    session_cache.record_results(["from-tty-one"])

    _fake_tty(monkeypatch, "/dev/tty-two")
    assert session_cache.load_last_results() == []


def test_corrupted_cache_file_is_treated_as_empty(isolated_omm_home, monkeypatch):
    _fake_tty(monkeypatch)
    session_cache.record_seen(["a"])

    from omm import config

    session_dir = config.OMM_HOME / "session"
    for f in session_dir.iterdir():
        f.write_text("{not valid json")

    assert session_cache.load_seen() == []
```

- [ ] **Step 2: 테스트 실행해서 실패 확인**

Run: `pytest tests/test_session_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'omm.session_cache'`

- [ ] **Step 3: 최소 구현**

`src/omm/session_cache.py`:

```python
"""Per-TTY session cache for `omm`: lets `search`/`list`/`recommend` results
be referenced later by number and pulled into Tab-completion, without any
in-memory state - Tab-completion runs as a fresh process on every keypress,
so state has to survive on disk. Best-effort only: never raises out of this
module. TTY-scoped so two terminal windows never see each other's results.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from omm import config

_MAX_SEEN = 50


def _session_path() -> Path | None:
    # Use the OS-level stdin fd (always 0 on POSIX) rather than
    # sys.stdin.fileno() - test runners and other harnesses often swap
    # sys.stdin for an object whose .fileno() raises before os.ttyname()
    # ever runs, which would short-circuit this to "no session" even when
    # fd 0 itself is a real tty.
    try:
        tty = os.ttyname(0)
    except OSError:
        return None
    digest = hashlib.sha1(tty.encode()).hexdigest()
    return config.OMM_HOME / "session" / f"{digest}.json"


def _load() -> dict[str, list[str]]:
    path = _session_path()
    if path is None or not path.exists():
        return {"seen": [], "last_results": []}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"seen": [], "last_results": []}
    return {
        "seen": list(data.get("seen", [])),
        "last_results": list(data.get("last_results", [])),
    }


def _save(data: dict[str, list[str]]) -> None:
    path = _session_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    except OSError:
        pass


def record_seen(refs: list[str]) -> None:
    if not refs:
        return
    data = _load()
    merged = list(refs) + [r for r in data["seen"] if r not in refs]
    data["seen"] = merged[:_MAX_SEEN]
    _save(data)


def record_results(refs: list[str]) -> None:
    data = _load()
    data["last_results"] = list(refs)
    merged = list(refs) + [r for r in data["seen"] if r not in refs]
    data["seen"] = merged[:_MAX_SEEN]
    _save(data)


def load_seen() -> list[str]:
    return _load()["seen"]


def load_last_results() -> list[str]:
    return _load()["last_results"]
```

- [ ] **Step 4: 테스트 실행해서 통과 확인**

Run: `pytest tests/test_session_cache.py -v`
Expected: 9 passed

- [ ] **Step 5: 커밋**

```bash
git add src/omm/session_cache.py tests/test_session_cache.py
git commit -m "feat: add TTY-scoped session cache for search/list results"
```

---

## Task 2: Tab 자동완성에 세션 캐시 병합

**Files:**
- Modify: `src/omm/completion.py`
- Test: `tests/test_completion.py`

**Interfaces:**
- Consumes: `session_cache.load_seen() -> list[str]` (Task 1)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_completion.py`에 추가:

```python
def test_complete_install_name_includes_session_seen_names(monkeypatch):
    monkeypatch.setattr(completion.predictor, "load_cached_model", lambda: None)
    monkeypatch.setattr(
        completion.session_cache,
        "load_seen",
        lambda: ["org/repo:some-file-Q4_K_M.gguf"],
    )

    result = completion.complete_install_name("org/repo")

    assert result == ["org/repo:some-file-Q4_K_M.gguf"]
```

- [ ] **Step 2: 테스트 실행해서 실패 확인**

Run: `pytest tests/test_completion.py -v`
Expected: FAIL — `AttributeError: module 'omm.completion' has no attribute 'session_cache'`

- [ ] **Step 3: 구현**

`src/omm/completion.py` 전체를 다음으로 교체:

```python
"""Shell tab-completion callbacks for omm's Typer CLI. These must never
make a network call, so `install` completion reads only the already-cached
recommend-model artifact (never a live fetch)."""

from __future__ import annotations

from omm import hub, predictor, registry, session_cache


def complete_install_name(incomplete: str) -> list[str]:
    names = set(hub.CURATED_INDEX.keys())

    artifact = predictor.load_cached_model()
    if artifact:
        names.update(c.get("name") for c in artifact.get("candidates", []) if c.get("name"))

    names.update(session_cache.load_seen())

    return sorted(n for n in names if n.startswith(incomplete))


def complete_remove_filename(incomplete: str) -> list[str]:
    filenames = registry.load_registry().keys()
    return sorted(f for f in filenames if f.startswith(incomplete))
```

- [ ] **Step 4: 테스트 실행해서 통과 확인**

Run: `pytest tests/test_completion.py -v`
Expected: 5 passed

- [ ] **Step 5: 커밋**

```bash
git add src/omm/completion.py tests/test_completion.py
git commit -m "feat: pull session-seen model refs into install tab-completion"
```

---

## Task 3: `search`/`list`/`recommend`에 번호 출력 + 세션 기록

**Files:**
- Modify: `src/omm/cli.py` (`search()` ~line 338, `list_models()` ~line 312, `recommend()` ~line 144)
- Modify: `tests/test_cli_search.py`
- Create: `tests/test_cli_list.py`

**Interfaces:**
- Consumes: `session_cache.record_results(refs: list[str])`, `session_cache.record_seen(refs: list[str])` (Task 1)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_cli_search.py`에 추가 (기존 두 테스트는 그대로 둔다):

```python
def test_search_prints_numbered_refs_and_records_session(monkeypatch):
    monkeypatch.setattr(cli, "load_config", lambda: {"model_url": None})
    monkeypatch.setattr(
        cli.search_mod,
        "local_candidate_pool",
        lambda model_url: [
            {
                "name": "tinyllama-1.1b-q4",
                "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                "description": "Curated default",
            },
        ],
    )
    monkeypatch.setattr(cli.search_mod, "search_huggingface", lambda query, **kwargs: [])
    recorded = []
    monkeypatch.setattr(cli.session_cache, "record_results", lambda refs: recorded.append(refs))

    result = runner.invoke(cli.app, ["search", "tiny"])

    assert result.exit_code == 0, result.stdout
    assert "[1] tinyllama-1.1b-q4" in result.stdout
    assert recorded == [["tinyllama-1.1b-q4"]]
```

`tests/test_cli_list.py` (신규):

```python
from typer.testing import CliRunner

from omm import cli, registry

runner = CliRunner()


def test_list_shows_index_column_and_records_session(isolated_omm_home, monkeypatch):
    registry.save_registry(
        {
            "a.gguf": {"size_bytes": 0, "linked": {"lmstudio": False, "ollama": False}},
            "b.gguf": {"size_bytes": 0, "linked": {"lmstudio": False, "ollama": True}},
        }
    )
    recorded = []
    monkeypatch.setattr(cli.session_cache, "record_results", lambda refs: recorded.append(refs))

    result = runner.invoke(cli.app, ["list"])

    assert result.exit_code == 0, result.stdout
    assert recorded == [["a.gguf", "b.gguf"]]


def test_list_empty_registry_does_not_touch_session(isolated_omm_home, monkeypatch):
    recorded = []
    monkeypatch.setattr(cli.session_cache, "record_results", lambda refs: recorded.append(refs))

    result = runner.invoke(cli.app, ["list"])

    assert result.exit_code == 0, result.stdout
    assert recorded == []
```

- [ ] **Step 2: 테스트 실행해서 실패 확인**

Run: `pytest tests/test_cli_search.py tests/test_cli_list.py -v`
Expected: FAIL — `AttributeError: module 'omm.cli' has no attribute 'session_cache'` (search), `assert '[1] tinyllama-1.1b-q4' in ...` 실패

- [ ] **Step 3: 구현**

`src/omm/cli.py` 상단 import 블록 (line 15-21 부근)에 추가:

```python
from omm import benchmark, linker, predictor, registry, rules as rules_mod, search as search_mod, session_cache, telemetry
```

(`session_cache`만 기존 임포트 라인에 알파벳 순으로 끼워 넣는다.)

`search()` (line 338-364)를 다음으로 교체:

```python
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
    refs: list[str] = []
    for family in sorted(groups):
        console.print(f"[bold cyan]==> {family}[/bold cyan]")
        for c in groups[family]:
            ref = search_mod.install_ref(c)
            refs.append(ref)
            desc = c.get("description") or ""
            console.print(f"  [{len(refs)}] {ref}  [dim]{desc}[/dim]")
        console.print()

    session_cache.record_results(refs)
```

`list_models()` (line 312-335)를 다음으로 교체:

```python
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
```

`recommend()`의 ML 경로 (line 155-174, `if artifact and artifact.get("candidates"):` 블록 안) 중 `choices = [...]` 부분을 다음으로 교체:

```python
        refs = [f"{c['repo_id']}:{c['filename']}" for c, speed in viable]
        session_cache.record_seen(refs)
        choices = [
            questionary.Choice(
                title=f"{c['name']} (~{speed:.0f} tok/s predicted) - {c.get('description', '')}",
                value=ref,
            )
            for (c, speed), ref in zip(viable, refs)
        ]
```

규칙 폴백 경로 (line 196-198, `choices = [...]` 부분)를 다음으로 교체:

```python
    session_cache.record_seen([r["name"] for r in matches])
    choices = [
        questionary.Choice(title=f"{r['name']} - {r['description']}", value=r["name"])
        for r in matches
    ]
```

- [ ] **Step 4: 테스트 실행해서 통과 확인**

Run: `pytest tests/test_cli_search.py tests/test_cli_list.py tests/test_cli_recommend_escape.py -v`
Expected: all passed

- [ ] **Step 5: 커밋**

```bash
git add src/omm/cli.py tests/test_cli_search.py tests/test_cli_list.py
git commit -m "feat: number search/list results and record them into the session cache"
```

---

## Task 4: `install`/`remove` 숫자 인자 해석

**Files:**
- Modify: `src/omm/cli.py` (`install()` line 199-202, `remove()` line 276-278)
- Create: `tests/test_cli_index_ref.py`

**Interfaces:**
- Consumes: `session_cache.load_last_results() -> list[str]` (Task 1)
- Produces: `_resolve_ref(arg: str) -> str` (raises `typer.Exit(1)` on bad index, used by Task 5's `remove()` too)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_cli_index_ref.py` (신규):

```python
from typer.testing import CliRunner

from omm import cli
from omm.hub import ModelResolutionError

runner = CliRunner()


def test_install_resolves_numeric_arg_from_last_results(isolated_omm_home, monkeypatch):
    monkeypatch.setattr(cli.session_cache, "load_last_results", lambda: ["org/repo:file.gguf"])
    monkeypatch.setattr(cli, "_print_install_suggestions", lambda name: None)
    seen = {}

    def fake_resolve_model(name):
        seen["name"] = name
        raise ModelResolutionError("stop here, we only care about the arg")

    monkeypatch.setattr(cli, "resolve_model", fake_resolve_model)

    runner.invoke(cli.app, ["install", "1"])

    assert seen["name"] == "org/repo:file.gguf"


def test_install_numeric_arg_out_of_range(monkeypatch):
    monkeypatch.setattr(cli.session_cache, "load_last_results", lambda: ["only-one"])

    result = runner.invoke(cli.app, ["install", "5"])

    assert result.exit_code == 1
    assert "5" in result.stdout


def test_install_numeric_arg_with_no_prior_results(monkeypatch):
    monkeypatch.setattr(cli.session_cache, "load_last_results", lambda: [])

    result = runner.invoke(cli.app, ["install", "1"])

    assert result.exit_code == 1
    assert "omm search" in result.stdout or "omm list" in result.stdout


def test_install_non_numeric_arg_is_unaffected(isolated_omm_home, monkeypatch):
    monkeypatch.setattr(cli.session_cache, "load_last_results", lambda: ["should-not-be-used"])
    monkeypatch.setattr(cli, "_print_install_suggestions", lambda name: None)
    seen = {}

    def fake_resolve_model(name):
        seen["name"] = name
        raise ModelResolutionError("stop")

    monkeypatch.setattr(cli, "resolve_model", fake_resolve_model)

    runner.invoke(cli.app, ["install", "tinyllama-1.1b-q4"])

    assert seen["name"] == "tinyllama-1.1b-q4"


def test_remove_resolves_numeric_arg_from_last_results(isolated_omm_home, monkeypatch):
    from omm import registry

    filename = "a.gguf"
    dest = cli.MODELS_DIR / filename
    dest.write_bytes(b"data")
    registry.save_registry({filename: {"linked": {"lmstudio": False, "ollama": False}}})
    monkeypatch.setattr(cli.session_cache, "load_last_results", lambda: [filename])

    result = runner.invoke(cli.app, ["remove", "1"])

    assert result.exit_code == 0, result.stdout
    assert f"Removed {filename}" in result.stdout
```

- [ ] **Step 2: 테스트 실행해서 실패 확인**

Run: `pytest tests/test_cli_index_ref.py -v`
Expected: FAIL — install/remove try to resolve `"1"` as a literal model name instead of an index.

- [ ] **Step 3: 구현**

`src/omm/cli.py`에서 `install()` 함수 바로 위 (line 198 부근, `@app.command()` 앞)에 헬퍼 추가:

```python
def _resolve_ref(arg: str) -> str:
    """If `arg` is a bare integer, treat it as a 1-based index into the last
    `omm search`/`omm list` results shown in this terminal. Any non-numeric
    arg passes through unchanged."""
    if not arg.isdigit():
        return arg

    results = session_cache.load_last_results()
    if not results:
        console.print(
            "[red]번호로 설치/삭제하려면 먼저 omm search 또는 omm list를 실행하세요.[/red]"
        )
        raise typer.Exit(1)

    idx = int(arg)
    if idx < 1 or idx > len(results):
        console.print(f"[red]{idx}번은 없습니다 (1-{len(results)}).[/red]")
        raise typer.Exit(1)

    return results[idx - 1]
```

`install()` 시그니처 아래 (line 203, docstring 다음 줄) 첫 줄로 추가:

```python
@app.command()
def install(
    model_name: str = typer.Argument(..., autocompletion=complete_install_name),
) -> None:
    """Download a model into the central hub and link it into installed engines."""
    model_name = _resolve_ref(model_name)
    try:
        resolved = resolve_model(model_name)
```

`remove()` 시그니처 아래 (line 279, docstring 다음 줄) 첫 줄로 추가:

```python
@app.command()
def remove(
    filename: str = typer.Argument(..., autocompletion=complete_remove_filename),
) -> None:
    """Remove a model and clean up all symlinks/manifests."""
    filename = _resolve_ref(filename)
    reg = registry.load_registry()
```

- [ ] **Step 4: 테스트 실행해서 통과 확인**

Run: `pytest tests/test_cli_index_ref.py tests/test_cli_remove.py tests/test_cli_install_suggestions.py -v`
Expected: all passed

- [ ] **Step 5: 커밋**

```bash
git add src/omm/cli.py tests/test_cli_index_ref.py
git commit -m "feat: allow install/remove to take a numeric index from the last search/list"
```

---

## Task 5: `remove`의 미완료 설치 정리

**Files:**
- Modify: `src/omm/cli.py` (`remove()`, line 275 부근)
- Modify: `tests/test_cli_remove.py`

**Interfaces:**
- Produces: `_cleanup_incomplete_install(filename: str) -> bool`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_cli_remove.py`에 추가:

```python
def test_remove_cleans_up_orphaned_part_file(isolated_omm_home):
    part = cli.MODELS_DIR / "orphan.gguf.part"
    cli.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    part.write_bytes(b"partial")

    result = runner.invoke(cli.app, ["remove", "orphan.gguf"])

    assert result.exit_code == 0, result.stdout
    assert "orphan.gguf" in result.stdout
    assert not part.exists()


def test_remove_cleans_up_unregistered_complete_download(isolated_omm_home):
    dest = cli.MODELS_DIR / "orphan.gguf"
    cli.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"complete-but-unregistered")

    result = runner.invoke(cli.app, ["remove", "orphan.gguf"])

    assert result.exit_code == 0, result.stdout
    assert not dest.exists()


def test_remove_still_errors_when_nothing_on_disk(isolated_omm_home):
    result = runner.invoke(cli.app, ["remove", "nothing-here.gguf"])

    assert result.exit_code == 1
    assert "is not installed via omm" in result.stdout
```

- [ ] **Step 2: 테스트 실행해서 실패 확인**

Run: `pytest tests/test_cli_remove.py -v`
Expected: FAIL — 새 테스트 2개가 "is not installed via omm" 에러로 실패.

- [ ] **Step 3: 구현**

`src/omm/cli.py`의 `remove()` 함수 (Task 4에서 이미 `filename = _resolve_ref(filename)`가 맨 위에 추가됨) 전체를 다음으로 교체:

```python
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


@app.command()
def remove(
    filename: str = typer.Argument(..., autocompletion=complete_remove_filename),
) -> None:
    """Remove a model and clean up all symlinks/manifests."""
    filename = _resolve_ref(filename)
    reg = registry.load_registry()
    entry = reg.get(filename)
    if entry is None and not filename.lower().endswith(".gguf"):
        filename = f"{filename}.gguf"
        entry = reg.get(filename)
    if entry is None:
        if _cleanup_incomplete_install(filename):
            console.print(f"[green]미완료 설치 {filename} 정리 완료[/green]")
            raise typer.Exit(0)
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
```

(`_cleanup_incomplete_install`을 `remove()` 바로 위, `_resolve_ref` 바로 아래에 둔다.)

- [ ] **Step 4: 테스트 실행해서 통과 확인**

Run: `pytest tests/test_cli_remove.py -v`
Expected: all passed

- [ ] **Step 5: 커밋**

```bash
git add src/omm/cli.py tests/test_cli_remove.py
git commit -m "feat: let remove clean up orphaned partial/unregistered downloads"
```

---

## Task 6: `autoremove`의 미완료 설치 스캔

**Files:**
- Modify: `src/omm/cli.py` (`autoremove()`, line 445 부근)
- Modify: `tests/test_cli_autoremove.py`

**Interfaces:**
- Produces: `_autoremove_incomplete_installs() -> int`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_cli_autoremove.py`에 추가:

```python
def test_autoremove_cleans_up_orphaned_part_and_gguf_files(isolated_omm_home, monkeypatch):
    monkeypatch.setattr(linker, "is_lmstudio_installed", lambda: False)
    monkeypatch.setattr(linker, "is_ollama_installed", lambda: False)
    cli.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    orphan_part = cli.MODELS_DIR / "orphan.gguf.part"
    orphan_part.write_bytes(b"partial")
    orphan_full = cli.MODELS_DIR / "orphan2.gguf"
    orphan_full.write_bytes(b"complete")

    result = runner.invoke(cli.app, ["autoremove"])

    assert result.exit_code == 0, result.stdout
    assert "2" in result.stdout
    assert not orphan_part.exists()
    assert not orphan_full.exists()


def test_autoremove_leaves_registered_files_alone(isolated_omm_home, monkeypatch):
    from omm import registry

    monkeypatch.setattr(linker, "is_lmstudio_installed", lambda: False)
    monkeypatch.setattr(linker, "is_ollama_installed", lambda: False)
    cli.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    kept = cli.MODELS_DIR / "kept.gguf"
    kept.write_bytes(b"data")
    registry.save_registry({"kept.gguf": {"linked": {"lmstudio": False, "ollama": False}}})

    result = runner.invoke(cli.app, ["autoremove"])

    assert result.exit_code == 0, result.stdout
    assert kept.exists()
```

- [ ] **Step 2: 테스트 실행해서 실패 확인**

Run: `pytest tests/test_cli_autoremove.py -v`
Expected: FAIL — 첫 테스트에서 "No broken symlinks found" 출력, 파일들이 지워지지 않음.

기존 3개 테스트(`test_autoremove_reports_zero_when_nothing_broken`,
`test_autoremove_reports_counts_from_both_engines`,
`test_autoremove_skips_uninstalled_engines`)는 지금까지 `isolated_omm_home`
fixture 없이 돌아갔다 — `autoremove()`가 `MODELS_DIR`를 건드리지 않았기
때문이다. 이제 `_autoremove_incomplete_installs()`가 `MODELS_DIR`를 스캔하므로
이 3개 테스트에도 `isolated_omm_home` 인자를 추가해야 실제 `~/.omm`을 건드리지
않는다.

- [ ] **Step 3: 구현**

`src/omm/cli.py`의 `autoremove()` (line 445-462)를 다음으로 교체:

```python
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
    deleted without going through `omm remove`, plus any orphaned partial or
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
```

- [ ] **Step 4: 테스트 실행해서 통과 확인**

Run: `pytest tests/test_cli_autoremove.py -v`
Expected: all passed (5 tests)

- [ ] **Step 5: 커밋**

```bash
git add src/omm/cli.py tests/test_cli_autoremove.py
git commit -m "feat: let autoremove clean up orphaned partial/unregistered downloads"
```

---

## Task 7: `telemetry.send_event`에 `force` 파라미터

**Files:**
- Modify: `src/omm/telemetry.py`
- Create: `tests/test_telemetry.py`

**Interfaces:**
- Produces: `send_event(event: dict, force: bool = False) -> None`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_telemetry.py` (신규):

```python
from omm import telemetry


def test_send_event_skips_when_not_opted_in_and_not_forced(monkeypatch):
    monkeypatch.setattr(
        telemetry,
        "load_config",
        lambda: {"telemetry_opt_in": False, "telemetry_endpoint": "https://example.com"},
    )
    called = []
    monkeypatch.setattr(telemetry.requests, "post", lambda *a, **k: called.append((a, k)))

    telemetry.send_event({"x": 1})

    assert called == []


def test_send_event_sends_when_forced_even_if_not_opted_in(monkeypatch):
    monkeypatch.setattr(
        telemetry,
        "load_config",
        lambda: {"telemetry_opt_in": False, "telemetry_endpoint": "https://example.com"},
    )
    called = []
    monkeypatch.setattr(telemetry.requests, "post", lambda *a, **k: called.append((a, k)))

    telemetry.send_event({"x": 1}, force=True)

    assert len(called) == 1


def test_send_event_forced_still_requires_endpoint(monkeypatch):
    monkeypatch.setattr(
        telemetry,
        "load_config",
        lambda: {"telemetry_opt_in": False, "telemetry_endpoint": None},
    )
    called = []
    monkeypatch.setattr(telemetry.requests, "post", lambda *a, **k: called.append((a, k)))

    telemetry.send_event({"x": 1}, force=True)

    assert called == []


def test_send_event_sends_when_opted_in_without_force(monkeypatch):
    monkeypatch.setattr(
        telemetry,
        "load_config",
        lambda: {"telemetry_opt_in": True, "telemetry_endpoint": "https://example.com"},
    )
    called = []
    monkeypatch.setattr(telemetry.requests, "post", lambda *a, **k: called.append((a, k)))

    telemetry.send_event({"x": 1})

    assert len(called) == 1
```

- [ ] **Step 2: 테스트 실행해서 실패 확인**

Run: `pytest tests/test_telemetry.py -v`
Expected: FAIL — `test_send_event_sends_when_forced_even_if_not_opted_in`이 `force`를 모르는 `send_event`에서 `TypeError` 발생.

- [ ] **Step 3: 구현**

`src/omm/telemetry.py` 전체를 다음으로 교체:

```python
"""Opt-in (or explicitly forced), best-effort telemetry. Never raises,
never blocks the CLI."""

from __future__ import annotations

from typing import Any

import requests

from omm.config import load_config


def send_event(event: dict[str, Any], force: bool = False) -> None:
    config = load_config()
    if not force and not config.get("telemetry_opt_in"):
        return
    endpoint = config.get("telemetry_endpoint")
    if not endpoint:
        return
    try:
        requests.post(endpoint, json=event, timeout=5)
    except requests.RequestException:
        pass
```

- [ ] **Step 4: 테스트 실행해서 통과 확인**

Run: `pytest tests/test_telemetry.py -v`
Expected: 4 passed

- [ ] **Step 5: 커밋**

```bash
git add src/omm/telemetry.py tests/test_telemetry.py
git commit -m "feat: let telemetry.send_event bypass the opt-in check when forced"
```

---

## Task 8: `install` 끝에 벤치마크+전송 y/n 동의

**Files:**
- Modify: `src/omm/cli.py` (`install()` line 262-265, `_report_telemetry()` line 465-482)
- Create: `tests/test_cli_install_confirm.py`

**Interfaces:**
- Consumes: `telemetry.send_event(event, force=True)` (Task 7), `benchmark.benchmark_ollama(tag) -> float | None` (existing)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_cli_install_confirm.py` (신규):

```python
from typer.testing import CliRunner

from omm import cli, linker
from omm.hub import ResolvedModel

runner = CliRunner()


def _stub_successful_install(monkeypatch, isolated_omm_home):
    filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    monkeypatch.setattr(
        cli,
        "resolve_model",
        lambda name: ResolvedModel(url="https://example.com/x.gguf", filename=filename, repo_id="org/repo"),
    )

    def fake_download(url, dest):
        dest.write_bytes(b"fake-gguf")

    monkeypatch.setattr(cli, "download_file", fake_download)
    monkeypatch.setattr(cli, "sha256_file", lambda dest: "deadbeef")
    monkeypatch.setattr(linker, "is_lmstudio_installed", lambda: False)
    monkeypatch.setattr(linker, "is_ollama_installed", lambda: True)
    monkeypatch.setattr(linker, "link_ollama", lambda dest, tag: True)
    monkeypatch.setattr(linker, "sanitize_ollama_tag", lambda filename: "tinyllama")
    return filename


def test_install_runs_benchmark_and_telemetry_on_yes(isolated_omm_home, monkeypatch):
    filename = _stub_successful_install(monkeypatch, isolated_omm_home)
    monkeypatch.setattr(cli.benchmark, "benchmark_ollama", lambda tag: 42.0)
    sent = []
    monkeypatch.setattr(cli.telemetry, "send_event", lambda event, force=False: sent.append((event, force)))

    result = runner.invoke(cli.app, ["install", "tinyllama-1.1b-q4"], input="y\n")

    assert result.exit_code == 0, result.stdout
    assert "42.0" in result.stdout or "42" in result.stdout
    assert len(sent) == 1
    assert sent[0][1] is True


def test_install_skips_benchmark_and_telemetry_on_no(isolated_omm_home, monkeypatch):
    _stub_successful_install(monkeypatch, isolated_omm_home)
    bench_calls = []
    monkeypatch.setattr(cli.benchmark, "benchmark_ollama", lambda tag: bench_calls.append(tag) or 42.0)
    sent = []
    monkeypatch.setattr(cli.telemetry, "send_event", lambda event, force=False: sent.append((event, force)))

    result = runner.invoke(cli.app, ["install", "tinyllama-1.1b-q4"], input="n\n")

    assert result.exit_code == 0, result.stdout
    assert bench_calls == []
    assert sent == []
```

- [ ] **Step 2: 테스트 실행해서 실패 확인**

Run: `pytest tests/test_cli_install_confirm.py -v`
Expected: FAIL — 지금은 프롬프트 없이 항상 벤치마크+전송이 실행되므로 `input="n\n"` 케이스가 실패 (`bench_calls`가 채워짐).

- [ ] **Step 3: 구현**

`src/omm/cli.py`의 `install()` 안, line 262-265 (`if linked["ollama"]: console.print("Benchmarking...") ...`)를 다음으로 교체:

```python
    if linked["ollama"]:
        if typer.confirm("모델 속도를 측정하고 결과를 서버로 보낼까요?", default=False):
            console.print("Benchmarking...")
            tokens_per_sec = benchmark.benchmark_ollama(ollama_tag)
            if tokens_per_sec:
                console.print(f"[cyan]{tokens_per_sec:.1f} tok/s[/cyan]")
            _report_telemetry(filename, repo_id, tokens_per_sec)
```

`_report_telemetry()` (line 465-482) 안의 `telemetry.send_event(` 호출을 다음으로 교체 (딕셔너리 내용은 그대로, `force=True`만 추가):

```python
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
        },
        force=True,
    )
```

- [ ] **Step 4: 테스트 실행해서 통과 확인**

Run: `pytest tests/test_cli_install_confirm.py -v`
Expected: 2 passed

- [ ] **Step 5: 커밋**

```bash
git add src/omm/cli.py tests/test_cli_install_confirm.py
git commit -m "feat: gate install-time benchmark+telemetry behind a single y/n prompt"
```

---

## Task 9: 전체 스위트 회귀 확인

**Files:** 없음 (검증만)

- [ ] **Step 1: 전체 테스트 실행**

Run: `pytest -v`
Expected: 기존 43+ 테스트 + 이번 계획에서 추가한 테스트(session_cache 9, completion 1, search/list 3, index_ref 5, remove 2, autoremove 2, telemetry 4, install_confirm 2 = 약 28개 신규) 전부 통과, 실패/에러 0.

- [ ] **Step 2: 결과가 다르면 원인 파악 후 수정, 통과할 때까지 반복.** (새 커밋 없이 이전 태스크 커밋에 대한 후속 수정 커밋으로 처리)
