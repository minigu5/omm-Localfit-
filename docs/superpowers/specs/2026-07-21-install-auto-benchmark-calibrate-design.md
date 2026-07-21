# `omm install` 자동 벤치마크+calibrate, 업로드만 확인 설계

## 배경

[[project-omm]]. 직전 스펙([[2026-07-20-cli-session-ux-design]] 4번)에서
`install` 끝의 벤치마크+텔레메트리 전송을 하나의 y/n 확인으로 묶어 매번 물어보게
바꿨다 (`cli.py:810-812`, `"Benchmark this model's speed and send the result to
the server?"`). 이번 요청은 그 확인을 다시 쪼갠다:

1. 벤치마크는 확인 없이 항상 자동 실행.
2. 벤치마크 성공하면 로컬 calibration(`omm calibrate`와 동일 로직)도 확인 없이
   자동 실행 — 서버 전송이 아니라 `~/.omm`에 로컬 저장이므로 동의 불필요.
3. 서버로 결과를 실제로 보낼 때만 y/n 확인.

동시에 `omm calibrate`를 독립 명령에서 `omm setting calibrate`로 옮긴다
([[feedback_terminal_subcommand_minimalism]] 방침 — 최상위 명령 개수를 줄이는
연장선. 이번 변경으로 `omm install`이 calibrate를 흡수하면서 수동 실행 빈도가
줄어들어 최상위 명령으로 남겨둘 이유도 약해진다).

## 1. `omm calibrate` → `omm setting calibrate`

- `src/omm/cli.py`의 `@app.command()\ndef calibrate(...)` (현재 1186행 부근)를
  `@setting_app.command(name="calibrate")`로 옮긴다. 함수 본문은 그대로.
- 하위 호환 alias 없음 — 완전 이동.
- `setting_menu`(bare `omm setting` 대화형 메뉴, 1302행 부근) 선택지에
  `questionary.Choice("Calibrate", value="calibrate")` 추가. 선택 시 모델명을
  텍스트로 물어보고(빈 입력이면 `None` → 최소 크기 모델 자동 선택) `calibrate(name
  or None)` 호출.

## 2. `_install_impl` 확인 흐름 분리

현재 (`cli.py:807-827`):

```python
tokens_per_sec = None
telemetry_sent = False
if linked["ollama"]:
    should_benchmark = auto_benchmark or _ask_confirm(
        "Benchmark this model's speed and send the result to the server?"
    )
    if should_benchmark:
        ...benchmark...
        telemetry_sent = _report_telemetry(filename, repo_id, tokens_per_sec)
    else:
        telemetry.log_attempt("declined_by_user", filename)
else:
    telemetry.log_attempt("not_attempted_no_ollama_link", filename)
```

변경 후:

```python
tokens_per_sec = None
telemetry_sent = False
if linked["ollama"]:
    console.print("Benchmarking...")
    try:
        tokens_per_sec = _run_interruptible(
            lambda: benchmark.benchmark_ollama(ollama_tag), stop_event
        )
    except _Interrupted as e:
        raise ContributionStopped(filename) from e

    if tokens_per_sec:
        console.print(f"[cyan]{tokens_per_sec:.1f} tok/s[/cyan]")
        _maybe_auto_calibrate(filename, repo_id, dest, tokens_per_sec)

        want_upload = auto_upload or _ask_confirm(
            "Send this machine's benchmark result to the server?"
        )
        if want_upload:
            telemetry_sent = _report_telemetry(filename, repo_id, tokens_per_sec)
        else:
            telemetry.log_attempt("declined_by_user", filename)
    else:
        telemetry_sent = _report_telemetry(filename, repo_id, tokens_per_sec)
else:
    telemetry.log_attempt("not_attempted_no_ollama_link", filename)
```

- 파라미터 이름 변경: `_install_impl(..., auto_benchmark: bool = False, ...)` →
  `auto_upload: bool = False`. 의미도 바뀐다 — "벤치마크를 스킵 없이 실행할지"가
  아니라 "업로드 확인을 스킵할지". 벤치마크 자체는 이제 무조건 실행되므로 옛
  이름은 더 이상 맞지 않는다.
- 호출부 `_run_contribution_loop`(1752행 부근)의
  `_install_impl(resolved, auto_benchmark=True, ...)` →
  `_install_impl(resolved, auto_upload=True, ...)`로 변경. `omm contribute`는
  시작할 때 이미 "unattended, no per-model confirmation" 동의를 받으므로 계속
  확인 없이 업로드.
- `tokens_per_sec`가 `None`이거나 `0` 이하(벤치마크 실패/Ollama 데몬 응답 없음)면
  보낼 결과 자체가 없으므로 업로드 확인 없이 바로 `_report_telemetry`만 호출 —
  내부적으로 `skipped_daemon_unreachable`을 로깅하고 `False`를 반환하는 기존
  동작 그대로 유지.

## 3. 자동 calibration (`_maybe_auto_calibrate`, 신규 헬퍼)

`omm setting calibrate`(구 `omm calibrate`)와 동일한 예측→비교 로직을 재사용하되,
이미 측정된 `tokens_per_sec`를 그대로 쓰고 별도 벤치마크를 다시 돌리지 않는다.

```python
def _maybe_auto_calibrate(
    filename: str, repo_id: str | None, dest: Path, tokens_per_sec: float
) -> None:
    """Best-effort local calibration right after a successful benchmark.
    Silent no-op if there's no cached model to compare against — this must
    never block or fail the install."""
    artifact = predictor.load_cached_model()
    if not artifact or not artifact.get("trees"):
        return
    hardware = scan_hardware()
    candidate = {
        "repo_id": repo_id,
        "filename": filename,
        "size_bytes": dest.stat().st_size if dest.exists() else None,
    }
    try:
        predicted, _, _ = predictor.predict_speed_interval(
            artifact["trees"], hardware, candidate, engine="ollama",
            apply_calibration=False,
        )
    except (ValueError, KeyError, TypeError, IndexError):
        return
    if predicted <= 0:
        return
    factor = calibration.record_calibration(
        hardware,
        measured_tokens_per_sec=tokens_per_sec,
        predicted_tokens_per_sec=predicted,
        engine="ollama",
    )
    console.print(
        f"[dim]Local calibration updated: correction ×{factor:.2f} "
        "(not uploaded).[/dim]"
    )
```

- 캐시된 추천 모델이 없거나 예측 실패 시 조용히 넘어간다 — 설치 자체를 막지
  않는다.
- `omm contribute` 루프에서도 `_install_impl`을 그대로 타므로 부수적으로
  calibration이 쌓인다. 의도된 효과(부작용 없음, 로컬 전용)이며 별도 분기
  불필요.
- 콘솔 메시지는 기존 `omm setting calibrate` 명령의 "not uploaded" 문구 톤을
  그대로 따른다.

## 4. `install()` 커맨드 출력

기존 출력(`cli.py:854-861`)은 그대로 둔다 — calibration/업로드 관련 메시지는
`_install_impl` 내부에서 이미 출력하므로 `install()` 자체는 수정 불필요.

## 5. 테스트 변경

- `tests/test_cli_calibrate_local.py`: `runner.invoke(cli.app, ["calibrate",
  filename])` → `["setting", "calibrate", filename]`.
- `tests/test_cli_install_confirm.py`:
  - `test_install_runs_benchmark_and_telemetry_on_yes` → 확인에 응답하는 대상이
    "업로드 여부"이므로 이름과 assert는 유지 가능(그대로 여전히 참).
  - `test_install_skips_benchmark_and_telemetry_on_no`는 더 이상 성립하지 않음
    (벤치마크는 항상 실행됨). `test_install_runs_benchmark_but_skips_upload_on_no`로
    교체: confirm=False에서도 `benchmark_ollama`는 호출되지만 `send_event`는
    호출 안 됨을 검증.
- `tests/test_install_impl.py`: `auto_benchmark=True` 사용하는 두 테스트를
  `auto_upload=True`로 이름 변경. `test_auto_benchmark_skips_confirm_prompt_and_sends_telemetry`
  → `test_auto_upload_skips_confirm_prompt_and_sends_telemetry`로 리네임하고,
  이제 확인이 없는 것은 "업로드 확인"뿐이라는 점을 반영(벤치마크는 애초에 항상
  확인 없이 실행되므로 이 테스트가 검증하던 "no prompt" 어서션은 업로드 단계에도
  여전히 유효 — `_ask_confirm` 호출 자체가 없어야 함).
  - `test_stop_event_set_during_benchmark_raises_contribution_stopped`은
    `auto_benchmark=True` 인자를 제거해도 됨(벤치마크가 항상 실행되므로 굳이
    필요 없음) — 유지해도 무해.
- 새 헬퍼 `_maybe_auto_calibrate`를 위한 단위 테스트 추가
  (`tests/test_install_impl.py` 또는 신규 파일): 캐시된 트리 있을 때
  `calibration.record_calibration` 호출됨, 캐시 없을 때 조용히 스킵됨 두 케이스.

## 6. 영향받지 않는 것

- `telemetry.send_event`의 `telemetry_opt_in` / `force` 로직은 그대로.
- `omm setting telemetry`(엔드포인트/opt-in 영구 설정)는 그대로 — 이번 변경은
  "설치 1회당 업로드 여부를 매번 확인"이라는 기존 정책을 유지하되, 그 확인
  대상에서 벤치마크와 calibration을 분리해내는 것뿐.
- `omm quality-eval`은 별개 명령(로컬 전용, 업로드 없음) — 변경 없음.
