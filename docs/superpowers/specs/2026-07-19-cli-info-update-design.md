# OMM CLI `info` / `update` 설계

## 배경

`omm`은 Homebrew 스타일 로컬 LLM(GGUF) 패키지 매니저 ([[project-omm]]). 지금은
설치된 모델의 상세 정보를 보거나, 원본 HuggingFace 레포가 파일을 갱신했을 때
재다운로드할 방법이 없다. 이번 설계는 3가지 요청을 다룬다:

1. `omm info <model>` — 이름, 버전, 크기, 연결된 프로그램, 실행 명령어 표시.
2. `omm update <model>` — 해당 모델만 최신 버전으로 갱신 (이미 최신이면
   up to date).
3. `omm update` (인자 없음) — 설치된 모든 모델을 최신 버전으로 갱신.

GGUF 모델은 HuggingFace에 시맨틱 버전이 없다 — 레포+파일명으로만 존재한다.
"최신 버전" 판단은 설치 시 저장한 sha256과 레포의 현재 파일 sha256(LFS
해시)을 비교하는 방식으로 정의한다.

## 1. registry에 `version` 필드 추가

`registry.upsert_entry()` 호출부(`install`, `update`)에서 `version=sha256[:7]`
(git commit hash 스타일 축약 해시)를 같이 저장한다. 기존에 설치된 항목은
`version` 필드가 없을 수 있으므로, 읽는 쪽은 항상
`entry.get("version") or entry.get("sha256", "")[:7] or "unknown"` 폴백을
쓴다 — 마이그레이션 스크립트는 두지 않는다.

## 2. `hub.py`: 원격 파일 해시 조회

```python
HF_PATHS_INFO = "https://huggingface.co/api/models/{repo_id}/paths-info/main"

def remote_file_sha256(repo_id: str, filename: str) -> str | None:
    ...
```

HF의 `paths-info` API에 `POST {"paths": [filename]}`로 요청 — 응답 배열의
`[0]["lfs"]["sha256"]`을 반환한다. 다음 경우 전부 `None`을 반환해 "버전 확인
불가"로 취급한다: 네트워크/HTTP 에러, 응답이 빈 배열, 파일이 LFS가 아님(`lfs`
키 없음 — gguf는 항상 LFS라 실사용에선 거의 발생하지 않음).

## 3. `omm info <model_name_or_index>`

기존 `_resolve_ref`로 번호(직전 `search`/`list` 결과)/이름을 해석한 뒤
registry를 조회한다. 없으면 `"{filename} is not installed via omm. See
\`omm list\`."` 에러로 `Exit(1)`.

있으면 테이블 출력:

| 필드 | 값 |
|---|---|
| Name | filename |
| Repo | `entry["repo_id"]` 또는 없으면 `"(direct URL install)"` |
| Version | 위 폴백 로직 |
| Size | `size_bytes` → GB 2자리 |
| Installed at | `installed_at` |
| LM Studio | linked면 `"linked (visible in LM Studio app)"`, 아니면 `"not linked"` |
| Ollama | linked면 `f"ollama run {ollama_name}"`, 아니면 `"not linked"` |

LM Studio는 CLI 실행 명령어 자체가 없는 앱이라 "실행 명령어"는 안내 문구로
대체한다.

## 4. `omm update [model_name]`

인자는 옵션(`typer.Argument(None, ...)`). 규칙:

- 인자 없음, 또는 `all`(대소문자 무시, `uninstall all`과 동일 관례) → 전체
  갱신 모드.
- 그 외 → `_resolve_ref`로 해석 후 단일 모델 갱신.

### 공통 로직: `_update_one(filename, entry) -> str`

반환값은 `"updated"` / `"up_to_date"` / `"skipped"` 중 하나.

- **`entry["repo_id"]`가 있는 경우** (HF 레포 설치):
  1. `hub.remote_file_sha256(repo_id, filename)` 호출.
  2. `None`이면 콘솔에 `"{filename}: 버전 확인 불가 (repo/LFS 정보 없음),
     건너뜀"` 출력 후 `"skipped"`.
  3. `entry["sha256"]`와 같으면 `"up_to_date"`.
  4. 다르면: 기존 `install()`과 동일한 방식으로 `download_file(url, dest)`
     호출해 같은 파일명 위치에 재다운로드(`.part` → rename으로 기존 파일
     덮어씀) → `sha256_file(dest)`로 재계산 → `registry.upsert_entry(filename,
     sha256=new_sha256, version=new_sha256[:7], size_bytes=...,
     installed_at=now)` → `install()`과 동일한 링크 재생성
     (`linker.link_lmstudio` / `linker.link_ollama`, 실패는 경고만 하고
     계속) → `"updated"`.

- **`entry["repo_id"]`가 없는 경우** (직접 URL 설치): 원격에서 해시를 값싸게
  조회할 API가 없으므로, `entry["source"]` URL을 임시 경로
  (`dest.with_name(dest.name + ".update")`)로 전부 재다운로드 →
  `sha256_file(tmp)` 계산 → `entry["sha256"]`와 비교.
  - 같으면: 임시 파일 삭제 → `"up_to_date"`.
  - 다르면: `tmp.replace(dest)`로 원자적 교체(기존 파일을 자동으로 덮어써
    중복 파일이 남지 않음) → registry 갱신 → 재연결 → `"updated"`.

### 단일 모델 모드

`_update_one` 결과에 맞춰 메시지 1줄 출력:
- `up_to_date` → `"{filename} is already up to date ({version})."`
- `updated` → `"{filename} updated to {new_version}."`
- `skipped` → (이미 `_update_one` 내부에서 사유 출력됨)

registry에 없는 모델(비-등록)을 인자로 주면 `omm info`와 동일하게
`"is not installed via omm"` 에러.

### 전체 모드

registry가 비어 있으면 `"No models installed via omm yet."` 출력 후
`Exit(0)`. 아니면 `_ask_confirm(f"Check {n} model(s) for updates?")`로 확인
받고(취소 시 `"Cancelled."` 출력 후 `Exit(0)`), 각 항목에 `_update_one` 실행
후 결과를 집계해 마지막에 `"{updated}개 업데이트, {up_to_date}개 최신,
{skipped}개 건너뜀."` 형태로 요약 출력.

## 테스트

- `tests/test_cli_info.py`: 정상 조회, 미설치 모델 에러, version 필드 없는
  기존 항목의 폴백 표시, 번호 참조(`_resolve_ref`) 라우팅.
- `tests/test_cli_update.py`: `hub.remote_file_sha256`과
  `downloader.download_file`을 monkeypatch로 대체.
  - 단일 모델: 해시 동일(up to date) / 해시 다름(재다운로드 후 registry
    갱신 및 재연결 호출 확인) / `remote_file_sha256`이 `None`(skip) 케이스.
  - repo_id 없는 모델: 재다운로드 후 해시 동일(임시파일 삭제, 원본 유지) /
    해시 다름(원자적 교체, 중복 파일 없음 확인) 케이스.
  - 전체 모드: confirm 취소 시 아무 것도 안 건드림 / 여러 모델 혼합 결과
    요약 문구 확인 / 빈 registry.

기존 `isolated_omm_home` 픽스처 재사용.
