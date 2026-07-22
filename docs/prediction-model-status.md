# Localfit 예측 모델 작업 현황

기준일: 2026-07-22  
브랜치: `matwaetle/prediction-evaluation`

## 한 줄 요약

예측 모델 코드와 안전장치는 로컬 구현 완료. 새 모델 공개는 아직 금지. 새 형식 실측 데이터가 부족함.

## 완료한 것

- 실측 3회 실행 후 중앙값 기록
- 모델 크기·양자화·실제 실행 옵션을 담는 v5 데이터 형식
- 학습과 실제 추천이 같은 feature를 사용하도록 수정
- 특이한 파일명도 metadata로 파라미터 수를 전달
- quant 선택 화면의 예측 추천 경로 수정
- 깨진 예측 모델 파일 차단
- Firebase v5 입력 규칙 작성
- Firebase 공개 URL에는 admin token을 보내지 않도록 수정
- self-host 데이터 서버만 admin token 사용
- 같은 PC·요청 후보가 학습/시험에 나뉘는 평가 누수 제거
- RMSLE, P90 오차, top-1, regret, fit 오탐 품질 게이트 추가
- 데이터나 평가 증거가 부족하면 공개 모델 교체 차단

## 검증 결과

- 관리자 터미널 전체 테스트: `392 passed / 0 failed`
- 일반 터미널의 symlink 권한 실패 7개도 관리자 권한에서는 재현되지 않음
- commit·push·PR: 아직 안 함
- 공개 모델 파일: 아직 안 바꿈

## 현재 데이터

- 원본 행: 148
- 정상 행: 147
- 중복 제거 구성: 37
- 새 direct-v5 구성: 0
- 누수 없는 시험 결과는 후보가 기존 모델보다 좋아 보임
- 하지만 선택 묶음 1개뿐이고 fit 실패 증거가 없어 품질 게이트가 배포 거부

즉, 코드 문제보다 새 실측 데이터 부족이 현재 핵심 문제다.

## 내가 지금 할 일

1. 팀원에게 현재 diff 리뷰 요청

   - 새 모델 자체가 아니라 데이터 수집·평가·배포 안전장치 PR이라고 설명
   - 리뷰 전에는 commit·push하지 않기

2. 프로젝트 소유자 작업 요청

   - `database.rules.json`을 Firebase production에 배포
   - Actions에 `LOCALFIT_TELEMETRY_EXPORT_URL` 설정
   - self-host 서버를 쓸 때만 `LOCALFIT_ADMIN_TOKEN` 설정

3. Tailscale SSH PC 13대에서 v5 실측 수집

   - 관리자 터미널에서 SSH 사용
   - PC당 최소 8개, 권장 10개 구성
   - 총 130개를 목표로 잡아 거부·중복 여유 확보
   - 작은 모델, 중간 모델, 메모리 한계 근처 모델 포함
   - Q4·Q5·Q8과 context 2048·4096을 섞기
   - 사용자 동의를 받은 경우에만 업로드
   - 장비가 멈출 정도의 큰 모델은 실행하지 않기

4. 데이터 조건 확인

   - direct-v5 고유 구성 100개 이상
   - 비교 가능한 선택 묶음 3개 이상
   - fit 성공·실패 양쪽 사례 존재

5. GitHub Actions 재학습 실행

   - 품질 게이트 통과: 새 모델 공개 검토
   - 품질 게이트 실패: 기존 모델 유지하고 데이터·모델 개선

6. 팀 리뷰 후 merge 검토

## 팀 docs에 올릴 짧은 상태

- `[로컬 구현 완료]` 예측 모델 v4 평가·배포 게이트
- `[완료]` 학습·추론 feature parity
- `[코드 완료 / 배포 대기]` Firebase v5 telemetry rules
- `[준비 중]` Tailscale SSH PC 13대 실측 수집
- `[보류]` 새 예측 모델 공개 — direct-v5 데이터 부족
- `[완료]` 관리자 터미널 전체 테스트: 392 passed
- `[실행 대기]` GitHub Actions

## 완료 기준

- 관리자 터미널과 GitHub CI 모두 통과
- direct-v5 고유 구성 100개 이상
- 선택·fit 증거 충족
- 모든 품질 지표 통과
- Firebase rules 배포 완료
- 팀 리뷰와 PR 완료

이 조건 전에는 `published/recommend-model.json`을 교체하지 않는다.
