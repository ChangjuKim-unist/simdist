# 설계 문서 19: 실기 Go1에서 "가장 좋은 baseline이 눈에 보이게 실패하는 과제" 찾기 (2026-11-12 등록)

사용자 결정(2026-11-12): 설계 18(§8~9)의 결론 — DR이 덮는 간극(friction·motor·mass 범위 안)에서는 넓은 DR expert를 BC한 policy가 전 조건 넘어짐 0이고 planner·adaptation·selector가 더하는 것은 최대 +3 % — 를 받아들이고, 방법을 더 고르기 전에 **실패 과제를 실기에서 먼저 찾는다**. SimDist 논문(arXiv 2603.15759, RSS 2026)의 실패 과제는 DR 축이 아니라 시뮬이 모델링하지 않는 물리(PTFE 슬로프 + 발 필름, memory foam)로 만들어졌고, 비교군에 넓은 DR zero-shot policy가 없다. 이 설계는 그 빈 칸을 Go1으로 메운다.

## 0. 질문 (한 줄)

시뮬이 모델링하지 않는 표면(미끄러운 판, 폼)에서 **넓은 DR policy(planner 없음, adaptation 없음)** 가 눈에 보이게 실패하는가. 실패한다면 SimDist의 dynamics adaptation + planner가 그것을 고치는가, 그리고 실패 표면을 DR에 넣어 다시 학습한 policy도 고치는가.

## 1. 팔 (arm)

| 팔 | 내용 | 컨트롤러 설정 | 학습·적응 필요 | 회차 |
|---|---|---|---|---|
| P1 논문 설정 DR policy | `go1_paper`의 policy head만 실행(prior_only), 논문 레시피 DR(마찰 0.2~1.2, 질량 −1~+3, 모터 DR 없음) | `simdist_controller_go1.yaml`을 `checkpoint: go1_paper`, `prior_only: true`로 복사한 `hw_P1_paperpolicy.yaml` | 없음 | 1 |
| P2 넓은 DR policy | `student_broad`(모터 0.3~1.1·마찰 0.2~1.2·질량 −1~+3 expert의 BC student)만 실행 | `hw_M2_student.yaml` (prior_only: true) | 없음 | 1 |
| P3 SimDist planner, 적응 없음 | `go1_paper` world model + policy head prior + MPPI 200×3 (zero-shot) | `simdist_controller_go1.yaml`을 `checkpoint: go1_paper`로 복사한 `hw_P3_zeroshot.yaml` | 없음 | 1 |
| P4 SimDist, 적응 | 실패 표면에서 모은 real data ≈ 30분으로 latent dynamics만 fine-tuning(`go1_round.sh`, RECIPE=dyn, 40 epoch) + MPPI 200×3 | `hw_P4_adapted.yaml` (checkpoint = 새 이름 `go1_paper_hw19_dyn`) | 데스크톱 GPU 1~2시간(1일차 밤) | 2 |
| P5 실패 표면 DR policy | 1일차에 잰 실패 표면의 마찰·경사 추정값을 DR 범위에 넣어 expert를 다시 학습(PPO, gpu1)하고 student로 BC(`train_student_prior.py`, heads filter 경로가 아니므로 감사 버그와 무관) | `hw_P5_surfdr.yaml` (student = 새 이름 `student_surf19`) | gpu1 PPO ≈ 5시간 + 데이터 생성·student 학습 ≈ 3시간(1일차 밤~2일차 오전) | 2 |

P5의 DR 값: 미끄러운 판은 마찰 계수 하한을 0.05로(지금 0.2), 경사는 지형 커리큘럼의 경사면 0~6°를 켠다. 폼은 시뮬에 compliance 축이 없으므로 **대리**로 접촉 강성·감쇠 랜덤화(Isaac Lab `physics_material` 반발 0~0.3은 이미 있음, 추가로 지면 강성 scale 0.3~1.0)를 넣되, 이것이 폼을 덮는다는 보장은 없다고 적는다. 1일차 끝에 P2가 두 표면 모두에서 실패하지 않으면(갈래 1 조건) P5는 돌리지 않는다.

이전 하드웨어 설정(M1~M5, `docs/go1_run_manual_ko.md`)과의 관계: P2 = M2, P3 = M3에서 student를 뺀 것(논문 설정 그대로), P4 = M4의 표면판. `go1_finetuned_r3`(평지 29.5분 적응, M1)는 평지 데이터라 이 설계의 팔이 아니지만, 시간이 남으면 참고로 폼에서만 10회 돌린다.

## 2. 과제 (표면)

| 표면 | 구성 | 목표 길이 | 속도 (m/s) | 논문 대응 |
|---|---|---|---|---|
| A 미끄러운 판 | PTFE 시트가 있으면 PTFE, 없으면 아크릴/코팅 합판. 발에 PE 필름 또는 매끈한 테이프. **경사 0°에서 시작**, P2 성공률 ≥ 0.8이면 ≈5°(길이 1.8 m에 높이 0.16 m, 받침목)로 올려 다시 잰다. 실제 경사·판 재질·마찰 추정값(로봇을 끌어 당기는 힘 ÷ 무게)을 기록 | 1.8 m | 0.1 / 0.3 / 0.5 | Slippery Slope (PTFE, 3.0°·5.7°, 1.82 m, 0.1/0.3/0.5) |
| B 폼 | memory foam 5 cm 두 장(없으면 요가 매트 겹쳐 5 cm, 종류·두께 기록) | 3.0 m | 0.2 / 0.5 / 0.8 | Foam (5 cm, 3.0 m, 0.2/0.7/1.2). Go1 토크 상한 1.0(23.7 N·m)과 `go1.yaml` 속도 제한 때문에 상한 속도를 0.8로 낮춤 |

속도는 조종기 스틱이 아니라 `go1.yaml`의 `wirelesscontroller_maxvels`를 그 속도로 고정하고 스틱을 끝까지 밀어 명령한다(명령 속도를 로그로 확인).

## 3. 측정과 시행

- **칸** = (팔, 표면, 속도). **칸당 10회**가 기본. 배터리·온도·시간 때문에 못 채우면 채운 횟수와 사유를 결과 표에 적는다(최소 5회 미만인 칸은 판정에 쓰지 않는다).
- 시행: 출발선에 STAND → Start → 목표 길이 통과 또는 넘어짐 또는 30 s. 끈은 느슨하게 잡고, 넘어짐 판정은 RECOVERY 진입 또는 끈으로 받친 경우.
- **주 지표: 전진 거리 (m, 클수록 좋음)** — leg odometry가 미끄러운 판에서 틀릴 수 있으므로 바닥 눈금(0.2 m 간격 테이프)으로 사람이 읽고, 로그 odometry는 보조. 목표 길이에서 잘라 `min(거리, 목표)`.
- 부 지표: 성공(목표 통과, 넘어짐 없음) 비율, 넘어짐 비율, 명령 속도 대비 실제 평균 속도 비.
- 짝짓기: 같은 (표면, 속도)에서 팔을 섞어 교대로 돌리고(순서는 미리 난수로 뽑아 기록), 팔 간 차이는 칸 단위 bootstrap 95 % CI(시행 단위 재표집).
- 실행 전 준비물(2026-11-12 커밋): 컨트롤러 설정 `hw_P1_paperpolicy.yaml`·`hw_P3_zeroshot.yaml`·`hw_P4_adapted.yaml`·`hw_P5_surfdr.yaml`(P2는 `hw_M2_student.yaml`), 시행 기록지 `results/real/stage19/trials_template.csv`(열: arm, surface, slope_deg, speed, trial, distance_m, success, fell, note), 팔 순서표 `results/real/stage19/order.csv`(seed 19, 칸·시행마다 팔 순서), 집계 `scripts/hw/summarize_design19.py`(표·§4 판정 자동).
- 로그 이름 `hw_<팔>_<표면><경사>_<속도>` (예: `hw_P2_slip0_0.3`, `hw_P4_foam_0.5`). 분석 `scripts/analyze_hardware_runs.py` + 이 설계용 집계 `scripts/hw/summarize_design19.py`(실행 전 작성·커밋).

## 4. 판정 경계 (숫자, 실행 전 고정)

"팔 X가 표면 S에서 **눈에 보이게 실패**" = 세 속도 중 둘 이상에서 성공률 ≤ 0.5, **또는** 세 속도 평균 전진 거리 ≤ 목표의 0.6.
"팔 X가 표면 S에서 **문제 없음**" = 세 속도 모두 성공률 ≥ 0.8 **그리고** 평균 전진 거리 ≥ 목표의 0.9.

| 갈래 | 조건 (둘 다 표면 A는 최종 경사에서 판정) | 결론 |
|---|---|---|
| 1 분석 논문 | P2가 표면 A·B 모두 "문제 없음" | 논문의 이득은 넓은 DR policy를 비교에 넣지 않은 탓. §8~9 + 이 표가 본문. 방법 개발 없음 |
| 2 방법 논문 | 어느 표면 S에서 P2 "눈에 보이게 실패" **그리고** P4 − P2 성공률 ≥ +0.3(세 속도 합산, bootstrap CI가 0 제외) **그리고** P4 전진 ≥ 목표의 0.9 **그리고** P5가 S에서 "눈에 보이게 실패"하거나 P4 − P5 성공률 ≥ +0.3(CI 0 제외) | SimDist 기제가 넓은 DR·표면 DR로 대체되지 않음. 다음 설계는 그 기제의 약점(한 축 적응의 망각, value head 불일치, 후보 순위 실패)을 고치는 방법 |
| 3 범위 밖 | 어느 표면 S에서 P2·P4·P5 모두 "눈에 보이게 실패" | 시뮬 expert 너머가 필요(real-data RL). 이번 논문 범위 밖 → 갈래 1의 분석 논문에 "SimDist도 못 고치는 표면"으로 한 절 추가 |
| 불명 | 위 어느 것도 아님 (예: P2 성공률 0.6, P4 − P2 +0.2) | 칸당 10회 추가 후 재판정. 그래도 불명이면 "불명"으로 기록하고 방법을 고르지 않는다 |

갈래 2의 추가 조건(사용자 보강 3): P5가 SimDist보다 뚜렷이 못하지 않으면(P4 − P5 < +0.3) "표면을 DR에 넣는 것이 더 싸다"로 갈래 1에 합친다.

## 5. 예측 (실행 전, 제 확률)

- 표면 B(폼)에서 P2 "문제 없음": 0.6. 논문의 zero-shot(world model + planner)도 0.2 m/s에서 3/5였고, 넓은 DR student는 평지 실기에서 안정적이었다.
- 표면 A 경사 0°에서 P2 "문제 없음": 0.5. 5°에서 P2 "눈에 보이게 실패": 0.6.
- P1(논문 설정 DR policy)은 P2보다 모든 칸에서 같거나 나쁨: 0.8.
- P3(planner zero-shot)는 P2보다 나쁨(설계 14 "planner 해악"의 실기판): 0.6.
- P2가 실패한 표면에서 P4 − P2 ≥ +0.3: 0.35. P5 ≥ P4: 0.5.
- 갈래 확률: 1 = 0.4, 2 = 0.2, 3 = 0.15, 불명 = 0.25.

## 6. 실행 순서와 일정 (이틀 기준)

**1일차 (실기, 데스크톱 GPU는 로봇 전용 — `docker ps` 비어 있어야 함)**
1. 표면 설치, 눈금, 마찰 추정(판을 당기는 힘). 배터리 2개 이상 충전.
2. 평지에서 P2로 서기·0.3 m/s 보행 확인(매뉴얼 6단계 1~4).
3. **policy만 먼저**: 표면 A 0° → P1·P2 교대(칸당 10회 × 3속도). P2 "문제 없음"이면 5°로 올려 반복.
4. 표면 B: P1·P2 교대.
5. **그 다음 planner**: P3를 표면 A(최종 경사)·B에서. 넘어짐 2회 연속이면 그 칸은 5회에서 끊고 사유 기록.
6. **적응 데이터 수집**: 1~5에서 넘어짐이 가장 적었던 컨트롤러(예상 P2)로 표면 A·B 각 15분(속도 섞음, `dataset_name:=hw_P4_collect_<표면>`). 논문은 적응 중인 planner로 on-policy 수집(Algorithm 1)이지만, 1회차는 가장 안전한 제어기로 off-policy 수집한다는 차이를 적는다. 시간이 남으면 P4로 2회차 10분을 더 모아 `go1_paper_hw19_dyn_r2`를 만든다.
7. 밤: `PREV=go1_paper DATASET=hw_P4_collect NEW=go1_paper_hw19_dyn RECIPE=dyn scripts/go1_round.sh` (데스크톱). 동시에 gpu1(빈 장만)에서 P5의 expert PPO → 데이터 생성 → student.

**2일차**
1. P4를 표면 A·B에서(칸당 10회 × 3속도), P2를 각 표면 1속도씩 5회 재측정(날짜 간 drift 확인).
2. P5가 준비됐으면 같은 칸. 안 됐으면 사유 기록, 3일차 오전으로.
3. 집계, 표 4의 판정, 문서 §8.

**안전**: 토크 상한 1.0(23.7 N·m, 시작 로그 `TORQUE CAP` 확인), 끈 필수, 모터 60 °C에서 휴식, 배터리 30 % 이하 교체. 미끄러운 판 주변 1 m 비움. 경사 판은 아래쪽에 받침 고정.

## 7. 논문 수치와의 대조 (원문 표 IX, Go2; 전진 m / 5회 중 성공)

| 과제 | 속도 | zero-shot (world model + planner) | 1스텝 BC | IQL | RLPD | SimDist (≈33~36분 적응) |
|---|---|---|---|---|---|---|
| Slippery Slope | 0.1 | 0.70 / 0 | 1.49 / 2 | 0.00 / 0 | 0.32 / 0 | 1.78 / 4 |
| Slippery Slope | 0.3 | 0.43 / 0 | 1.43 / 1 | 0.39 / 0 | 0.34 / 0 | 1.82 / 5 |
| Slippery Slope | 0.5 | 0.90 / 0 | 0.62 / 0 | 0.46 / 0 | 0.35 / 0 | 1.82 / 5 |
| Foam | 0.2 | 2.98 / 3 | 2.07 / 1 | 0.92 / 1 | (제외: 로봇 불안정) | 3.00 / 5 |
| Foam | 0.7 | 2.39 / 2 | 1.54 / 1 | 2.25 / 2 | — | 3.00 / 5 |
| Foam | 1.2 | 1.70 / 0 | 2.45 / 2 | 2.73 / 3 | — | 3.00 / 5 |

출처: arXiv 2603.15759 HTML v2, Experiments 절 표 IX(2026-11-12 열람). 논문의 비교군에는 넓은 DR zero-shot policy와 history adaptation(RMA류)이 없다. 우리 P1 ≈ 논문의 "1스텝 BC"(단, 우리는 policy head의 25스텝 chunk 중 첫 행동 실행), P3 ≈ 논문의 zero-shot, P4 ≈ SimDist.

**Go2(논문) ↔ Go1(우리) 차이**: 로봇 Go1 EDU(RR 발 힘센서 고장, leg odometry로 속도 추정, lidar 없음); 토크 상한 23.7 N·m 전 관절(실제 calf 35.55 N·m를 시뮬 자산에 맞춰 자름); planner 200×3(논문 450×8); 속도 상한 0.8(폼); 표면은 실험실에 있는 재료로 대체(재질·경사 기록); 적응 데이터는 1회차 off-policy. 논문 숫자와 우리 숫자를 같은 표에 넣지 않고 나란히 둔다.

## 8. 규칙

- 등록·커밋·푸시 후 실행. 실행 전에 바꾼 것은 이 문서에 날짜와 함께 적는다.
- 판정 경계(§4)와 예측(§5)은 실행 뒤 바꾸지 않는다. 칸 수 부족·장비 문제는 사유만 기록.
- 원시 로그(`datasets/real/.../raw/<날짜>/`)·집계 CSV(`results/real/stage19/`)는 NAS `/nas_data/cjkim/simdist/results/stage19/`에 백업. 삭제 금지.
- gpu1은 읽기 전용 상태 확인 뒤 빈 장만(P5의 PPO·데이터 생성).
- 결과 보고: 표 하나(행 팔, 열 표면×속도, 칸 = 전진 거리 평균 / 성공 횟수, 숫자마다 무엇을 잰 것인지), §4 갈래 판정 한 줄, §5 예측 성립/기각 한 줄씩, 논문 표 IX와 나란히.
