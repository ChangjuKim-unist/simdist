# 자산 목록 (2026-11-12 정리)

세 곳에 흩어진 문서·코드·결과·모델·데이터를 한 표로 모은다. 삭제는 하지 않았고, §6에 삭제·이동 후보를 묶어 두었다(사용자 승인 뒤 실행).

## 1. 문서 (`docs/`)

| 문서 | 상태 | 내용 |
|---|---|---|
| `research_log_ko.md` | **현행** | 설계 1~19 한 줄 판정표 + 남은 자산 + 교훈. 새 작업은 여기서 시작 |
| `design_branch_headroom_ko.md` | **현행** (설계 18, 완료) | 관문 0(§1~7), 보강 §8, 관문 1-축소판 §9(선택기, 재계획 구간 §9.7). 결론: DR 범위 안에서는 방법 개발 중단, 분석 논문 |
| `design19_real_failure_ko.md` | **현행** (설계 19, 등록·실기 대기) | 실기 실패 과제 찾기: 팔 P1~P5, 표면 2, 판정 경계, 이틀 일정 |
| `audit_upstream_ko.md` | 현행 (참고) | 원본 SimDist 대비 포크 감사. 버그 2건은 `audit-fixes` 브랜치(미병합) |
| `go1_run_manual_ko.md`, `deployment_go1.md`, `server_runbook_ko.md` | 현행 (운용) | 실기 절차, Go1 배포, gpu1·NAS 사용법 |
| `inventory_ko.md` | 현행 | 이 문서 |
| `adaptation.md`, `installation.md`, `pretraining_go2.md`, `deployment_go2.md` | 원본 SimDist 문서 | 수정하지 않음 |
| `docs/archive/` (18개) | 종료 | 설계 1~17 문서, `research_plan_ko.md`, `sim2sim_adaptation.md`. 판정은 research_log에 요약됨 |

Git-ignored 보고서: `reports/`(572 KB: sim2real 조사, 방법론 후보, 아이디어 탐색), `research_notes/`(2.3 MB). 저장소 밖이므로 NAS 백업 대상(§6 그룹 A).

## 2. 코드 (`scripts/`)

| 위치 | 현행 | 비고 |
|---|---|---|
| `scripts/*.py` 원본 9개 | generate_data, train_rl, train_model, finetune_model, process_data, aggregate_realworld_data, simulate_go2, play_rl, export_policies | 원본 SimDist. 손대지 않음 |
| `scripts/` 포크 추가 | `simulate_batch.py`(배치 평가기), `train_student_prior.py`(student BC), `analyze_hardware_runs.py`, `go1_round.sh`(실기 적응 라운드) | 설계 19가 씀 |
| `scripts/hw/` | `summarize_design19.py` | 설계 19 집계·판정 |
| `scripts/sim2sim/` | `branch_eval.py`, `select_offline.py`, `stage18_{gate0,supp,features,horizon}.sh`, `summarize_branch*.py`, `summarize_horizon.py`, `mark_terminated.py`, `wait_supp.sh` (설계 18); 분석 도구 `paired_stats.py`, `calib_head.py`, `calib_shift.py`, `gen_chunks.py`, `identify_physics.py`, `dr_from_posterior.py`, `return_error.py`, `trust_probe.py`, `run_pipeline.sh` | 현행 또는 재사용 가능 |
| `scripts/archive/` (2026-11-12 이동) | density_ratio_weights, prior_consistency_weights, distill_planner_to_student, repack_processed_data | 설계 7·9 전용 |
| `scripts/sim2sim/archive/` (60 + 2026-11-12 이동 11) | 설계 1~17 드라이버; ambiguity_by_condition, build_round_dataset, switch_trace, test_context_cpu, value_error, value_forgetting, visibility_{apply,diag,labelfree,probe}, summarize | 종료된 설계 전용 |
| `scripts/audit/` | equivalence_check.py, reproduce_go2.sh | 감사용 |

브랜치: `go1-support`(현행), `audit-fixes`(버그 수정 2건, 미병합, policy head 재학습 전에 병합 필요), 태그 `pre-restart-2026-11-11`.

## 3. 결과 (`results/`)

| 설계 | 로컬 `results/` | gpu1 | NAS `/nas_data/cjkim/simdist/` | git 추적 |
|---|---|---|---|---|
| 18 (현행) | `stage18/` 960 MB (분기 CSV 6 × 35 MB, 특징 npz 6 × 58 MB, 앵커, 요약 md 4, select_results.json) | `/mnt/newssd/cjkim/simdist/results/stage18/` 959 MB (원본) | `results/stage18/` 959 MB, 59 파일 | 요약 md·json 7개 |
| 17 | `stage17/` 580 KB | `~/simdist/results/stage17/` 31 MB | `results/stage17/` 31 MB, `traces_stage17/` 9.3 GB (31 npz) | 12 |
| 16 | `stage16/` 306 MB (트레이스 1개) | `~/simdist/results/stage16/` 307 MB | (NAS results에 stage16 없음 — gpu1 사본만) | 7 |
| 15 | `stage15/` 16 MB (gen_broad.npz 포함) | 16 MB | — | 6 |
| 14 | `stage14/` 2.4 GB (트레이스 8 × 320 MB) | 19 MB | `traces_stage14/` 5.5 GB (49 npz) | 17 |
| 13 | `stage13/` 2.4 GB (트레이스 8 × 320 MB) | — | `results/stage13/` 152 KB, `traces_stage13/` 17 GB (54 npz) | 0 |
| 4 | `stage4/` 359 MB (w_*.npz 가중) | — | 359 MB | 0 |
| 0~12, prior_coverage, sim2sim, prehw, real | 각 ≤ 8 MB | — | 같은 이름으로 있음 (+ server_logs, tests) | real 2 |
| audit | 4.8 MB (repro_go2.csv 등) | 200 KB | — | 5 |

설계 19 예정: 로컬 `results/real/stage19/`(순서표·기록지 템플릿 커밋됨) → NAS `results/stage19/`.

## 4. 모델 (`checkpoints/`)

| 종류 | 로컬 | NAS | 현행 |
|---|---|---|---|
| RL expert | `rl/` 37개, 57 GB | `checkpoints/rl/` 37개, 56 GB | **2026-10-01_10-38-36**(정상, go1_paper의 teacher), **2026-10-20_07-40-06**(넓은 motor DR 0.3~1.1, broadwm·generator·student_broad의 teacher). 나머지 35개는 설계 6·12 DR 폭 sweep 등 종료 |
| World model | `models/` 116개, 739 MB | 113개, 721 MB | **go1_paper**, **go1_broadwm**(+s2, s3), go1_ctx(+s2), go1_scratch는 NAS만, **go1_finetuned_r2/r3**(실기 적응), go1_paper_very_weak_motor_dyn_b45000_term·go1_paper_friction0.15_dyn_b45000_term(설계 18의 적응 모델). 나머지 ≈100개는 설계 1~12의 적응 변형(각 6~12 MB) |
| Student | `students/` 24개, 280 MB | 24개 | **student_broad**(설계 19 P2), student_narrow; w03~w10 폭 sweep, pfi*는 종료 |
| 기타 | — | `probes/` 3.6 MB, go1_broadwm_s2/s3 사본 | |
| generator | `results/stage15/gen_broad.npz` | — (git 추적) | 설계 18 base |

## 5. 데이터 (`datasets/`)

| 데이터 | 로컬 | gpu1 | NAS | 용도 |
|---|---|---|---|---|
| 2026-10-02_01-18-59 논문 사전학습 (102M 스텝) | 빈 디렉터리 | — | **141 GB** | go1_paper 학습 데이터. 유일본 |
| 2026-11-05_11-39-08 넓은 DR (19.2M 스텝, 물리 라벨) | 27 GB | `/mnt/newssd/.../datasets/sim/` 27 GB | 27 GB | broadwm·generator·student 학습. 설계 19 P5의 비교 기준 |
| 2026-10-02_01-04-39 | 83 MB | — | — | 초기 smoke 데이터 |
| `datasets/real/go1_simdist_controller` 실기 로그 (r1~r3, 29.5분) | 516 MB | — | 516 MB | 유일한 실기 데이터 |
| `datasets/real/sim2sim_*` 95개 (설계 1~12 적응 로그) | 8.1 GB | — | ≈ 8 GB | 종료된 설계의 sim2sim 적응 입력. 재생성 가능 |

NAS 전체: 우리 몫 ≈ 290 GB (datasets 176, checkpoints 57, traces 32, results 1.4, simdist_docker 이미지 24). NAS는 97 TB 중 95 TB 사용(99 %, 연구실 전체). gpu1 root 디스크 97 %(우리 홈은 0.5 GB뿐), `/mnt/newssd` 44 %.

## 6. 정리 후보 (승인 필요, 아직 실행 안 함)

| 그룹 | 내용 | 크기 | 제안 |
|---|---|---|---|
| A 백업 | `reports/`, `research_notes/`(git-ignored), `results/stage16/`·`stage15/`(NAS에 없음) | 0.3 GB | NAS `results/`로 복사 (삭제 아님, 승인만) |
| B 로컬 트레이스 | `results/stage13/traces`, `stage14/traces`, `stage16/traces` (17 npz) | 5.1 GB | NAS `traces_stage13/14`에 더 큰 집합이 있음(stage16은 gpu1에만) → stage16 트레이스를 NAS로 보낸 뒤 로컬 삭제 |
| C NAS 트레이스 | `traces_stage13/14/17` (134 npz) | 32 GB | 설계 13·14·17 종료, 판정은 research_log에 요약. 2026-11-11에 사용자가 보류한 그룹. 재검토 요청 |
| D sim2sim 적응 로그 | `datasets/real/sim2sim_*` 로컬 + NAS | 8 + 8 GB | 설계 1~12 입력. 모델 체크포인트가 남아 있고 재생성 가능 |
| E 적응 모델 변형 | `checkpoints/models/` 중 설계 1~12 변형 ≈100개 | 0.6 GB (로컬·NAS 각) | 작아서 보관 비용 없음. 목록만 research_log에 남기고 유지 권장 |
| F RL expert sweep | `checkpoints/rl/` 35개 (두 teacher 제외) | ≈ 53 GB (로컬·NAS 각) | 설계 6·12 DR 폭 sweep. 2026-11-11에 보류한 그룹. 재검토 요청 |
| G gpu1 홈 | `~/simdist/results/` 372 MB, 로그 | — | 루트 디스크 압박의 원인이 아님. 그대로 |

실행 규칙: 삭제는 그룹 단위로 승인받은 것만, 삭제 전 NAS 사본 확인, NAS 쓰기는 컨테이너를 통해(`~/simdist-docker/run.sh`), 로컬 root 소유 파일은 로컬 컨테이너로.

## 7. 정리 실행 (2026-11-14, 사용자 승인)
삭제 전 확인: 논문 데이터 gpu1 사본은 NAS 원본과 파일 단위 동일(12개), RL 37개는 로컬에 같은 크기(2026-11-05_06-23-44는 로컬이 더 완전), 로컬 stage14 트레이스는 NAS 집합에 포함.

| 묶음 | 삭제한 것 | 회수 |
|---|---|---|
| gpu1 논문 데이터 사본 | `/mnt/newssd/cjkim/simdist/datasets/sim/2026-10-02_01-18-59` + 저장소 심볼릭 링크 | gpu1 데이터 디스크 150 GB (52 → 44 %) |
| RL expert sweep NAS 사본 | NAS `checkpoints/rl/` 34개(teacher 2026-10-01_10-38-36·2026-10-20_07-40-06, go2_2026-02-11 유지). **로컬 사본 37개는 유지(유일본)** | NAS 51 GB |
| planner 트레이스 | NAS `traces_stage13`(17 GB)·`traces_stage17`(9.3 GB), 로컬 `results/stage13/traces`(설계 13 go1_ctx 8개, NAS에 없던 것)·`results/stage14/traces`. **NAS `traces_stage14` 5.5 GB 유지**(설계 18 §9.7 모순 건 미해결) | NAS 26 GB, 로컬 4.8 GB |
| sim2sim 적응 로그 | 설계 1~12 입력 `datasets/real/sim2sim_*` NAS 83개·로컬 85개. 유지 11개: `go1_simdist_controller`(실기), `sim2sim_{friction0.15,very_weak_motor,nominal}_go1_paper*`(설계 14·18·20 입력) | NAS 8 GB, 로컬 7 GB |
| 잡동사니 | 로컬 `wandb/`, `outputs/`, `checkpoints/models/smoke20_*` 3개 | 로컬 0.9 GB |

유지 결정: 설계 1~12 적응 변형 체크포인트(0.6 GB), 로컬 `results/stage16/traces`(0.3 GB, NAS 백업 안 함), student sweep. 결과: NAS 우리 몫 266 → 181 GB(datasets 169, traces_stage14 5.5, checkpoints 5.6, results 1.4), 로컬 datasets 29 GB·checkpoints 58 GB·results 1.7 GB. gpu1 root 100 %는 다른 사용자의 멈춘 컨테이너(≈105 GB)·미사용 이미지(68 GB) 때문이며 우리 홈은 3.4 GB — 관리자에게 prune 요청 필요.
