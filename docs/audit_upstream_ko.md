# 원본 SimDist 대비 구현 감사 (2026-11-11)

포크 `go1-support`(170 커밋)가 원본 SimDist(CLeARoboticsLab/simdist)에서 Go2→Go1 이식, 모터 DR 축 추가, 평가기 수정을 거치며 (a) 방법이 달라졌는지, (b) 버그가 들어갔는지 확인한다. 논문에서 "SimDist 기준선"이라 부를 근거가 목적이다.

분류: **[동일]** 원본과 같은 동작 · **[의도·문서화됨]** 설계 문서나 코드 주석에 근거가 있음 · **[의도·미문서]** 의도된 변경이지만 기록이 없었음(이 문서가 기록) · **[버그]** 의도치 않은 동작 · **[원본 의심]** 원본에도 있는 문제.

## 0. 요약

- **심각: 없음.** 모든 옵션을 끈 포크는 같은 체크포인트·같은 입력·같은 난수에서 원본 코드와 모델 출력·계획기 출력이 비트 단위로 일치한다(§3). 환경 정의(관절 순서·게인·보상·종료·명령·DR)는 Go1 자산 교체를 빼면 원본 코드 그대로다. Go2 저자 체크포인트의 시뮬 재현은 §4.
- **보통 (버그 2건, `audit-fixes` 브랜치에서 수정, 병합은 사용자 결정):** ① heads 필터가 최상위 모듈이 아니라 경로 안의 키를 찾아 `policy_only`가 정책 출력층 `policy_dec`를 학습시키지 않았고(동결) 동결해야 할 `prior_anchor.policy`를 포함했음 → 정책 헤드 재학습을 쓴 설계 3·8·10·11·12의 모델은 출력층이 사전학습 값 그대로였음(재실행 최소 11 GPU시간, §5) ② 단일 로봇 평가기 로거의 넘어짐 플래그가 첫 에피소드 길이로 모든 에피소드를 판정 → 시뮬-투-시뮬 로그의 `terminated` 열 오류, 영향은 설계 1·2의 가치 적응(이미 기각된 가지)뿐.
- **보통 (방법이 달라진 의도적 변경, 논문의 "SimDist 대비 변경점" 표):** ① 평가기가 리셋마다 뽑던 관절 게인 ±10 %·관절 마찰 0~0.05 랜덤화를 끄고 고정 공칭값으로 평가(원본 평가기는 난이도 1.0으로 랜덤화를 유지) ② 넓은 모터 DR 전문가의 PPO 표본 수가 논문 레시피의 1/4(1024 env × 24 스텝 × 5000) ③ 평가 프로토콜(평지 100 m 슬랩, 에피소드 100 s, 5000 스텝, 18대, 고정 게인) ④ 하드웨어 계획기 예산 200 표본 × 3 반복(논문 450 × 8).
- **경미:** ① 학습기의 검증 손실이 학습 모드(관측 잡음 증강)로 계산됨(원본도 같음) ② 체크포인트 로더가 모델·체크포인트의 교집합만 복원하고 나머지는 개수만 출력(맥락 슬롯 warm start가 이 동작에 의존하므로 유지) ③ 배치 평가기의 `run_reward_per_step` 분모가 리셋 직후 워밍업 25스텝을 포함(모델 간 비교에는 영향 없음) ④ 배치 평가기가 컴파일 스텝 뒤 계획기를 리셋하지 않았음(수정, 영향 없음) ⑤ 단일 로봇 평가기(원본)는 시간 초과 리셋 때 제어기를 리셋하지 않음(1000스텝 평가에서는 발생 안 함; 포크는 수정) ⑥ 모터 배율 > 1은 토크-속도 곡선 때문에 거의 무효(DR 상한 1.1의 라벨 해석에만 관련).

## 1. 기준점

| 항목 | 값 |
|---|---|
| 원본 저장소 | `origin` = https://github.com/CLeARoboticsLab/simdist.git, `origin/main` |
| 분기점 (merge-base) | `b63f76b` "Update readme and gif" |
| 분기 이후 원본 커밋 | **없음** (origin/main == 분기점; 이식할 원본 수정 없음) |
| 포크 커밋 수 | 170 (`b63f76b..e00c537`) |
| 비교 방법 | 원본을 `/home/rml/simdist-upstream`에 worktree로 체크아웃해 파일별 diff, 동치 테스트는 같은 컨테이너에서 `PYTHONPATH`로 원본 패키지를 앞세워 실행 |
| 코드 diff 규모 | simdist/ +1,900 −130, config/ +520, scripts/simulate_go2.py +100 −7, 신규 scripts/simulate_batch.py·mppi_batch.py·go1.py 등 (docs/paper/results 제외) |

원본 Isaac Lab 서브모듈(0.33.17, Isaac Sim 4.5)은 두 쪽이 같은 커밋이다.

## 2. 변경점 표

### A. 환경·자산

| 파일 | 내용 | 분류 | 결과에 미치는 영향 | 조치 |
|---|---|---|---|---|
| simdist/rl/go1.py | 로봇 자산을 Isaac Lab `UNITREE_GO1_CFG`(go1.usd)로 교체. DC 모터 모델 토크 한계 23.7 N·m / 속도 30 rad/s (Go2 23.5 / 30), Kp 25·Kd 0.5·관절 마찰 0은 Go2와 동일. 기본 자세 ISAAC 값 동일 | [의도·문서화됨] | 실제 Go1 종아리(35.55 / 20.06)와 다르다는 한계는 go1-calf 메모·sim2real 조사에 기록됨 | 없음 (하드웨어 토크 캡 1.0 유지) |
| simdist/rl/go1.py | 몸통 이름 `base`→`trunk` (높이 스캐너·IMU·질량 DR·외력·질량 관측·종료 접촉) ; IMU 오프셋을 go1_description URDF 값으로 | [의도·문서화됨] | IMU 가속도는 기록용 관측에만 있고 월드모델 입력(35차원)에는 없음 → 모델 무관 | 없음 |
| simdist/rl/go1.py | 종료 접촉 몸체 `trunk, .*_thigh` (Go2: `base, .*_thigh, Head.*`; Go1은 머리 몸체 없음) | [동일] | 넘어짐 정의 동일 | 없음 |
| simdist/rl/go2.py (공유) | 관절 순서(Unitree 순서, preserve_order), 행동 스케일 0.25, 제어 50 Hz(dt 0.005 × 4), 관측 항·잡음(선속도 0.05, 각속도 0.1, 중력 0.025, 관절각 0.01, 관절속도 0.75, 높이 0.02), 보상 항·가중치, 종료(접촉 + 자세 60°), 명령 범위(vx −0.5~1.5, vy ±0.5, yaw ±1.0), 지형 커리큘럼, DR(질량 −1~+3 kg, 마찰 0.2~1.2, 반발 0~0.3, 게인 ±10 %, 관절 마찰 0~0.05), 외력 밀기 꺼짐 | [동일] | Go1 환경은 Go2EnvCfg를 상속하고 위 항목만 바꿈. 포크의 go2.py 변경은 물리 라벨 기록기(PhysicsParamsRecorder) 추가뿐 | 없음 |
| config/system/go1.yaml, go1dr.yaml | go2.yaml과 `name`만 다름(관측 차원·증강 잡음·행동 잡음 범위 동일) | [동일] | — | 없음 |
| simdist/rl/go1.py `Go1DR*` | 모터 세기 DR 축: 리셋마다 `effort_limit`(토크 캡)을 U(0.3, 1.1) 배 (saturation 곡선은 그대로). 환경변수로 마찰·질량·모터 범위 덮어쓰기 가능 | [의도·문서화됨] (설계 4·6·11) | 데이터 생성(Go1DRRecordEnvCfg)·평가(`sim.motor_strength`, effort_limit만 스케일)·되감기 식별(PD 토크 요구를 scale × 23.7로 캡)·하드웨어 브리지(호스트 PD 토크를 ±scale × 23.7로 클램프) 네 곳이 같은 정의. 단 하드웨어 캡에는 시뮬 DC 모터의 토크-속도 곡선(23.7 × (1 − q̇/30))이 없음 — 실제 모터의 물리 곡선이 대신함 | 없음 |
| scripts/simulate_batch.py, scripts/simulate_go2.py | 평가 환경에서 리셋마다 뽑던 관절 게인 ±10 %·관절 마찰 0~0.05 랜덤화를 끔(`events.joint_stiffness_and_damping = None`, `joint_friction = None`), env 시드 고정. **원본 평가기(Go2SimEnvCfg)는 커리큘럼 항만 끄는데 난이도 초기값이 1.0이라 두 랜덤화가 전폭으로 켜진 채 평가함** | [의도·문서화됨 (코드 리뷰 2026-11-03)] → 논문 대비 **[의도·미문서]** | 원본의 시뮬 수치는 게인·관절 마찰이 매 에피소드 흔들리는 조건이고 우리 수치는 공칭 고정. 2026-11-03 이후 모든 표(설계 12~18)가 고정 조건. 논문의 변경점 표에 적을 것 | 이 문서에 기록 |
| scripts/simulate_batch.py, simulate_go2.py | 지형 테두리 100 m(원본 20 m: 15 m 지형 + 20 m 테두리 = 반폭 27.5 m. 1000스텝 × 1.2 m/s = 24 m라 원본 프로토콜에서는 안 닿고, 우리 5000스텝 런에서는 닿아 넘어짐으로 섞였음) | [의도·문서화됨] | 원본 프로토콜(1000스텝)에는 영향 없음 | 없음 |
| scripts/simulate_go2.py | 시간 초과(truncated)에도 제어기 리셋 (원본은 terminated에만 리셋: 시간 초과 때 로봇은 환경이 옮기지만 제어기 히스토리·MPPI 상태는 이어짐) | [원본 의심] (포크에서 수정) | 원본 프로토콜(1000스텝 = 에피소드 20 s)에서는 시간 초과가 런 끝과 겹쳐 발생하지 않음 | 없음 |
| scripts/simulate_go2.py, simulate_batch.py | `sim.motor_strength`, `sim.episode_length_s`, RL 전문가로 행동(`policy.rl_run`), 학생 prior, 실데이터 형식 로깅, 전진 거리·리셋 수 지표 추가 | [의도·문서화됨] | 기본값은 원본과 같음 | 없음 |

### B. 데이터

| 파일 | 내용 | 분류 | 결과에 미치는 영향 | 조치 |
|---|---|---|---|---|
| checkpoints/rl/2026-10-01_10-38-36 (go1_paper 전문가) | PPO 설정은 Go2 설정 상속(`UnitreeGo1RoughPPORunnerCfg`, 5000 반복). 환경 수 1024 × 반복당 96 스텝 (논문 4096 × 24) = 4.9억 스텝으로 총 표본은 동일, 롤아웃 길이만 다름 | [의도·미문서] | 총 표본 같음. 전문가 품질은 go1_paper 정상 조건 성능으로 간접 확인(§4) | 이 문서에 기록 |
| checkpoints/rl/2026-10-20_07-40-06 (넓은 모터 DR 전문가) | 1024 env × 24 스텝 × 5000 반복 = 1.2억 스텝 (논문 레시피의 1/4) | [의도·미문서] | 설계 4~18의 넓은 prior·go1_broadwm·생성기가 이 전문가에서 나옴. 전문가가 덜 수렴했을 가능성은 표본 수만으로는 판단 불가(설계 17에서 이 전문가는 전 조건 넘어짐 0) | 이 문서에 기록. 재학습은 요청 시 |
| datasets/sim/2026-10-02_01-18-59 (go1_paper 데이터) | 생성 설정(전문가 확률 0.5, 비전문가 체크포인트 11개, 행동 섭동 구간 U[1,50]/U[25,500], never-noise 0.025)은 원본 config와 동일. 384 env × 266,667 스텝 = 1.024억 스텝 (논문 4096 × 25,000 = 1.024억) | [동일 (env 수만 다름)] | 에피소드 20 s라 env 수 차이는 에피소드 구성에 영향 없음 | 없음 |
| datasets/sim/2026-11-05_11-39-08 (넓은 DR 데이터) | 시스템 go1dr(모터 DR 포함), 384 × 50,000 = 1,920만 스텝, 물리 라벨 포함 | [의도·문서화됨] (설계 13) | — | 없음 |
| simdist/data/data_processor.py, paths.py | terminal.hdf5(에피소드 끝까지 스텝 수, 할인 reward-to-go, 넘어짐 여부)·physics.hdf5 추가, HDF5 청크를 시간축으로 | [의도·문서화됨] | 기존 파일(proprio/extero/acts/cmds/rewards/values/flags)과 정규화 통계 계산은 동일. 체크포인트의 정규화 통계가 Go1 데이터의 것인지는 §3 | 없음 |
| simdist/data/dataset.py | 라벨에 steps_to_end·rtg·terminated·physics·prior_weight 추가(파일 없으면 기본값), spawn 워커용 pickle 처리 | [동일] | 원본 손실은 이 키를 쓰지 않음 | 없음 |
| simdist/data/episode_aggregator.py | 선택 키(reward, terminated) 복사, mtime 동률 시 경로 순, `max_steps` 예산 | [의도·문서화됨] | — | 없음 |
| simdist/modeling/trainer.py | 학습/검증 분할을 `random_split`(무작위 창)에서 앞 99.8 % / 뒤 0.2 %(통째 에피소드)로 | [의도·미문서] (RAM) | 검증 손실의 정의만 바뀜(학습 창과 겹치지 않는 에피소드). 모델 선택에 쓰지 않음(최신 스텝 사용) | 이 문서에 기록 |

### C. 모델·손실·학습

| 파일 | 내용 | 분류 | 결과에 미치는 영향 | 조치 |
|---|---|---|---|---|
| simdist/modeling/models.py | 가치 헤드 생성을 `_build_value_head`로 분리, 잠재 동역학을 `_latent_dynamics`로, 가치 계산을 `value_from_latents`로 분리. 선택 모듈(맥락 슬롯, 잔차 동역학, 잔차 가치, 가치 앵커, 정책 앵커)은 설정이 켜질 때만 생성 | [동일] (옵션 꺼짐) | §3 동치 테스트로 확인 | 없음 |
| simdist/modeling/losses.py | world_model 손실에 context·context_consistency 항(슬롯 없으면 0), prior_weight·boundary_weight 옵션(기본 꺼짐). 손실 `world_model_value_adapt`, `policy_refresh` 신규 | [의도·문서화됨] | 기본 가중치(latent 1, reward 1, value 1, action 4) 동일 | 없음 |
| simdist/modeling/trainer.py | 검증 루프 직전 `training = True` → 검증 데이터도 증강(관측 잡음) 켜진 채 평가. 원본 코드도 같은 줄(원본 trainer.py:234) | [원본 의심] | 검증 손실이 학습 분포로 계산될 뿐 학습·체크포인트에 영향 없음 | 유지(원본과 같은 정의 유지) |
| simdist/modeling/trainer.py | DataLoader 워커 forkserver, 적응 훅(잔차·앵커·맥락 추가), 잔차 전용 학습률, 시뮬 리플레이 | [의도·문서화됨] | 기본 경로(옵션 꺼짐)는 원본과 같은 연산 | 없음 |
| simdist/utils/model.py `load_model_from_ckpt` | GPU 체크포인트의 CPU 복원 대체 경로; 체크포인트와 모델의 교집합만 복원하고 나머지는 개수만 출력 | [의도·미문서] | 파라미터가 빠진 체크포인트를 조용히 초기값으로 채울 수 있음. 맥락 슬롯 warm start(사전학습 모델 + 새 슬롯)가 이 동작에 기댐 | 유지. 출력 줄로 확인 가능 |
| checkpoints/models/go1_paper | 출처: 데이터 2026-10-02_01-18-59, config = 원본 train_model.yaml(배치 512, 2 에폭, lr 2e-4→1e-4 코사인, warmup 10k, 증강 잡음 켬), 최종 368,920 스텝(논문 3.69 × 10⁵) | [동일] | — | 없음 |
| config/model/quadruped_world_model.yaml | 구조(latent 64, 동역학 2층 8헤드, 보상·가치 1층, 정책 4층 8헤드, H = T = 25) 동일; `context` 블록(기본 꺼짐) 추가 | [동일] | — | 없음 |

### D. 계획기

| 파일 | 내용 | 분류 | 결과에 미치는 영향 | 조치 |
|---|---|---|---|---|
| config/control/mppi.yaml | 표본 450, 정책 표본 22(0.05 × 450), 반복 8, 초기 std 2.0, 최소 std 0.05, 정책 잡음 0.05, 엘리트 64, 온도 0.25, 모멘텀 0, 할인 0.99, 지연 1 — 원본과 동일. 추가 옵션(init_mode shift = 원본 warm start, prior_only, noise_corr 0, noise_schedule·consistency_gate·calib·gen 꺼짐) | [동일] | — | 없음 |
| simdist/control/mppi.py | `prior_only` 경로(정책 헤드의 25스텝 행동열을 계획 없이 그대로 반환, 첫 행동 실행)와 외부 prior 훅. 기본 경로는 원본 그대로 | [의도·문서화됨] | prior_only = 정책 헤드의 첫 행동 = 행동 복제 정책. §3에서 정책 헤드 출력과 일치 확인 | 없음 |
| simdist/control/mppi_batch.py | N대 로봇을 한 배치로 계획. 알고리즘(정책 표본 잡음, shift 초기화, 반복별 top-k·softmax 가중 평균·std 갱신·clip, 수익 = Σγᵗr + γ^(T+1)V)은 원본과 같은 식. 난수 키만 로봇별 `fold_in(key(seed), reset_count)` | [의도·문서화됨] | 같은 키를 주면 단일 계획기와 같은 결과(§3). 난수열 자체는 원본 단일 런과 다르지만 분포는 같음 | 없음 |
| 행동 클리핑 | 원본·포크 모두 계획기에 행동 클리핑 없음(환경의 관절 한계만) | [동일] | — | 없음 |
| go2_ros2_ws/.../simdist_controller_go1.yaml | 하드웨어: 표본 200, 반복 3, 엘리트 32(RTX 3060, 20 ms 예산) | [의도·문서화됨] | 하드웨어 라운드에만 해당 | 없음 |

### E. 적응

| 파일 | 내용 | 분류 | 결과에 미치는 영향 | 조치 |
|---|---|---|---|---|
| config/finetune_model.yaml, heads/world_model_dynamics_only.yaml | 기본 레시피 동일: 손실 latent_dynamics만, 학습 대상 `dynamics`(경로에 "dynamics"가 있는 파라미터 = 잠재 동역학 트랜스포머; 인코더·행동 임베딩·보상·가치·정책 동결), lr 2e-4(warmup 2000), 배치 32, 100 에폭, 분할 0.95 | [동일] | 포크의 `adapt:` 블록은 전부 기본 꺼짐 | 없음 |
| simdist/utils/model.py `create_param_filter` | 원본 그대로(PathContains). 새 모듈 이름(dyn_res, value_res, prior_anchor, context)은 기존 필터 "dynamics"/"policy"/"value"에 잡히지 않도록 명명 | [동일] | — | 없음 |
| 수집 절차 | 시뮬-투-시뮬: 평가기 로깅(실데이터 형식 + 보상 + terminated) → aggregate → process → finetune. 하드웨어: 컨트롤러 노드 로거 + 넘어짐 이벤트 | [의도·문서화됨] | — | 없음 |

### F. 평가기

| 항목 | 원본 (simulate_go2.py) | 포크 (simulate_batch.py) | 분류 |
|---|---|---|---|
| 넘어짐 정의 | 환경 종료(base_contact ∪ bad_orientation) | 같음; `time_out`은 따로 셈(`num_timeouts`) | [동일] |
| 리셋 처리 | terminated에만 제어기 리셋, 리셋 후 25스텝 영행동·영명령 | terminated ∪ truncated에 제어기 리셋, 25스텝 영행동·영명령 동일 | [의도·문서화됨] |
| 조건 주입 | 마찰(정·동 동일값), 반발, 질량 | 같음 + 모터 세기(effort_limit 스케일) | [의도·문서화됨] |
| 보상/스텝 | 첫 에피소드 보상 / (길이 − 25) | 런 전체: 워밍업 외 스텝 보상 합 / 전체 스텝 수(워밍업 포함) → 리셋 수에 비례해 조금 낮게 나옴 | [경미] (모델 간 비교는 동일 정의) |
| 넘어짐/1000스텝 | 없음 | num_falls / (total_steps / 1000) | [의도·문서화됨] |
| 전진 | 없음 | 첫 에피소드의 x 변위(리셋 직전 위치 − 시작 위치) | [의도·문서화됨] |
| 속도 명령 | WalkStraightCommand(위치 오차 PD) 1대 | 같은 항을 로봇별 forward_vel로 패치(x_des·오차 계산식 동일) | [동일] |
| 지형 | 15 m × 15 m 1칸, 테두리 20 m | 테두리 100 m, 시드 고정, 게인·관절 마찰 랜덤화 끔 | §A 참조 |

### G. 하드웨어 브리지 (후순위)

| 항목 | Go2 (원본) | Go1 (포크) | 분류 |
|---|---|---|---|
| 관절 순서·기본 자세·행동 스케일 | Unitree 순서, 0.25 | 같음 (`config/go1.yaml`, `simdist_controller_go1.yaml`) | [동일] |
| 보행 게인 | Kp 25 / Kd 0.5 | Kp 25 / Kd 0.5 (2026-11-03까지 Kd 1.0이었음) | [의도·문서화됨] |
| 토크 캡 | 없음 | 호스트 PD 토크 ±scale × 23.7 N·m, 항상 켬 | [의도·문서화됨] |
| 기저 속도 | Point-LIO/모캡 | 다리 오도메트리 + IMU + 접촉 | [의도·문서화됨] (deployment_go1.md) |
| 높이 맵 | 라이다 elevation map | 평지 가정 평면 맵(스탠스 다리로 몸 높이) | [의도·문서화됨] |
| 계획기 | 실행 중에만 | 서 있을 때도 계획기를 돌려 분포를 예열(결과는 버림) | [의도·문서화됨] |
| 안전 | — | 명령 끊김 watchdog, 넘어짐 이벤트 → 로그 terminated | [의도·문서화됨] |

## 3. 동치 테스트

스크립트 `scripts/audit/equivalence_check.py`, 로컬 컨테이너(JAX 0.6.0). 체크포인트 go1_paper(368,920 스텝), 입력은 사전학습 데이터의 실제 창 8개(시드 0, 창 시작 1.69M~86.9M), 계획기는 창 0의 히스토리 26스텝을 버퍼에 넣고 2스텝 실행(두 번째 스텝은 shift warm start 검사). 원본은 `PYTHONPATH=/workspace/upstream`로 원본 패키지를 먼저 올려 같은 컨테이너에서 실행. 결과 파일 `results/audit/{upstream,fork}_cpu.npz`, 로그 `results/audit/run_cpu2.log`.

| 비교 | 최대 절대 차이 (CPU) | 기준 크기 | 의미 |
|---|---|---|---|
| 모델 출력 latents / rewards / values / actions (원본 코드 vs 포크, 창 8개) | 0 / 0 / 0 / 0 | 2.07 / 0.055 / 3.89 / 3.87 | 옵션을 끈 포크의 순전파는 원본과 비트 단위 동일 |
| 단일 로봇 MPPI 난수 키 | 0 | — | 같은 시드에서 같은 키 |
| 단일 로봇 MPPI 행동열 1스텝 / 2스텝 (원본 vs 포크) | 0 / 0 | 2.80 / 2.77 | 기본 경로의 계획기는 원본과 비트 단위 동일 |
| 배치 계획기 vs 단일 계획기 (같은 키, N = 1) 1스텝 / 2스텝 | 9.5 × 10⁻⁷ / 7.2 × 10⁻⁷ | 2.80 | 엘리트 가중 평균을 `Σw·a` 꼴로 쓰는 부동소수 차이뿐 |
| prior_only 계획 vs 정책 헤드 행동열 (배치 / 단일) | 1.4 × 10⁻⁶ / 1.4 × 10⁻⁶ | 3.87 | prior만 실행 = 정책 헤드의 25스텝 행동열, 첫 행동 실행 |
| 체크포인트 정규화 통계 vs Go1 데이터셋 `scaler_params.json` (6개 키 × 평균·표준편차) | ≤ 1.2 × 10⁻⁷ | — | Go1 데이터의 통계. Go2 값 없음 |

GPU(RTX 3060, 기본 TF32 행렬곱)에서는 같은 비교가 latents 2.4 × 10⁻³, actions 3.6 × 10⁻³(상대 1e-3), 배치 vs 단일 계획기 4.7 × 10⁻²로 나온다. 커널 선택·누적 순서 차이가 커지는 TF32 특성이고, 같은 코드끼리도 생기는 수준이다(`results/audit/run3.log`). 결론은 CPU 결과로 판단한다. 원본 저장소의 단위 테스트 2개(`tests/test_control.py`, `tests/test_models.py`, 두 저장소에서 동일 파일)도 포크에서 통과.

## 4. 재현 테스트

- **Go2 (원본 설정)**: 저자가 원본 저장소에 커밋한 체크포인트 `checkpoints/models/go2_2026-02-12`(원본 커밋 `c0d9eac` "Add pretrained checkpoints", 367,004 스텝)를 우리 배치 평가기로 돌렸다. 논문 부록 D의 프로토콜(속도 × 마찰 × 지형 × 난이도 전 조합, 조건당 1 에피소드, 1000스텝 또는 넘어짐까지, 지표 = 에피소드 누적 보상의 평균)과 표 I의 SimDist 사족 수치 **22.78**이 기준. 축소 격자: 지형 6종(boxes, random_rough, pyramid_stairs, pyramid_stairs_inv, hf_pyramid_slope, hf_pyramid_slope_inv) × 마찰 {0.2, 0.7, 1.2} × 난이도 {0.2, 0.6, 1.0} × 속도 {0.2, 0.4, 0.6, 0.8, 1.0, 1.2}(한 런에 6대) = 54런, 324 에피소드, 시드 42. 드라이버 `scripts/audit/reproduce_go2.sh`, 결과 `results/audit/repro_go2.csv`.

(진행 중 — 완료 후 채움)

- **Go1 기준값**: go1_paper의 정상 조건 성능(배치 평가기, 평지, 18대 × 5000스텝, 설계 12 이후 고정 게인 조건)은 `results/stage10/eval_core.csv`의 수치를 기준값으로 둔다. 설계 17 기록: 보상/스텝 0.0352(prior만), 넘어짐/1000스텝 0.011(go1_broadwm nominal).

## 5. 고친 버그와 커밋

브랜치 `audit-fixes`(go1-support 위, 커밋 `c475eca`, `mine`에 푸시, **go1-support에 병합하지 않음**). 회귀 테스트 `tests/test_audit_fixes.py`(4개, 컨테이너 CPU에서 통과). 병합은 사용자 결정.

| 버그 | 위치 | 수정 | 영향받는 과거 결과 | 재실행 |
|---|---|---|---|---|
| **heads 필터가 최상위 모듈이 아니라 경로 안의 키를 찾음.** `nnx.PathContains("policy")`는 경로 원소가 정확히 `policy`인 파라미터를 고르므로 `policy_dec`(정책 출력 선형층)는 빠지고 `prior_anchor.policy`(동결해야 할 복사본)는 들어감. `config/heads/policy_only.yaml`의 주석("policy, policy_dec를 잡음")이 틀렸음. 원본의 `create_param_filter`와 같은 코드이지만 원본은 `dynamics` 필터만 썼으므로 원본에는 증상이 없음 | `simdist/utils/model.py:125`, `config/heads/policy_only.yaml` | 한 토막 이름은 최상위 속성 이름과 정확히 일치(`_TopLevel`), `policy_only.yaml`에 `policy_dec` 추가. 다른 heads 파일(dynamics, value_emb/value/value_dec, value_res, dyn_res, context)은 전부 최상위 이름이라 선택 결과 불변 | **정책 헤드 재학습을 쓴 설계 전부**: 설계 3 관문 2(policy_refresh; 앵커 복사본의 기울기는 stop_gradient라 0이므로 앵커는 실제로 안 움직였고, `policy_dec` 동결만 영향), 설계 8 `go1_paper_broadhead`, 설계 10 `go1_paper_targethead`, 설계 11(stage7, targethead 씨드), 설계 12(stage9, 폭 규칙 헤드·마찰 블랭킷 헤드). 이 모델들은 정책 트랜스포머만 재학습되고 64→12 출력층은 사전학습 값 그대로였다. 학습은 됐으나(설계 8 모터 0.4 넘어짐 4.12→1.49) 제약된 파라미터화였으므로 "정책 헤드만 재학습" 결론(정상 보상 −22 %, 마찰 2배 악화)의 크기는 재확인 대상. 설계 13~18(go1_broadwm은 heads=all, 생성기는 별도 코드)은 무관 | 최소: `go1_paper_broadhead`·`go1_paper_targethead` 재학습(각 60k 스텝 ≈ 1.3 GPU시간) + 설계 8·10 핵심 평가(nominal·motor0.4·friction0.15 × prior만·계획기, 18대 × 5000스텝 ≈ 40분/런 × 12런 ≈ 8 GPU시간) ≈ **11 GPU시간**. 설계 11·12의 파생 헤드까지 포함하면 약 25 GPU시간. 사용자 승인 뒤 |
| **단일 로봇 평가기 로거의 넘어짐 플래그가 첫 에피소드 길이로 결정됨.** `self.episode_length`는 첫 리셋 뒤 멈추는데 매 리셋의 `logger.close(terminated=episode_length < limit − 5)`가 이를 씀 → 한 런(5000스텝)의 2번째 이후 에피소드는 전부 첫 에피소드의 플래그 | `scripts/simulate_go2.py:298` | 에피소드별 카운터 `self.steps` 사용, 판정은 `simdist/data/episode_logger.py::fell_before_limit` | `logging.enabled=true`로 모은 시뮬-투-시뮬 로그의 `terminated` 열(`datasets/real/sim2sim_*`, 2026-10-14 이후). 이 열을 쓰는 것은 `loss.terminal.enabled`(설계 1 `*_term` 레시피: 넘어짐 직전 창의 가치 목표를 reward-to-go로 교체)와 `boundary_weight`(설계 2)뿐이며, 동역학만 적응(`dyn`) 레시피는 쓰지 않는다. 즉 설계 1·2의 가치 적응 표(모두 기각·부분 결론)에 라벨 오류가 섞였고, 설계 11·12·18이 쓴 `*_dyn_b45000_term` 체크포인트는 이름에 term이 있어도 동역학만 학습이라 영향 없음(§3의 정책 헤드 동일성 확인과 별개로, 동역학 손실은 이 열을 읽지 않음) | 설계 1·2는 이미 기각된 가지라 재실행 권하지 않음. 필요하면 `scripts/sim2sim/mark_terminated.py`(시간 간격 규칙)로 라벨을 다시 매기고 `*_term` 적응만 재학습(런당 ≈ 30분) |
| 배치 평가기가 컴파일용 계획 1스텝을 돌린 뒤 리셋하지 않음(원본 단일 평가기는 initialize 뒤 reset). 첫 실제 스텝이 0 평균이 아니라 한 스텝 진행된 평균·키에서 시작 | `scripts/simulate_batch.py:171` | 컴파일 뒤 `controller.reset` 호출 | 리셋 직후 25스텝은 어차피 영행동이고 계획기 상태는 그 동안 25번 갱신되므로 실질 영향은 난수열의 1스텝 차이뿐. 과거 표 재실행 불필요. 회귀 테스트는 Isaac Sim이 필요해 붙이지 못함(다음 평가 런이 스모크) | 없음 |

고치지 않고 기록만 한 것(의도적 변경 또는 원본과 같은 정의 유지): §2의 [의도·미문서] 항목들, 검증 손실의 증강 잡음(원본과 동일), `run_reward_per_step` 분모, 체크포인트 로더의 교집합 복원.

## 6. 독립 검토

결론을 알려 주지 않은 별도 검토자(에이전트)에게 두 체크아웃과 A~F 파일 목록만 주고 차이·버그를 찾게 했다. 13개 지적을 받았고 전부 코드로 재확인했다.

| 검토자 지적 | 재확인 | 처리 |
|---|---|---|
| 검증 손실이 증강 잡음을 켠 채 계산됨(원본도 같음; 포크는 partial로 묶어 플래그로 못 바꿈) | 맞음(원본 trainer.py:234도 `training = True`) | §2-C 기록, 원본 정의 유지 |
| 단일 평가기 로거의 terminated 플래그가 첫 에피소드 길이 사용 | 맞음 | §5에서 수정 |
| 평가 물리에서 게인·관절 마찰 랜덤화 제거, 시드, 테두리 100 m | 맞음(난이도 초기값 1.0 확인) | §2-A 기록(논문 변경점 표) |
| 원본 평가기는 시간 초과에 제어기 리셋 안 함 | 맞음, 1000스텝 프로토콜에는 영향 없음 | §2-A |
| 학습/검증 분할 무작위 → 연속 | 맞음 | §2-B |
| forkserver 워커 | 수치 영향 없음 | 기록 |
| 모터 배율 > 1은 관절이 토크 방향으로 움직일 때 거의 무효(토크-속도 곡선이 그대로) → 라벨 1.0~1.1 구간이 데이터에서 1.0과 구분 안 됨 | 맞음(DCMotor `_clip_effort`) | 기록: 맥락 슬롯 라벨·DR 폭 해석 시 유의. 결과 표에는 영향 없음(평가는 0.4·0.5 등 < 1) |
| 배치 평가기가 컴파일 스텝 뒤 리셋 안 함 | 맞음 | §5에서 수정 |
| `run_reward_per_step` 분모에 워밍업 포함 | 맞음 | §2-F 기록(모델 간 비교 동일 정의) |
| 체크포인트 로더가 구조 검증을 안 함 | 맞음 | §2-C 기록(맥락 슬롯 warm start가 의존) |
| `policy_only` 필터가 `prior_anchor.policy`도 선택 | 맞음. 검토자가 놓친 것: 같은 이유로 `policy_dec`는 선택되지 않음 | §5에서 수정 |
| 시뮬 기록기 데이터셋은 terminated가 없어 terminal 옵션이 작동 불가 | 맞음(선택 손실에만 해당) | 기록 |
| 배치 평가기의 로봇들이 같은 원점에 겹침 | Isaac Lab `filter_collisions`로 환경 간 충돌 없음. 관문 0 충실도 검사에서 648대가 겹친 채 섭동 없는 복제본 SD 0.0000이므로 물리 상호작용 없음 | 기록. 단 §4의 재현 런에서 워밍업 중 넘어짐이 속도별로 다르게 나온 현상은 확인 중 |

검토자가 동일하다고 확인한 것: 모델 순전파(옵션 꺼짐), 손실 가중치, 스케일러, 적응 기본 레시피, MPPI 식과 하이퍼파라미터, 환경 정의(관절 순서·자세·게인·주기·관측·보상·종료·명령·DR), 시스템 설정, 데이터 파이프라인, 평가 지표 정의.

## 7. 확인하지 못한 항목과 이유

- **원본의 Go2 시뮬 수치(22.78) 재현의 정확한 격자**: 논문 부록 표 X의 값이 HTML에서 깨져 속도·마찰 0.2~1.2, 난이도 0.2~1.0, 지형 6종(6 × 6 × 6 × 5 = 1080)으로 추정해 축소 격자(마찰 3 × 난이도 3)로 돌렸다(§4). 저자에게 격자를 확인하기 전에는 수치 비교가 근사다.
- **RL 전문가의 수렴 정도**: 넓은 모터 DR 전문가(2026-10-20)는 논문 레시피의 1/4 표본(1024 env × 24 × 5000)으로 학습됐다. 품질은 평가 성능(설계 17: 전 조건 넘어짐 0)으로만 간접 확인했고, 4096 env로 재학습해 비교하지는 않았다(≈ 6 GPU시간).
- **Go1 USD 자산의 질량·관성·충돌 형상**이 실제 Go1과 맞는지는 이 감사 범위 밖(자산은 Isaac Lab 제공 go1.usd 그대로).
- **배치 평가기의 `terms_csv`·`trace_npz`·`context_csv` 등 부가 출력 경로**는 결과 표에 쓰인 지표가 아니라 검토하지 않았다.
- **하드웨어 브리지(G)**는 설정값 대조만 했고(§2-G), 관측 벡터의 실제 배열 순서를 로봇 없이 실행해 확인하지는 못했다. 과거 하드웨어 라운드가 걸은 사실로 간접 확인.
- **원본 평가기의 게인·관절 마찰 랜덤화가 켜진 조건에서의 수치**는 다시 돌리지 않았다. 논문 변경점 표에는 "평가는 공칭 고정 게인"으로 적는다.

## 8. 이미 알려진 항목 재확인

| 항목 | 현재 상태 |
|---|---|
| 식별 스크립트의 Unitree↔Isaac 관절 순서 버그 | 수정 유지. `identify_physics.py:86-91` articulation 순서 → Unitree 순서 매핑 `u2n`를 런타임에 계산하고 출력 |
| 평가기: 지형 끝 리셋이 넘어짐에 섞임 | 수정 유지(테두리 100 m, `simulate_batch.py`); 넘어짐은 base_contact ∪ bad_orientation만 셈 |
| 모터 약화 정의 불일치 | 수정 유지: 네 곳 모두 effort_limit(토크 캡)만 스케일(§A) |
| Go1 자산 한계 23.7 / 30 vs 실제 종아리 35.55 / 20.06 | 유지(토크 캡 1.0) |
| 외력 밀기 꺼짐, 지연 0 | 외력 밀기는 Isaac Lab Go2 rough 설정(`push_robot = None`)이라 원본도 꺼짐. 계획기 `delay` = 1은 원본과 같음 |
| 헤드 재학습 때 마지막 선형층 동결·틀린 주석 | **고쳐져 있지 않았음.** `create_param_filter`의 `PathContains("policy")`는 경로 원소가 정확히 `policy`인 것만 고르므로 `policy_dec`는 동결, `prior_anchor.policy`는 학습 대상(컨테이너에서 확인). `policy_only.yaml`의 주석은 틀렸음. `audit-fixes`에서 수정·테스트(§5) |
| 처음부터 학습(150k 스텝) 모델의 미달 | 코드 문제가 아니라 학습 예산(설계 17): 논문 레시피는 369k 스텝 |
