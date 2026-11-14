# 설계 문서 20: 저용량 동역학 적응과 망각 측정 (2026-11-13, 시뮬 전용)

사용자 지시(2026-11-13): "설계 20 — 저용량 동역학 적응과 망각 측정 (카드 5, 시뮬 전용)". 드라이버 `scripts/sim2sim/stage20_lowcap.sh`(gpu1, 4장), 결과 `results/stage20/`. 아이디어 출처: `reports/sim-to-real 적응 아이디어 탐색.md` 카드 5·6·7, 선행연구 점검 `research_notes/sim-to-real 적응 아이디어 탐색/_tools/prior_work_card5_7.md`.

## 0. 배경과 가설
SimDist는 실데이터로 latent dynamics 전체를 fine-tuning한다. 설계 13(go1_ctx 저마찰 적응: 모터 0.4 넘어짐 0.27→0.77)과 설계 14(go1_paper 마찰 적응: 마찰 0.15 1.16→0.22, 모터 0.4 3.23→4.14)에서 한 축에 적응하면 다른 축이 나빠졌다. 가설: 망각은 전 층을 자유롭게 바꾸는 데서 오고, 갱신 자유도를 낮추면 적응 축 이득은 대부분 유지되면서 다른 축이 보존된다.

## 1. 실험 전 코드 사실 (2026-11-13 확인)
1. **시뮬 전이 혼합: 없음.** `finetune_model.py` 기본 레시피(`loss=world_model_dynamics_only heads=world_model_dynamics_only`)는 실데이터 창만 쓴다. 시뮬 replay(`adapt.replay.dataset_name`)는 가치 적응 손실(`world_model_value_adapt`, 설계 1)의 가치 라벨 앵커로만 쓰였고 동역학 손실은 replay 인자를 무시했다(`losses.WorldModelLoss`에 replay 처리가 없었음). 설계 13 L3·설계 14의 적응도 replay 없음.
2. **정규화 층: 있음.** latent dynamics(`TransformerDecoder` 2층)에 층마다 `nnx.LayerNorm` 3개(scale·bias 각 64) = affine 파라미터 768개. V4(affine만) 가능.
3. **적응 때 갱신되는 파라미터** (`heads=world_model_dynamics_only` = 경로에 `dynamics`가 있는 Param): 52텐서 133,504개 / 전체 667,758개(20.0 %). 구성: 층당 self-attention·cross-attention 각 q/k/v/out(64×8×8 + bias) + MLP 64→256→64 + LayerNorm 3개. encoder·fut_acts_embed·reward·value·policy는 동결. 마지막 층(V5 = `dynamics/layers/1/dense_proj`) 33,088개. rank-1 LoRA 3,328개(2.5 %), rank-4 13,312개. 확인 스크립트 `scripts/sim2sim/stage20_params.py`.
4. **audit-fixes 영향: 이 경로에 없음.** 버그 ①(heads 필터가 경로 안 키를 찾음)은 `dynamics`라는 키가 `self.dynamics` 밖에 없어 동역학 적응에는 영향 없음(policy_only에만 영향). 버그 ②(단일 로봇 로거의 terminated 플래그)는 `loss.terminal`·`boundary_weight`만 읽고 동역학 손실은 읽지 않음. 따라서 merge 없이 진행한다(merge는 사용자 승인 뒤).
5. (추가) 설계 14의 두 적응은 warmup이 달랐다(마찰 2000, 모터 200). 이 설계는 SimDist 기본값(warmup 2000, lr 2e-4→1e-4, 3000스텝, 배치 32)으로 통일한다(설계 13 L3과 같음).

## 2. 변형 (손실·데이터·동결 범위는 SimDist 그대로, 갱신 파라미터만 다름)
| 변형 | 갱신 파라미터 | 개수 | 구현 |
|---|---|---|---|
| V0 | 없음 | 0 | 사전학습 모델 |
| V1 | dynamics 전체 | 133,504 | `VARIANTS=dyn` (SimDist 원안, 기준) |
| V2 | 전 층 rank-1 LoRA, B=0 초기화 | 3,328 | `lora1`: `simdist/modeling/lora.py`, `heads=lora` |
| V3 | rank-4 LoRA | 13,312 | `lora4` |
| V4 | LayerNorm affine만 | 768 | `affine`: `heads=dynamics_affine` |
| V5 | 마지막 층 MLP만 | 33,088 | `last`: `heads=dynamics_last` |
| V6 | 전체 갱신 + 시뮬 전이 replay 50 % | 133,504 | `dyn_replay`: 손실 = 0.5·실 + 0.5·시뮬(배치 32+32), 시뮬 = 그 모델의 사전학습 데이터 |
LoRA: 선형층 20개(어텐션 q/k/v/out 16 + MLP 4)에 `W x + (x A) B`, A ~ N(0, 1/in), B = 0, scale 1. 학습 스텝·학습률은 V1과 같고 변형별 튜닝 없음. 동치 테스트 `tests/test_stage20_lora.py`(B=0이면 출력 동일, split/merge 왕복, 파라미터 수) 통과; 20스텝 smoke(lora1·dyn_replay·affine) 통과; LoRA 체크포인트 재적재 시 기반 가중치 불변·정책 헤드 출력 동일 확인.

## 3. 프로토콜
- 모델: `go1_broadwm`(넓은 DR), `go1_paper`(논문 설정).
- 적응 데이터(15분 = 45k 스텝, 설계 13 방식 = 그 모델의 planner로 3라운드 × 3속도 × 5000스텝): go1_paper는 기존 `sim2sim_friction0.15_go1_paper`·`sim2sim_very_weak_motor_go1_paper`(설계 11·14와 동일), go1_broadwm은 새로 수집. 순서 (a) 마찰 0.15, (b) 모터 0.4. fine-tuning 시드 42·43.
- 평가 조건: 마찰 0.15, 모터 0.4, nominal, mass3. 평가기 = 설계 13~18과 동일(`simulate_batch.py`, 6시드 × 3속도, 5000스텝, 에피소드 100 s). 지표: 넘어짐(1000스텝당, ↓), 수익/스텝(↑), 전진 거리(↑), 짝지은 부트스트랩 CI(`paired_stats.py`).
- 평가 팔: ① planner(MPPI, SimDist 방식) ② 정책 헤드만(`control.prior_only=true`; 동역학과 무관해야 함) ③ 모델 수준: 조건별 **held-out 기록**(그 모델의 planner로 1라운드 × 3속도 × 5000스텝, 수집 시드 2000대, 학습에 안 씀)에서 25스텝 잠재 예측 오차(`scripts/sim2sim/latent_error.py`: 평균·1스텝·25스텝).
- 재방문 A→B→A: (a) 마찰 적응 체크포인트에서 (b) 모터 데이터로 이어 적응(LoRA는 같은 어댑터를 계속 학습) → 마찰 0.15·모터 0.4 재평가. V1·V2·V6만.
- 지표 정의: 적응 이득 = 적응 축에서 V0 대비 개선; 망각 = 적응 안 한 축에서 V0 대비 악화(넘어짐·예측 오차 둘 다); 유지율 = 다른 축 성능의 V0 대비 비율; 갱신 파라미터 수·학습 시간.

## 4. 예측과 판정 (사전 등록)
| # | 예측 | 확률 |
|---|---|---|
| P1 | go1_broadwm 마찰 적응에서 V2의 마찰 0.15 넘어짐이 V1 대비 +30 % 이내 | 0.5 |
| P2 | 같은 칸에서 V2의 모터 0.4 넘어짐이 V0 대비 +50 % 이내; V1은 설계 13처럼 2배 이상 악화 | 0.5 |
| P3 | 정책 헤드만 실행한 팔은 모든 변형에서 V0와 차이 없음(넘어짐 동일, 수익 차 < 1e-3). 차이가 나면 구현 오류로 보고 멈춘다 | 0.9 |
| P4 | 모델 수준 망각(다른 축 예측 오차 증가)은 V1 > V6 > V2 순 | 0.4 |
| P5 | V6(replay)가 모터 0.4 넘어짐을 V0 대비 +30 % 이내로 지킴 | 0.5 |
결정 규칙: P1·P2가 두 모델 중 하나 이상에서 성립 → 저용량 갱신 채택, 카드 7 전제 확인(부록)으로 넘어감. V2~V5가 적응 이득을 절반도 못 내면 "저용량은 접촉 축 격차를 못 담는다"로 기록하고 V6 결과로 replay 쪽을 판단. V1에서 망각이 재현되지 않으면(설계 13과 다르면) 원인을 먼저 찾는다.

## 5. 부록 실험 (카드 7 전제, CPU)
go1_paper 세 모델(원본, 마찰 적응 V1 s42, 모터 적응 V1 s42; 보조로 V2)을 같은 기록에 대어, 최근 2초(100스텝) 창의 1스텝 잠재 예측 오차가 가장 작은 모델이 그 조건의 모델인지 본다. 기록: nominal·마찰 0.15·모터 0.4 × 수집 정책 2종 = 정책 헤드만(`prior_only`, 시드 3000대)과 planner(held-out 기록 재사용). 생성기 평균은 go1_paper에서 걷지 못하므로(설계 17) planner로 대체. 지표: 창 단위 3분류 정확도, 혼동 행렬(`scripts/sim2sim/stage20_appendix.py`). 예측: 마찰 0.15 ≥ 0.8 (0.6), 모터 0.4 < 0.6 (0.6). 0.8 미만이면 카드 7은 그 축에서 접는다.

## 6. 비용·일정 (gpu1 2080 Ti 4장, 다른 사용자 없음 확인 2026-11-13 19:40)
수집 16건 ≈ 1 h → 처리 ≈ 0.3 h → fine-tuning 48 + 재방문 12 ≈ 1.5 h → 정책 헤드만 248런(≈ 5분/런) ≈ 5 h + 잠재 오차 ≈ 1 h → planner 224런(≈ 50분/런) ≈ 47 h. 합계 ≈ 2.3일. planner 큐는 중요한 칸부터(broadwm 마찰 적응 V0/V1/V2/V6 → paper → 모터 적응 → V3~V5 → 시드 43 → mass3 → 재방문). 중간 결과는 CSV가 쌓이는 대로 보고.

실행 기록: 20:03 시작. 수집·처리·fine-tuning C1(23:01)·C2(23:23) 완료. go1_paper V6(dyn_replay) 4건(마찰·모터 × 시드 42·43)은 22:12~22:57에 "Dataset directory not found: 2026-10-02_01-18-59"로 실패 — 논문 시뮬 데이터를 gpu1 데이터 디스크로 복사했으나 저장소 `datasets/sim/` 안의 심볼릭 링크를 만들지 않은 설정 실수(코드 문제 아님). 23:25 링크 추가 후 4건과 재방문 2건을 D 단계 큐 앞에 끼워 넣어 재실행(해당 체크포인트의 평가는 큐 뒤쪽이라 순서 영향 없음).

P3 관문(D 단계 끝, 11/14 04:45): 시뮬 기준으로는 정책 헤드만 팔 4,446행 중 1,584행이 V0와 달랐다(넘어짐 수가 다른 행 1,285). 원인 확인: (1) 같은 체크포인트(V0)를 같은 명령으로 두 번 돌려도 결과가 다르다(로컬 GPU, 마찰 0.15, 1000스텝 × 3속도 × 2시드: 보상이 모두 다르고 속도 0.9 시드 12는 넘어짐 0 대 2) → 시뮬레이터(GPU 병렬 물리) 자체가 실행마다 비결정적이라 "넘어짐 동일·보상 차 < 1e-3" 기준은 V0 자신도 통과 못 한다. (2) 변형별 정책 헤드만 성적은 V0 주변의 잡음이며 적응 축 쪽으로 체계적 이동이 없다(go1_paper 마찰 0.15: V0 1.17, 적응 모델 0.91~1.26; go1_broadwm 모터 0.4: V0 0.24, 적응 모델 0.18~0.43 넘어짐/1000스텝). (3) 결정적 검사로 대체: 같은 입력 창 512개에 대해 60개 체크포인트 전부 정책 헤드 출력이 V0와 **비트 단위로 동일**(최대 |차| = 0.0), 반면 latent 예측은 모두 달라짐(최대 |차| 0.32~1.71) — `scripts/sim2sim/stage20_p3_actions.py`, `results/stage20/p3_actions.csv`, 반복 실행 비교 `results/stage20/p3check/`. 판정: **P3 성립**(구현 오류 없음), 단 시뮬 수준 동일성 기준은 잘못 세운 것이라 결정적 기준으로 바꾼다. 영향: planner 비교에도 실행 잡음이 섞이므로 6시드 × 3속도 짝지은 CI로 판단(설계 13~18과 동일). E 단계는 멈추지 않고 계속.

범위 축소(사용자 지시, 11/14 15:10): planner 평가에서 mass3 조건 50런과 V3~V5(rank-4·affine·마지막 층)의 시드 43 36런을 큐에서 제외했다(사전 등록 대비 축소; 정책 헤드만 팔과 latent 오차는 mass3 포함 전부 완료됨). 남은 planner 런 94개, 예상 종료 11/15 오후.

## 7. 규칙
시뮬만. 삭제 없음. 끝나면 NAS 백업, 커밋·푸시, 연구 기록 갱신. 보고: 표 1(변형 × 조건 넘어짐, planner), 표 2(예측 오차 유지율), 표 3(A→B→A), P1~P5 판정 한 줄씩, 결정 규칙 결론 한 줄.

중단(사용자 지시, 11/14 저녁): 결정 규칙의 답이 중간 결과로 이미 나와(P1·P2 두 모델 모두 기각, V1 망각 재현, V6은 paper × 마찰에서만 이득 유지) 남은 planner 런 82개(V3~V5 모터 적응 일부, 시드 43, 재방문 A→B→A)를 돌리지 않고 E 단계를 멈췄다. 완료된 planner 행 936개·latent 오차·정책 헤드만 팔·체크포인트 60개는 보존. 부록(카드 7 전제)은 미실행. §8 최종 표는 연구 방향 재설정 후 작성.
