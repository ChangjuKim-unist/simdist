# Go1 실험 매뉴얼 (SimDist 배포)

매 세션 같은 순서로 진행한다. 컴퓨터는 연구실 데스크톱(RTX 3060, `~/simdist`), 로봇은 Unitree Go1 EDU(RR 발 힘센서 고장).

## 0. 세션 시작 전 체크리스트
- [ ] 배터리 50 % 이상 (보행 수집은 완충 권장). 교체는 전원 끈 뒤 배 아래 배터리 양쪽 레버.
- [ ] 바닥: 평평한 매트 약 3 m × 2 m, 주변 2 m 비움.
- [ ] 끈: 로봇 윗면 손잡이에 끈을 걸어 사람이 위에서 느슨하게 잡는다 (간이 하네스).
- [ ] 랜선: USB-C 허브 랜 포트 ↔ Go1 **뒤쪽** 랜 포트. 선은 항상 로봇 **뒤로** 빠지게.
- [ ] 조종기 전원 ON, 비상 시 누를 버튼(Start 외 아무 버튼 = 엎드리기) 확인.
- [ ] 이 컴퓨터에서 시뮬레이션 실험이 돌고 있지 않은지: `docker ps` 가 비어 있어야 한다 (GPU 하나를 로봇 제어와 나눠 쓸 수 없음).
- [ ] 역할: ① 조종기 ② 끈·랜선 ③ (선택) 터미널.

## 1. 네트워크 확인 (호스트 터미널)
```bash
ip -4 addr show enx00e04c680024 | grep inet      # 192.168.123.162/24 가 보여야 함
# 안 보이면:  sudo nmcli con up go1
ping -c 2 192.168.123.10                           # 제어보드 응답
```

## 2. 로봇 전원 → 저수준 모드 (★ 컴퓨터 쪽을 켜기 전에 ★)
1. Go1 전원 ON → 평소처럼 일어설 때까지 대기.
2. 조종기: **`L2+A` → `L2+A` → `L2+B` → `L1+L2+Start`** → 로봇이 스스로 엎드리고 힘이 빠짐 = 저수준 모드.
3. 이 상태를 확인한 **다음에** 3단계로 간다. 서 있는 로봇에 브리지를 켜면 다리 힘이 빠지며 주저앉는다.

## 3. 컨테이너와 전체 스택 시작
터미널 1 (호스트):
```bash
cd ~/simdist
./go2_ros2_ws/scripts/run.sh      # 컨테이너 진입 (첫 10~20 초는 simdist 설치)
source scripts/setup.sh
tmuxp load tmuxp_go1.yaml         # tmux 창들이 열림
```
tmux 창 구성:
| 창 | 내용 | 할 일 |
|---|---|---|
| rviz | 시각화 | 자동 실행 |
| state machine | `launch_state_machine.py robot_config:=go1.yaml` | 자동 실행. **여기서 키보드로 상태 전환** |
| (포커스) | `ros2 launch bringup launch_go1.py` 입력돼 있음 | **Enter** → 5 초 경고 후 브리지·센서 노드 시작 |
| control | `launch_control.py controller_config:=simdist_controller_go1.yaml robot_config:=go1.yaml` | 보행 단계에서만 **Enter** |

tmux 창 이동: `Ctrl+b` 다음 화살표. 창 포커스가 있어야 키 입력이 그 창으로 간다.

브리지 로그에 `Receiving LowState from the Go1` 이 뜨면 통신 성공. `No LowState ... (is it in low-level mode?)` 가 계속이면 2단계를 다시.

## 4. 수신 확인 (모니터)
터미널 2 (호스트에서 새 터미널):
```bash
docker exec -it go2_ros2 bash
source scripts/setup.sh
python3 scripts/go1_monitor.py
```
확인: `/lowstate msgs: 250` (=500 Hz), 배터리 %, 모터 온도 < 60 °C, 발 힘(들었을 때 ≈100, 접지 판정은 200 이상), IMU roll/pitch ≈ 0.

## 5. 상태머신 조작
| 현재 상태 | 입력 (키보드 = 상태머신 창 / 조종기) | 결과 |
|---|---|---|
| OFF | **스페이스** / **Start** | PRONE (엎드린 채 약한 Kp 유지) |
| PRONE | 스페이스 / Start | STANDING → 3 초 뒤 STAND |
| STAND | 스페이스 / Start | WALKING (컨트롤러가 켜져 있어야 함) |
| STAND | **그 외 아무 키 / Start 외 버튼** | PRONING → PRONE |
| WALKING | 스페이스 / Start | STAND (멈춤) |
| WALKING | 그 외 아무 키 / 버튼 | RECOVERY (복구 자세) → 스페이스로 다시 서기, 다른 키로 엎드리기 |

## 6. 단계별 진행과 중단 기준
| 단계 | 하는 것 | 통과 기준 | 즉시 엎드리기 |
|---|---|---|---|
| 1 서기 | PRONE → 스페이스 (끈 잡은 채) | 떨림·소음 없이 3 초 안에 서고 10 초 유지 | 떨림, 윙윙 소리, 한쪽 기움 |
| 2 반복 | 서기↔엎드리기 3 회 | 매번 같은 자세 | 자세가 달라짐 |
| 3 컨트롤러 | control 창 Enter → `Controller initialized` 확인 → STAND | 선 채로 유지 | 다리 떨림, 미끄러짐 |
| 4 첫 보행 | STAND → 스페이스 → 조종기 스틱 살짝 (최대 0.3 m/s로 제한돼 있음) | 3 m 직진, 안 넘어짐 | 넘어짐 2 회 → 중단, 로그 분석 |
| 5 속도 | 0.3 m/s 이상, 회전 (go1.yaml 제한 상향 후) | 안정 | — |
| 6 수집 | 안정된 속도로 5~10 분씩 보행 (WALKING 동안 자동 기록) | 온도 < 60 °C | 60 °C 경고 → 5~10 분 휴식 |

자동 안전장치(따로 조작 불필요): 명령 끊김 0.5 s → 댐핑 · 모터 70 °C → 댐핑 · 넘어짐(roll/pitch 40°) → 댐핑, 세운 뒤 PRONE으로 되돌려야 해제 · 배터리 20 % 경고 · 전력 80 % 제한.

## 7. 데이터 위치와 파인튜닝
- 보행 로그: `~/simdist/datasets/real/go1_simdist_controller/raw/<날짜>/<시각>.hdf5` (WALKING 상태 동안만 기록, 에피소드당 파일 1개)
- 세션 후 처리·파인튜닝 (simdist 컨테이너, `~/simdist-docker/run.sh`):
```bash
cd ~/simdist-docker
./run.sh python scripts/aggregate_realworld_data.py dataset_name=go1_simdist_controller
./run.sh python scripts/process_data.py dataset_name=go1_simdist_controller system=go1
./run.sh python scripts/finetune_model.py data.dataset_name=go1_simdist_controller system=go1 \
    checkpoint.resume_checkpoint=go1_paper run_name=go1_finetuned
```
- 재배포: `simdist_controller_go1.yaml` 의 `model.checkpoint` 를 `go1_finetuned` 로 바꾸고 컨테이너 안에서 `colcon build`.

## 8. 종료
1. 상태머신으로 PRONE.
2. 각 tmux 창 `Ctrl+C` → `exit` 로 컨테이너 종료 (브리지가 꺼지면 모터는 댐핑 상태).
3. 로봇 전원 OFF. 평소 모드로 쓰려면 껐다 켜기.
4. 배터리 충전.

## 9. 자주 생기는 문제
| 증상 | 원인 / 조치 |
|---|---|
| `No LowState from the Go1 yet` | 저수준 모드 아님 → 2단계. 또는 랜선/IP → 1단계 |
| `docker: unknown runtime nvidia` | 옛 `run.sh` → 최신 브랜치(`go1-support`)로 갱신 |
| 서기 중 `PowerProtect triggered` | `go1.yaml` 의 `power_protect_level` 8 → 10 |
| 서기 자세가 이상 (한쪽만 펴짐 등) | 즉시 엎드리기, 모니터에서 관절값 확인 후 보고 |
| 발 힘센서가 들어도 200 이상 | `go1.yaml` 의 `foot_force_offset` 을 그날 값으로 조정 |
| 로봇 넘어진 뒤 명령이 안 먹음 | 넘어짐 잠금 상태: 로봇을 세우고 상태머신을 PRONE 으로 → 해제 로그 확인 |
| 온도 경고 | 5~10 분 휴식. 70 °C 에서는 자동 댐핑 |

## 10. 설정 파일 (Go1 전용)
- `go2_ros2_ws/src/config/config/go1.yaml` — 로봇 IP, 안전 한계, 접지 임계값·오프셋, 상태머신 Kp/Kd, 속도 제한
- `go2_ros2_ws/src/config/config/simdist_controller_go1.yaml` — 체크포인트, MPPI, 로깅 데이터셋 이름
- 수정 후 컨테이너 안에서 `colcon build --packages-select config` 또는 컨테이너 재시작.
