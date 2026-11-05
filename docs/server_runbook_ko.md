# 연구실 서버에서 simdist 돌리기 — 처음부터 끝까지

서버 4대(포트 52700, 계정 cjkim, Ubuntu 18.04, docker 그룹 있음, 공유 NAS `/nas_data`):
| 별칭 | 주소 | GPU | 드라이버 | 루트 여유 | 데이터 디스크 | 적합 |
|---|---|---|---|---|---|---|
| gpu1 | 10.20.22.101 | RTX 2080 Ti 11 GB × 4 | 530 | 51 GB | – | 가능 (평가 1개/장) |
| gpu2 | 10.20.22.105 | Quadro RTX 8000 48 GB × 4 | 470 | 42 GB | – | 불가 (CUDA 12 안 됨) |
| gpu3 | 10.20.22.71 | RTX 3090 24 GB × 6 | 545 | 25 GB | – | 디스크 없음 |
| gpu4 | 10.20.22.166 | RTX 3090 24 GB × 4 | 530 | 30 GB | /data1 505 GB | **주력** (2026-11-05 세팅 완료) |

**gpu4에서 실제로 필요했던 수정 (2026-11-05)**
1. 서버 도커 20.10.7의 기본 seccomp 프로필이 `clone3`를 EPERM으로 막아 컨테이너 안 glibc의 스레드 생성이 실패(numpy 임포트 시 OpenBLAS `pthread_create failed: Operation not permitted`). 해결: 기본 프로필 + `clone3` 허용 규칙(`scripts/server/seccomp-clone3.json`, 도커 20.10.10의 수정과 같은 효과; 호스트 커널 4.15에는 clone3가 없어 ENOSYS → glibc가 clone으로 폴백)을 `run.sh`의 docker run 인자에 `--security-opt seccomp=$HOME/simdist-docker/seccomp-clone3.json`로 추가.
2. `git clone`만으로는 IsaacLab 서브모듈이 없어 `No module named isaaclab` → `git submodule update --init IsaacLab`.
3. 루트 디스크가 작아 `~/simdist/datasets/sim` → `/data1/cjkim/datasets_sim` 링크, `~/docker/isaac-sim` → `/data1/cjkim/isaac-sim` 링크, `run.sh`에 `-v /data1:/data1 -v /nas_data:/nas_data` 추가.
4. 이미지는 ssh 대신 NAS 경유: 로컬 `docker save simdist:latest > /nas_data/cjkim/simdist_docker/simdist_image.tar`, 서버 `docker load < 그 파일`.
5. `cuStreamGetGreenCtx` 경고(드라이버 530)는 무해.

아래 원래 절차(gpu1 기준)는 그대로 유효.
전제: 본인 계정 + docker 그룹 + 디스크 150 GB. 아래 명령은 표시가 없으면 **로컬 PC**에서 실행.

## 0. 접속 설정 (한 번)
```bash
cat >> ~/.ssh/config <<'X'
Host gpu1
    HostName 10.20.22.101
    Port 52700
    User <계정>
X
ssh-copy-id gpu1            # 비밀번호 1회
ssh gpu1 'nvidia-smi --query-gpu=index,name,memory.used --format=csv; docker ps; df -h ~'
```
GPU `memory.used`가 0에 가까운 장번호를 적어 둔다(아래 `GPUS_LIST`).

## 1. 도커 이미지 전송 (25 GB, 1 Gbps ≈ 5분; NGC 로그인 불필요)
```bash
docker save simdist:latest | ssh gpu1 'docker load'
```
서버에 캐시 디렉터리: `ssh gpu1 'mkdir -p ~/docker/isaac-sim ~/simdist-docker'`

## 2. 코드·실행 스크립트·체크포인트
```bash
ssh gpu1 'git clone -b go1-support git@github.com:ChangjuKim-unist/simdist.git ~/simdist'   # 서버에 깃허브 키 없으면 https URL
scp ~/simdist-docker/run.sh gpu1:~/simdist-docker/
rsync -a ~/simdist/checkpoints/models/go1_paper gpu1:~/simdist/checkpoints/models/     # 8 MB
rsync -a ~/simdist/checkpoints/rl/2026-10-20_07-40-06 gpu1:~/simdist/checkpoints/rl/   # 2.3 GB, 데이터 생성용 전문가
```
평가만 할 거면 평가할 모델 디렉터리(`checkpoints/models/<이름>`)만 더 복사.

## 3. 동작 확인 (서버, 2분)
```bash
ssh gpu1
cd ~/simdist && GPUS='"device=0"' ~/simdist-docker/run.sh python -c "import jax; print(jax.devices())"
GPUS='"device=0"' ~/simdist-docker/run.sh python scripts/simulate_batch.py model.checkpoint=go1_paper sim.terrain=plane \
  sim.total_steps=200 'batch.speeds=[0.6]' 'batch.seeds=[11]' output.csv=/workspace/simdist/results/smoke.csv --headless
```
첫 실행은 셰이더 캐시 생성으로 5–10분 더 걸림. `results/smoke.csv`에 한 줄 생기면 끝.

## 4. 실험 실행 (서버, tmux 안에서)
```bash
tmux new -s s11
cd ~/simdist && GPUS_LIST="0 1 2 3" nohup bash scripts/sim2sim/stage11_context_server.sh > results/stage11_server.log 2>&1 &
```
`Ctrl-b d`로 나옴. 흐름: 데이터(1장, ≈ 2.5 h) → 학습 2개 병렬(2장, ≈ 5 h) → 평가 16회를 4장에 분산(≈ 4 h) → **≈ 12 h**.
진행 확인: `ssh gpu1 'grep -E "^==|^###" ~/simdist/results/stage11_server.log | tail'`

## 5. 결과 회수
```bash
rsync -a gpu1:~/simdist/results/stage11/ ~/simdist/results/stage11_server/
rsync -a gpu1:~/simdist/checkpoints/models/go1_ctx gpu1:~/simdist/checkpoints/models/go1_broadwm ~/simdist/checkpoints/models/
```

## 6. 예절·정리
- 시작 전 `nvidia-smi`로 빈 장만 쓰고, 끝나면 `docker ps`에 남은 컨테이너 없는지 확인(`docker stop <id>`).
- 데이터 55 GB는 끝나면 `rm -rf ~/simdist/datasets/sim/<이름>`으로 지운다(체크포인트·CSV만 남기면 됨).
- 다른 실험(평가만)을 서버에서 돌릴 때는 `GPUS='"device=N"'`을 붙여 장을 지정한다.
