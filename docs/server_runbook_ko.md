# 연구실 서버(gpu1)에서 simdist 돌리기 — 처음부터 끝까지

서버: `10.20.22.101`, 포트 52700, RTX 2080 Ti 11 GB × 4, 드라이버 595, Docker 있음, 공유 NAS `/nas_data`.
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
