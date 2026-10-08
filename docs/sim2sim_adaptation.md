## Sim-to-Sim Adaptation Experiments (selective value adaptation)

- [Motivation](#motivation)
- [Setup](#setup)
- [Adaptation recipes](#adaptation-recipes)
- [Metrics](#metrics)
- [Running an experiment](#running-an-experiment)
- [Files](#files)

### Motivation

SimDist adapts to the real world by finetuning only the latent dynamics on ~30 min of real data, with the encoder, reward and value heads frozen. The paper reports that unfreezing the value head causes catastrophic forgetting, and leaves "selectively updating value functions" as future work. That negative result rests on one recipe (a plain TD update of the full head on real data, no anchor, no sim replay). These experiments ask:

1. Can the value function be adapted **without forgetting** by training only a small, zero-initialized residual on top of the frozen simulation value, with returns bootstrapped by a frozen anchor?
2. How does the gain depend on the amount of "real" data (5 / 15 / 30 min)?

A perturbed Isaac Sim world stands in for the real world, so every ablation can be run without the robot. The real Go1 is reserved for confirming the best recipe.

### Setup

**Target conditions** (`scripts/sim2sim/run_pipeline.sh`, `condition_overrides`): physics changes of the Go1 simulation that the pretrained model has not seen, deployed with the same MPPI planner.

| name | change |
|---|---|
| `nominal` | friction 0.8, no payload, nominal motors (= pretraining distribution) |
| `low_friction` | friction 0.3 |
| `payload` | +3 kg on the trunk |
| `weak_motor` | motor torque limits scaled to 60% (`sim.motor_strength`) |
| `combo` | friction 0.4, +2 kg, motors 80% |

**"Real" data**: `scripts/simulate_go2.py` with `logging.enabled=true` rolls out the pretrained model + MPPI in the target condition at several commanded speeds and logs each episode in the real-world data format (`datasets/real/<name>/raw/...`), with the simulator reward stored alongside. `aggregate_realworld_data.py max_steps=N` applies a data budget (N steps at 50 Hz: 15000 = 5 min), then `process_data.py` builds the training windows as for real data.

**Pretrained model**: `go1_paper` (Go1, paper-scale pretraining).

### Adaptation recipes

All recipes start from the pretrained checkpoint and train for the same number of gradient steps on the same windows.

| variant | trainable | value loss | bootstrap target |
|---|---|---|---|
| `dyn` | latent dynamics | none (SimDist) | – |
| `res` | dynamics + residual value head δ(z, cmd), zero-initialized | truncated returns | current head (= frozen V_sim, since the head itself is frozen) |
| `res_anchor` | same as `res` | truncated returns | frozen copy of V_sim |
| `full` | dynamics + the whole value head | truncated returns | the head being trained (SimDist's forgetting ablation) |
| `full_anchor` | dynamics + the whole value head | truncated returns | frozen copy of V_sim |

The value target for the state at step t+k of a window is the SGFT-style truncated return `G_k = Σ_{j=k}^{T-1} γ^{j-k} r_j + γ^{T-k} V_boot(z_{t+T})`, computed on the encoded true future latents (so the value update is decoupled from dynamics error). Implemented in `simdist/modeling/losses.py::WorldModelValueAdaptLoss`; the residual head and anchor live in `simdist/modeling/models.py` (`value_from_latents`, `add_value_residual`, `add_value_anchor`) and are switched on through `config/finetune_model.yaml` (`adapt:`), `config/heads/dynamics_value_*.yaml` and `config/loss/world_model_value_adapt.yaml`. The planner picks up `V_sim + δ` automatically because it reads the model's `values` output.

### Metrics

- **Deployment in the target condition** (`simulate_go2.py`, 1000 steps, several seeds): total reward, forward progress [m] of the first episode, and whether the robot fell (episode length < 1000).
- **Deployment in the nominal condition**: the same, to detect behavioral forgetting.
- **Value forgetting** (`scripts/sim2sim/value_forgetting.py`): on windows of the nominal pretraining dataset, RMSE between the adapted and the pretrained value predictions and the correlation of each with the simulation critic's labels.

`scripts/sim2sim/summarize.py` turns `results/sim2sim/<condition>/eval.csv` and `forgetting.json` into a markdown table.

### Running an experiment

From the host (uses `~/simdist-docker/run.sh`):

```bash
# one condition, 5-minute budget (1 round x 3 speeds x 100 s), all recipes
COND=low_friction scripts/sim2sim/run_pipeline.sh all

# stages separately, e.g. a data-budget sweep after one 30-minute collection
COND=low_friction ROUNDS=6 COLLECT_STEPS=5000 SPEEDS="0.3 0.6 0.9" scripts/sim2sim/run_pipeline.sh collect   # 6 x 3 x 100 s = 30 min
COND=low_friction BUDGETS="15000 45000 90000" scripts/sim2sim/run_pipeline.sh process
COND=low_friction BUDGETS="15000 45000 90000" scripts/sim2sim/run_pipeline.sh finetune
COND=low_friction BUDGETS="15000 45000 90000" scripts/sim2sim/run_pipeline.sh evaluate
COND=low_friction BUDGETS="15000 45000 90000" scripts/sim2sim/run_pipeline.sh forget
COND=low_friction scripts/sim2sim/run_pipeline.sh summarize
```

Checkpoints are named `<base>_<condition>_<variant>_b<budget>` in `checkpoints/models/`, results go to `results/sim2sim/<condition>/`. Every stage skips work it finds already done (evaluation rows, existing checkpoints are reused).

### Files

| file | role |
|---|---|
| `scripts/simulate_go2.py`, `config/simulate_go2.yaml` | `sim.motor_strength`, `logging.*`, `forward_progress` metric |
| `simdist/data/episode_logger.py` | ROS-free episode logger in the real-world format (+ reward) |
| `simdist/data/episode_aggregator.py`, `config/aggregate_realworld_data.yaml` | `reward` pass-through, `max_steps` budget |
| `simdist/modeling/models.py` | residual value head, frozen value anchor, `value_from_latents` |
| `simdist/modeling/losses.py` | `world_model_value_adapt` loss |
| `simdist/modeling/trainer.py`, `config/finetune_model.yaml` | `adapt.value_residual`, `adapt.value_anchor` |
| `config/heads/dynamics_value_residual.yaml`, `config/heads/dynamics_value_full.yaml`, `config/loss/world_model_value_adapt.yaml` | recipe configs |
| `scripts/sim2sim/run_pipeline.sh`, `value_forgetting.py`, `summarize.py` | experiment driver, forgetting metric, table |
