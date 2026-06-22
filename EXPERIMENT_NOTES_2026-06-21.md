# Spades RL Experiment Notes - 2026-06-21

## Current Best

- Current promoted checkpoint: `checkpoints/play_ev_v2_10k_s4_headonly_from_best.pt`.
- Confirmed duplicate eval: `+11.17` raw points/hand at 2048 hands, CI `[+10.29, +12.06]`.
- Do not promote `ppo_play_anchor_ladder_*`: unanchored play PPO drifted badly.
- Do not promote V3 or unanchored V4: both regressed against conservative duplicate eval.
- Do not promote strong-anchor V4 weight `30.0`: it got close at 512 but missed V2 on 2048 confirmation.

## Results So Far

### Unanchored PPO ladder from smoke

- Best 512-hand duplicate checkpoint: `ppo_play_anchor_ladder_000131072.pt`, `+4.20` raw points/hand.
- Smoke baseline on same 512 eval: `+5.21` raw points/hand.
- Later checkpoints fell to about `-5` raw points/hand.
- Interpretation: bidding stayed stable, but play PPO drifted and destroyed hand outcomes.

### Play-KL anchored PPO, weight 1.0

Command family:

```bash
uv run spades-train-puffer \
  --load-path checkpoints/ppo_play_anchor_smoke.pt \
  --checkpoint-prefix checkpoints/ppo_play_anchor_playkl_w1_lr3e5 \
  --total-timesteps 262144 \
  --learning-rate 3e-5 \
  --clip-coef 0.1 \
  --vf-clip-coef 0.1 \
  --ent-coef 0.002 \
  --play-anchor-path checkpoints/ppo_play_anchor_smoke.pt \
  --play-anchor-weight 1.0
```

512-hand duplicate ranking:

```text
000098304: +5.32 raw
000032768: +5.29 raw
000065536: +5.25 raw
smoke:     +5.21 raw
```

2048-hand confirmation:

```text
000098304: +4.39 raw, CI [+3.51, +5.26]
smoke:     +4.34 raw, CI [+3.47, +5.22]
```

Interpretation: play anchor prevents collapse, but weight `1.0` is too conservative or PPO signal is too weak. No meaningful promotion.

## Next Experiment Running

Run a lower play-anchor-weight ladder to test whether PPO can move more while still avoiding collapse:

- Name: `ppo_play_anchor_playkl_w03_lr3e5`
- Base: `checkpoints/ppo_play_anchor_smoke.pt`
- Play teacher: `checkpoints/ppo_play_anchor_smoke.pt`
- Play anchor weight: `0.3`
- LR: `3e-5`
- Total timesteps: `262144`
- Checkpoint interval: `32768`
- Promotion gate: must beat smoke by at least `+1.0` raw point/hand at 512, then confirm at 2048.

Status:

- A100 run `ppo_play_anchor_playkl_w03_lr3e5` completed successfully.
- No illegal actions.
- 8 checkpoints were saved from `32768` through `262144` steps.
- Evaluation sweep completed under `logs/playkl_w03_eval/`.

512-hand duplicate ranking:

```text
000262144: +5.56 raw
000229376: +5.49 raw
000163840: +5.31 raw
smoke:     +5.21 raw
000098304: +5.18 raw
000196608: +5.17 raw
000032768: +5.15 raw
000065536: +4.99 raw
000131072: +4.91 raw
```

Interpretation: lower play-KL PPO moved in the right direction late, but the best 512-hand gain over smoke is only about `+0.35` raw points/hand. This does not meet the `+1.0` raw promotion gate, so skip 2048 confirmation for now and spend the A100 on play-EV data.

## Implementation Direction

If weight `0.3` does not produce a clear duplicate-eval improvement, stop PPO-only tuning and implement play-EV labels:

- collect active play states,
- reconstruct full game state by replaying bids and played cards,
- evaluate each legal card by rollouts,
- train supervised play policy/Q heads,
- resume PPO only after play has a better supervised prior.

## Play-EV Implementation Progress

Added a first play-EV dataset generator:

- commands:
  - `spades-generate-play-ev`
  - `spades-summarize-play-ev`
- supports state and rollout actors as either `bot:<name>` or `checkpoint:<path>`,
- collects active play states,
- stores enough replay data to reconstruct each state:
  - dealer,
  - initial hands,
  - previous bid actions,
  - previous play actions,
  - team scores/bags,
- evaluates every legal card by rolling out the hand,
- writes NPZ fields matching the bid-EV style:
  - `observations`,
  - `legal_mask`,
  - `evaluated_mask`,
  - `q_values`,
  - `q_std`,
  - `visit_counts`,
  - `best_action`,
  - replay metadata.

Validation:

```text
uv run ruff check src tests
uv run pytest
79 passed, 1 warning
```

Next after the `w03` eval:

- if `w03` has no clear improvement, push play-EV generator to the A100 and generate the first real play dataset,
- initial dataset target should be small enough to complete tonight:
  - `states=2000`,
  - `rollout_samples=4`,
  - `state_actor=checkpoint:checkpoints/ppo_play_anchor_smoke.pt`,
  - `rollout_actor=bot:conservative`.

## Play-EV Trainer Implementation

Added supervised play-EV training/eval tooling:

- commands:
  - `spades-train-play-ev`
  - `spades-eval-play-ev`
- trains existing card-action logits from play-EV soft labels,
- keeps checkpoint format unchanged,
- supports `--play-heads-only` for the first conservative pass so bidding/encoder behavior is not disturbed,
- reports policy regret, top-1, mean best EV, and mean chosen EV.

Validation:

```text
uv run ruff check src tests
uv run pytest
84 passed, 1 warning
```

Next A100 job should:

1. wait for the current `w03` duplicate sweep to finish,
2. generate a small real play-EV dataset,
3. train a play-head-only model from `ppo_play_anchor_smoke.pt`,
4. evaluate it on the play-EV dataset and duplicate eval against `bot:conservative`.

Status:

- queued script: `/tmp/spades_play_ev_pipeline.sh`
- log: `logs/play_ev_v1/pipeline.log`
- smoke play-EV generation and smoke training completed successfully,
- real dataset generation `data/play_ev_v1_2k_s4_smoke_state_conservative_rollout.npz` started at `2026-06-21T20:58:35+00:00`,
- pipeline completed at `2026-06-21T21:04:08+00:00`.

Dataset summary:

```text
rows: 2000
rollout_samples: 4
state_actor: checkpoint:checkpoints/ppo_play_anchor_smoke.pt
rollout_actor: bot:conservative
mean_best_ev: 42.54
mean_best_second_gap: 4.11
```

Play-EV fit from smoke:

```text
smoke on dataset:
  top1: 46.65%
  regret: 4.48
  mean_policy_ev: 38.06

play_ev_v1_2k_s4_headonly:
  top1: 49.75%
  regret: 3.43
  mean_policy_ev: 39.11
```

512-hand duplicate:

```text
play_ev_v1_2k_s4_headonly: +5.38 raw, CI [+3.59, +7.16]
smoke baseline in same eval family: about +5.21 raw
best w03 512: +5.56 raw
```

Interpretation: play-EV supervision learns its labels and transfers slightly to duplicate eval, but this first dataset/checkpoint is not a clear promotion. Next quick test is to apply the same play-head-only supervised fit starting from the best `w03` checkpoint (`ppo_play_anchor_playkl_w03_lr3e5_000262144.pt`).

## Play-EV From Best W03 Checkpoint

Quick combination test:

- base: `checkpoints/ppo_play_anchor_playkl_w03_lr3e5_000262144.pt`
- dataset: `data/play_ev_v1_2k_s4_smoke_state_conservative_rollout.npz`
- output: `checkpoints/play_ev_v1_2k_s4_headonly_from_w03_262144.pt`
- mode: `--play-heads-only`

Play-EV dataset eval:

```text
w03 base:
  top1: 46.35%
  regret: 4.44
  mean_policy_ev: 38.11

play_ev_v1_2k_s4_headonly_from_w03_262144:
  top1: 48.75%
  regret: 3.57
  mean_policy_ev: 38.97
```

512-hand duplicate:

```text
play_ev_v1_2k_s4_headonly_from_w03_262144: +6.21 raw, CI [+4.34, +8.08]
contract failures: 211
illegal actions: 0
```

Interpretation: this is the first checkpoint to clear the promotion gate against conservative at 512 hands. It combines late protected PPO with supervised play-EV and looks materially better than smoke (`+5.21`) and best raw `w03` (`+5.56`) on 512. Started 2048-hand confirmation at `2026-06-21T21:09:20+00:00`.

2048-hand confirmation:

```text
play_ev_v1_2k_s4_headonly_from_w03_262144: +6.36 raw, CI [+5.44, +7.27]
contract failures: A=853, B=911
illegal actions: 0
```

Decision:

- promote `checkpoints/play_ev_v1_2k_s4_headonly_from_w03_262144.pt` as the current best checkpoint,
- keep `checkpoints/ppo_play_anchor_playkl_w03_lr3e5_000262144.pt` as the best PPO-only base,
- next experiment should use the promoted checkpoint for another play-EV iteration.

Important limitation: current play-EV rollouts use one rollout actor for all four seats. That is not exactly duplicate evaluation, where policy A controls one team and conservative controls the other. The v1 result is still useful, but the next implementation pass should add team-aware state collection/rollout actors for higher-quality labels.

## Play-EV V2 From Promoted V1 Checkpoint

Second play-EV iteration:

- base: `checkpoints/play_ev_v1_2k_s4_headonly_from_w03_262144.pt`
- state actor: `checkpoint:checkpoints/play_ev_v1_2k_s4_headonly_from_w03_262144.pt`
- rollout actor: `bot:conservative`
- dataset: `data/play_ev_v2_10k_s4_best_state_conservative_rollout.npz`
- output: `checkpoints/play_ev_v2_10k_s4_headonly_from_best.pt`
- mode: `--play-heads-only`

Dataset eval:

```text
base on v2 dataset:
  top1: 46.16%
  regret: 4.52
  mean_policy_ev: 37.84

play_ev_v2_10k_s4_headonly_from_best:
  top1: 46.87%
  regret: 3.92
  mean_policy_ev: 38.44
```

Duplicate eval:

```text
512 hands:  +11.42 raw, CI [+9.65, +13.19]
2048 hands: +11.17 raw, CI [+10.29, +12.06]
contract failures at 2048: A=732, B=1154
illegal actions: 0
```

Decision:

- promote `checkpoints/play_ev_v2_10k_s4_headonly_from_best.pt` as the current best checkpoint,
- this is a large confirmed improvement over v1 (`+6.36` raw at 2048) and smoke (`~+4.34` raw at 2048 from earlier confirmation),
- continue with another self-iteration using the v2 checkpoint as state actor/base while keeping the known limitation about single-actor rollouts in mind.

## Play-EV V3 From Promoted V2 Checkpoint

Third play-EV iteration:

- base: `checkpoints/play_ev_v2_10k_s4_headonly_from_best.pt`
- state actor: `checkpoint:checkpoints/play_ev_v2_10k_s4_headonly_from_best.pt`
- rollout actor: `bot:conservative`
- dataset: `data/play_ev_v3_10k_s4_v2_state_conservative_rollout.npz`
- output: `checkpoints/play_ev_v3_10k_s4_headonly_from_v2.pt`
- mode: `--play-heads-only`

Dataset eval:

```text
base on v3 dataset:
  top1: 46.18%
  regret: 4.21
  mean_policy_ev: 38.25

play_ev_v3_10k_s4_headonly_from_v2:
  top1: 46.37%
  regret: 4.16
  mean_policy_ev: 38.31
```

Duplicate eval:

```text
512 hands: +10.00 raw, CI [+8.09, +11.92]
contract failures at 512: A=190, B=284
illegal actions: 0
```

Decision:

- do not promote V3,
- keep `checkpoints/play_ev_v2_10k_s4_headonly_from_best.pt` as the current best checkpoint,
- skip 2048 confirmation because V3 did not clear the `+10.5` raw 512-hand gate and is below V2's `+11.42` 512-hand result.

Interpretation: naive self-iteration appears to be plateauing. V3 improved only slightly on its own play-EV dataset and regressed in duplicate eval. The next implementation step should fix label quality by making play-EV rollouts team-aware: when evaluating a candidate action for the learning policy's team, complete the hand with the policy actor on that team and the opponent actor on the other team, matching duplicate eval more closely.

## Team-Aware Play-EV Generator

Implemented support for duplicate-style play-EV labels:

- `spades-generate-play-ev` now accepts `--state-ally-actor`, `--state-opponent-actor`, `--rollout-ally-actor`, and `--rollout-opponent-actor`.
- Team-aware state collection alternates the ally team by deal and stores only states where the acting player belongs to that ally team.
- Candidate-action rollouts now complete the acting player's team with the ally actor and the other team with the opponent actor.
- Datasets now include `target_team` plus metadata flags for team-aware state collection and rollouts.

Validation:

```text
uv run ruff check src tests
uv run pytest

86 passed, 1 warning
```

## Play-EV V4 Team-Aware 2k

First team-aware label run:

- base: `checkpoints/play_ev_v2_10k_s4_headonly_from_best.pt`
- state ally: `checkpoint:checkpoints/play_ev_v2_10k_s4_headonly_from_best.pt`
- state opponent: `bot:conservative`
- rollout ally: `checkpoint:checkpoints/play_ev_v2_10k_s4_headonly_from_best.pt`
- rollout opponent: `bot:conservative`
- dataset: `data/play_ev_v4_team_aware_2k_s4_v2_vs_conservative.npz`
- output: `checkpoints/play_ev_v4_team_aware_2k_s4_from_v2.pt`
- mode: `--play-heads-only`

Dataset summary:

```text
rows: 2000
rollout samples: 4
mean_best_ev: 48.27
mean_best_second_gap: 2.76
team_aware_state_collection: true
team_aware_rollouts: true
```

Dataset eval:

```text
base on v4 team-aware dataset:
  top1: 43.45%
  regret: 3.35
  mean_policy_ev: 44.92

play_ev_v4_team_aware_2k_s4_from_v2:
  top1: 50.50%
  regret: 2.21
  mean_policy_ev: 46.06
```

Duplicate eval:

```text
512 hands: +3.99 raw, CI [+2.20, +5.77]
contract failures at 512: A=239, B=188
illegal actions: 0
```

Decision:

- do not promote V4,
- keep `checkpoints/play_ev_v2_10k_s4_headonly_from_best.pt` as the current best checkpoint,
- do not scale this team-aware setup to 10k yet.

Interpretation: the team-aware generator worked mechanically and the supervised fit learned the new labels, but transfer got much worse. Bidding counts are unchanged from the base policy, so the regression is in play decisions. This likely means the one-step team-aware play labels are too high-variance or mismatched with the changed policy's future continuation at this dataset size. Before scaling this path, inspect label quality and training dynamics: try smaller LR, evaluate intermediate epochs, and compare against a holdout duplicate set. Another likely fix is to train from team-aware labels with a stronger behavior anchor to the base play logits rather than pure label fitting.

## Play-EV V4 Behavior Anchor Sweep

Added a behavior-anchor term to `spades-train-play-ev`:

- `--behavior-anchor-path`
- `--behavior-anchor-weight`
- `--behavior-anchor-temperature`

The anchor is a KL from the base checkpoint's legal play-action distribution to the student policy's legal play-action distribution. It is intended to prevent the team-aware play-EV fit from moving too far away from the promoted V2 behavior.

Validation:

```text
uv run ruff check src tests
uv run pytest

87 passed, 1 warning
```

Sweep on `data/play_ev_v4_team_aware_2k_s4_v2_vs_conservative.npz`, loaded from `checkpoints/play_ev_v2_10k_s4_headonly_from_best.pt`:

```text
weight  label_regret  label_top1  duplicate512_raw  duplicate512_ci95
0.1     2.57          49.70%      +3.41             [+1.57, +5.25]
0.3     2.27          48.75%      +8.01             [+6.30, +9.72]
1.0     2.35          48.95%      +7.33             [+5.55, +9.11]
3.0     2.48          48.95%      +9.23             [+7.38, +11.07]
```

Decision:

- do not promote any anchor-sweep checkpoint,
- keep `checkpoints/play_ev_v2_10k_s4_headonly_from_best.pt` as current best,
- behavior anchoring helped relative to unanchored V4 (`+3.99` raw), but still did not recover V2's confirmed strength.

Interpretation: stronger anchoring improves transfer, and the best result is the strongest tested anchor. This suggests the team-aware labels may contain useful signal but the supervised update is still too disruptive. Next ablation should push anchor weight higher and/or reduce update size rather than scaling data yet.

## Play-EV V4 Strong Behavior Anchor Sweep

Follow-up sweep on the same team-aware V4 2k/s4 dataset:

- base: `checkpoints/play_ev_v2_10k_s4_headonly_from_best.pt`
- dataset: `data/play_ev_v4_team_aware_2k_s4_v2_vs_conservative.npz`
- training mode: `--play-heads-only`
- LR: `1e-4`
- epochs: `30`
- behavior anchor path: `checkpoints/play_ev_v2_10k_s4_headonly_from_best.pt`
- weights tested: `10.0`, `30.0`, `100.0`

512-hand duplicate results:

```text
weight  label_regret  label_top1  duplicate512_raw  duplicate512_ci95
10.0    3.28          44.25%      +11.27            [+9.54, +13.00]
30.0    3.23          43.95%      +11.58            [+9.80, +13.37]
100.0   3.33          43.25%      +10.81            [+9.03, +12.59]
```

Interpretation:

- This is the first V4 team-aware variant to get back to the V2 range.
- The best 512-hand result is weight `30.0`, slightly above V2's previous 512-hand result (`+11.42` raw), but the confidence intervals overlap.
- The label metrics are worse than the lower-anchor sweeps, which is expected: a high behavior anchor intentionally resists matching high-variance labels. The useful signal is the duplicate result, not label top-1.

2048-hand confirmation:

```text
checkpoint: checkpoints/play_ev_v4_team_aware_2k_s4_strong_anchor_w30p0_from_v2.pt
eval: 2048-hand duplicate against bot:conservative
raw mean: +10.78
CI: [+9.92, +11.63]
contract failures: A=733, B=1096
illegal actions: 0
```

Decision:

- do not promote the strong-anchor V4 checkpoint,
- keep `checkpoints/play_ev_v2_10k_s4_headonly_from_best.pt` as the current best,
- the 512-hand result was a useful screen but did not survive 2048 confirmation.

## Play-EV V5 Team-Aware 10k Strong Anchor

Launched a larger team-aware data-scaling run after the V4 strong-anchor no-promotion.

Important correction:

- The first V5 launch used the generator default `--device cpu`, because the pipeline runner did not pass a device into `spades-generate-play-ev`.
- This made the A100 sit at `0%` GPU while CPU load saturated the 14-core box.
- Stopped that partial run after a few hundred EV states.
- Fixed `scripts/run_play_ev_team_aware_pipeline.sh` to default `DEVICE=cuda` and pass `--device "$DEVICE"` during dataset generation.
- Restarted as `play_ev_v5_team_aware_10k_s4_v2_strong_anchor_cuda`.
- After generation completed, training initially failed because the runner still passed an invalid `--device cuda` argument to `spades-train-play-ev`.
- Fixed the runner to remove that train arg and added `SKIP_GENERATE=1` resume support.
- Resumed the same CUDA run from the completed dataset instead of regenerating rollouts.

- runner: `scripts/run_play_ev_team_aware_pipeline.sh`
- run name: `play_ev_v5_team_aware_10k_s4_v2_strong_anchor_cuda`
- base: `checkpoints/play_ev_v2_10k_s4_headonly_from_best.pt`
- state ally / rollout ally: same V2 checkpoint
- state opponent / rollout opponent: `bot:conservative`
- dataset: `data/play_ev_v5_team_aware_10k_s4_v2_strong_anchor_cuda.npz`
- states: `10000`
- rollout samples: `4`
- seed: `5205`
- train mode: `--play-heads-only`
- LR: `1e-4`
- epochs: `30`
- behavior anchor path: V2 checkpoint
- anchor weights: `10.0`, `30.0`, `100.0`
- duplicate screening: 512 hands per trained checkpoint
- log dir: `logs/play_ev_v5_team_aware_10k_s4_v2_strong_anchor_cuda/`

Launch command shape:

```bash
UV_BIN=/home/ubuntu/.local/bin/uv \
BASE_CHECKPOINT=checkpoints/play_ev_v2_10k_s4_headonly_from_best.pt \
RUN_NAME=play_ev_v5_team_aware_10k_s4_v2_strong_anchor_cuda \
STATES=10000 \
ROLLOUT_SAMPLES=4 \
SEED=5205 \
WEIGHTS="10.0 30.0 100.0" \
LEARNING_RATE=0.0001 \
EPOCHS=30 \
BATCH_SIZE=512 \
DEVICE=cuda \
DUPLICATE_HANDS=512 \
WANDB=1 \
WANDB_GROUP=play-ev-v5-team-aware-10k \
scripts/run_play_ev_team_aware_pipeline.sh
```

Status:

- CPU run started on the A100 at about `2026-06-21T23:40Z` and was aborted after discovering the device issue,
- CUDA run started at about `2026-06-21T23:48Z`,
- early CUDA health check: GPU utilization around `39%`, GPU memory around `595 MiB`, and state collection at `5010/10000`,
- auto-confirm watcher `scripts/auto_confirm_play_ev_summary.sh` is running for this run with threshold `+10.5` raw at 512 hands,
- expected runtime is roughly 3 hours for dataset generation plus a few minutes for training/eval.

Decision gate:

- if a V5 checkpoint has at least `+10.5` raw at 512 hands, the watcher will run a 2048 confirmation automatically,
- otherwise keep V2 as current best and stop scaling this exact team-aware label recipe.

Screening results:

```text
weight  label_regret  label_top1  duplicate512_raw
30.0    3.3594        0.4441      +11.22
100.0   3.3949        0.4454      +10.58
10.0    3.4606        0.4417      +10.49
```

The auto-confirm watcher selected weight `30.0` for 2048-hand duplicate confirmation.

2048-hand confirmation:

```text
checkpoint: checkpoints/play_ev_v5_team_aware_10k_s4_v2_strong_anchor_cuda_anchor_w30p0.pt
raw mean: +10.45
CI: [+9.58, +11.32]
contract failures: A=760, B=1122
illegal actions: 0
```

Decision:

- do not promote V5,
- keep `checkpoints/play_ev_v2_10k_s4_headonly_from_best.pt` as the current best checkpoint,
- V5's larger team-aware label set did not beat V2 despite a promising 512-hand screen.

## Protected PPO From V2

Implemented `scripts/run_protected_ppo_ladder.sh` to test whether PPO can improve the promoted V2 play policy without repeating the prior PPO drift.

Experiment design:

- initialize from `checkpoints/play_ev_v2_10k_s4_headonly_from_best.pt`,
- disable bid PPO with `--bid-ppo-weight 0.0`,
- freeze bid heads,
- anchor bidding to V2 with bid policy KL and bid-Q loss,
- train play decisions with PPO,
- anchor play logits to V2 with play KL,
- checkpoint every fixed step interval,
- evaluate base and all checkpoints on the same duplicate deals,
- write `summary.json`,
- auto-confirm the best checkpoint at 2048 hands if it clears the screen threshold.

Default command shape:

```bash
UV_BIN=/home/ubuntu/.local/bin/uv \
BASE_CHECKPOINT=checkpoints/play_ev_v2_10k_s4_headonly_from_best.pt \
RUN_NAME=ppo_protected_v2_playkl_w1_lr3e5_524k \
TOTAL_TIMESTEPS=524288 \
CHECKPOINT_INTERVAL_STEPS=65536 \
LEARNING_RATE=0.00003 \
PLAY_ANCHOR_WEIGHT=1.0 \
BID_ANCHOR_WEIGHT=1.0 \
BID_Q_ANCHOR_WEIGHT=0.1 \
BID_PPO_WEIGHT=0.0 \
PLAY_PPO_WEIGHT=1.0 \
DUPLICATE_HANDS=512 \
CONFIRM_THRESHOLD_RAW=10.8 \
WANDB=1 \
WANDB_GROUP=ppo-protected-v2 \
scripts/run_protected_ppo_ladder.sh
```

Interpretation target:

- A checkpoint must beat V2 in duplicate eval, not just improve online PPO reward.
- If protected PPO cannot beat V2, the next implementation direction is better search/label quality rather than more PPO scale.

Launch status:

- run name: `ppo_protected_v2_playkl_w1_lr3e5_524k`
- launched on A100 at `2026-06-22T03:33:39+00:00`
- W&B run: `https://wandb.ai/veerpareek12/spades-rl/runs/hh5pow8b`
- initial health: GPU utilization about `72%`, memory about `72 GiB / 80 GiB`,
- early PPO logs showed no illegal actions and low play-anchor KL around `0.0027`.

Results:

```text
checkpoint  duplicate512_raw
base V2     +11.42
065536      +10.75
131072      +11.00
196608      +10.94
262144      +10.70
327680      +10.28
393216      +11.66
458752      +10.87
524288      +11.33
final       +11.33
```

The `393216` checkpoint was the only one that beat the same-seed V2 base on the 512-hand screen, so the runner auto-confirmed it at 2048 hands.

2048-hand confirmation:

```text
checkpoint: checkpoints/ppo_protected_v2_playkl_w1_lr3e5_524k_000393216.pt
raw mean: +10.79
CI: [+9.88, +11.70]
contract failures: A=761, B=1160
illegal actions: 0
```

Decision:

- do not promote the protected PPO checkpoint,
- keep `checkpoints/play_ev_v2_10k_s4_headonly_from_best.pt` as current best,
- protected PPO can produce short-screen candidates, but this `play_anchor_weight=1.0` run did not beat V2 at 2048 hands.

Next ablation:

- run the same protected PPO recipe with lower play-anchor weight (`0.3`),
- keep bid PPO disabled and bid heads frozen,
- keep bid anchors active,
- use the same checkpoint interval and duplicate screen,
- require a 512-hand checkpoint screen above the same-run V2 base before spending 2048-hand confirmation,
- require a 2048 duplicate result above V2 before promotion.

## Protected PPO From V2, Play Anchor 0.3

Launched the lower play-anchor ablation to test whether the `1.0` play KL anchor was too restrictive.

Run configuration:

- run name: `ppo_protected_v2_playkl_w03_lr3e5_524k`
- base: `checkpoints/play_ev_v2_10k_s4_headonly_from_best.pt`
- total timesteps: `524288`
- checkpoint interval: `65536`
- LR: `3e-5`
- bid PPO weight: `0.0`
- play PPO weight: `1.0`
- bid anchor weight: `1.0`
- bid-Q anchor weight: `0.1`
- play anchor weight: `0.3`
- duplicate screen: `512` hands
- auto-confirm: only if best checkpoint beats both `+10.8` raw and the same-run V2 base screen.

Launch status:

- launched on A100 at `2026-06-22T04:15:25+00:00`,
- W&B run: `https://wandb.ai/veerpareek12/spades-rl/runs/1u5qya5k`,
- initial health: GPU memory about `72 GiB / 80 GiB`, no illegal actions in early epochs.

Final status before teardown:

- training completed successfully through `524288` steps and wrote all interval checkpoints plus the final checkpoint,
- no illegal actions in training,
- the wrapper had just started the base 512-hand duplicate eval when the user asked to stop launching evals,
- stopped the wrapper before any completed w0.3 duplicate-eval JSON was written,
- no w0.3 checkpoint was evaluated, so no promotion decision can be made from this ablation yet.

Artifacts copied back locally under `cloud_artifacts_2026_06_21/`:

- `logs/`
- `checkpoints/`
- `data/`
- `wandb/`

The synced local artifact directory is about `1.4G`. Remote GPU was idle and no Spades/Puffer processes were running before cleanup.

Remote cleanup:

- removed `~/spades-rl`,
- removed uv/Python caches under `~/.cache` and `~/.local/share/uv`,
- verified GPU idle and no project processes running,
- remote home directory reduced to about `60M`,
- A100 instance is ready to tear down.
