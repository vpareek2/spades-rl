# Spades RL Experiment Notes - 2026-06-21

## Current Best

- Keep `checkpoints/ppo_play_anchor_smoke.pt` as the promoted/current-best checkpoint.
- Do not promote `ppo_play_anchor_ladder_*`: unanchored play PPO drifted badly.
- Do not promote `ppo_play_anchor_playkl_w1_lr3e5_000098304.pt`: play-KL anchoring stabilized PPO, but the 2048-hand duplicate gain over smoke was noise.

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
