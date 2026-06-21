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
- Evaluation sweep is running under `logs/playkl_w03_eval/`.

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
