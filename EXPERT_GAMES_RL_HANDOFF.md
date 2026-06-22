# Spades RL Expert Handoff

This document summarizes the current state of the Spades RL project for an external
games-RL expert. It is intended to be self-contained: what game variant we model,
what infrastructure exists, what experiments were run, what worked, what failed,
and where we need directional advice.

## Executive Summary

We have a working Spades+ Classic RL environment, GPU training integration, EV
label generation for bidding and play, duplicate head-to-head evaluation, and a
current best checkpoint.

Current promoted checkpoint:

```text
checkpoints/play_ev_v2_10k_s4_headonly_from_best.pt
```

Current best confirmed result:

```text
duplicate eval vs bot:conservative
hands: 2048 duplicate deals
raw margin: +11.17 points/hand
CI95 raw: [+10.29, +12.06]
illegal actions: 0
```

The project has moved past basic environment correctness and into the harder
problem: generating high-quality improvement signal for play decisions under
hidden information. Bidding can be bootstrapped with rollout EV. Play can be
improved with supervised EV labels, but naive self-iteration, team-aware labels,
and PPO have not yet beaten the current V2 checkpoint after 2048-hand
confirmation.

Our current belief:

- The infrastructure is mostly ready.
- The model is probably adequate for the next stage.
- The bottleneck is label/search quality for play, not raw model size.
- PPO alone is unstable or too noisy unless heavily protected.
- 512-hand duplicate screens are useful but have produced false positives; 2048
  duplicate confirmation is the current promotion gate.

## Repository And Artifact State

Repository:

```text
https://github.com/vpareek2/spades-rl.git
branch: codex/puffer-spades-training
latest pushed cleanup commit at the end of this run: cb69f50 Record remote cleanup
```

Important local artifact root:

```text
cloud_artifacts_2026_06_21/
```

This contains copied cloud artifacts:

```text
cloud_artifacts_2026_06_21/checkpoints/
cloud_artifacts_2026_06_21/data/
cloud_artifacts_2026_06_21/logs/
cloud_artifacts_2026_06_21/wandb/
```

The final A100 instance was cleaned after artifact sync:

- `~/spades-rl` removed remotely.
- uv/Python caches removed remotely.
- GPU verified idle.
- No project train/eval process remained.

Untracked local files intentionally remain:

```text
advice.md
cloud_artifacts_2026_06_21/
```

## Game Variant And Environment

The environment models four-player partnership Spades.

Teams:

```text
Team 0: players 0 and 2
Team 1: players 1 and 3
```

Default rule features:

- Individual player bids.
- Team score delta is sum of partners' individual deltas.
- Nil enabled.
- Blind nil enabled.
- Bags enabled with threshold 10 and penalty -100.
- Spades cannot be led until broken unless the player has only spades.
- Dealer, first bidder, and first trick leader rotate by hand.

Action space is fixed at size 67:

```text
0-51: play card by card id
52-64: normal bids 1-13
65: nil
66: blind nil
```

Card ids:

```text
card_id = suit * 13 + rank_index
suits: clubs, diamonds, hearts, spades
ranks: 2 ... A
```

Main environment and model files:

```text
src/spades/env.py
src/spades/rules.py
src/spades/scoring.py
src/spades/observations.py
src/spades/policy.py
src/spades/puffer.py
src/spades/bid_ev.py
src/spades/bid_ev_train.py
src/spades/play_ev.py
src/spades/play_ev_train.py
src/spades/eval.py
```

Testing status after major implementation work:

- Environment/rules/wrappers/EV tooling tests passed on cloud and local.
- Recent full cloud test count after play-EV/Puffer work was `71 passed, 1 warning`
  on the A100 setup.
- Earlier local test counts rose from 38 to 87 as bid/play EV and duplicate eval
  tests were added.

## Observation And Model

The flat observation size used by current checkpoints is:

```text
observation_size: 564
```

Observation includes, among other fields:

- Phase.
- Relative current player/dealer/leader.
- Trick number and turn index.
- Team scores and bags.
- Bid kind/value.
- Tricks taken.
- Own hand mask.
- Played cards mask.
- Current trick cards/seats.
- Spades-broken flag.
- Action mask.
- Public history fields.

Current transformer policy shape used for best checkpoints:

```text
d_model: 256
transformer_layers: 6
attention_heads: 8
ffn_size: 1024
dropout: 0.05
```

Policy has:

- Card/play logits over card tokens.
- Bid policy head.
- Bid-Q head.
- Value/margin heads.
- Auxiliary scaffolding for hidden-owner/void prediction.

At this stage we have not aggressively scaled the model because the strongest
signals suggest label/search quality is limiting progress more than
representation capacity.

## Evaluation Methodology

The main evaluation is duplicate head-to-head Spades:

```bash
uv run spades-eval-duplicate \
  --policy-a checkpoint:<candidate> \
  --policy-b bot:conservative \
  --hands <N>
```

Duplicate eval procedure:

- Generate a deal.
- Play it twice with team seats swapped.
- Policy A controls one partnership in one orientation and the other partnership
  in the swapped orientation.
- Conservative bot controls the opposing partnership.
- Report average A margin across duplicate orientations.

Reported fields include:

- `duplicate_margin_raw_mean`: raw points/hand.
- `duplicate_margin_ci95`: CI over scaled margin; multiply by 100 for raw points.
- `contract_failures`.
- `bid_counts`.
- `illegal_actions`.

Current screening/promotion rule:

- 512-hand duplicate is a cheap screen.
- 2048-hand duplicate is required for promotion.
- 512-hand wins have repeatedly failed to survive 2048 confirmation.

Important caveat:

The opponent is currently `bot:conservative`, not a population of strong or
adaptive opponents. Results should be interpreted as progress against this
specific baseline, not as general Spades strength.

## Baseline Bots

The project uses simple bots for rollouts and comparison:

- Conservative bidding/play bot.
- Lowest legal card bot.
- Highest legal card bot.
- Random legal bot.

Most important baseline in experiments:

```text
bot:conservative
```

It is the default rollout/eval opponent for most of the current work.

## Infrastructure Work Completed

### Local/Cloud Setup

We used cloud GPU boxes because PufferLib's native/CUDA backend could not be
used locally.

Key setup steps:

- Cloned repo on GPU instances.
- Installed `uv`.
- Installed compiler/CUDA dependencies.
- Built PufferLib float32 backend with:

```bash
./scripts/build_pufferlib_float.sh
```

PufferLib build issues encountered:

- Initial failure due missing CUDA compiler/toolchain.
- `nvcc` host compiler error:

```text
gcc: fatal error: cannot execute 'cc1plus': execvp: No such file or directory
nvcc fatal: Failed to preprocess host compiler properties.
```

- Fixed by installing/using proper `gcc-11`, `g++-11`, and CUDA toolkit paths.
- Later verified build produced:

```text
PufferLib/pufferlib/_C.cpython-312-x86_64-linux-gnu.so
precision_bytes 4
```

GPU hardware used:

- A6000 48GB first.
- A100 80GB later for faster turnaround.

### Scripts Added

Pipeline scripts added:

```text
scripts/build_pufferlib_float.sh
scripts/run_play_ev_team_aware_pipeline.sh
scripts/auto_confirm_play_ev_summary.sh
scripts/run_protected_ppo_ladder.sh
```

Notable script behavior:

- Team-aware play-EV runner supports resume after generation with `SKIP_GENERATE=1`.
- Play-EV generator now defaults/passes CUDA device correctly for generation.
- Protected PPO ladder writes interval checkpoints, duplicate screens them, writes
  `summary.json`, and optionally auto-confirms the best candidate.
- Auto-confirm gate was tightened so future protected PPO runs only spend 2048
  eval if a checkpoint beats both a fixed threshold and the same-run base screen.

## Bid-EV Stage

Goal:

Bootstrap bidding from rollout EV because bidding is sparse/delayed and naive PPO
often collapses to bad bid distributions.

Dataset:

```text
data/bid_ev_v1_5k_s8.npz
states: 5000
hidden_samples: 8
state_bot: conservative
rollout_bot: conservative
generation time: about 1h41m on A6000-era setup
```

Dataset summary:

```text
mean_best_ev: 38.63
mean_best_second_gap: 7.47
nil_best_rate: 0.0048
blind_nil_best_rate: 0.0060
```

Best-action distribution:

```text
action 52 / bid 1: 1261
action 53 / bid 2: 1625
action 54 / bid 3: 1232
action 55 / bid 4: 537
action 56 / bid 5: 214
action 57 / bid 6: 60
action 58 / bid 7: 12
action 59 / bid 8: 5
action 65 / nil: 24
action 66 / blind nil: 30
```

Initial quick bid model:

```text
checkpoints/bid_ev_transformer_v1_5k_s8.pt
```

It underfit/collapsed toward bid 2 in duplicate evaluation.

Stronger bid model:

```text
checkpoints/bid_ev_transformer_v1_5k_s8_stronger.pt
```

Training result:

```text
best_val_policy_regret: 3.39725
final val policy_top1: 0.57
final val q_mae: 21.92
```

Duplicate eval for stronger bid model vs conservative:

```text
hands: 2048
raw margin: +2.75 points/hand
CI95 raw: approximately [+1.90, +3.59]
illegal actions: 0
```

Interpretation:

Bid EV worked. It produced a model that bids conditionally and beats the
conservative bot. It is not enough by itself: many contracts still fail and
the model needs better play.

## PPO Before Play-EV

### Protected PPO Smoke

Goal:

Take the stronger bid-EV checkpoint and let PPO improve play while preserving
bidding.

Key ingredients:

- Load bid-EV model.
- Freeze bid heads.
- Disable bid PPO.
- Enable play PPO.
- Anchor bid policy/Q to the bid-EV checkpoint.

Representative command:

```bash
uv run spades-train-puffer \
  --load-path checkpoints/bid_ev_transformer_v1_5k_s8_stronger.pt \
  --save-path checkpoints/ppo_play_anchor_smoke.pt \
  --total-timesteps 131072 \
  --num-envs 32 \
  --horizon 32 \
  --minibatch-size 1024 \
  --learning-rate 1e-4 \
  --anneal-lr \
  --active-only-loss \
  --bid-ppo-weight 0.0 \
  --play-ppo-weight 1.0 \
  --freeze-bid-heads \
  --bid-anchor-path checkpoints/bid_ev_transformer_v1_5k_s8_stronger.pt \
  --bid-anchor-weight 1.0 \
  --bid-q-anchor-weight 0.05
```

Result:

```text
checkpoint: checkpoints/ppo_play_anchor_smoke.pt
duplicate 512 raw: +5.21
CI95 raw: [+3.43, +7.00]
illegal actions: 0
```

Interpretation:

This was the first meaningful improvement over pure bid-EV. It showed that
protected PPO could improve play somewhat without destroying bidding.

### Unanchored PPO Ladder From Smoke

Goal:

See if longer PPO from the smoke checkpoint improves play.

Result:

```text
best 512-hand checkpoint: ppo_play_anchor_ladder_000131072.pt
best raw: +4.20
smoke baseline same eval family: +5.21
later checkpoints: fell toward about -5 raw
```

Interpretation:

Unanchored play PPO drifted badly. Bidding stayed relatively stable, but play
quality degraded.

### Play-KL Anchored PPO From Smoke

We added play KL anchoring to reduce play drift.

Weight `1.0` run:

```text
best 512: +5.32 raw
smoke 512: +5.21 raw
2048 confirm best: +4.39 raw
smoke 2048: +4.34 raw
```

Weight `0.3` run:

```text
best 512: +5.56 raw at 262144 steps
smoke 512: +5.21 raw
promotion gate was +1.0 raw over smoke, so no 2048 confirm
```

Interpretation:

Play KL prevents collapse but yields small/no confirmed gains from smoke. This
motivated play-EV supervised labels.

## Play-EV Supervised Learning

Goal:

Collect play states, evaluate legal card choices by rollout, and train play
logits from EV-derived labels. This is similar in spirit to using search/rollout
targets for policy improvement.

Implemented commands:

```text
spades-generate-play-ev
spades-summarize-play-ev
spades-train-play-ev
spades-eval-play-ev
```

Stored dataset fields:

- `observations`
- `legal_mask`
- `evaluated_mask`
- `q_values`
- `q_std`
- `visit_counts`
- `best_action`
- replay metadata:
  - dealer
  - initial hands
  - previous bid actions
  - previous play actions
  - team scores/bags

Training:

- Uses card-action logits.
- Can train play heads only.
- Can add behavior-anchor KL to a teacher/base checkpoint.
- Keeps checkpoint format compatible with Puffer policy.

### Play-EV V1, 2k States

Dataset:

```text
data/play_ev_v1_2k_s4_smoke_state_conservative_rollout.npz
states: 2000
rollout_samples: 4
state_actor: checkpoint:checkpoints/ppo_play_anchor_smoke.pt
rollout_actor: bot:conservative
mean_best_ev: 42.54
mean_best_second_gap: 4.11
```

Fit from smoke:

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

Duplicate:

```text
play_ev_v1_2k_s4_headonly 512: +5.38 raw
```

Then trained same labels from best w0.3 PPO checkpoint:

```text
checkpoint: play_ev_v1_2k_s4_headonly_from_w03_262144.pt
512 raw: +6.21
2048 raw: +6.36
CI95 raw: [+5.44, +7.27]
```

Decision:

Promote V1-from-w03 as current best at that point.

Interpretation:

Play-EV labels transferred better than PPO-only and gave the first clear
confirmed gain over smoke.

### Play-EV V2, 10k States

Second play-EV iteration using V1 as state actor/base:

```text
base: checkpoints/play_ev_v1_2k_s4_headonly_from_w03_262144.pt
state_actor: same V1 checkpoint
rollout_actor: bot:conservative
dataset: data/play_ev_v2_10k_s4_best_state_conservative_rollout.npz
states: 10000
rollout_samples: 4
```

Dataset summary:

```text
mean_best_ev: 42.36
mean_best_second_gap: 4.29
```

Dataset eval:

```text
base on V2 dataset:
  top1: 46.16%
  regret: 4.52
  mean_policy_ev: 37.84

play_ev_v2_10k_s4_headonly_from_best:
  top1: 46.87%
  regret: 3.92
  mean_policy_ev: 38.44
```

Duplicate:

```text
512 raw: +11.42
2048 raw: +11.17
CI95 raw: [+10.29, +12.06]
contract failures at 2048: A=732, B=1154
illegal actions: 0
```

Decision:

Promote V2. This remains the current champion.

Interpretation:

This was the biggest confirmed jump. Scaling single-actor play-EV from 2k to
10k states produced a large improvement.

### Play-EV V3, 10k Self-Iteration From V2

Third iteration:

```text
base: V2 champion
state_actor: V2 champion
rollout_actor: bot:conservative
states: 10000
rollout_samples: 4
```

Dataset eval:

```text
base:
  top1: 46.18%
  regret: 4.21
  mean_policy_ev: 38.25

V3 model:
  top1: 46.37%
  regret: 4.16
  mean_policy_ev: 38.31
```

Duplicate:

```text
512 raw: +10.00
CI95 raw: [+8.09, +11.92]
```

Decision:

Do not promote. Keep V2.

Interpretation:

Naive self-iteration plateaued/regressed. More of the same single-actor play-EV
data was not sufficient.

## Team-Aware Play-EV

Motivation:

Previous play-EV rollouts used one rollout actor for all four seats. This does
not match duplicate eval, where the candidate policy controls one partnership
and the conservative bot controls the opponents. We implemented team-aware
state collection and rollouts.

New generator options:

```text
--state-ally-actor
--state-opponent-actor
--rollout-ally-actor
--rollout-opponent-actor
```

Mechanics:

- Alternate ally team by deal.
- Store only states where acting player belongs to ally team.
- Candidate rollouts complete the acting team with ally actor and opponents with
  opponent actor.
- Store `target_team` and metadata flags.

### V4 Team-Aware 2k, Unanchored

Dataset:

```text
base: V2
state ally: V2
state opponent: bot:conservative
rollout ally: V2
rollout opponent: bot:conservative
states: 2000
rollout_samples: 4
mean_best_ev: 48.27
mean_best_second_gap: 2.76
```

Dataset eval:

```text
base:
  top1: 43.45%
  regret: 3.35
  mean_policy_ev: 44.92

trained V4:
  top1: 50.50%
  regret: 2.21
  mean_policy_ev: 46.06
```

Duplicate:

```text
512 raw: +3.99
CI95 raw: [+2.20, +5.77]
```

Decision:

Do not promote.

Interpretation:

The model learned the team-aware labels, but transfer to duplicate eval collapsed.
This strongly suggests the labels were high variance, distribution-mismatched,
or too disruptive when fit directly.

### V4 Team-Aware With Behavior Anchor

Added behavior-anchor KL to `spades-train-play-ev`:

```text
--behavior-anchor-path
--behavior-anchor-weight
--behavior-anchor-temperature
```

Anchor is KL from teacher/base legal play-action distribution to student
legal play-action distribution.

Sweep on same V4 2k team-aware dataset:

```text
weight  label_regret  label_top1  duplicate512_raw
0.1     2.57          49.70%      +3.41
0.3     2.27          48.75%      +8.01
1.0     2.35          48.95%      +7.33
3.0     2.48          48.95%      +9.23
```

Decision:

Do not promote. Anchoring helped but did not recover V2.

Strong anchor sweep:

```text
weight  label_regret  label_top1  duplicate512_raw
10.0    3.28          44.25%      +11.27
30.0    3.23          43.95%      +11.58
100.0   3.33          43.25%      +10.81
```

Best 512 candidate:

```text
checkpoint: play_ev_v4_team_aware_2k_s4_strong_anchor_w30p0_from_v2.pt
512 raw: +11.58
```

2048 confirmation:

```text
raw: +10.78
CI95 raw: [+9.92, +11.63]
```

Decision:

Do not promote. Keep V2.

Interpretation:

High behavior anchoring can prevent catastrophic transfer failure, and 512-hand
screen can look competitive, but it did not beat V2 in 2048 confirmation.

### V5 Team-Aware 10k Strong Anchor

Question:

Was V4 limited by only 2k team-aware states? Try 10k states with strong behavior
anchor.

Dataset/run:

```text
run name: play_ev_v5_team_aware_10k_s4_v2_strong_anchor_cuda
base: V2
state ally: V2
state opponent: bot:conservative
rollout ally: V2
rollout opponent: bot:conservative
states: 10000
rollout_samples: 4
seed: 5205
train mode: play-heads-only
LR: 1e-4
epochs: 30
behavior anchor: V2
weights: 10.0, 30.0, 100.0
```

Important engineering issue:

- First V5 run accidentally used CPU for generation.
- Fixed pipeline to pass `--device cuda` to `spades-generate-play-ev`.
- Training runner also initially passed an invalid `--device` arg to
  `spades-train-play-ev`; fixed and resumed with `SKIP_GENERATE=1`.

512 screen:

```text
weight  label_regret  label_top1  duplicate512_raw
30.0    3.3594        44.41%      +11.22
100.0   3.3949        44.54%      +10.58
10.0    3.4606        44.17%      +10.49
```

2048 confirmation for weight 30:

```text
raw: +10.45
CI95 raw: [+9.58, +11.32]
contract failures: A=760, B=1122
illegal actions: 0
```

Decision:

Do not promote. Keep V2.

Interpretation:

Scaling team-aware labels from 2k to 10k did not beat V2. The issue is likely
not simply dataset size. Label quality/search mismatch remains suspect.

## Protected PPO From V2

After supervised team-aware scaling failed, we tested PPO from the V2 champion
with stricter protection.

Implemented:

```text
scripts/run_protected_ppo_ladder.sh
```

Recipe:

- Initialize from V2.
- Disable bid PPO: `--bid-ppo-weight 0.0`.
- Freeze bid heads.
- Anchor bid policy to V2.
- Anchor bid-Q to V2.
- Enable play PPO.
- Anchor play logits to V2.
- Save interval checkpoints.
- Duplicate-screen all checkpoints.
- Auto-confirm best if it beats threshold.

### Protected PPO, Play Anchor 1.0

Run:

```text
run name: ppo_protected_v2_playkl_w1_lr3e5_524k
base: V2
total timesteps: 524288
checkpoint interval: 65536
learning rate: 3e-5
play anchor weight: 1.0
bid anchor weight: 1.0
bid-Q anchor weight: 0.1
bid PPO weight: 0.0
play PPO weight: 1.0
```

Training:

```text
completed 524288 steps
illegal actions: 0
play anchor KL around 0.0026-0.0028
```

512 duplicate screen:

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

Best 512 candidate:

```text
checkpoint: ppo_protected_v2_playkl_w1_lr3e5_524k_000393216.pt
512 raw: +11.66
```

2048 confirmation:

```text
raw: +10.79
CI95 raw: [+9.88, +11.70]
contract failures: A=761, B=1160
illegal actions: 0
```

Decision:

Do not promote. Keep V2.

Interpretation:

Protected PPO can generate 512-hand candidates in the V2 range, but this run
did not improve after 2048 confirmation. The PPO signal is not obviously useless,
but it remains too noisy to rely on without better evaluation/search guidance.

### Protected PPO, Play Anchor 0.3

Run:

```text
run name: ppo_protected_v2_playkl_w03_lr3e5_524k
base: V2
total timesteps: 524288
checkpoint interval: 65536
learning rate: 3e-5
play anchor weight: 0.3
bid anchor weight: 1.0
bid-Q anchor weight: 0.1
```

Training completed:

```text
agent_steps: 524288
illegal actions: 0
final env/hand_score: 0.0566
final loss/play_anchor_kl: 0.00267
saved checkpoints: every 65536 steps through 524288
```

No completed duplicate eval was run because the user asked to stop launching new
training/eval runs. The wrapper had just started the base 512 duplicate eval and
was stopped before any completed JSON was written.

Decision:

No promotion decision possible for w0.3 from V2 yet. Checkpoints are preserved
locally in artifacts.

## Current Best Checkpoint And Why

Current best remains:

```text
checkpoints/play_ev_v2_10k_s4_headonly_from_best.pt
```

Why it is best:

- It has the strongest 2048-hand confirmed duplicate result.
- It beats conservative by +11.17 raw points/hand.
- Later attempts either regressed or had 512-hand gains that vanished at 2048.

Confirmed comparison table:

```text
model / run                                      2048 raw      CI95 raw              decision
smoke PPO-ish baseline                           +4.34         [+3.47, +5.22]       obsolete
play-EV V1 from w03                              +6.36         [+5.44, +7.27]       obsolete
play-EV V2                                       +11.17        [+10.29, +12.06]     current best
V4 team-aware strong anchor w30                  +10.78        [+9.92, +11.63]      no promote
V5 team-aware 10k strong anchor w30              +10.45        [+9.58, +11.32]      no promote
protected PPO from V2, play KL 1.0 at 393216     +10.79        [+9.88, +11.70]      no promote
```

## What We Think The Results Mean

### Bidding Is No Longer The Main Bottleneck

The bid-EV model got us to a reasonable bidding policy. Later improvements mostly
changed play. Bid counts stayed similar across V2/V4/V5/PPO variants, and no
illegal actions occurred. Contract failures remain high but the deltas suggest
play quality and contract execution matter heavily.

### Play-EV Supervision Is The Best Lever So Far

The largest confirmed gain came from V2 play-EV:

```text
V1 2048: +6.36
V2 2048: +11.17
```

This suggests supervised policy improvement from rollout/search targets is more
sample-efficient and stable than PPO for this environment.

### Naive Self-Iteration Plateaued

V3 used the current best policy as state actor/base but did not improve. This
looks like a policy-iteration plateau or label mismatch.

### Team-Aware Labels Are Not Automatically Better

Team-aware rollout labeling is conceptually closer to duplicate evaluation, but
unanchored V4 collapsed. Strong anchoring recovered most of the loss but did not
beat V2. Scaling to 10k did not fix it.

Hypotheses:

- Team-aware labels are higher variance because the ally policy's future behavior
  changes when the student is updated.
- One-step action EV under fixed continuation policy may not be policy-improvement
  safe in this setting.
- Four rollout samples per legal card may be too noisy.
- Candidate action values can be close; mean-best-second-gap was lower for
  team-aware data, so supervised targets may be fragile.
- Behavior KL was necessary but may also block useful learning.

### PPO Can Move The Policy But Is Not Reliable Yet

Unanchored PPO drifts badly. Protected PPO avoids catastrophic collapse and can
produce plausible 512-hand candidates. However, a 512 win did not hold at 2048.

Possible interpretations:

- PPO is overfitting to short-run stochasticity.
- Sparse hand-end rewards make credit assignment hard.
- Self-play/on-policy distribution is not aligned with duplicate vs conservative.
- Current value function is weak for trick-level credit.
- The policy needs a better prior/search target before PPO.

## Key Open Questions For An Expert

We would like advice on the next strategic direction.

### 1. How Should We Generate Better Play Targets?

Current play-EV is one-step legal card evaluation with rollout completion. The
best confirmed model came from this, but later iterations plateaued.

Questions:

- Should we move toward information-set MCTS / PIMC / determinization-based
  search for each play decision?
- Is rollout EV with 4 samples per action too low to be useful after V2?
- Should labels be values, advantage rankings, pairwise preferences, or policy
  targets from a search distribution?
- How should we handle hidden card belief?
- Should belief be learned by a network or sampled from legal-card constraints?

### 2. How Should Search Handle Partial Observability?

Spades is not poker, but it has hidden hands, partnership, void inference, and
public trick/bid history.

Questions:

- Would a belief model over remaining cards by opponent be enough?
- Should we sample determinized deals consistent with public history and run
  rollouts/search over those?
- How should we avoid strategy fusion / non-locality problems from naive
  determinization?
- Is an ISMCTS-style method warranted, or is a simpler sampled-rollout policy
  iteration likely sufficient?

### 3. What Should The Next Training Regime Be?

Options we are considering:

1. Better supervised play targets from belief/search, then train policy/Q.
2. Search-improved policy iteration: generate data from current best + search
   correction, train, repeat.
3. PPO only after supervised/search improvement.
4. Population/self-play league after model is no longer just exploiting
   conservative.

Questions:

- Is it premature to do self-play PPO?
- Should we first build a stronger bot/search opponent for evaluation?
- How do we avoid overfitting to `bot:conservative`?
- Should duplicate eval eventually be against a policy population, not a single
  bot?

### 4. Is The Current Model Architecture Adequate?

Current model is a moderate transformer over the flat/card-token observation.

Questions:

- Is there an obvious architecture gap for trick-taking games?
- Should we explicitly encode each player's void suits, inferred hand ranges,
  partnership contract pressure, and trick context as structured tokens?
- Should bidding and play use separate encoders?
- Is a value/Q head over legal cards more important than a policy-only head?

### 5. How Should We Evaluate Progress?

Current eval:

- Duplicate vs conservative.
- 512 screen, 2048 confirm.

Questions:

- Is duplicate margin vs one bot too narrow?
- Should we run multiple seeds/opponents for confirmation?
- How many duplicate hands are needed for reliable promotion decisions?
- Should we evaluate match-level score over many hands rather than independent
  hand deltas?
- Should nil/blind-nil be disabled initially for a clearer curriculum, then
  reintroduced?

## Suggested Next Step From Our Perspective

Our current recommendation, before expert input, is:

1. Stop running larger PPO from V2.
2. Implement a stronger play-target generator:
   - sample hidden deals consistent with public history,
   - use current best policy for ally continuation,
   - use conservative and/or a small opponent pool for opponents,
   - evaluate legal cards across many determinizations,
   - optionally use shallow search/rollout rather than pure rollout.
3. Train play policy/Q heads with a behavior anchor to V2.
4. Evaluate by duplicate 512 screen and 2048 confirmation.
5. Only after a new supervised/search checkpoint beats V2 should we resume PPO
   or self-play.

The central problem to solve is not "can the system train?" It can. The central
problem is "what improvement target is trustworthy under hidden information and
partner/opponent continuation mismatch?"

## Most Important Artifact Paths

Current best:

```text
cloud_artifacts_2026_06_21/checkpoints/play_ev_v2_10k_s4_headonly_from_best.pt
cloud_artifacts_2026_06_21/logs/play_ev_v2/duplicate_play_ev_v2_10k_s4_headonly_from_best_vs_conservative_2048.json
cloud_artifacts_2026_06_21/logs/play_ev_v2/train_play_ev_v2_10k_s4_headonly_from_best.json
cloud_artifacts_2026_06_21/data/play_ev_v2_10k_s4_best_state_conservative_rollout.npz
```

Bid-EV:

```text
cloud_artifacts_2026_06_21/data/bid_ev_v1_5k_s8.npz
cloud_artifacts_2026_06_21/checkpoints/bid_ev_transformer_v1_5k_s8_stronger.pt
cloud_artifacts_2026_06_21/logs/bid_ev_transformer_v1_5k_s8_stronger.json
```

Team-aware failed/near-miss runs:

```text
cloud_artifacts_2026_06_21/logs/play_ev_v4_team_aware_2k_s4/
cloud_artifacts_2026_06_21/logs/play_ev_v4_anchor_sweep/
cloud_artifacts_2026_06_21/logs/play_ev_v4_strong_anchor_sweep/
cloud_artifacts_2026_06_21/logs/play_ev_v5_team_aware_10k_s4_v2_strong_anchor_cuda/
```

Protected PPO from V2:

```text
cloud_artifacts_2026_06_21/logs/ppo_protected_v2_playkl_w1_lr3e5_524k/
cloud_artifacts_2026_06_21/checkpoints/ppo_protected_v2_playkl_w1_lr3e5_524k_000393216.pt
cloud_artifacts_2026_06_21/logs/ppo_protected_v2_playkl_w1_lr3e5_524k/duplicate_000393216_2048.json
```

Unevaluated protected PPO w0.3 checkpoints:

```text
cloud_artifacts_2026_06_21/checkpoints/ppo_protected_v2_playkl_w03_lr3e5_524k_*.pt
cloud_artifacts_2026_06_21/logs/ppo_protected_v2_playkl_w03_lr3e5_524k/train.json
```

## Glossary Of Metrics

`duplicate_margin_raw_mean`:

Average raw point margin per hand for policy A over duplicate deal orientations.
Positive means policy A beat policy B.

`duplicate_margin_mean`:

Scaled version of the raw margin. In our setup reward scale is usually `0.01`,
so raw points are `duplicate_margin_mean / 0.01`.

`policy_regret`:

Average EV gap between the dataset's best legal action and the model's selected
action on EV-labeled rows.

`policy_top1`:

Fraction of rows where the model selects the EV-labeled best action.

`mean_best_second_gap`:

Average gap between best and second-best EV action in the generated dataset.
Lower values imply noisier/more fragile labels.

`contract_failures`:

Count of individual player contract failures attributed to policy A or B across
duplicate games. Useful for debugging bidding/play execution, but not the primary
promotion metric.

`illegal_actions`:

Should always be zero. If nonzero, the checkpoint/evaluator is invalid.

## Bottom Line

The strongest verified path so far is:

```text
bid-EV supervised learning
  -> protected PPO smoke
  -> play-EV V1
  -> play-EV V2
```

The current champion is good enough to serve as a serious baseline, but subsequent
attempts show that the simple improvement loop is saturated. We need expert
guidance on search/belief/target generation for play decisions under partial
observability, and on when/how to reintroduce PPO or self-play.
