Spades+ Classic RL Environment
==============================

This repository contains a Python implementation of a turn-based, multi-agent
Spades+ Classic environment for reinforcement learning and self-play.

The current priority is correctness and testability of the rules engine. PufferLib
training integration is intentionally deferred until running on a machine that can
support it.

Project Layout
--------------

- `src/spades/env.py`: core `SpadesPlusEnv`
- `src/spades/config.py`: game, scoring, and reward configuration
- `src/spades/actions.py`: fixed action-space helpers
- `src/spades/rules.py`: legal-play and trick-resolution helpers
- `src/spades/scoring.py`: hand scoring, nil/blind nil, and bag penalties
- `src/spades/observations.py`: dict observations and flat RL vectors
- `src/spades/wrappers/`: Gymnasium and PettingZoo AEC wrappers
- `src/spades/bots.py`: simple baseline bots for smoke tests
- `src/spades/rollout.py`: quick match rollout helper
- `tests/`: unit, controlled-hand, wrapper, and rollout tests
- `env.md`: implementation spec for the Spades+ rules variant

Rules Summary
-------------

The default environment models a four-player partnership Spades game:

- Players `0` and `2` are partners; players `1` and `3` are partners.
- Each player bids individually.
- Team score delta is the sum of both partners' individual score deltas.
- Nil and blind nil are enabled by default.
- Bags are enabled with a threshold of `10` and penalty of `-100`.
- Spades cannot be led until broken unless the player has only spades.
- Dealer, first bidder, and first trick leader rotate by hand.

Action Space
------------

The environment uses one fixed discrete action space of size `67`:

- `0-51`: play card by card id
- `52-64`: normal bids `1-13`
- `65`: nil
- `66`: blind nil

Card ids are encoded as `suit * 13 + rank_index`, with suits ordered clubs,
diamonds, hearts, spades and ranks ordered `2` through `A`.

Basic Usage
-----------

```python
from spades import SpadesPlusEnv

env = SpadesPlusEnv()
obs = env.reset(seed=0)

while True:
    legal_actions = env.legal_actions()
    action = legal_actions[0]
    obs, rewards, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

For manual debugging:

```bash
uv run spades-repl
```

For a simple import smoke test:

```bash
uv run spades
```

Testing
-------

Run the full local test suite:

```bash
uv run pytest
```

Current coverage includes card/action encoding, legal action masks, deterministic
dealing, bidding/play phase transitions, scoring and bag penalties, controlled
hands, history/debug APIs, rollout smoke tests, and Gymnasium/PettingZoo wrapper
checks.

PufferLib Training
------------------

The Spades training entrypoint uses PufferLib's PyTorch PPO backend with a Python
vector environment and an action-masked policy:

```bash
uv run spades-train-puffer --help
```

On the GPU training box, build PufferLib's float32 backend first:

```bash
./scripts/build_pufferlib_float.sh
```

Then run a smoke train:

```bash
uv run spades-train-puffer \
  --num-envs 4 \
  --total-timesteps 1024 \
  --horizon 16 \
  --minibatch-size 64 \
  --d-model 64 \
  --transformer-layers 1 \
  --attention-heads 4 \
  --ffn-size 128 \
  --dropout 0.0 \
  --cuda-buffers \
  --save-path checkpoints/transformer_smoke.pt
```

By default, PPO episodes terminate at Spades hand boundaries because the
environment scores contracts per hand. Use `--match-episodes` to train across
full multi-hand matches instead.

For an easier first curriculum, constrain bidding and remove nil/blind nil:

```bash
uv run spades-train-puffer \
  --num-envs 256 \
  --total-timesteps 5242880 \
  --max-normal-bid 5 \
  --no-nil \
  --no-blind-nil \
  --cuda-buffers
```

Continue from a checkpoint with `--load-path`:

```bash
uv run spades-train-puffer \
  --load-path checkpoints/spades_curriculum_5m.pt \
  --save-path checkpoints/spades_curriculum_next.pt \
  --max-normal-bid 5 \
  --no-nil \
  --no-blind-nil \
  --cuda-buffers
```

If PPO collapses to always bidding one, bootstrap the bidding logits from the
conservative bid bot before continuing PPO:

```bash
uv run spades-pretrain-puffer \
  --load-path checkpoints/spades_curriculum_25m.pt \
  --save-path checkpoints/spades_curriculum_25m_bidbot.pt \
  --samples 262144 \
  --max-normal-bid 5 \
  --no-nil \
  --no-blind-nil \
  --freeze-encoder
```

Evaluate a checkpoint greedily:

```bash
uv run spades-eval-puffer checkpoints/spades_curriculum_25m.pt \
  --max-normal-bid 5 \
  --no-nil \
  --no-blind-nil
```

Run duplicate head-to-head eval against a baseline bot:

```bash
uv run spades-eval-duplicate \
  --policy-a bot:lowest \
  --policy-b bot:highest \
  --hands 1024 \
  --max-normal-bid 13 \
  --nil \
  --blind-nil \
  --cpu
```

Positive `duplicate_margin_mean` means policy A beat policy B after replaying
each deal with team seats swapped.

Generate rollout-EV bidding labels:

```bash
uv run spades-generate-bid-ev \
  --states 1024 \
  --hidden-samples 8 \
  --state-bot conservative \
  --rollout-bot conservative \
  --max-normal-bid 13 \
  --nil \
  --blind-nil \
  --output data/bid_ev_v1.npz

uv run spades-summarize-bid-ev data/bid_ev_v1.npz
```

The EV dataset stores one bidding observation per row plus candidate-bid
Q-values estimated by rollout. Q-values are unscaled current-team hand score
deltas, so they can train a bid-Q/policy head directly.

The best pre-public-history/pre-transformer checkpoint is preserved as a
historical artifact in `artifacts/spades_curriculum_25m_bidbot/checkpoint.pt`.
It used the legacy MLP policy and flat observation format and will not load on
branch tips that include public-history observation fields or the transformer
policy. It scored
`mean=-0.010721435770392418` over 16,384 greedy hands with normal bids capped at
5, nil disabled, blind nil disabled, and `0` illegal actions. With the full
action space enabled it scored `mean=-0.025358887389302254` over 4,096 hands and
also produced `0` illegal actions.

Current Limitations
-------------------

- Reward shaping config exists, but shaped rewards are not implemented yet.
- The Gymnasium wrapper is a single-controller wrapper over the current player.
- The PettingZoo wrapper is minimal and intended for compatibility smoke tests.
- The current PufferLib training adapter uses Python Spades envs with CUDA model
  buffers; a native C/PufferLib env would be the next performance step.
