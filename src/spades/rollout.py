"""Rollout helpers for smoke tests and quick experiments."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from spades.bots import Bot
from spades.env import SpadesPlusEnv


def play_match(
    env: SpadesPlusEnv,
    bots: Sequence[Bot],
    seed: int | None = None,
    max_steps: int = 5000,
) -> dict[str, Any]:
    if len(bots) != 4:
        raise ValueError("play_match requires exactly four bots")

    rng = np.random.default_rng(seed)
    obs = env.reset(seed=seed)
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    steps = 0
    while not terminated and steps < max_steps:
        player = env.current_player()
        legal_actions = env.legal_actions()
        action = int(bots[player].act(obs, legal_actions, rng))
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1

    return {
        "terminated": terminated,
        "truncated": truncated or not terminated,
        "steps": steps,
        "hands": env.hand_number + 1,
        "team_scores": env.team_scores.copy(),
        "team_bags": env.team_bags.copy(),
        "winning_team": env.winning_team,
        "info": info,
    }
