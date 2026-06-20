"""Gymnasium-style single-controller wrapper over the core turn env."""

from typing import Any

import gymnasium
import numpy as np

from spades.actions import ACTION_SPACE_SIZE
from spades.config import SpadesPlusConfig
from spades.env import IllegalActionError, SpadesPlusEnv
from spades.observations import FLAT_OBSERVATION_SIZE, flatten_observation


class SpadesGymnasiumEnv(gymnasium.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: SpadesPlusConfig | None = None):
        self.env = SpadesPlusEnv(config)
        self.action_space = gymnasium.spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = gymnasium.spaces.Box(
            low=-1_000_000_000,
            high=1_000_000_000,
            shape=(FLAT_OBSERVATION_SIZE,),
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        obs = self.env.reset(seed=seed)
        return flatten_observation(obs), {"action_mask": obs["action_mask"].copy()}

    def step(self, action: int):
        player = self.env.current_player()
        try:
            obs, rewards, terminated, truncated, info = self.env.step(int(action))
            reward = float(rewards[player])
        except IllegalActionError as exc:
            obs = self.env.observe()
            rewards = np.zeros(4, dtype=np.float32)
            terminated = False
            truncated = False
            reward = -1.0
            info = {
                "phase": self.env.phase,
                "current_player": player,
                "legal_actions": self.env.legal_actions(),
                "illegal_action": True,
                "error": str(exc),
            }
        if terminated:
            flat_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            action_mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        else:
            flat_obs = flatten_observation(obs)
            action_mask = obs["action_mask"].copy()
        info = dict(info)
        info["all_rewards"] = rewards.copy()
        info["action_mask"] = action_mask
        return flat_obs, reward, terminated, truncated, info
