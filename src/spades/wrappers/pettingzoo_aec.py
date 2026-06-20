"""Minimal PettingZoo AEC wrapper."""

from typing import Any

import gymnasium
import numpy as np
from pettingzoo.utils.env import AECEnv

from spades.actions import ACTION_SPACE_SIZE
from spades.config import SpadesPlusConfig
from spades.env import SpadesPlusEnv
from spades.observations import FLAT_OBSERVATION_SIZE, flatten_observation


class SpadesAECEnv(AECEnv):
    metadata = {"name": "spades_plus_v0"}

    def __init__(self, config: SpadesPlusConfig | None = None):
        super().__init__()
        self.env = SpadesPlusEnv(config)
        self.possible_agents = [f"player_{idx}" for idx in range(4)]
        self.agents = self.possible_agents[:]
        self.agent_selection = self.possible_agents[self.env.current_player()]
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}
        self._set_current_action_masks()
        self._observation_space = gymnasium.spaces.Box(
            low=-1_000_000_000,
            high=1_000_000_000,
            shape=(FLAT_OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self._action_space = gymnasium.spaces.Discrete(ACTION_SPACE_SIZE)

    def observation_space(self, agent: str):
        return self._observation_space

    def action_space(self, agent: str):
        return self._action_space

    def observe(self, agent: str):
        player = int(agent.split("_")[1])
        return flatten_observation(self.env.observe(player))

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        self.env.reset(seed=seed)
        self.agents = self.possible_agents[:]
        self.agent_selection = self.possible_agents[self.env.current_player()]
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}
        self._set_current_action_masks()

    def step(self, action: int):
        if not self.agents:
            return
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self._cumulative_rewards[agent] = 0.0
        obs, rewards, terminated, truncated, info = self.env.step(int(action))
        self.rewards = {
            agent: float(rewards[idx]) for idx, agent in enumerate(self.possible_agents)
        }
        self.terminations = {agent: terminated for agent in self.possible_agents}
        self.truncations = {agent: truncated for agent in self.possible_agents}
        self.infos = {agent: dict(info) for agent in self.possible_agents}
        for agent_info in self.infos.values():
            agent_info["action_mask"] = (
                np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
                if terminated
                else obs["action_mask"].astype(np.int8, copy=True)
            )
        if terminated or truncated:
            self.agents = []
        else:
            self.agent_selection = self.possible_agents[self.env.current_player()]
        self._accumulate_rewards()

    def _set_current_action_masks(self) -> None:
        mask = self.env.action_mask().astype(np.int8, copy=True)
        for agent in self.possible_agents:
            self.infos[agent]["action_mask"] = mask.copy()
