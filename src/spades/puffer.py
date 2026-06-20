"""PufferLib training adapter for the Python Spades environment."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from spades.actions import ACTION_SPACE_SIZE
from spades.config import SpadesPlusConfig
from spades.env import IllegalActionError, SpadesPlusEnv
from spades.observations import FLAT_OBSERVATION_SIZE, flatten_observation


MASK_OFFSET = FLAT_OBSERVATION_SIZE - ACTION_SPACE_SIZE


class _CudaPtr:
    """Wrap a raw CUDA pointer so torch can copy it without a C++ extension."""

    def __init__(self, ptr: int, shape: tuple[int, ...], dtype: torch.dtype):
        typestr = {
            torch.float32: "<f4",
            torch.uint8: "|u1",
        }[dtype]
        self.__cuda_array_interface__ = {
            "data": (ptr, False),
            "shape": shape,
            "typestr": typestr,
            "version": 2,
        }


@dataclass
class SpadesPufferConfig:
    num_envs: int = 64
    seed: int = 0
    max_steps_per_match: int = 5000
    reward_scale: float = 0.01
    use_cuda_buffers: bool = False


class SpadesPufferVecEnv:
    """Minimal vector env implementing the interface used by PufferLib 4.

    Each Spades table contributes four Puffer agent slots. Only the current
    player's slot can choose a real action on a given step; inactive slots are
    masked to action 0 and ignored by the environment.
    """

    obs_dtype = "FloatTensor"
    num_atns = 1
    act_sizes = (ACTION_SPACE_SIZE,)
    obs_size = FLAT_OBSERVATION_SIZE

    def __init__(self, config: SpadesPufferConfig | None = None):
        self.config = config or SpadesPufferConfig()
        if self.config.num_envs <= 0:
            raise ValueError("num_envs must be positive")

        self.num_envs = int(self.config.num_envs)
        self.num_agents = 4
        self.total_agents = self.num_envs * self.num_agents
        self.gpu = bool(self.config.use_cuda_buffers)
        if self.gpu and not torch.cuda.is_available():
            raise RuntimeError("CUDA buffers requested, but torch.cuda.is_available() is false")

        self.envs = [
            SpadesPlusEnv(SpadesPlusConfig())
            for _ in range(self.num_envs)
        ]
        self._env_steps = np.zeros(self.num_envs, dtype=np.int32)
        self._match_returns = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)
        self._obs = np.zeros((self.total_agents, self.obs_size), dtype=np.float32)
        self._rewards = np.zeros(self.total_agents, dtype=np.float32)
        self._terminals = np.zeros(self.total_agents, dtype=np.float32)
        self._latest_scores = np.zeros((self.num_envs, 2), dtype=np.int32)
        self._latest_bags = np.zeros((self.num_envs, 2), dtype=np.int32)

        self.completed_matches = 0
        self.completed_hands = 0
        self.illegal_actions = 0
        self.truncated_matches = 0
        self.total_steps = 0
        self._recent_match_returns: deque[float] = deque(maxlen=256)
        self._recent_hand_scores: deque[float] = deque(maxlen=512)

        self._gpu_obs: torch.Tensor | None = None
        self._gpu_rewards: torch.Tensor | None = None
        self._gpu_terminals: torch.Tensor | None = None
        if self.gpu:
            self._gpu_obs = torch.zeros_like(torch.from_numpy(self._obs), device="cuda")
            self._gpu_rewards = torch.zeros_like(torch.from_numpy(self._rewards), device="cuda")
            self._gpu_terminals = torch.zeros_like(torch.from_numpy(self._terminals), device="cuda")

        self.reset()

    @property
    def obs_ptr(self) -> int:
        return int(self._obs.ctypes.data)

    @property
    def rewards_ptr(self) -> int:
        return int(self._rewards.ctypes.data)

    @property
    def terminals_ptr(self) -> int:
        return int(self._terminals.ctypes.data)

    @property
    def gpu_obs_ptr(self) -> int:
        assert self._gpu_obs is not None
        return int(self._gpu_obs.data_ptr())

    @property
    def gpu_rewards_ptr(self) -> int:
        assert self._gpu_rewards is not None
        return int(self._gpu_rewards.data_ptr())

    @property
    def gpu_terminals_ptr(self) -> int:
        assert self._gpu_terminals is not None
        return int(self._gpu_terminals.data_ptr())

    def reset(self) -> None:
        for env_idx, env in enumerate(self.envs):
            env.reset(seed=self.config.seed + env_idx)
            self._env_steps[env_idx] = 0
            self._match_returns[env_idx] = 0.0
        self._rewards.fill(0.0)
        self._terminals.fill(0.0)
        self._refresh_observations()
        self._sync_to_gpu()

    def cpu_step(self, actions_ptr: int) -> None:
        raw = (ctypes.c_float * (self.total_agents * self.num_atns)).from_address(actions_ptr)
        actions = np.ctypeslib.as_array(raw).reshape(self.total_agents, self.num_atns)
        self._step(actions[:, 0].astype(np.int64, copy=False))

    def gpu_step(self, actions_ptr: int) -> None:
        actions = torch.as_tensor(
            _CudaPtr(actions_ptr, (self.total_agents, self.num_atns), torch.float32),
            device="cuda",
        )
        self._step(actions[:, 0].to(dtype=torch.int64, device="cpu").numpy())

    def _step(self, actions: np.ndarray) -> None:
        self._rewards.fill(0.0)
        self._terminals.fill(0.0)

        for env_idx, env in enumerate(self.envs):
            active_player = env.current_player()
            slot = self._slot(env_idx, active_player)
            action = int(actions[slot])
            legal = env.legal_actions()
            if action not in legal:
                self.illegal_actions += 1
                action = int(legal[0])

            try:
                _obs, rewards, terminated, truncated, info = env.step(action)
            except IllegalActionError:
                self.illegal_actions += 1
                _obs, rewards, terminated, truncated, info = env.step(int(legal[0]))

            scaled_rewards = np.asarray(rewards, dtype=np.float32) * self.config.reward_scale
            base = self._slot(env_idx, 0)
            self._rewards[base : base + self.num_agents] = scaled_rewards
            self._match_returns[env_idx] += scaled_rewards
            self._env_steps[env_idx] += 1
            self.total_steps += 1

            if info.get("hand_completed"):
                self.completed_hands += 1
                self._latest_scores[env_idx] = info["team_scores"]
                self._latest_bags[env_idx] = info["team_bags"]
                self._recent_hand_scores.append(float(np.mean(scaled_rewards)))

            timed_out = self._env_steps[env_idx] >= self.config.max_steps_per_match
            if terminated or truncated or timed_out:
                self.completed_matches += 1
                self.truncated_matches += int(timed_out and not terminated)
                self._terminals[base : base + self.num_agents] = 1.0
                self._recent_match_returns.append(float(np.mean(self._match_returns[env_idx])))
                env.reset(seed=self.config.seed + self.completed_matches * self.num_envs + env_idx)
                self._env_steps[env_idx] = 0
                self._match_returns[env_idx] = 0.0

        self._refresh_observations()
        self._sync_to_gpu()

    def _refresh_observations(self) -> None:
        for env_idx, env in enumerate(self.envs):
            active_player = env.current_player()
            for player in range(self.num_agents):
                flat = flatten_observation(env.observe(player)).astype(np.float32, copy=True)
                if player != active_player:
                    flat[MASK_OFFSET:] = 0.0
                    flat[MASK_OFFSET] = 1.0
                self._obs[self._slot(env_idx, player)] = flat

    def _sync_to_gpu(self) -> None:
        if not self.gpu:
            return
        assert self._gpu_obs is not None
        assert self._gpu_rewards is not None
        assert self._gpu_terminals is not None
        self._gpu_obs.copy_(torch.from_numpy(self._obs).to("cuda"))
        self._gpu_rewards.copy_(torch.from_numpy(self._rewards).to("cuda"))
        self._gpu_terminals.copy_(torch.from_numpy(self._terminals).to("cuda"))

    def _slot(self, env_idx: int, player: int) -> int:
        return env_idx * self.num_agents + player

    def log(self) -> dict[str, float]:
        mean_match_return = float(np.mean(self._recent_match_returns)) if self._recent_match_returns else 0.0
        mean_hand_score = float(np.mean(self._recent_hand_scores)) if self._recent_hand_scores else 0.0
        return {
            "score": mean_match_return,
            "hand_score": mean_hand_score,
            "completed_matches": float(self.completed_matches),
            "completed_hands": float(self.completed_hands),
            "illegal_actions": float(self.illegal_actions),
            "truncated_matches": float(self.truncated_matches),
            "mean_team0_score": float(np.mean(self._latest_scores[:, 0])),
            "mean_team1_score": float(np.mean(self._latest_scores[:, 1])),
            "mean_team0_bags": float(np.mean(self._latest_bags[:, 0])),
            "mean_team1_bags": float(np.mean(self._latest_bags[:, 1])),
        }

    def render(self, env_id: int = 0) -> None:
        env = self.envs[int(env_id)]
        state = env.get_debug_state()
        print(
            f"env={env_id} phase={state['phase'].value} player={state['current_player']} "
            f"scores={state['team_scores'].tolist()} bags={state['team_bags'].tolist()}"
        )

    def close(self) -> None:
        self._gpu_obs = None
        self._gpu_rewards = None
        self._gpu_terminals = None


class MaskedSpadesPolicy(nn.Module):
    """Small MLP policy that applies the flattened Spades action mask."""

    def __init__(self, obs_size: int, action_size: int, hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        in_size = obs_size
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_size, hidden_size), nn.GELU()])
            in_size = hidden_size
        self.encoder = nn.Sequential(*layers)
        self.action_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def initial_state(self, batch_size: int, device: str | torch.device):
        return ()

    def forward_eval(self, observations: torch.Tensor, state=()):
        logits, values = self._forward_flat(observations)
        return logits, values, state

    def forward(self, observations: torch.Tensor):
        batch, horizon = observations.shape[:2]
        flat_obs = observations.reshape(batch * horizon, observations.shape[-1])
        logits, values = self._forward_flat(flat_obs)
        return logits, values.reshape(batch, horizon)

    def _forward_flat(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = observations.float()
        hidden = self.encoder(x)
        logits = self.action_head(hidden)
        mask = x[:, MASK_OFFSET : MASK_OFFSET + ACTION_SPACE_SIZE] > 0.5
        fallback = torch.zeros_like(mask)
        fallback[:, 0] = True
        mask = torch.where(mask.any(dim=1, keepdim=True), mask, fallback)
        logits = logits.masked_fill(~mask, -1.0e9)
        values = self.value_head(hidden)
        return logits, values


def build_train_args(cli_args: argparse.Namespace) -> dict[str, Any]:
    return {
        "env_name": "spades",
        "world_size": 1,
        "gpu_id": 0,
        "checkpoint_dir": cli_args.checkpoint_dir,
        "log_dir": cli_args.log_dir,
        "sweep": {"metric": "score"},
        "train": {
            "total_timesteps": cli_args.total_timesteps,
            "learning_rate": cli_args.learning_rate,
            "anneal_lr": cli_args.anneal_lr,
            "min_lr_ratio": 0.0,
            "gamma": cli_args.gamma,
            "gae_lambda": cli_args.gae_lambda,
            "replay_ratio": cli_args.replay_ratio,
            "clip_coef": cli_args.clip_coef,
            "vf_coef": cli_args.vf_coef,
            "vf_clip_coef": cli_args.vf_clip_coef,
            "max_grad_norm": cli_args.max_grad_norm,
            "ent_coef": cli_args.ent_coef,
            "anneal_ent_coef": 0,
            "min_ent_coef_ratio": 0.1,
            "beta1": cli_args.beta1,
            "beta2": cli_args.beta2,
            "eps": cli_args.eps,
            "minibatch_size": cli_args.minibatch_size,
            "horizon": cli_args.horizon,
            "vtrace_rho_clip": cli_args.vtrace_rho_clip,
            "vtrace_c_clip": cli_args.vtrace_c_clip,
            "prio_alpha": cli_args.prio_alpha,
            "prio_beta0": cli_args.prio_beta0,
        },
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Spades with PufferLib's PyTorch PPO backend")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--minibatch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.90)
    parser.add_argument("--replay-ratio", type=float, default=1.0)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=2.0)
    parser.add_argument("--vf-clip-coef", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.95)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--vtrace-rho-clip", type=float, default=1.0)
    parser.add_argument("--vtrace-c-clip", type=float, default=1.0)
    parser.add_argument("--prio-alpha", type=float, default=0.8)
    parser.add_argument("--prio-beta0", type=float, default=0.2)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps-per-match", type=int, default=5000)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--metrics-path", type=str, default="")
    parser.add_argument("--cuda-buffers", action="store_true", default=False)
    parser.add_argument("--cpu-buffers", action="store_false", dest="cuda_buffers")
    parser.add_argument("--anneal-lr", action="store_true", default=False)
    return parser


def train(cli_args: argparse.Namespace) -> dict[str, Any]:
    from pufferlib import pufferl
    from pufferlib.torch_pufferl import PuffeRL

    torch.manual_seed(cli_args.seed)
    np.random.seed(cli_args.seed)

    vec = SpadesPufferVecEnv(
        SpadesPufferConfig(
            num_envs=cli_args.num_envs,
            seed=cli_args.seed,
            max_steps_per_match=cli_args.max_steps_per_match,
            reward_scale=cli_args.reward_scale,
            use_cuda_buffers=cli_args.cuda_buffers,
        )
    )
    args = build_train_args(cli_args)
    policy = MaskedSpadesPolicy(
        obs_size=vec.obs_size,
        action_size=ACTION_SPACE_SIZE,
        hidden_size=cli_args.hidden_size,
        num_layers=cli_args.num_layers,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    trainer = PuffeRL(args, vec, policy, verbose=False)
    final_logs: dict[str, Any] = {}
    try:
        while trainer.epoch < trainer.total_epochs:
            trainer.rollouts()
            trainer.train()
            if trainer.epoch % cli_args.log_interval == 0 or trainer.epoch == trainer.total_epochs:
                final_logs = dict(pufferl.unroll_nested_dict(trainer.log()))
                print(
                    "epoch={epoch} steps={steps:.0f} sps={sps:.0f} "
                    "score={score:.4f} hand_score={hand_score:.4f} "
                    "hands={hands:.0f} matches={matches:.0f} illegal={illegal:.0f}".format(
                        epoch=int(final_logs.get("epoch", trainer.epoch)),
                        steps=float(final_logs.get("agent_steps", trainer.global_step)),
                        sps=float(final_logs.get("SPS", 0.0)),
                        score=float(final_logs.get("env/score", 0.0)),
                        hand_score=float(final_logs.get("env/hand_score", 0.0)),
                        hands=float(final_logs.get("env/completed_hands", 0.0)),
                        matches=float(final_logs.get("env/completed_matches", 0.0)),
                        illegal=float(final_logs.get("env/illegal_actions", 0.0)),
                    ),
                    flush=True,
                )

        if cli_args.save_path:
            os.makedirs(os.path.dirname(cli_args.save_path) or ".", exist_ok=True)
            trainer.save_weights(cli_args.save_path)
    finally:
        trainer.close()

    final_logs["model_path"] = cli_args.save_path
    final_logs["completed_at"] = time.time()
    if cli_args.metrics_path:
        os.makedirs(os.path.dirname(cli_args.metrics_path) or ".", exist_ok=True)
        with open(cli_args.metrics_path, "w") as f:
            json.dump(final_logs, f, indent=2, sort_keys=True)
    return final_logs


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
