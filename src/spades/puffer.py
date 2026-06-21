"""PufferLib training adapter for the Python Spades environment."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from spades.actions import ACTION_SPACE_SIZE, BID_ACTION_START
from spades.actions import bid_action_to_bid
from spades.bots import ConservativeBidBot
from spades.config import SpadesPlusConfig
from spades.env import IllegalActionError, SpadesPlusEnv
from spades.observations import FLAT_OBSERVATION_SIZE, flatten_observation
from spades.policy import SpadesTransformerPolicy, load_transformer_state
from spades.state import Phase


MASK_OFFSET = FLAT_OBSERVATION_SIZE - ACTION_SPACE_SIZE
PHASE_INDEX = 0
CURRENT_PLAYER_REL_INDEX = 1


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
    hand_episodes: bool = True
    max_normal_bid: int = 13
    nil_enabled: bool = True
    blind_nil_enabled: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.max_normal_bid <= 13:
            raise ValueError("max_normal_bid must be in [1, 13]")


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
            SpadesPlusEnv(
                SpadesPlusConfig(
                    nil_enabled=self.config.nil_enabled,
                    blind_nil_enabled=self.config.blind_nil_enabled,
                    blind_nil_policy="always" if self.config.blind_nil_enabled else "disabled",
                )
            )
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
        self.completed_episodes = 0
        self.illegal_actions = 0
        self.truncated_matches = 0
        self.total_steps = 0
        self._recent_episode_returns: deque[float] = deque(maxlen=512)
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
            legal = self._training_legal_actions(env)
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
                hand_score = float(np.mean(scaled_rewards))
                self._recent_hand_scores.append(hand_score)
                if self.config.hand_episodes:
                    self.completed_episodes += 1
                    self._recent_episode_returns.append(hand_score)
                    self._terminals[base : base + self.num_agents] = 1.0

            timed_out = self._env_steps[env_idx] >= self.config.max_steps_per_match
            if terminated or truncated or timed_out:
                self.completed_matches += 1
                self.completed_episodes += 1
                self.truncated_matches += int(timed_out and not terminated)
                self._terminals[base : base + self.num_agents] = 1.0
                match_return = float(np.mean(self._match_returns[env_idx]))
                self._recent_match_returns.append(match_return)
                if not self.config.hand_episodes:
                    self._recent_episode_returns.append(match_return)
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
                flat[MASK_OFFSET:] = self._action_mask_for_observation(env, player == active_player)
                self._obs[self._slot(env_idx, player)] = flat

    def _action_mask_for_observation(self, env: SpadesPlusEnv, active: bool) -> np.ndarray:
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        if not active:
            mask[0] = 1.0
            return mask
        for action in self._training_legal_actions(env):
            mask[action] = 1.0
        return mask

    def _training_legal_actions(self, env: SpadesPlusEnv) -> list[int]:
        legal = env.legal_actions()
        if env.phase != Phase.BIDDING:
            return legal
        max_action = 51 + self.config.max_normal_bid
        return [action for action in legal if action <= max_action or action >= 65]

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
        mean_episode_return = (
            float(np.mean(self._recent_episode_returns)) if self._recent_episode_returns else 0.0
        )
        mean_match_return = float(np.mean(self._recent_match_returns)) if self._recent_match_returns else 0.0
        mean_hand_score = float(np.mean(self._recent_hand_scores)) if self._recent_hand_scores else 0.0
        return {
            "score": mean_episode_return,
            "hand_score": mean_hand_score,
            "match_score": mean_match_return,
            "completed_episodes": float(self.completed_episodes),
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


def build_policy_from_args(args: argparse.Namespace, obs_size: int, device: str | torch.device) -> SpadesTransformerPolicy:
    return SpadesTransformerPolicy(
        obs_size=obs_size,
        action_size=ACTION_SPACE_SIZE,
        d_model=args.d_model,
        num_layers=args.transformer_layers,
        num_heads=args.attention_heads,
        ffn_size=args.ffn_size,
        dropout=args.dropout,
    ).to(device)


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
            "active_only_loss": cli_args.active_only_loss,
            "bid_ppo_weight": cli_args.bid_ppo_weight,
            "play_ppo_weight": cli_args.play_ppo_weight,
            "bid_anchor_weight": cli_args.bid_anchor_weight,
            "bid_q_anchor_weight": cli_args.bid_q_anchor_weight,
            "bid_anchor_temperature": cli_args.bid_anchor_temperature,
        },
    }


def freeze_bid_heads(policy: SpadesTransformerPolicy) -> None:
    for module in (policy.bid_policy_head, policy.bid_q_head):
        for param in module.parameters():
            param.requires_grad = False


def _decision_phase_weights(
    observations: torch.Tensor,
    *,
    active_only_loss: bool,
    bid_ppo_weight: float,
    play_ppo_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    active = observations[..., CURRENT_PLAYER_REL_INDEX] == 0
    bidding = observations[..., PHASE_INDEX] == 0
    playing = ~bidding
    weights = torch.where(
        bidding,
        torch.full_like(observations[..., PHASE_INDEX], float(bid_ppo_weight)),
        torch.full_like(observations[..., PHASE_INDEX], float(play_ppo_weight)),
    )
    if active_only_loss:
        weights = weights * active.float()
    return weights, active, bidding, playing


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    denom = weights.sum()
    if float(denom.detach().item()) <= 0.0:
        return values.sum() * 0.0
    return (values * weights).sum() / denom.clamp_min(1.0)


def bid_anchor_losses(
    policy: SpadesTransformerPolicy,
    teacher: SpadesTransformerPolicy | None,
    observations: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if teacher is None:
        zero = observations.sum() * 0.0
        return zero, zero, 0
    if temperature <= 0:
        raise ValueError("bid anchor temperature must be positive")

    flat_obs = observations.reshape(-1, observations.shape[-1])
    active = flat_obs[:, CURRENT_PLAYER_REL_INDEX] == 0
    bidding = flat_obs[:, PHASE_INDEX] == 0
    bid_rows = active & bidding
    if not bool(bid_rows.any().item()):
        zero = flat_obs.sum() * 0.0
        return zero, zero, 0

    bid_obs = flat_obs[bid_rows]
    legal = bid_obs[:, MASK_OFFSET + BID_ACTION_START : MASK_OFFSET + ACTION_SPACE_SIZE] > 0.5
    valid = legal.any(dim=1)
    if not bool(valid.any().item()):
        zero = flat_obs.sum() * 0.0
        return zero, zero, 0
    bid_obs = bid_obs[valid]
    legal = legal[valid]

    student_heads = policy.forward_heads(bid_obs, apply_mask=False)
    with torch.no_grad():
        teacher_heads = teacher.forward_heads(bid_obs, apply_mask=False)

    student_logits = student_heads["logits"][:, BID_ACTION_START:ACTION_SPACE_SIZE]
    teacher_logits = teacher_heads["logits"][:, BID_ACTION_START:ACTION_SPACE_SIZE]
    student_logits = student_logits.masked_fill(~legal, -1.0e9) / temperature
    teacher_logits = teacher_logits.masked_fill(~legal, -1.0e9) / temperature
    teacher_probs = torch.softmax(teacher_logits, dim=1)
    student_log_probs = torch.log_softmax(student_logits, dim=1)
    policy_kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

    student_q = student_heads["bid_q"]
    teacher_q = teacher_heads["bid_q"]
    q_anchor = F.smooth_l1_loss(student_q[legal], teacher_q[legal])
    return policy_kl, q_anchor, int(bid_obs.shape[0])


def _wandb_config(cli_args: argparse.Namespace) -> dict[str, Any]:
    keys = [
        "num_envs",
        "total_timesteps",
        "horizon",
        "minibatch_size",
        "learning_rate",
        "gamma",
        "gae_lambda",
        "replay_ratio",
        "clip_coef",
        "vf_coef",
        "vf_clip_coef",
        "max_grad_norm",
        "ent_coef",
        "d_model",
        "transformer_layers",
        "attention_heads",
        "ffn_size",
        "dropout",
        "seed",
        "reward_scale",
        "max_normal_bid",
        "nil",
        "blind_nil",
        "hand_episodes",
        "load_path",
        "save_path",
        "active_only_loss",
        "bid_ppo_weight",
        "play_ppo_weight",
        "freeze_bid_heads",
        "bid_anchor_path",
        "bid_anchor_weight",
        "bid_q_anchor_weight",
        "bid_anchor_temperature",
    ]
    return {key: getattr(cli_args, key) for key in keys}


def _init_wandb(cli_args: argparse.Namespace):
    if not cli_args.wandb:
        return None, None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B logging requested with --wandb, but wandb is not installed") from exc
    run = wandb.init(
        project=cli_args.wandb_project,
        group=cli_args.wandb_group or None,
        name=cli_args.wandb_run_name or None,
        mode=cli_args.wandb_mode,
        config=_wandb_config(cli_args),
    )
    return wandb, run


def _wandb_artifact_name(cli_args: argparse.Namespace) -> str:
    if cli_args.wandb_artifact_name:
        return cli_args.wandb_artifact_name
    if cli_args.save_path:
        return os.path.splitext(os.path.basename(cli_args.save_path))[0]
    return "spades_puffer"


def attach_log_history(final_logs: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = dict(final_logs)
    metrics["history"] = [dict(row) for row in history]
    return metrics


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
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=6)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--ffn-size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps-per-match", type=int, default=5000)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--max-normal-bid", type=int, default=13)
    parser.add_argument("--nil", action="store_true", default=True)
    parser.add_argument("--no-nil", action="store_false", dest="nil")
    parser.add_argument("--blind-nil", action="store_true", default=True)
    parser.add_argument("--no-blind-nil", action="store_false", dest="blind_nil")
    parser.add_argument("--hand-episodes", action="store_true", default=True)
    parser.add_argument("--match-episodes", action="store_false", dest="hand_episodes")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--load-path", type=str, default="")
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--metrics-path", type=str, default="")
    parser.add_argument("--cuda-buffers", action="store_true", default=False)
    parser.add_argument("--cpu-buffers", action="store_false", dest="cuda_buffers")
    parser.add_argument("--anneal-lr", action="store_true", default=False)
    parser.add_argument("--active-only-loss", action="store_true", default=False)
    parser.add_argument("--bid-ppo-weight", type=float, default=1.0)
    parser.add_argument("--play-ppo-weight", type=float, default=1.0)
    parser.add_argument("--freeze-bid-heads", action="store_true", default=False)
    parser.add_argument("--bid-anchor-path", type=str, default="")
    parser.add_argument("--bid-anchor-weight", type=float, default=0.0)
    parser.add_argument("--bid-q-anchor-weight", type=float, default=0.0)
    parser.add_argument("--bid-anchor-temperature", type=float, default=1.0)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb-project", type=str, default="spades-rl")
    parser.add_argument("--wandb-group", type=str, default="ppo")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-artifact-name", type=str, default="")
    return parser


def _filter_bidding_mask(flat: np.ndarray, max_normal_bid: int) -> np.ndarray:
    flat = flat.copy()
    mask = flat[MASK_OFFSET:].copy()
    allowed = np.zeros_like(mask)
    allowed[:52] = mask[:52]
    allowed[52 : 52 + max_normal_bid] = mask[52 : 52 + max_normal_bid]
    allowed[65:] = mask[65:]
    flat[MASK_OFFSET:] = allowed
    return flat


def _filter_bidding_legal_actions(legal_actions: list[int], max_normal_bid: int) -> list[int]:
    max_action = 51 + max_normal_bid
    return [action for action in legal_actions if action <= max_action or action >= 65]


def train(cli_args: argparse.Namespace) -> dict[str, Any]:
    from pufferlib import pufferl
    from pufferlib.torch_pufferl import (
        Profile,
        PuffeRL,
        _actions_for_vec_step,
        compute_puff_advantage,
        sample_logits,
    )

    if cli_args.bid_ppo_weight < 0 or cli_args.play_ppo_weight < 0:
        raise ValueError("PPO phase weights must be non-negative")
    if cli_args.bid_anchor_weight < 0 or cli_args.bid_q_anchor_weight < 0:
        raise ValueError("bid anchor weights must be non-negative")
    if cli_args.bid_anchor_temperature <= 0:
        raise ValueError("bid anchor temperature must be positive")

    class AlignedPuffeRL(PuffeRL):
        """PuffeRL variant that stores step rewards with the sampled action.

        PufferLib's generic PyTorch rollout loop mirrors the native callback
        convention where reward buffers are read before the next action is
        sampled. For sparse hand-end rewards in this Python turn env, storing
        the reward after stepping gives much cleaner credit assignment.
        """

        def __init__(self, *args, bid_teacher: SpadesTransformerPolicy | None = None, **kwargs):
            super().__init__(*args, **kwargs)
            self.bid_teacher = bid_teacher

        def rollouts(self):
            prof = self.profile
            config = self.config
            device = self.device
            horizon = config["horizon"]

            self.state = tuple(torch.zeros_like(s) for s in self.state) if self.state else ()
            o = self.vec_obs

            prof.mark(0)
            for t in range(horizon):
                o_device = torch.as_tensor(o, device=device)

                prof.mark(1)
                with torch.no_grad():
                    logits, value, state = self.policy.forward_eval(o_device, self.state)
                    action, logprob, _ = sample_logits(logits)
                prof.mark(2)

                with torch.no_grad():
                    self.state = state
                    self.observations[t] = o_device
                    self.actions[t] = action
                    self.logprobs[t] = logprob
                    self.values[t] = value.flatten()

                actions_flat = _actions_for_vec_step(action)
                if self.gpu:
                    actions_flat = actions_flat.cuda()
                    self._vec.gpu_step(actions_flat.data_ptr())
                    torch.cuda.synchronize()
                else:
                    if actions_flat.is_cuda:
                        actions_flat = actions_flat.cpu()
                    self._vec.cpu_step(actions_flat.data_ptr())

                o = self.vec_obs
                with torch.no_grad():
                    self.rewards[t] = torch.as_tensor(self.vec_rewards, device=device)
                    self.terminals[t] = torch.as_tensor(self.vec_terminals, device=device).float()
                prof.mark(3)
                prof.elapsed(Profile.EVAL_GPU, 1, 2)
                prof.elapsed(Profile.EVAL_ENV, 2, 3)

            prof.mark(1)
            prof.elapsed(Profile.ROLLOUT, 0, 1)
            self.global_step += self.total_agents * horizon
            self.env_logs = self._vec.log()

        def train(self):
            prof = self.profile
            losses = defaultdict(float)
            counts = defaultdict(float)
            config = self.config
            device = self.device

            b0 = config["prio_beta0"]
            a = config["prio_alpha"]
            clip_coef = config["clip_coef"]
            vf_clip = config["vf_clip_coef"]
            anneal_beta = b0 + (1 - b0) * a * self.epoch / self.total_epochs
            self.ratio[:] = 1

            learning_rate = config["learning_rate"]
            if config["anneal_lr"] and self.epoch > 0:
                lr_ratio = self.epoch / self.total_epochs
                lr_min = config["learning_rate"] * config["min_lr_ratio"]
                learning_rate = lr_min + 0.5 * (learning_rate - lr_min) * (1 + np.cos(np.pi * lr_ratio))
                self.optimizer.param_groups[0]["lr"] = learning_rate

            obs = self.observations.transpose(0, 1).contiguous()
            act = self.actions.transpose(0, 1).contiguous()
            val = self.values.T.contiguous()
            lp = self.logprobs.T.contiguous()
            rew = self.rewards.T.contiguous().clamp(-1, 1)
            ter = self.terminals.T.contiguous()

            prof.mark(0)
            num_minibatches = int(config["replay_ratio"] * self.batch_size / config["minibatch_size"])
            for _mb in range(num_minibatches):
                shape = val.shape
                advantages = torch.zeros(shape, device=device)
                advantages = compute_puff_advantage(
                    val,
                    rew,
                    ter,
                    self.ratio,
                    advantages,
                    config["gamma"],
                    config["gae_lambda"],
                    config["vtrace_rho_clip"],
                    config["vtrace_c_clip"],
                )

                adv = advantages.abs().sum(axis=1)
                prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
                prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)
                idx = torch.multinomial(prio_probs, self.minibatch_segments, replacement=True)
                mb_prio = (self.total_agents * prio_probs[idx, None]) ** -anneal_beta

                mb_obs = obs[idx]
                mb_actions = act[idx]
                mb_logprobs = lp[idx]
                mb_values = val[idx]
                mb_returns = advantages[idx] + mb_values
                mb_advantages = advantages[idx]

                prof.mark(1)
                logits, newvalue = self.policy(mb_obs)
                _actions, newlogprob, entropy = sample_logits(logits, action=mb_actions)
                prof.mark(2)
                prof.elapsed(Profile.TRAIN_FORWARD, 1, 2)

                newlogprob = newlogprob.reshape(mb_logprobs.shape)
                entropy = entropy.reshape(mb_logprobs.shape)
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()
                self.ratio[idx] = ratio.detach()

                weights, active_mask, bid_mask, play_mask = _decision_phase_weights(
                    mb_obs,
                    active_only_loss=bool(config["active_only_loss"]),
                    bid_ppo_weight=float(config["bid_ppo_weight"]),
                    play_ppo_weight=float(config["play_ppo_weight"]),
                )

                with torch.no_grad():
                    old_approx_kl = _weighted_mean(-logratio, weights)
                    approx_kl = _weighted_mean((ratio - 1) - logratio, weights)
                    clipfrac = _weighted_mean(
                        ((ratio - 1.0).abs() > config["clip_coef"]).float(),
                        weights,
                    )

                adv = mb_advantages
                adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)
                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = _weighted_mean(torch.max(pg_loss1, pg_loss2), weights)

                newvalue = newvalue.view(mb_returns.shape)
                v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
                v_loss_unclipped = (newvalue - mb_returns) ** 2
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss = 0.5 * _weighted_mean(torch.max(v_loss_unclipped, v_loss_clipped), weights)

                entropy_loss = _weighted_mean(entropy, weights)
                anchor_kl, anchor_q, anchor_rows = bid_anchor_losses(
                    self.policy,
                    self.bid_teacher,
                    mb_obs,
                    temperature=float(config["bid_anchor_temperature"]),
                )
                loss = (
                    pg_loss
                    + config["vf_coef"] * v_loss
                    - config["ent_coef"] * entropy_loss
                    + float(config["bid_anchor_weight"]) * anchor_kl
                    + float(config["bid_q_anchor_weight"]) * anchor_q
                )
                val[idx] = newvalue.detach().float()

                losses["policy_loss"] += float(pg_loss.detach().item())
                losses["value_loss"] += float(v_loss.detach().item())
                losses["entropy"] += float(entropy_loss.detach().item())
                losses["old_approx_kl"] += float(old_approx_kl.detach().item())
                losses["approx_kl"] += float(approx_kl.detach().item())
                losses["clipfrac"] += float(clipfrac.detach().item())
                losses["importance"] += float(_weighted_mean(ratio, weights).detach().item())
                losses["bid_anchor_kl"] += float(anchor_kl.detach().item())
                losses["bid_q_anchor_loss"] += float(anchor_q.detach().item())
                counts["active_rows"] += float(active_mask.sum().item())
                counts["bid_rows"] += float((active_mask & bid_mask).sum().item())
                counts["play_rows"] += float((active_mask & play_mask).sum().item())
                counts["ppo_weighted_rows"] += float(weights.sum().item())
                counts["bid_anchor_rows"] += float(anchor_rows)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config["max_grad_norm"])
                self.optimizer.step()
                self.optimizer.zero_grad()

            prof.mark(1)
            prof.elapsed(Profile.TRAIN, 0, 1)

            denom = max(num_minibatches, 1)
            self.losses = {key: value / denom for key, value in losses.items()}
            self.losses.update({key: value / denom for key, value in counts.items()})
            y_pred = val.flatten()
            y_true = advantages.flatten() + val.flatten()
            var_y = y_true.var()
            self.losses["explained_variance"] = (
                float("nan")
                if float(var_y.detach().item()) == 0.0
                else float((1 - (y_true - y_pred).var() / var_y).item())
            )
            self.losses["learning_rate"] = float(learning_rate)
            self.epoch += 1

    torch.manual_seed(cli_args.seed)
    np.random.seed(cli_args.seed)

    vec = SpadesPufferVecEnv(
        SpadesPufferConfig(
            num_envs=cli_args.num_envs,
            seed=cli_args.seed,
            max_steps_per_match=cli_args.max_steps_per_match,
            reward_scale=cli_args.reward_scale,
            use_cuda_buffers=cli_args.cuda_buffers,
            hand_episodes=cli_args.hand_episodes,
            max_normal_bid=cli_args.max_normal_bid,
            nil_enabled=cli_args.nil,
            blind_nil_enabled=cli_args.blind_nil,
        )
    )
    args = build_train_args(cli_args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = build_policy_from_args(cli_args, vec.obs_size, device)
    if cli_args.load_path:
        load_transformer_state(policy, cli_args.load_path, next(policy.parameters()).device)
    if cli_args.freeze_bid_heads:
        freeze_bid_heads(policy)

    bid_teacher = None
    anchor_path = cli_args.bid_anchor_path
    if not anchor_path and (cli_args.bid_anchor_weight > 0 or cli_args.bid_q_anchor_weight > 0):
        anchor_path = cli_args.load_path
    if anchor_path:
        bid_teacher = build_policy_from_args(cli_args, vec.obs_size, device)
        load_transformer_state(bid_teacher, anchor_path, next(bid_teacher.parameters()).device)
        bid_teacher.eval()
        for param in bid_teacher.parameters():
            param.requires_grad = False
    elif cli_args.bid_anchor_weight > 0 or cli_args.bid_q_anchor_weight > 0:
        raise ValueError("bid anchor weights require --bid-anchor-path or --load-path")

    wandb_module, wandb_run = _init_wandb(cli_args)
    trainer = AlignedPuffeRL(args, vec, policy, verbose=False, bid_teacher=bid_teacher)
    final_logs: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    try:
        while trainer.epoch < trainer.total_epochs:
            trainer.rollouts()
            trainer.train()
            if trainer.epoch % cli_args.log_interval == 0 or trainer.epoch == trainer.total_epochs:
                final_logs = dict(pufferl.unroll_nested_dict(trainer.log()))
                history.append(final_logs)
                print(
                    "epoch={epoch} steps={steps:.0f} sps={sps:.0f} "
                    "score={score:.4f} hand_score={hand_score:.4f} bid_kl={bid_kl:.4f} "
                    "episodes={episodes:.0f} hands={hands:.0f} "
                    "matches={matches:.0f} illegal={illegal:.0f}".format(
                        epoch=int(final_logs.get("epoch", trainer.epoch)),
                        steps=float(final_logs.get("agent_steps", trainer.global_step)),
                        sps=float(final_logs.get("SPS", 0.0)),
                        score=float(final_logs.get("env/score", 0.0)),
                        hand_score=float(final_logs.get("env/hand_score", 0.0)),
                        bid_kl=float(final_logs.get("loss/bid_anchor_kl", 0.0)),
                        episodes=float(final_logs.get("env/completed_episodes", 0.0)),
                        hands=float(final_logs.get("env/completed_hands", 0.0)),
                        matches=float(final_logs.get("env/completed_matches", 0.0)),
                        illegal=float(final_logs.get("env/illegal_actions", 0.0)),
                    ),
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(final_logs, step=int(final_logs.get("agent_steps", trainer.global_step)))

        if cli_args.save_path:
            os.makedirs(os.path.dirname(cli_args.save_path) or ".", exist_ok=True)
            trainer.save_weights(cli_args.save_path)
    finally:
        trainer.close()
        if wandb_run is not None:
            if cli_args.save_path and os.path.exists(cli_args.save_path):
                artifact = wandb_module.Artifact(_wandb_artifact_name(cli_args), type="model")
                artifact.add_file(cli_args.save_path)
                wandb_run.log_artifact(artifact)
            wandb_run.finish()

    final_logs = attach_log_history(final_logs, history)
    final_logs["model_path"] = cli_args.save_path
    final_logs["loaded_model_path"] = cli_args.load_path
    final_logs["bid_anchor_path"] = anchor_path
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


def evaluate_policy(args: argparse.Namespace) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    policy = build_policy_from_args(args, FLAT_OBSERVATION_SIZE, device)
    load_transformer_state(policy, args.checkpoint, device)
    policy.eval()

    env_config = SpadesPlusConfig(
        nil_enabled=args.nil,
        blind_nil_enabled=args.blind_nil,
        blind_nil_policy="always" if args.blind_nil else "disabled",
    )
    env = SpadesPlusEnv(env_config)
    env.reset(seed=args.seed)

    hand_scores: list[float] = []
    team_deltas: list[list[int]] = []
    bid_counts: dict[int, int] = {}
    illegal_actions = 0

    progress = tqdm(total=args.hands, desc="Evaluate hands", disable=not args.progress)
    try:
        while len(hand_scores) < args.hands:
            obs = env.observe()
            flat = flatten_observation(obs).astype(np.float32)
            if obs["phase"] == 0:
                flat = _filter_bidding_mask(flat, args.max_normal_bid)
            x = torch.from_numpy(flat).float().to(device).unsqueeze(0)
            with torch.no_grad():
                logits, _values, _state = policy.forward_eval(x, ())
                action = int(torch.argmax(logits, dim=1).item())

            legal = env.legal_actions()
            if obs["phase"] == 0:
                legal = _filter_bidding_legal_actions(legal, args.max_normal_bid)
            if action not in legal:
                illegal_actions += 1
                action = int(legal[0])
            if obs["phase"] == 0:
                bid_counts[action] = bid_counts.get(action, 0) + 1

            _obs, rewards, terminated, truncated, info = env.step(action)
            if info.get("hand_completed"):
                hand_scores.append(float(np.mean(rewards) * args.reward_scale))
                team_deltas.append([int(value) for value in info["team_score_delta"]])
                progress.update(1)
                if hand_scores:
                    progress.set_postfix(mean=f"{np.mean(hand_scores):.4f}", illegal=illegal_actions)
            if terminated or truncated:
                env.reset(seed=args.seed + len(hand_scores))
    finally:
        progress.close()

    scores = np.asarray(hand_scores, dtype=np.float32)
    deltas = np.asarray(team_deltas, dtype=np.float32)
    top_bids = [
        {
            "action": action,
            "bid": str(bid_action_to_bid(action)),
            "count": count,
        }
        for action, count in sorted(bid_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
    ]
    return {
        "checkpoint": args.checkpoint,
        "hands": args.hands,
        "mean": float(scores.mean()),
        "last100": float(scores[-100:].mean()),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "illegal_actions": illegal_actions,
        "mean_team_delta": deltas.mean(axis=0).tolist() if len(deltas) else [0.0, 0.0],
        "top_bids": top_bids,
    }


def make_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Spades Puffer checkpoint greedily")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--hands", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=6)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--ffn-size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--max-normal-bid", type=int, default=13)
    parser.add_argument("--nil", action="store_true", default=True)
    parser.add_argument("--no-nil", action="store_false", dest="nil")
    parser.add_argument("--blind-nil", action="store_true", default=True)
    parser.add_argument("--no-blind-nil", action="store_false", dest="blind_nil")
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-progress", action="store_false", dest="progress")
    parser.add_argument("--output", type=str, default="")
    return parser


def eval_main() -> None:
    parser = make_eval_parser()
    args = parser.parse_args()
    metrics = evaluate_policy(args)
    rendered = json.dumps(metrics, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(rendered)
            f.write("\n")


def _collect_bidding_batch(
    env: SpadesPlusEnv,
    bot: ConservativeBidBot,
    rng: np.random.Generator,
    batch_size: int,
    max_normal_bid: int,
) -> tuple[np.ndarray, np.ndarray]:
    observations: list[np.ndarray] = []
    actions: list[int] = []
    while len(actions) < batch_size:
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        env.reset(seed=seed)
        while env.phase == Phase.BIDDING and len(actions) < batch_size:
            obs = env.observe()
            legal = _filter_bidding_legal_actions(env.legal_actions(), max_normal_bid)
            action = int(bot.act(obs, legal, rng))
            flat = _filter_bidding_mask(flatten_observation(obs).astype(np.float32), max_normal_bid)
            observations.append(flat)
            actions.append(action)
            env.step(action)

    return np.stack(observations).astype(np.float32), np.asarray(actions, dtype=np.int64)


def pretrain_bidding(args: argparse.Namespace) -> dict[str, Any]:
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 1 <= args.max_normal_bid <= 13:
        raise ValueError("max_normal_bid must be in [1, 13]")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    policy = build_policy_from_args(args, FLAT_OBSERVATION_SIZE, device)
    if args.load_path:
        load_transformer_state(policy, args.load_path, device)

    if args.freeze_encoder:
        encoder_modules = (policy.tokenizer, policy.transformer, policy.final_norm)
        for module in encoder_modules:
            for param in module.parameters():
                param.requires_grad = False

    trainable_params = [param for param in policy.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable policy parameters remain after freezing")

    optimizer = torch.optim.Adam(trainable_params, lr=args.learning_rate)
    env = SpadesPlusEnv(
        SpadesPlusConfig(
            nil_enabled=args.nil,
            blind_nil_enabled=args.blind_nil,
            blind_nil_policy="always" if args.blind_nil else "disabled",
        )
    )
    bot = ConservativeBidBot()
    rng = np.random.default_rng(args.seed)

    samples_seen = 0
    batches = 0
    total_correct = 0
    total_loss = 0.0
    policy.train()
    while samples_seen < args.samples:
        batch_size = min(args.batch_size, args.samples - samples_seen)
        observations, actions = _collect_bidding_batch(
            env,
            bot,
            rng,
            batch_size,
            args.max_normal_bid,
        )
        x = torch.from_numpy(observations).to(device)
        y = torch.from_numpy(actions).to(device)

        logits, _values, _state = policy.forward_eval(x, ())
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            correct = int((logits.argmax(dim=1) == y).sum().item())
        samples_seen += batch_size
        batches += 1
        total_correct += correct
        total_loss += float(loss.item()) * batch_size

        if batches % args.log_interval == 0 or samples_seen >= args.samples:
            print(
                "batch={batch} samples={samples} loss={loss:.4f} acc={acc:.4f}".format(
                    batch=batches,
                    samples=samples_seen,
                    loss=total_loss / samples_seen,
                    acc=total_correct / samples_seen,
                ),
                flush=True,
            )

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        torch.save(policy.state_dict(), args.save_path)

    metrics = {
        "samples": samples_seen,
        "batches": batches,
        "loss": total_loss / samples_seen,
        "accuracy": total_correct / samples_seen,
        "model_path": args.save_path,
        "loaded_model_path": args.load_path,
        "completed_at": time.time(),
    }
    if args.metrics_path:
        os.makedirs(os.path.dirname(args.metrics_path) or ".", exist_ok=True)
        with open(args.metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
    return metrics


def make_pretrain_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bootstrap Spades Puffer bidding logits from the conservative bid bot"
    )
    parser.add_argument("--samples", type=int, default=131_072)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=6)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--ffn-size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-normal-bid", type=int, default=13)
    parser.add_argument("--nil", action="store_true", default=True)
    parser.add_argument("--no-nil", action="store_false", dest="nil")
    parser.add_argument("--blind-nil", action="store_true", default=True)
    parser.add_argument("--no-blind-nil", action="store_false", dest="blind_nil")
    parser.add_argument("--load-path", type=str, default="")
    parser.add_argument("--save-path", type=str, default="checkpoints/spades_bidding_pretrain.pt")
    parser.add_argument("--metrics-path", type=str, default="")
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--freeze-encoder", action="store_true", default=False)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser


def pretrain_main() -> None:
    parser = make_pretrain_parser()
    args = parser.parse_args()
    pretrain_bidding(args)


if __name__ == "__main__":
    main()
