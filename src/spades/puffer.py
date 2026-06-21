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
from torch.nn import functional as F

from spades.actions import ACTION_SPACE_SIZE
from spades.actions import bid_action_to_bid
from spades.bots import ConservativeBidBot
from spades.config import SpadesPlusConfig
from spades.env import IllegalActionError, SpadesPlusEnv
from spades.observations import FLAT_OBSERVATION_SIZE, flatten_observation
from spades.policy import SpadesTransformerPolicy, load_transformer_state
from spades.state import Phase


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
    from pufferlib.torch_pufferl import Profile, PuffeRL, _actions_for_vec_step, sample_logits

    class AlignedPuffeRL(PuffeRL):
        """PuffeRL variant that stores step rewards with the sampled action.

        PufferLib's generic PyTorch rollout loop mirrors the native callback
        convention where reward buffers are read before the next action is
        sampled. For sparse hand-end rewards in this Python turn env, storing
        the reward after stepping gives much cleaner credit assignment.
        """

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

    trainer = AlignedPuffeRL(args, vec, policy, verbose=False)
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
                    "episodes={episodes:.0f} hands={hands:.0f} "
                    "matches={matches:.0f} illegal={illegal:.0f}".format(
                        epoch=int(final_logs.get("epoch", trainer.epoch)),
                        steps=float(final_logs.get("agent_steps", trainer.global_step)),
                        sps=float(final_logs.get("SPS", 0.0)),
                        score=float(final_logs.get("env/score", 0.0)),
                        hand_score=float(final_logs.get("env/hand_score", 0.0)),
                        episodes=float(final_logs.get("env/completed_episodes", 0.0)),
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
    final_logs["loaded_model_path"] = cli_args.load_path
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
        if terminated or truncated:
            env.reset(seed=args.seed + len(hand_scores))

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
