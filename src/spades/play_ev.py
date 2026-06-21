"""Rollout-EV dataset generation for Spades play decisions."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from spades.actions import ACTION_SPACE_SIZE
from spades.bid_ev import make_ev_env, replay_previous_bids
from spades.config import SpadesPlusConfig
from spades.env import SpadesPlusEnv
from spades.eval import Actor, make_actor
from spades.observations import FLAT_OBSERVATION_SIZE, flatten_observation
from spades.puffer import _filter_bidding_legal_actions
from spades.rules import player_team
from spades.state import Phase


@dataclass(frozen=True)
class PlayState:
    observation: np.ndarray
    legal_actions: list[int]
    acting_player: int
    target_team: int
    dealer: int
    hands: list[list[int]]
    team_scores: list[int]
    team_bags: list[int]
    previous_bid_actions: list[int]
    previous_play_actions: list[int]


def _legal_actions_for_generation(env: SpadesPlusEnv, max_normal_bid: int) -> list[int]:
    legal = env.legal_actions()
    if env.phase == Phase.BIDDING:
        legal = _filter_bidding_legal_actions(legal, max_normal_bid)
    return legal


def _descriptor_actor(
    descriptor: str,
    *,
    max_normal_bid: int,
    d_model: int,
    transformer_layers: int,
    attention_heads: int,
    ffn_size: int,
    dropout: float,
    device: str,
) -> Actor:
    if not descriptor.startswith(("bot:", "checkpoint:")):
        descriptor = f"bot:{descriptor}"
    return make_actor(
        descriptor,
        label="actor",
        max_normal_bid=max_normal_bid,
        d_model=d_model,
        transformer_layers=transformer_layers,
        attention_heads=attention_heads,
        ffn_size=ffn_size,
        dropout=dropout,
        device=device,
    )


def replay_previous_plays(env: SpadesPlusEnv, previous_play_actions: list[int]) -> None:
    for action in previous_play_actions:
        if env.phase != Phase.PLAYING:
            raise RuntimeError("Cannot replay play action outside playing phase")
        _obs, _rewards, terminated, truncated, info = env.step(int(action))
        if info.get("hand_completed") or terminated or truncated:
            raise RuntimeError("Replay completed hand before target play state")


def collect_play_states(
    *,
    count: int,
    seed: int,
    state_actor: Actor,
    state_ally_actor: Actor | None = None,
    state_opponent_actor: Actor | None = None,
    max_normal_bid: int,
    nil: bool,
    blind_nil: bool,
    progress: bool = True,
) -> list[PlayState]:
    rng = np.random.default_rng(seed)
    states: list[PlayState] = []
    deal_idx = 0
    team_aware = state_opponent_actor is not None
    ally_actor = state_ally_actor or state_actor
    with tqdm(total=count, desc="Collect play states", disable=not progress) as bar:
        while len(states) < count:
            target_team = deal_idx % 2
            env = SpadesPlusEnv(
                SpadesPlusConfig(
                    nil_enabled=nil,
                    blind_nil_enabled=blind_nil,
                    blind_nil_policy="always" if blind_nil else "disabled",
                    debug=True,
                )
            )
            env.reset(seed=seed + deal_idx)
            debug = env.get_debug_state()
            initial_hands = [list(hand) for hand in debug["hands"]]
            dealer = int(env.dealer)
            team_scores = env.team_scores.astype(int).tolist()
            team_bags = env.team_bags.astype(int).tolist()
            previous_bid_actions: list[int] = []
            previous_play_actions: list[int] = []

            while len(states) < count:
                obs = env.observe()
                legal = _legal_actions_for_generation(env, max_normal_bid)
                acting_player = env.current_player()
                acting_team = player_team(acting_player)
                should_collect = not team_aware or acting_team == target_team
                if env.phase == Phase.PLAYING and should_collect:
                    states.append(
                        PlayState(
                            observation=flatten_observation(obs).astype(np.float32),
                            legal_actions=list(legal),
                            acting_player=acting_player,
                            target_team=acting_team,
                            dealer=dealer,
                            hands=[list(hand) for hand in initial_hands],
                            team_scores=list(team_scores),
                            team_bags=list(team_bags),
                            previous_bid_actions=list(previous_bid_actions),
                            previous_play_actions=list(previous_play_actions),
                        )
                    )
                    bar.update(1)

                actor = state_actor
                if team_aware:
                    actor = ally_actor if acting_team == target_team else state_opponent_actor
                    if actor is None:
                        raise RuntimeError("state_opponent_actor is required for team-aware collection")
                action = int(actor.act(obs, legal, rng))
                if action not in legal:
                    action = int(legal[0])
                if env.phase == Phase.BIDDING:
                    previous_bid_actions.append(action)
                else:
                    previous_play_actions.append(action)
                _obs, _rewards, terminated, truncated, info = env.step(action)
                if info.get("hand_completed") or terminated or truncated:
                    break
            deal_idx += 1
    return states


def make_play_ev_env(
    *,
    state: PlayState,
    nil: bool,
    blind_nil: bool,
) -> SpadesPlusEnv:
    env = make_ev_env(
        dealer=state.dealer,
        hands=state.hands,
        team_scores=state.team_scores,
        team_bags=state.team_bags,
        nil=nil,
        blind_nil=blind_nil,
    )
    replay_previous_bids(env, state.previous_bid_actions)
    replay_previous_plays(env, state.previous_play_actions)
    if env.phase != Phase.PLAYING or env.current_player() != state.acting_player:
        raise RuntimeError("Replayed env does not match target play state")
    return env


def _complete_hand_with_actor(
    env: SpadesPlusEnv,
    actor: Actor,
    rng: np.random.Generator,
    max_normal_bid: int,
) -> dict[str, Any]:
    info: dict[str, Any] = {}
    while True:
        obs = env.observe()
        legal = _legal_actions_for_generation(env, max_normal_bid)
        action = int(actor.act(obs, legal, rng))
        if action not in legal:
            action = int(legal[0])
        _obs, _rewards, terminated, truncated, info = env.step(action)
        if info.get("hand_completed") or terminated or truncated:
            return info


def _complete_hand_with_team_actors(
    env: SpadesPlusEnv,
    ally_actor: Actor,
    opponent_actor: Actor,
    target_team: int,
    rng: np.random.Generator,
    max_normal_bid: int,
) -> dict[str, Any]:
    info: dict[str, Any] = {}
    while True:
        obs = env.observe()
        legal = _legal_actions_for_generation(env, max_normal_bid)
        actor = ally_actor if player_team(env.current_player()) == target_team else opponent_actor
        action = int(actor.act(obs, legal, rng))
        if action not in legal:
            action = int(legal[0])
        _obs, _rewards, terminated, truncated, info = env.step(action)
        if info.get("hand_completed") or terminated or truncated:
            return info


def evaluate_play_candidate_action(
    state: PlayState,
    candidate_action: int,
    rollout_actor: Actor | None = None,
    *,
    rollout_ally_actor: Actor | None = None,
    rollout_opponent_actor: Actor | None = None,
    rollout_samples: int,
    max_normal_bid: int,
    nil: bool,
    blind_nil: bool,
    seed: int,
) -> tuple[float, float, int]:
    if rollout_ally_actor is None:
        if rollout_actor is None:
            raise ValueError("rollout_actor or rollout_ally_actor is required")
        rollout_ally_actor = rollout_actor
    if rollout_opponent_actor is None:
        rollout_opponent_actor = rollout_actor or rollout_ally_actor

    values: list[float] = []
    actor_team = player_team(state.acting_player)
    for sample_idx in range(rollout_samples):
        env = make_play_ev_env(state=state, nil=nil, blind_nil=blind_nil)
        legal = env.legal_actions()
        if candidate_action not in legal:
            continue
        _obs, _rewards, terminated, truncated, info = env.step(int(candidate_action))
        if not (info.get("hand_completed") or terminated or truncated):
            rng = np.random.default_rng(seed + sample_idx)
            info = _complete_hand_with_team_actors(
                env,
                rollout_ally_actor,
                rollout_opponent_actor,
                state.target_team,
                rng,
                max_normal_bid,
            )
        values.append(float(info["team_score_delta"][actor_team]))
    if not values:
        return 0.0, 0.0, 0
    arr = np.asarray(values, dtype=np.float32)
    return float(arr.mean()), float(arr.std(ddof=0)), int(len(values))


def _metadata(args: argparse.Namespace) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        commit = ""
    state_ally_actor = getattr(args, "state_ally_actor", "") or args.state_actor
    state_opponent_actor = getattr(args, "state_opponent_actor", "")
    rollout_ally_actor = getattr(args, "rollout_ally_actor", "") or args.rollout_actor
    rollout_opponent_actor = getattr(args, "rollout_opponent_actor", "") or args.rollout_actor
    return {
        "git_commit": commit,
        "observation_size": FLAT_OBSERVATION_SIZE,
        "action_space_size": ACTION_SPACE_SIZE,
        "states": args.states,
        "rollout_samples": args.rollout_samples,
        "seed": args.seed,
        "state_actor": args.state_actor,
        "state_ally_actor": state_ally_actor,
        "state_opponent_actor": state_opponent_actor,
        "rollout_actor": args.rollout_actor,
        "rollout_ally_actor": rollout_ally_actor,
        "rollout_opponent_actor": rollout_opponent_actor,
        "team_aware_state_collection": bool(state_opponent_actor),
        "team_aware_rollouts": rollout_ally_actor != rollout_opponent_actor,
        "max_normal_bid": args.max_normal_bid,
        "nil": args.nil,
        "blind_nil": args.blind_nil,
    }


def generate_play_ev_dataset(args: argparse.Namespace) -> dict[str, Any]:
    if args.states <= 0:
        raise ValueError("states must be positive")
    if args.rollout_samples <= 0:
        raise ValueError("rollout_samples must be positive")
    if not 1 <= args.max_normal_bid <= 13:
        raise ValueError("max_normal_bid must be in [1, 13]")
    if not args.output:
        raise ValueError("output is required")

    device = "cuda" if args.device == "cuda" else "cpu"
    state_ally_descriptor = getattr(args, "state_ally_actor", "") or args.state_actor
    state_opponent_descriptor = getattr(args, "state_opponent_actor", "")
    rollout_ally_descriptor = getattr(args, "rollout_ally_actor", "") or args.rollout_actor
    rollout_opponent_descriptor = getattr(args, "rollout_opponent_actor", "") or args.rollout_actor

    state_actor = _descriptor_actor(
        state_ally_descriptor,
        max_normal_bid=args.max_normal_bid,
        d_model=args.d_model,
        transformer_layers=args.transformer_layers,
        attention_heads=args.attention_heads,
        ffn_size=args.ffn_size,
        dropout=args.dropout,
        device=device,
    )
    state_opponent_actor = (
        _descriptor_actor(
            state_opponent_descriptor,
            max_normal_bid=args.max_normal_bid,
            d_model=args.d_model,
            transformer_layers=args.transformer_layers,
            attention_heads=args.attention_heads,
            ffn_size=args.ffn_size,
            dropout=args.dropout,
            device=device,
        )
        if state_opponent_descriptor
        else None
    )
    rollout_ally_actor = _descriptor_actor(
        rollout_ally_descriptor,
        max_normal_bid=args.max_normal_bid,
        d_model=args.d_model,
        transformer_layers=args.transformer_layers,
        attention_heads=args.attention_heads,
        ffn_size=args.ffn_size,
        dropout=args.dropout,
        device=device,
    )
    rollout_opponent_actor = _descriptor_actor(
        rollout_opponent_descriptor,
        max_normal_bid=args.max_normal_bid,
        d_model=args.d_model,
        transformer_layers=args.transformer_layers,
        attention_heads=args.attention_heads,
        ffn_size=args.ffn_size,
        dropout=args.dropout,
        device=device,
    )

    states = collect_play_states(
        count=args.states,
        seed=args.seed,
        state_actor=state_actor,
        state_opponent_actor=state_opponent_actor,
        max_normal_bid=args.max_normal_bid,
        nil=args.nil,
        blind_nil=args.blind_nil,
        progress=args.progress,
    )

    observations = np.zeros((args.states, FLAT_OBSERVATION_SIZE), dtype=np.float32)
    legal_mask = np.zeros((args.states, ACTION_SPACE_SIZE), dtype=np.bool_)
    evaluated_mask = np.zeros((args.states, ACTION_SPACE_SIZE), dtype=np.bool_)
    q_values = np.zeros((args.states, ACTION_SPACE_SIZE), dtype=np.float32)
    q_std = np.zeros((args.states, ACTION_SPACE_SIZE), dtype=np.float32)
    visit_counts = np.zeros((args.states, ACTION_SPACE_SIZE), dtype=np.int32)
    best_action = np.full(args.states, -1, dtype=np.int16)
    acting_player = np.zeros(args.states, dtype=np.int8)
    target_team = np.zeros(args.states, dtype=np.int8)
    dealer = np.zeros(args.states, dtype=np.int8)
    previous_bid_actions = np.full((args.states, 4), -1, dtype=np.int16)
    previous_play_actions = np.full((args.states, 52), -1, dtype=np.int16)

    for idx, state in enumerate(tqdm(states, desc="Evaluate play EV", disable=not args.progress)):
        observations[idx] = state.observation
        acting_player[idx] = state.acting_player
        target_team[idx] = state.target_team
        dealer[idx] = state.dealer
        for bid_idx, action in enumerate(state.previous_bid_actions[:4]):
            previous_bid_actions[idx, bid_idx] = action
        for play_idx, action in enumerate(state.previous_play_actions[:52]):
            previous_play_actions[idx, play_idx] = action
        for action in state.legal_actions:
            legal_mask[idx, action] = True

        for action in state.legal_actions:
            mean, std, visits = evaluate_play_candidate_action(
                state,
                action,
                rollout_ally_actor=rollout_ally_actor,
                rollout_opponent_actor=rollout_opponent_actor,
                rollout_samples=args.rollout_samples,
                max_normal_bid=args.max_normal_bid,
                nil=args.nil,
                blind_nil=args.blind_nil,
                seed=args.seed * 1_000_000 + idx * 1000 + action,
            )
            if visits:
                evaluated_mask[idx, action] = True
                q_values[idx, action] = mean
                q_std[idx, action] = std
                visit_counts[idx, action] = visits

        if evaluated_mask[idx].any():
            masked_q = np.where(evaluated_mask[idx], q_values[idx], -np.inf)
            best_action[idx] = int(np.argmax(masked_q))

    metadata_json = json.dumps(_metadata(args), sort_keys=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(
        args.output,
        observations=observations,
        legal_mask=legal_mask,
        evaluated_mask=evaluated_mask,
        q_values=q_values,
        q_std=q_std,
        visit_counts=visit_counts,
        best_action=best_action,
        acting_player=acting_player,
        target_team=target_team,
        dealer=dealer,
        previous_bid_actions=previous_bid_actions,
        previous_play_actions=previous_play_actions,
        metadata_json=np.asarray(metadata_json),
    )
    return summarize_play_ev(args.output)


def summarize_play_ev(path: str) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    evaluated_mask = data["evaluated_mask"].astype(bool)
    q_values = data["q_values"]
    best_action = data["best_action"].astype(np.int64)
    visit_counts = data["visit_counts"]
    valid_rows = best_action >= 0
    best_values = q_values[np.arange(len(best_action))[valid_rows], best_action[valid_rows]]

    gaps: list[float] = []
    for idx in np.flatnonzero(valid_rows):
        row_values = q_values[idx][evaluated_mask[idx]]
        if len(row_values) >= 2:
            top = np.sort(row_values)[-2:]
            gaps.append(float(top[-1] - top[-2]))

    best_counts = {
        str(action): int(count)
        for action, count in zip(*np.unique(best_action[valid_rows], return_counts=True))
    }
    action_coverage = {
        str(action): int(evaluated_mask[:, action].sum())
        for action in range(52)
        if evaluated_mask[:, action].any()
    }
    metadata = json.loads(str(data["metadata_json"]))
    return {
        "path": path,
        "rows": int(len(best_action)),
        "valid_rows": int(valid_rows.sum()),
        "observation_size": int(data["observations"].shape[1]),
        "mean_best_ev": float(best_values.mean()) if len(best_values) else 0.0,
        "mean_best_second_gap": float(np.mean(gaps)) if gaps else 0.0,
        "mean_visits_per_evaluated_action": (
            float(visit_counts[evaluated_mask].mean()) if evaluated_mask.any() else 0.0
        ),
        "best_action_distribution": best_counts,
        "action_coverage": action_coverage,
        "metadata": metadata,
    }


def make_generate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate rollout-EV labels for Spades play decisions")
    parser.add_argument("--states", type=int, default=1024)
    parser.add_argument("--rollout-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--state-actor", type=str, default="bot:conservative")
    parser.add_argument("--state-ally-actor", type=str, default="")
    parser.add_argument("--state-opponent-actor", type=str, default="")
    parser.add_argument("--rollout-actor", type=str, default="bot:conservative")
    parser.add_argument("--rollout-ally-actor", type=str, default="")
    parser.add_argument("--rollout-opponent-actor", type=str, default="")
    parser.add_argument("--max-normal-bid", type=int, default=13)
    parser.add_argument("--nil", action="store_true", default=True)
    parser.add_argument("--no-nil", action="store_false", dest="nil")
    parser.add_argument("--blind-nil", action="store_true", default=True)
    parser.add_argument("--no-blind-nil", action="store_false", dest="blind_nil")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=6)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--ffn-size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-progress", action="store_false", dest="progress")
    parser.add_argument("--output", required=True)
    return parser


def make_summary_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize a Spades play EV NPZ dataset")
    parser.add_argument("path")
    return parser


def generate_main() -> None:
    parser = make_generate_parser()
    args = parser.parse_args()
    summary = generate_play_ev_dataset(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


def summarize_main() -> None:
    parser = make_summary_parser()
    args = parser.parse_args()
    summary = summarize_play_ev(args.path)
    print(json.dumps(summary, indent=2, sort_keys=True))
