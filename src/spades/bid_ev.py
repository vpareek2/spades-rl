"""Rollout-EV dataset generation for Spades bidding decisions."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any

import numpy as np

from spades.actions import (
    ACTION_SPACE_SIZE,
    BLIND_NIL_ACTION,
    NIL_ACTION,
    NORMAL_BID_ACTION_START,
)
from spades.bots import (
    ConservativeBidBot,
    HighestLegalCardBot,
    LowestLegalCardBot,
    RandomLegalBot,
)
from spades.config import SpadesPlusConfig
from spades.env import SpadesPlusEnv
from spades.observations import FLAT_OBSERVATION_SIZE, flatten_observation
from spades.puffer import _filter_bidding_legal_actions
from spades.rules import player_team
from spades.state import Phase


BOT_FACTORIES = {
    "random": RandomLegalBot,
    "conservative": ConservativeBidBot,
    "lowest": LowestLegalCardBot,
    "highest": HighestLegalCardBot,
}


@dataclass(frozen=True)
class BiddingState:
    observation: np.ndarray
    legal_actions: list[int]
    acting_player: int
    dealer: int
    hands: list[list[int]]
    team_scores: list[int]
    team_bags: list[int]
    previous_bid_actions: list[int]


def make_bot(name: str):
    if name not in BOT_FACTORIES:
        options = ", ".join(sorted(BOT_FACTORIES))
        raise ValueError(f"Unknown bot '{name}'. Expected one of: {options}")
    return BOT_FACTORIES[name]()


def make_ev_env(
    *,
    dealer: int,
    hands: list[list[int]],
    team_scores: list[int] | np.ndarray,
    team_bags: list[int] | np.ndarray,
    nil: bool,
    blind_nil: bool,
) -> SpadesPlusEnv:
    env = SpadesPlusEnv(
        SpadesPlusConfig(
            first_dealer=dealer,
            nil_enabled=nil,
            blind_nil_enabled=blind_nil,
            blind_nil_policy="always" if blind_nil else "disabled",
            debug=True,
        )
    )
    env.reset(seed=0)
    env.set_hands(hands)
    env.set_scores(team_scores, team_bags)
    return env


def replay_previous_bids(env: SpadesPlusEnv, previous_bid_actions: list[int]) -> None:
    for action in previous_bid_actions:
        if env.phase != Phase.BIDDING:
            raise RuntimeError("Cannot replay bid after bidding phase ended")
        env.step(int(action))


def collect_bidding_states(
    *,
    count: int,
    seed: int,
    state_bot_name: str,
    max_normal_bid: int,
    nil: bool,
    blind_nil: bool,
) -> list[BiddingState]:
    rng = np.random.default_rng(seed)
    bot = make_bot(state_bot_name)
    states: list[BiddingState] = []
    deal_idx = 0
    while len(states) < count:
        env = SpadesPlusEnv(
            SpadesPlusConfig(
                nil_enabled=nil,
                blind_nil_enabled=blind_nil,
                blind_nil_policy="always" if blind_nil else "disabled",
            )
        )
        env.reset(seed=seed + deal_idx)
        previous_bid_actions: list[int] = []
        while env.phase == Phase.BIDDING and len(states) < count:
            obs = env.observe()
            legal = _filter_bidding_legal_actions(env.legal_actions(), max_normal_bid)
            debug = env.get_debug_state()
            states.append(
                BiddingState(
                    observation=flatten_observation(obs).astype(np.float32),
                    legal_actions=list(legal),
                    acting_player=env.current_player(),
                    dealer=int(env.dealer),
                    hands=[list(hand) for hand in debug["hands"]],
                    team_scores=env.team_scores.astype(int).tolist(),
                    team_bags=env.team_bags.astype(int).tolist(),
                    previous_bid_actions=list(previous_bid_actions),
                )
            )
            action = int(bot.act(obs, legal, rng))
            env.step(action)
            previous_bid_actions.append(action)
        deal_idx += 1
    return states


def resample_hidden_hands(state: BiddingState, rng: np.random.Generator) -> list[list[int]]:
    acting_player = state.acting_player
    own_hand = list(state.hands[acting_player])
    unknown_cards = [card for card in range(52) if card not in set(own_hand)]
    rng.shuffle(unknown_cards)
    hands: list[list[int]] = [[] for _ in range(4)]
    hands[acting_player] = sorted(own_hand)
    cursor = 0
    for player in range(4):
        if player == acting_player:
            continue
        hands[player] = sorted(map(int, unknown_cards[cursor : cursor + 13]))
        cursor += 13
    return hands


def hidden_samples_for_state(
    state: BiddingState,
    *,
    samples: int,
    seed: int,
) -> list[list[list[int]]]:
    rng = np.random.default_rng(seed)
    return [resample_hidden_hands(state, rng) for _ in range(samples)]


def _complete_hand_with_bot(
    env: SpadesPlusEnv,
    bot,
    rng: np.random.Generator,
    max_normal_bid: int,
) -> dict:
    info: dict[str, Any] = {}
    while True:
        obs = env.observe()
        legal = env.legal_actions()
        if env.phase == Phase.BIDDING:
            legal = _filter_bidding_legal_actions(legal, max_normal_bid)
        action = int(bot.act(obs, legal, rng))
        if action not in legal:
            action = int(legal[0])
        _obs, _rewards, terminated, truncated, info = env.step(action)
        if info.get("hand_completed") or terminated or truncated:
            return info


def evaluate_candidate_action(
    state: BiddingState,
    candidate_action: int,
    hidden_samples: list[list[list[int]]],
    *,
    rollout_bot_name: str,
    max_normal_bid: int,
    nil: bool,
    blind_nil: bool,
    seed: int,
) -> tuple[float, float, int]:
    bot = make_bot(rollout_bot_name)
    values: list[float] = []
    actor_team = player_team(state.acting_player)
    for sample_idx, hands in enumerate(hidden_samples):
        env = make_ev_env(
            dealer=state.dealer,
            hands=hands,
            team_scores=state.team_scores,
            team_bags=state.team_bags,
            nil=nil,
            blind_nil=blind_nil,
        )
        replay_previous_bids(env, state.previous_bid_actions)
        legal = _filter_bidding_legal_actions(env.legal_actions(), max_normal_bid)
        if candidate_action not in legal:
            continue
        env.step(int(candidate_action))
        rng = np.random.default_rng(seed + sample_idx)
        info = _complete_hand_with_bot(env, bot, rng, max_normal_bid)
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
    return {
        "git_commit": commit,
        "observation_size": FLAT_OBSERVATION_SIZE,
        "action_space_size": ACTION_SPACE_SIZE,
        "states": args.states,
        "hidden_samples": args.hidden_samples,
        "seed": args.seed,
        "state_bot": args.state_bot,
        "rollout_bot": args.rollout_bot,
        "max_normal_bid": args.max_normal_bid,
        "nil": args.nil,
        "blind_nil": args.blind_nil,
        "action_mapping": {
            "normal_bid_start": NORMAL_BID_ACTION_START,
            "nil": NIL_ACTION,
            "blind_nil": BLIND_NIL_ACTION,
        },
    }


def generate_bid_ev_dataset(args: argparse.Namespace) -> dict[str, Any]:
    if args.states <= 0:
        raise ValueError("states must be positive")
    if args.hidden_samples <= 0:
        raise ValueError("hidden_samples must be positive")
    if not 1 <= args.max_normal_bid <= 13:
        raise ValueError("max_normal_bid must be in [1, 13]")
    if not args.output:
        raise ValueError("output is required")

    states = collect_bidding_states(
        count=args.states,
        seed=args.seed,
        state_bot_name=args.state_bot,
        max_normal_bid=args.max_normal_bid,
        nil=args.nil,
        blind_nil=args.blind_nil,
    )

    observations = np.zeros((args.states, FLAT_OBSERVATION_SIZE), dtype=np.float32)
    legal_mask = np.zeros((args.states, ACTION_SPACE_SIZE), dtype=np.bool_)
    evaluated_mask = np.zeros((args.states, ACTION_SPACE_SIZE), dtype=np.bool_)
    q_values = np.zeros((args.states, ACTION_SPACE_SIZE), dtype=np.float32)
    q_std = np.zeros((args.states, ACTION_SPACE_SIZE), dtype=np.float32)
    visit_counts = np.zeros((args.states, ACTION_SPACE_SIZE), dtype=np.int32)
    best_action = np.full(args.states, -1, dtype=np.int16)
    acting_player = np.zeros(args.states, dtype=np.int8)
    dealer = np.zeros(args.states, dtype=np.int8)
    previous_bid_actions = np.full((args.states, 4), -1, dtype=np.int16)

    for idx, state in enumerate(states):
        observations[idx] = state.observation
        acting_player[idx] = state.acting_player
        dealer[idx] = state.dealer
        for bid_idx, action in enumerate(state.previous_bid_actions[:4]):
            previous_bid_actions[idx, bid_idx] = action
        for action in state.legal_actions:
            legal_mask[idx, action] = True

        hidden_samples = hidden_samples_for_state(
            state,
            samples=args.hidden_samples,
            seed=args.seed * 100_000 + idx,
        )
        for action in state.legal_actions:
            mean, std, visits = evaluate_candidate_action(
                state,
                action,
                hidden_samples,
                rollout_bot_name=args.rollout_bot,
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
        dealer=dealer,
        previous_bid_actions=previous_bid_actions,
        metadata_json=np.asarray(metadata_json),
    )
    return summarize_bid_ev(args.output)


def summarize_bid_ev(path: str) -> dict[str, Any]:
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
        for action in range(ACTION_SPACE_SIZE)
        if evaluated_mask[:, action].any()
    }
    summary = {
        "path": path,
        "rows": int(len(best_action)),
        "valid_rows": int(valid_rows.sum()),
        "observation_size": int(data["observations"].shape[1]),
        "best_action_distribution": best_counts,
        "action_coverage": action_coverage,
        "mean_best_ev": float(best_values.mean()) if len(best_values) else 0.0,
        "mean_best_second_gap": float(np.mean(gaps)) if gaps else 0.0,
        "nil_best_rate": float(np.mean(best_action[valid_rows] == NIL_ACTION)) if valid_rows.any() else 0.0,
        "blind_nil_best_rate": (
            float(np.mean(best_action[valid_rows] == BLIND_NIL_ACTION)) if valid_rows.any() else 0.0
        ),
        "mean_visits_per_evaluated_action": (
            float(visit_counts[evaluated_mask].mean()) if evaluated_mask.any() else 0.0
        ),
        "metadata": json.loads(str(data["metadata_json"])),
    }
    return summary


def make_generate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate rollout-EV labels for Spades bidding")
    parser.add_argument("--states", type=int, default=1024)
    parser.add_argument("--hidden-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--state-bot", choices=sorted(BOT_FACTORIES), default="conservative")
    parser.add_argument("--rollout-bot", choices=sorted(BOT_FACTORIES), default="conservative")
    parser.add_argument("--max-normal-bid", type=int, default=13)
    parser.add_argument("--nil", action="store_true", default=True)
    parser.add_argument("--no-nil", action="store_false", dest="nil")
    parser.add_argument("--blind-nil", action="store_true", default=True)
    parser.add_argument("--no-blind-nil", action="store_false", dest="blind_nil")
    parser.add_argument("--output", required=True)
    return parser


def make_summary_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize a Spades bidding EV NPZ dataset")
    parser.add_argument("path")
    return parser


def generate_main() -> None:
    parser = make_generate_parser()
    args = parser.parse_args()
    summary = generate_bid_ev_dataset(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


def summarize_main() -> None:
    parser = make_summary_parser()
    args = parser.parse_args()
    summary = summarize_bid_ev(args.path)
    print(json.dumps(summary, indent=2, sort_keys=True))
