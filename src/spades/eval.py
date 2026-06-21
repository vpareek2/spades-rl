"""Evaluation harnesses for comparing Spades policies."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from tqdm.auto import tqdm

from spades.actions import ACTION_SPACE_SIZE, BLIND_NIL_ACTION, NIL_ACTION
from spades.bots import (
    ConservativeBidBot,
    HighestLegalCardBot,
    LowestLegalCardBot,
    RandomLegalBot,
)
from spades.config import SpadesPlusConfig
from spades.env import SpadesPlusEnv
from spades.observations import FLAT_OBSERVATION_SIZE, flatten_observation
from spades.puffer import (
    _filter_bidding_legal_actions,
    _filter_bidding_mask,
)
from spades.policy import SpadesTransformerPolicy, load_transformer_state
from spades.state import BidKind, Phase


class Actor(Protocol):
    name: str

    def act(
        self,
        obs: dict,
        legal_actions: list[int],
        rng: np.random.Generator,
    ) -> int: ...


@dataclass
class CheckpointActor:
    name: str
    checkpoint: str
    max_normal_bid: int
    d_model: int = 256
    transformer_layers: int = 6
    attention_heads: int = 8
    ffn_size: int = 1024
    dropout: float = 0.05
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.policy = SpadesTransformerPolicy(
            FLAT_OBSERVATION_SIZE,
            ACTION_SPACE_SIZE,
            d_model=self.d_model,
            num_layers=self.transformer_layers,
            num_heads=self.attention_heads,
            ffn_size=self.ffn_size,
            dropout=self.dropout,
        ).to(self.device)
        load_transformer_state(self.policy, self.checkpoint, self.device)
        self.policy.eval()

    def act(
        self,
        obs: dict,
        legal_actions: list[int],
        rng: np.random.Generator,
    ) -> int:
        del legal_actions, rng
        flat = flatten_observation(obs).astype(np.float32)
        if obs["phase"] == 0:
            flat = _filter_bidding_mask(flat, self.max_normal_bid)
        x = torch.from_numpy(flat).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _values, _state = self.policy.forward_eval(x, ())
        return int(torch.argmax(logits, dim=1).item())


@dataclass
class BotActor:
    name: str
    bot: object

    def act(
        self,
        obs: dict,
        legal_actions: list[int],
        rng: np.random.Generator,
    ) -> int:
        return int(self.bot.act(obs, legal_actions, rng))


@dataclass(frozen=True)
class Deal:
    hands: list[list[int]]
    dealer: int


def generate_deal(seed: int) -> Deal:
    env = SpadesPlusEnv(SpadesPlusConfig(debug=True))
    env.reset(seed=seed)
    state = env.get_debug_state()
    return Deal(hands=[list(hand) for hand in state["hands"]], dealer=int(state["dealer"]))


def make_env_for_deal(deal: Deal, nil: bool, blind_nil: bool) -> SpadesPlusEnv:
    env = SpadesPlusEnv(
        SpadesPlusConfig(
            first_dealer=deal.dealer,
            nil_enabled=nil,
            blind_nil_enabled=blind_nil,
            blind_nil_policy="always" if blind_nil else "disabled",
            debug=True,
        )
    )
    env.reset(seed=0)
    env.set_hands(deal.hands)
    env.set_scores([0, 0], [0, 0])
    return env


def _legal_actions_for_eval(env: SpadesPlusEnv, max_normal_bid: int) -> list[int]:
    legal = env.legal_actions()
    if env.phase == Phase.BIDDING:
        legal = _filter_bidding_legal_actions(legal, max_normal_bid)
    return legal


def _policy_for_player(player: int, a_is_team0: bool) -> str:
    if a_is_team0:
        return "A" if player in (0, 2) else "B"
    return "B" if player in (0, 2) else "A"


def _increment_bid_stats(stats: dict[str, dict], policy_name: str, action: int) -> None:
    bid_counts = stats["bid_counts"][policy_name]
    bid_counts[str(action)] = bid_counts.get(str(action), 0) + 1
    if action == NIL_ACTION:
        stats["nil_attempts"][policy_name] += 1
    elif action == BLIND_NIL_ACTION:
        stats["blind_nil_attempts"][policy_name] += 1


def _record_hand_stats(stats: dict[str, dict], info: dict, a_is_team0: bool) -> None:
    bids = info.get("player_bids", [])
    made_contract = info.get("made_contract")
    nil_success = info.get("nil_success")
    for player, bid in enumerate(bids):
        policy_name = _policy_for_player(player, a_is_team0)
        if made_contract is not None and not bool(made_contract[player]):
            stats["contract_failures"][policy_name] += 1
        if nil_success is None:
            continue
        if bid.kind == BidKind.NIL and bool(nil_success[player]):
            stats["nil_successes"][policy_name] += 1
        elif bid.kind == BidKind.BLIND_NIL and bool(nil_success[player]):
            stats["blind_nil_successes"][policy_name] += 1


def play_duplicate_orientation(
    deal: Deal,
    actor_a: Actor,
    actor_b: Actor,
    *,
    a_is_team0: bool,
    max_normal_bid: int,
    nil: bool,
    blind_nil: bool,
    reward_scale: float,
    seed: int,
    stats: dict[str, dict],
) -> dict[str, float | list[int]]:
    env = make_env_for_deal(deal, nil=nil, blind_nil=blind_nil)
    rng = np.random.default_rng(seed)
    info: dict = {}
    while True:
        player = env.current_player()
        policy_name = _policy_for_player(player, a_is_team0)
        actor = actor_a if policy_name == "A" else actor_b
        obs = env.observe()
        legal = _legal_actions_for_eval(env, max_normal_bid)
        action = int(actor.act(obs, legal, rng))
        if action not in legal:
            stats["illegal_actions"][policy_name] += 1
            action = int(legal[0])
        if env.phase == Phase.BIDDING:
            _increment_bid_stats(stats, policy_name, action)

        _obs, _rewards, terminated, truncated, info = env.step(action)
        if info.get("hand_completed"):
            break
        if terminated or truncated:
            break

    team_delta = [int(value) for value in info["team_score_delta"]]
    if a_is_team0:
        a_margin = reward_scale * float(team_delta[0] - team_delta[1])
    else:
        a_margin = reward_scale * float(team_delta[1] - team_delta[0])
    welfare = reward_scale * float(sum(team_delta)) / 2.0
    _record_hand_stats(stats, info, a_is_team0)
    return {
        "a_margin": a_margin,
        "table_welfare": welfare,
        "team_delta": team_delta,
    }


def _empty_stats() -> dict[str, dict]:
    return {
        "illegal_actions": {"A": 0, "B": 0},
        "bid_counts": {"A": {}, "B": {}},
        "nil_attempts": {"A": 0, "B": 0},
        "nil_successes": {"A": 0, "B": 0},
        "blind_nil_attempts": {"A": 0, "B": 0},
        "blind_nil_successes": {"A": 0, "B": 0},
        "contract_failures": {"A": 0, "B": 0},
    }


def _stderr(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def evaluate_duplicate(
    actor_a: Actor,
    actor_b: Actor,
    *,
    hands: int,
    seed: int,
    max_normal_bid: int,
    nil: bool,
    blind_nil: bool,
    reward_scale: float,
    progress: bool = True,
) -> dict:
    stats = _empty_stats()
    paired_margins: list[float] = []
    orientation0_margins: list[float] = []
    orientation1_margins: list[float] = []
    table_welfare: list[float] = []
    team_deltas_by_orientation: dict[str, list[list[int]]] = {"a_team0": [], "a_team1": []}

    progress_bar = tqdm(total=hands, desc="Duplicate deals", disable=not progress)
    try:
        for hand_idx in range(hands):
            deal = generate_deal(seed + hand_idx)
            orientation0 = play_duplicate_orientation(
                deal,
                actor_a,
                actor_b,
                a_is_team0=True,
                max_normal_bid=max_normal_bid,
                nil=nil,
                blind_nil=blind_nil,
                reward_scale=reward_scale,
                seed=seed * 10_000 + hand_idx * 2,
                stats=stats,
            )
            orientation1 = play_duplicate_orientation(
                deal,
                actor_a,
                actor_b,
                a_is_team0=False,
                max_normal_bid=max_normal_bid,
                nil=nil,
                blind_nil=blind_nil,
                reward_scale=reward_scale,
                seed=seed * 10_000 + hand_idx * 2 + 1,
                stats=stats,
            )
            margin0 = float(orientation0["a_margin"])
            margin1 = float(orientation1["a_margin"])
            orientation0_margins.append(margin0)
            orientation1_margins.append(margin1)
            paired_margins.append((margin0 + margin1) / 2.0)
            table_welfare.append(float(orientation0["table_welfare"]))
            table_welfare.append(float(orientation1["table_welfare"]))
            team_deltas_by_orientation["a_team0"].append(orientation0["team_delta"])
            team_deltas_by_orientation["a_team1"].append(orientation1["team_delta"])
            progress_bar.update(1)
            progress_bar.set_postfix(mean=f"{np.mean(paired_margins):.4f}")
    finally:
        progress_bar.close()

    margins = np.asarray(paired_margins, dtype=np.float32)
    se = _stderr(margins)
    margin_mean = float(margins.mean()) if len(margins) else 0.0
    team_delta_a_team0 = np.asarray(team_deltas_by_orientation["a_team0"], dtype=np.float32)
    team_delta_a_team1 = np.asarray(team_deltas_by_orientation["a_team1"], dtype=np.float32)
    return {
        "policy_a": actor_a.name,
        "policy_b": actor_b.name,
        "hands": hands,
        "seed": seed,
        "max_normal_bid": max_normal_bid,
        "nil": nil,
        "blind_nil": blind_nil,
        "reward_scale": reward_scale,
        "duplicate_margin_mean": margin_mean,
        "duplicate_margin_stderr": se,
        "duplicate_margin_ci95": [margin_mean - 1.96 * se, margin_mean + 1.96 * se],
        "duplicate_margin_raw_mean": margin_mean / reward_scale,
        "orientation_0_margin_mean": float(np.mean(orientation0_margins)),
        "orientation_1_margin_mean": float(np.mean(orientation1_margins)),
        "table_welfare_mean": float(np.mean(table_welfare)),
        "mean_team_delta_by_orientation": {
            "a_team0": team_delta_a_team0.mean(axis=0).tolist(),
            "a_team1": team_delta_a_team1.mean(axis=0).tolist(),
        },
        **stats,
    }


def make_actor(
    descriptor: str,
    *,
    label: str,
    max_normal_bid: int,
    d_model: int,
    transformer_layers: int,
    attention_heads: int,
    ffn_size: int,
    dropout: float,
    device: str,
) -> Actor:
    if descriptor.startswith("checkpoint:"):
        path = descriptor.split(":", 1)[1]
        return CheckpointActor(
            name=descriptor,
            checkpoint=path,
            max_normal_bid=max_normal_bid,
            d_model=d_model,
            transformer_layers=transformer_layers,
            attention_heads=attention_heads,
            ffn_size=ffn_size,
            dropout=dropout,
            device=device,
        )

    if not descriptor.startswith("bot:"):
        raise ValueError(f"{label} must start with 'checkpoint:' or 'bot:'")

    bot_name = descriptor.split(":", 1)[1]
    bots = {
        "random": RandomLegalBot,
        "conservative": ConservativeBidBot,
        "lowest": LowestLegalCardBot,
        "highest": HighestLegalCardBot,
    }
    if bot_name not in bots:
        options = ", ".join(sorted(bots))
        raise ValueError(f"Unknown bot '{bot_name}'. Expected one of: {options}")
    return BotActor(name=descriptor, bot=bots[bot_name]())


def make_duplicate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Duplicate head-to-head Spades evaluation")
    parser.add_argument("--policy-a", required=True)
    parser.add_argument("--policy-b", required=True)
    parser.add_argument("--hands", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=123)
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
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-progress", action="store_false", dest="progress")
    parser.add_argument("--output", type=str, default="")
    return parser


def duplicate_main() -> None:
    parser = make_duplicate_parser()
    args = parser.parse_args()
    if args.hands <= 0:
        raise ValueError("hands must be positive")
    if not 1 <= args.max_normal_bid <= 13:
        raise ValueError("max_normal_bid must be in [1, 13]")
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    actor_a = make_actor(
        args.policy_a,
        label="policy-a",
        max_normal_bid=args.max_normal_bid,
        d_model=args.d_model,
        transformer_layers=args.transformer_layers,
        attention_heads=args.attention_heads,
        ffn_size=args.ffn_size,
        dropout=args.dropout,
        device=device,
    )
    actor_b = make_actor(
        args.policy_b,
        label="policy-b",
        max_normal_bid=args.max_normal_bid,
        d_model=args.d_model,
        transformer_layers=args.transformer_layers,
        attention_heads=args.attention_heads,
        ffn_size=args.ffn_size,
        dropout=args.dropout,
        device=device,
    )
    metrics = evaluate_duplicate(
        actor_a,
        actor_b,
        hands=args.hands,
        seed=args.seed,
        max_normal_bid=args.max_normal_bid,
        nil=args.nil,
        blind_nil=args.blind_nil,
        reward_scale=args.reward_scale,
        progress=args.progress,
    )
    rendered = json.dumps(metrics, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(rendered)
            f.write("\n")
