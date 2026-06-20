"""Scoring helpers."""

from dataclasses import dataclass

import numpy as np

from spades.config import SpadesPlusConfig
from spades.rules import player_team
from spades.state import Bid, BidKind


@dataclass(frozen=True)
class PlayerScore:
    score: int
    bags: int
    made_contract: bool
    nil_success: bool | None


@dataclass(frozen=True)
class HandScore:
    player_score_delta: np.ndarray
    bags_gained_by_player: np.ndarray
    team_score_delta: np.ndarray
    team_scores: np.ndarray
    team_bags: np.ndarray
    made_contract: np.ndarray
    nil_success: np.ndarray


def score_player_bid(bid: Bid, tricks: int, config: SpadesPlusConfig) -> PlayerScore:
    tricks = int(tricks)
    if bid.kind == BidKind.NORMAL:
        if tricks >= bid.value:
            return PlayerScore(10 * bid.value + (tricks - bid.value), tricks - bid.value, True, None)
        return PlayerScore(-10 * bid.value, 0, False, None)

    if bid.kind == BidKind.NIL:
        success = tricks == 0
        return PlayerScore(
            config.nil_bonus if success else config.nil_penalty,
            0 if success else tricks,
            success,
            success,
        )

    success = tricks == 0
    return PlayerScore(
        config.blind_nil_bonus if success else config.blind_nil_penalty,
        0 if success else tricks,
        success,
        success,
    )


def score_hand(
    player_bids: list[Bid],
    player_tricks: np.ndarray,
    team_scores: np.ndarray,
    team_bags: np.ndarray,
    config: SpadesPlusConfig,
) -> HandScore:
    player_score_delta = np.zeros(4, dtype=np.int32)
    bags_gained_by_player = np.zeros(4, dtype=np.int32)
    made_contract = np.zeros(4, dtype=np.bool_)
    nil_success = np.zeros(4, dtype=np.bool_)
    nil_success[:] = False

    for player, bid in enumerate(player_bids):
        scored = score_player_bid(bid, int(player_tricks[player]), config)
        player_score_delta[player] = scored.score
        bags_gained_by_player[player] = scored.bags
        made_contract[player] = scored.made_contract
        if scored.nil_success is not None:
            nil_success[player] = scored.nil_success

    team_score_delta = np.zeros(2, dtype=np.int32)
    next_team_bags = np.array(team_bags, dtype=np.int32).copy()
    next_team_scores = np.array(team_scores, dtype=np.int32).copy()

    for player in range(4):
        team = player_team(player)
        team_score_delta[team] += player_score_delta[player]
        next_team_bags[team] += bags_gained_by_player[player]

    if config.bags_enabled:
        for team in range(2):
            while next_team_bags[team] >= config.bag_threshold:
                team_score_delta[team] += config.bag_penalty
                next_team_bags[team] -= config.bag_threshold

    next_team_scores += team_score_delta
    return HandScore(
        player_score_delta=player_score_delta,
        bags_gained_by_player=bags_gained_by_player,
        team_score_delta=team_score_delta,
        team_scores=next_team_scores,
        team_bags=next_team_bags,
        made_contract=made_contract,
        nil_success=nil_success,
    )
