from types import SimpleNamespace

import numpy as np
import pytest

from spades.config import SpadesPlusConfig
from spades.rules import (
    abs_to_rel,
    bidding_order,
    first_bidder,
    first_trick_leader,
    game_result,
    legal_card_ids,
    rel_to_abs,
    resolve_trick_winner,
    should_break_spades,
)
from spades.scoring import score_hand, score_player_bid
from spades.state import Bid, BidKind, TrickPlay


def test_seat_conversion_round_trips():
    for acting in range(4):
        for player in range(4):
            rel = abs_to_rel(player, acting)
            assert rel_to_abs(rel, acting) == player
    assert abs_to_rel(0, 2) == 2


def test_bidding_order_and_game_result_helpers():
    assert first_bidder(3) == 0
    assert first_trick_leader(3) == 0
    assert bidding_order(3) == [0, 1, 2, 3]
    assert game_result([249, 10], 250) == (False, None)
    assert game_result([250, 10], 250) == (True, 0)
    assert game_result([260, 270], 250) == (True, 1)
    assert game_result([260, 260], 250) == (False, None)


def test_legal_cards_enforce_follow_suit_and_spades_lead_rule():
    state = SimpleNamespace(
        hands=[[0, 39, 40], [13, 14, 41], [26, 42], [39, 40]],
        current_trick=[],
        spades_broken=False,
    )
    assert legal_card_ids(state, 0) == [0]
    assert legal_card_ids(state, 3) == [39, 40]

    state.current_trick = [TrickPlay(0, 13)]
    assert legal_card_ids(state, 1) == [13, 14]
    assert legal_card_ids(state, 2) == [26, 42]

    state.current_trick = []
    state.spades_broken = True
    assert legal_card_ids(state, 0) == [0, 39, 40]


def test_spades_break_only_on_spade_play():
    assert should_break_spades([], 39)
    assert should_break_spades([TrickPlay(0, 0)], 39)
    assert not should_break_spades([TrickPlay(0, 39)], 40)
    assert not should_break_spades([TrickPlay(0, 0)], 13)


def test_resolve_trick_winner():
    assert (
        resolve_trick_winner(
            [TrickPlay(0, 0), TrickPlay(1, 12), TrickPlay(2, 2), TrickPlay(3, 10)]
        )
        == 1
    )
    assert (
        resolve_trick_winner(
            [TrickPlay(0, 12), TrickPlay(1, 25), TrickPlay(2, 39), TrickPlay(3, 10)]
        )
        == 2
    )
    assert (
        resolve_trick_winner(
            [TrickPlay(0, 12), TrickPlay(1, 25), TrickPlay(2, 26), TrickPlay(3, 10)]
        )
        == 0
    )
    with pytest.raises(ValueError):
        resolve_trick_winner([TrickPlay(0, 0)])


def test_score_player_bid_cases():
    config = SpadesPlusConfig()
    assert score_player_bid(Bid(BidKind.NORMAL, 4), 4, config).score == 40
    over = score_player_bid(Bid(BidKind.NORMAL, 4), 6, config)
    assert (over.score, over.bags, over.made_contract) == (42, 2, True)
    failed = score_player_bid(Bid(BidKind.NORMAL, 4), 3, config)
    assert (failed.score, failed.bags, failed.made_contract) == (-40, 0, False)
    assert score_player_bid(Bid(BidKind.NIL), 0, config).score == 100
    nil_failed = score_player_bid(Bid(BidKind.NIL), 2, config)
    assert (nil_failed.score, nil_failed.bags, nil_failed.nil_success) == (-100, 2, False)
    assert score_player_bid(Bid(BidKind.BLIND_NIL), 0, config).score == 200


def test_score_hand_individual_contracts_and_bag_penalty_examples():
    config = SpadesPlusConfig()
    result = score_hand(
        [Bid(BidKind.NIL), Bid(BidKind.NORMAL, 1), Bid(BidKind.NORMAL, 4), Bid(BidKind.NORMAL, 1)],
        np.array([1, 0, 3, 0]),
        np.array([0, 0]),
        np.array([0, 0]),
        config,
    )
    assert result.player_score_delta.tolist() == [-100, -10, -40, -10]
    assert result.bags_gained_by_player.tolist() == [1, 0, 0, 0]
    assert result.team_score_delta.tolist() == [-140, -20]
    assert result.team_bags.tolist() == [1, 0]

    result = score_hand(
        [
            Bid(BidKind.NORMAL, 3),
            Bid(BidKind.NORMAL, 1),
            Bid(BidKind.BLIND_NIL),
            Bid(BidKind.NORMAL, 1),
        ],
        np.array([5, 0, 0, 0]),
        np.array([0, 0]),
        np.array([0, 0]),
        config,
    )
    assert result.team_score_delta.tolist() == [232, -20]
    assert result.team_bags.tolist() == [2, 0]

    result = score_hand(
        [
            Bid(BidKind.NORMAL, 1),
            Bid(BidKind.NORMAL, 2),
            Bid(BidKind.NORMAL, 1),
            Bid(BidKind.NORMAL, 3),
        ],
        np.array([1, 4, 1, 4]),
        np.array([0, 0]),
        np.array([0, 8]),
        config,
    )
    assert result.player_score_delta.tolist() == [10, 22, 10, 31]
    assert result.team_score_delta.tolist() == [20, -47]
    assert result.team_bags.tolist() == [0, 1]

    result = score_hand(
        [
            Bid(BidKind.NORMAL, 1),
            Bid(BidKind.NORMAL, 1),
            Bid(BidKind.NORMAL, 1),
            Bid(BidKind.NORMAL, 1),
        ],
        np.array([8, 0, 8, 0]),
        np.array([0, 0]),
        np.array([9, 0]),
        config,
    )
    assert result.team_score_delta.tolist()[0] == -166
    assert result.team_bags.tolist()[0] == 3
