import pytest

from spades.actions import (
    ACTION_SPACE_SIZE,
    BLIND_NIL_ACTION,
    NIL_ACTION,
    NORMAL_BID_ACTION_END,
    NORMAL_BID_ACTION_START,
    action_to_normal_bid,
    bid_action_to_bid,
    bid_to_action,
)
from spades.cards import card_name, card_rank, card_suit, is_spade
from spades.state import Bid, BidKind


def test_card_helpers_cover_full_deck():
    assert len({suit * 13 + rank for suit in range(4) for rank in range(13)}) == 52
    for card in range(52):
        assert card_suit(card) == card // 13
        assert card_rank(card) == card % 13
        assert is_spade(card) is (card >= 39)


def test_card_names_and_invalid_ids():
    assert card_name(0) == "2C"
    assert card_name(12) == "AC"
    assert card_name(51) == "AS"
    with pytest.raises(ValueError):
        card_suit(-1)
    with pytest.raises(ValueError):
        card_rank(52)


def test_fixed_action_mapping():
    assert ACTION_SPACE_SIZE == 67
    assert NORMAL_BID_ACTION_START == 52
    assert NORMAL_BID_ACTION_END == 64
    assert NIL_ACTION == 65
    assert BLIND_NIL_ACTION == 66
    assert [action_to_normal_bid(action) for action in range(52, 65)] == list(range(1, 14))
    with pytest.raises(ValueError):
        action_to_normal_bid(51)


def test_bid_action_round_trip():
    for action in range(52, 67):
        bid = bid_action_to_bid(action)
        assert bid_to_action(bid) == action
    assert bid_to_action(Bid(BidKind.NORMAL, 13)) == 64
    with pytest.raises(ValueError):
        bid_action_to_bid(0)
