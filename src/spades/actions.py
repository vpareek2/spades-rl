"""Fixed action-space helpers."""

from spades.state import Bid, BidKind

CARD_ACTION_START = 0
CARD_ACTION_END = 51

BID_ACTION_START = 52
NORMAL_BID_ACTION_START = 52
NORMAL_BID_ACTION_END = 64

NIL_ACTION = 65
BLIND_NIL_ACTION = 66

ACTION_SPACE_SIZE = 67


def is_card_action(action: int) -> bool:
    return CARD_ACTION_START <= action <= CARD_ACTION_END


def is_bid_action(action: int) -> bool:
    return BID_ACTION_START <= action < ACTION_SPACE_SIZE


def action_to_normal_bid(action: int) -> int:
    if not NORMAL_BID_ACTION_START <= action <= NORMAL_BID_ACTION_END:
        raise ValueError(f"Action {action} is not a normal bid action")
    return action - 51


def bid_action_to_bid(action: int) -> Bid:
    if NORMAL_BID_ACTION_START <= action <= NORMAL_BID_ACTION_END:
        return Bid(BidKind.NORMAL, action_to_normal_bid(action))
    if action == NIL_ACTION:
        return Bid(BidKind.NIL)
    if action == BLIND_NIL_ACTION:
        return Bid(BidKind.BLIND_NIL)
    raise ValueError(f"Action {action} is not a bid action")


def bid_to_action(bid: Bid) -> int:
    if bid.kind == BidKind.NORMAL:
        return bid.value + 51
    if bid.kind == BidKind.NIL:
        return NIL_ACTION
    if bid.kind == BidKind.BLIND_NIL:
        return BLIND_NIL_ACTION
    raise ValueError(f"Unsupported bid kind: {bid.kind}")
