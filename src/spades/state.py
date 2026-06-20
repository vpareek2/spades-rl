"""Shared state dataclasses and enums."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class BidKind(Enum):
    NORMAL = "normal"
    NIL = "nil"
    BLIND_NIL = "blind_nil"


@dataclass(frozen=True)
class Bid:
    kind: BidKind
    value: int = 0

    def __post_init__(self) -> None:
        if self.kind == BidKind.NORMAL:
            if not 1 <= self.value <= 13:
                raise ValueError("Normal bids must be in [1, 13]")
        elif self.value != 0:
            raise ValueError("Nil and blind nil bids must have value 0")


class Phase(Enum):
    BIDDING = "bidding"
    PLAYING = "playing"
    HAND_OVER = "hand_over"
    GAME_OVER = "game_over"


@dataclass(frozen=True)
class TrickPlay:
    player: int
    card_id: int


@dataclass(frozen=True)
class Event:
    type: Literal["bid", "play", "trick_end", "hand_end"]
    player: int | None = None
    action: int | None = None
    card_id: int | None = None
    bid: Bid | None = None
    trick_winner: int | None = None
    hand_index: int | None = None
    trick_index: int | None = None


BID_UNKNOWN = 0
BID_NORMAL = 1
BID_NIL = 2
BID_BLIND_NIL = 3
