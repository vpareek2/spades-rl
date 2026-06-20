"""Card encoding helpers for a standard 52-card deck."""

CLUBS = 0
DIAMONDS = 1
HEARTS = 2
SPADES = 3

SUITS = ["C", "D", "H", "S"]
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
NUM_CARDS = 52


def validate_card_id(card_id: int) -> None:
    if not 0 <= int(card_id) < NUM_CARDS:
        raise ValueError(f"Invalid card id: {card_id}")


def card_suit(card_id: int) -> int:
    validate_card_id(card_id)
    return int(card_id) // 13


def card_rank(card_id: int) -> int:
    validate_card_id(card_id)
    return int(card_id) % 13


def is_spade(card_id: int) -> bool:
    return card_suit(card_id) == SPADES


def card_name(card_id: int) -> str:
    validate_card_id(card_id)
    return f"{RANKS[card_rank(card_id)]}{SUITS[card_suit(card_id)]}"
