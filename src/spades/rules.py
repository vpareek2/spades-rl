"""Pure rule helpers."""

from collections.abc import Sequence

from spades.cards import SPADES, card_rank, card_suit, is_spade
from spades.state import TrickPlay


def player_team(player: int) -> int:
    return 0 if player in (0, 2) else 1


def abs_to_rel(abs_player: int, acting_player: int) -> int:
    return (abs_player - acting_player) % 4


def rel_to_abs(rel_seat: int, acting_player: int) -> int:
    return (acting_player + rel_seat) % 4


def first_bidder(dealer: int) -> int:
    return (dealer + 1) % 4


def first_trick_leader(dealer: int) -> int:
    return first_bidder(dealer)


def bidding_order(dealer: int) -> list[int]:
    return [(dealer + offset) % 4 for offset in range(1, 5)]


def game_result(team_scores, target_score: int) -> tuple[bool, int | None]:
    reached = [team_scores[0] >= target_score, team_scores[1] >= target_score]
    if not any(reached):
        return False, None
    if reached[0] and not reached[1]:
        return True, 0
    if reached[1] and not reached[0]:
        return True, 1
    if team_scores[0] > team_scores[1]:
        return True, 0
    if team_scores[1] > team_scores[0]:
        return True, 1
    return False, None


def legal_card_ids(state, player: int) -> list[int]:
    hand = list(state.hands[player])
    if len(state.current_trick) == 0:
        if state.spades_broken:
            return sorted(hand)
        non_spades = [card for card in hand if not is_spade(card)]
        return sorted(non_spades if non_spades else hand)

    led_suit = card_suit(state.current_trick[0].card_id)
    follow_suit_cards = [card for card in hand if card_suit(card) == led_suit]
    return sorted(follow_suit_cards if follow_suit_cards else hand)


def resolve_trick_winner(current_trick: Sequence[TrickPlay]) -> int:
    if len(current_trick) != 4:
        raise ValueError("A trick must contain exactly four plays")

    spade_plays = [play for play in current_trick if is_spade(play.card_id)]
    if spade_plays:
        return max(spade_plays, key=lambda play: card_rank(play.card_id)).player

    led_suit = card_suit(current_trick[0].card_id)
    led_suit_plays = [
        play for play in current_trick if card_suit(play.card_id) == led_suit
    ]
    return max(led_suit_plays, key=lambda play: card_rank(play.card_id)).player


def should_break_spades(current_trick: Sequence[TrickPlay], played_card: int) -> bool:
    if not is_spade(played_card):
        return False
    if len(current_trick) == 0:
        return True
    return card_suit(current_trick[0].card_id) != SPADES
