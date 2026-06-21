"""Observation encoding."""

from typing import Any

import numpy as np

from spades.actions import ACTION_SPACE_SIZE
from spades.cards import NUM_CARDS
from spades.cards import card_suit
from spades.rules import abs_to_rel, player_team
from spades.state import (
    BID_BLIND_NIL,
    BID_NIL,
    BID_NORMAL,
    BID_UNKNOWN,
    BidKind,
    Phase,
)


PHASE_TO_INT = {
    Phase.BIDDING: 0,
    Phase.PLAYING: 1,
    Phase.HAND_OVER: 2,
    Phase.GAME_OVER: 3,
}


def _seat_order(acting_player: int, seat_relative: bool) -> list[int]:
    if seat_relative:
        return [(acting_player + rel) % 4 for rel in range(4)]
    return [0, 1, 2, 3]


def _bid_encoding(bid) -> tuple[int, int]:
    if bid is None:
        return BID_UNKNOWN, 0
    if bid.kind == BidKind.NORMAL:
        return BID_NORMAL, bid.value
    if bid.kind == BidKind.NIL:
        return BID_NIL, 0
    return BID_BLIND_NIL, 0


def _current_hand_events(env) -> list:
    return [event for event in env.history if event.hand_index == env.hand_number]


def _public_history_encoding(env, acting_player: int, seat_relative: bool) -> dict[str, np.ndarray]:
    bid_history_seat = np.full(4, -1, dtype=np.int8)
    bid_history_kind = np.zeros(4, dtype=np.int8)
    bid_history_value = np.zeros(4, dtype=np.int8)

    play_history_card = np.full(NUM_CARDS, -1, dtype=np.int16)
    play_history_seat = np.full(NUM_CARDS, -1, dtype=np.int8)
    play_history_trick = np.full(NUM_CARDS, -1, dtype=np.int8)
    play_history_pos = np.full(NUM_CARDS, -1, dtype=np.int8)
    play_history_led_suit = np.full(NUM_CARDS, -1, dtype=np.int8)
    play_history_followed_suit = np.full(NUM_CARDS, -1, dtype=np.int8)

    trick_winner_by_trick = np.full(13, -1, dtype=np.int8)
    void_suits_abs = np.zeros((4, 4), dtype=np.bool_)
    play_position_by_trick = np.zeros(13, dtype=np.int8)
    led_suit_by_trick = np.full(13, -1, dtype=np.int8)
    bid_idx = 0

    for event in _current_hand_events(env):
        if event.type == "bid" and event.player is not None and event.bid is not None:
            if bid_idx < 4:
                seat = abs_to_rel(event.player, acting_player) if seat_relative else event.player
                bid_history_seat[bid_idx] = seat
                bid_history_kind[bid_idx], bid_history_value[bid_idx] = _bid_encoding(event.bid)
            bid_idx += 1
        elif event.type == "play" and event.player is not None and event.card_id is not None:
            trick_index = int(event.trick_index if event.trick_index is not None else 0)
            if not 0 <= trick_index < 13:
                continue
            pos = int(play_position_by_trick[trick_index])
            play_position_by_trick[trick_index] += 1
            card_id = int(event.card_id)
            suit = card_suit(card_id)
            if pos == 0:
                led_suit_by_trick[trick_index] = suit
            led_suit = int(led_suit_by_trick[trick_index])
            followed_suit = int(suit == led_suit)
            if pos > 0 and not followed_suit:
                void_suits_abs[event.player, led_suit] = True

            seat = abs_to_rel(event.player, acting_player) if seat_relative else event.player
            play_history_card[card_id] = card_id
            play_history_seat[card_id] = seat
            play_history_trick[card_id] = trick_index
            play_history_pos[card_id] = pos
            play_history_led_suit[card_id] = led_suit
            play_history_followed_suit[card_id] = followed_suit
        elif event.type == "trick_end" and event.trick_winner is not None:
            trick_index = int(event.trick_index if event.trick_index is not None else -1)
            if 0 <= trick_index < 13:
                trick_winner_by_trick[trick_index] = (
                    abs_to_rel(event.trick_winner, acting_player)
                    if seat_relative
                    else event.trick_winner
                )

    void_suits_by_rel_seat = np.zeros((4, 4), dtype=np.bool_)
    cards_remaining_by_rel_seat = np.zeros(4, dtype=np.int8)
    for abs_player in range(4):
        out_idx = abs_to_rel(abs_player, acting_player) if seat_relative else abs_player
        void_suits_by_rel_seat[out_idx] = void_suits_abs[abs_player]
        cards_remaining_by_rel_seat[out_idx] = len(env.hands[abs_player])

    return {
        "bid_history_seat": bid_history_seat,
        "bid_history_kind": bid_history_kind,
        "bid_history_value": bid_history_value,
        "play_history_card": play_history_card,
        "play_history_seat": play_history_seat,
        "play_history_trick": play_history_trick,
        "play_history_pos": play_history_pos,
        "play_history_led_suit": play_history_led_suit,
        "play_history_followed_suit": play_history_followed_suit,
        "trick_winner_by_trick": trick_winner_by_trick,
        "void_suits_by_rel_seat": void_suits_by_rel_seat,
        "cards_remaining_by_rel_seat": cards_remaining_by_rel_seat,
    }


def build_observation(env, acting_player: int) -> dict[str, Any]:
    seat_relative = env.config.seat_relative_observation
    order = _seat_order(acting_player, seat_relative)

    own_team = player_team(acting_player)
    opp_team = 1 - own_team
    team_scores = np.array([env.team_scores[own_team], env.team_scores[opp_team]], dtype=np.int32)
    team_bags = np.array([env.team_bags[own_team], env.team_bags[opp_team]], dtype=np.int32)

    bid_kind = np.zeros(4, dtype=np.int8)
    bid_value = np.zeros(4, dtype=np.int8)
    tricks_taken = np.zeros(4, dtype=np.int8)
    for out_idx, abs_player in enumerate(order):
        bid_kind[out_idx], bid_value[out_idx] = _bid_encoding(env.player_bids[abs_player])
        tricks_taken[out_idx] = env.player_tricks[abs_player]

    own_hand_mask = np.zeros(NUM_CARDS, dtype=np.bool_)
    for card in env.hands[acting_player]:
        own_hand_mask[card] = True

    played_cards_mask = np.zeros(NUM_CARDS, dtype=np.bool_)
    for card in env.played_cards:
        played_cards_mask[card] = True

    current_trick_cards_by_order = np.full(4, -1, dtype=np.int16)
    current_trick_seats_by_order = np.full(4, -1, dtype=np.int8)
    current_trick_card_by_rel_seat = np.full(4, -1, dtype=np.int16)
    for idx, play in enumerate(env.current_trick):
        seat = abs_to_rel(play.player, acting_player) if seat_relative else play.player
        current_trick_cards_by_order[idx] = play.card_id
        current_trick_seats_by_order[idx] = seat
        current_trick_card_by_rel_seat[seat] = play.card_id

    current_rel = abs_to_rel(env.current_player(), acting_player) if seat_relative else env.current_player()
    dealer_rel = abs_to_rel(env.dealer, acting_player) if seat_relative else env.dealer
    leader_rel = abs_to_rel(env.leader, acting_player) if seat_relative else env.leader

    mask = env.action_mask()
    assert mask.shape == (ACTION_SPACE_SIZE,)

    observation = {
        "phase": PHASE_TO_INT[env.phase],
        "current_player_rel": current_rel,
        "dealer_rel": dealer_rel,
        "leader_rel": leader_rel,
        "trick_number": env.trick_number,
        "turn_index_within_trick": env.turn_index_within_trick,
        "hand_number": env.hand_number,
        "team_scores": team_scores,
        "team_bags": team_bags,
        "bid_kind": bid_kind,
        "bid_value": bid_value,
        "tricks_taken": tricks_taken,
        "own_hand_mask": own_hand_mask,
        "played_cards_mask": played_cards_mask,
        "current_trick_cards_by_order": current_trick_cards_by_order,
        "current_trick_seats_by_order": current_trick_seats_by_order,
        "current_trick_card_by_rel_seat": current_trick_card_by_rel_seat,
        "spades_broken": bool(env.spades_broken),
        "action_mask": mask,
    }
    observation.update(_public_history_encoding(env, acting_player, seat_relative))
    return observation


def flatten_observation(obs: dict[str, Any]) -> np.ndarray:
    """Flatten the dict observation into a deterministic float32 vector."""

    parts = [
        np.array(
            [
                obs["phase"],
                obs["current_player_rel"],
                obs["dealer_rel"],
                obs["leader_rel"],
                obs["trick_number"],
                obs["turn_index_within_trick"],
                obs["hand_number"],
                int(obs["spades_broken"]),
            ],
            dtype=np.float32,
        ),
        np.asarray(obs["team_scores"], dtype=np.float32),
        np.asarray(obs["team_bags"], dtype=np.float32),
        np.asarray(obs["bid_kind"], dtype=np.float32),
        np.asarray(obs["bid_value"], dtype=np.float32),
        np.asarray(obs["tricks_taken"], dtype=np.float32),
        np.asarray(obs["own_hand_mask"], dtype=np.float32),
        np.asarray(obs["played_cards_mask"], dtype=np.float32),
        np.asarray(obs["current_trick_cards_by_order"], dtype=np.float32),
        np.asarray(obs["current_trick_seats_by_order"], dtype=np.float32),
        np.asarray(obs["current_trick_card_by_rel_seat"], dtype=np.float32),
        np.asarray(obs["bid_history_seat"], dtype=np.float32),
        np.asarray(obs["bid_history_kind"], dtype=np.float32),
        np.asarray(obs["bid_history_value"], dtype=np.float32),
        np.asarray(obs["play_history_card"], dtype=np.float32),
        np.asarray(obs["play_history_seat"], dtype=np.float32),
        np.asarray(obs["play_history_trick"], dtype=np.float32),
        np.asarray(obs["play_history_pos"], dtype=np.float32),
        np.asarray(obs["play_history_led_suit"], dtype=np.float32),
        np.asarray(obs["play_history_followed_suit"], dtype=np.float32),
        np.asarray(obs["trick_winner_by_trick"], dtype=np.float32),
        np.asarray(obs["void_suits_by_rel_seat"], dtype=np.float32).reshape(-1),
        np.asarray(obs["cards_remaining_by_rel_seat"], dtype=np.float32),
        np.asarray(obs["action_mask"], dtype=np.float32),
    ]
    return np.concatenate(parts).astype(np.float32, copy=False)


FLAT_OBSERVATION_SIZE = (
    8
    + 2
    + 2
    + 4
    + 4
    + 4
    + 52
    + 52
    + 4
    + 4
    + 4
    + 4
    + 4
    + 4
    + 52
    + 52
    + 52
    + 52
    + 52
    + 52
    + 13
    + 16
    + 4
    + ACTION_SPACE_SIZE
)
