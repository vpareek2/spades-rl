"""Structured transformer policy for Spades observations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from spades.actions import ACTION_SPACE_SIZE
from spades.cards import NUM_CARDS
from spades.observations import FLAT_OBSERVATION_SIZE


MASK_OFFSET = FLAT_OBSERVATION_SIZE - ACTION_SPACE_SIZE

GLOBAL_TOKEN_COUNT = 1
CARD_TOKEN_COUNT = 52
BID_EVENT_COUNT = 4
PLAY_EVENT_COUNT = 52
EVENT_TOKEN_COUNT = BID_EVENT_COUNT + PLAY_EVENT_COUNT
SEAT_TOKEN_COUNT = 4
TOKEN_COUNT = GLOBAL_TOKEN_COUNT + CARD_TOKEN_COUNT + EVENT_TOKEN_COUNT + SEAT_TOKEN_COUNT


@dataclass(frozen=True)
class ObservationSlices:
    scalars: slice = slice(0, 8)
    team_scores: slice = slice(8, 10)
    team_bags: slice = slice(10, 12)
    bid_kind: slice = slice(12, 16)
    bid_value: slice = slice(16, 20)
    tricks_taken: slice = slice(20, 24)
    own_hand_mask: slice = slice(24, 76)
    played_cards_mask: slice = slice(76, 128)
    current_trick_cards_by_order: slice = slice(128, 132)
    current_trick_seats_by_order: slice = slice(132, 136)
    current_trick_card_by_rel_seat: slice = slice(136, 140)
    bid_history_seat: slice = slice(140, 144)
    bid_history_kind: slice = slice(144, 148)
    bid_history_value: slice = slice(148, 152)
    play_history_card: slice = slice(152, 204)
    play_history_seat: slice = slice(204, 256)
    play_history_trick: slice = slice(256, 308)
    play_history_pos: slice = slice(308, 360)
    play_history_led_suit: slice = slice(360, 412)
    play_history_followed_suit: slice = slice(412, 464)
    trick_winner_by_trick: slice = slice(464, 477)
    void_suits_by_rel_seat: slice = slice(477, 493)
    cards_remaining_by_rel_seat: slice = slice(493, 497)
    action_mask: slice = slice(MASK_OFFSET, MASK_OFFSET + ACTION_SPACE_SIZE)


SLICES = ObservationSlices()


def _as_long(values: torch.Tensor) -> torch.Tensor:
    return values.round().long()


def _unknown_index(values: torch.Tensor, valid_count: int) -> torch.Tensor:
    values = _as_long(values)
    unknown = torch.full_like(values, valid_count)
    return torch.where((values >= 0) & (values < valid_count), values, unknown)


def _normalized_unknown(values: torch.Tensor, max_value: float) -> torch.Tensor:
    valid = values >= 0
    normalized = torch.where(valid, values / max_value, torch.zeros_like(values))
    return torch.stack([normalized, valid.float()], dim=-1)


class FlatObservationTokenizer(nn.Module):
    """Decode flat observations into transformer tokens."""

    def __init__(self, d_model: int, dropout: float = 0.05):
        super().__init__()
        card_ids = torch.arange(NUM_CARDS, dtype=torch.long)
        self.register_buffer("card_ids", card_ids, persistent=False)
        self.register_buffer("card_ranks", card_ids % 13, persistent=False)
        self.register_buffer("card_suits", card_ids // 13, persistent=False)
        self.register_buffer("card_is_spade", (card_ids // 13 == 3).float(), persistent=False)
        seat_ids = torch.arange(4, dtype=torch.long)
        self.register_buffer("seat_ids", seat_ids, persistent=False)

        self.card_id_embed = nn.Embedding(53, d_model)
        self.rank_embed = nn.Embedding(13, d_model)
        self.suit_embed = nn.Embedding(4, d_model)
        self.seat_embed = nn.Embedding(5, d_model)
        self.bid_kind_embed = nn.Embedding(4, d_model)
        self.phase_embed = nn.Embedding(4, d_model)
        self.role_embed = nn.Embedding(3, d_model)
        self.event_kind_embed = nn.Embedding(3, d_model)
        self.token_type_embed = nn.Embedding(4, d_model)

        self.global_proj = nn.Sequential(nn.Linear(20, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.card_proj = nn.Sequential(nn.Linear(19, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.event_proj = nn.Sequential(nn.Linear(13, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.seat_proj = nn.Sequential(nn.Linear(21, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        x = observations.float()
        global_tokens = self._global_tokens(x)
        card_tokens, current_winning_card = self._card_tokens(x)
        event_tokens = self._event_tokens(x, current_winning_card)
        seat_tokens = self._seat_tokens(x)

        tokens = torch.cat([global_tokens, card_tokens, event_tokens, seat_tokens], dim=1)
        token_types = torch.cat(
            [
                torch.zeros(GLOBAL_TOKEN_COUNT, dtype=torch.long, device=x.device),
                torch.ones(CARD_TOKEN_COUNT, dtype=torch.long, device=x.device),
                torch.full((EVENT_TOKEN_COUNT,), 2, dtype=torch.long, device=x.device),
                torch.full((SEAT_TOKEN_COUNT,), 3, dtype=torch.long, device=x.device),
            ]
        )
        tokens = tokens + self.token_type_embed(token_types).unsqueeze(0)
        return {
            "tokens": self.dropout(tokens),
            "card_token_slice": slice(1, 1 + CARD_TOKEN_COUNT),
            "seat_token_slice": slice(
                1 + CARD_TOKEN_COUNT + EVENT_TOKEN_COUNT,
                1 + CARD_TOKEN_COUNT + EVENT_TOKEN_COUNT + SEAT_TOKEN_COUNT,
            ),
        }

    def _global_tokens(self, x: torch.Tensor) -> torch.Tensor:
        scalars = x[:, SLICES.scalars]
        phase = _unknown_index(scalars[:, 0], 4)
        current_rel = _normalized_unknown(scalars[:, 1], 3.0).reshape(x.shape[0], -1)
        dealer_rel = _normalized_unknown(scalars[:, 2], 3.0).reshape(x.shape[0], -1)
        leader_rel = _normalized_unknown(scalars[:, 3], 3.0).reshape(x.shape[0], -1)
        trick_number = (scalars[:, 4:5] / 12.0).clamp(0.0, 1.0)
        turn_index = (scalars[:, 5:6] / 3.0).clamp(0.0, 1.0)
        hand_number = (scalars[:, 6:7] / 32.0).clamp(0.0, 1.0)
        spades_broken = scalars[:, 7:8].clamp(0.0, 1.0)

        team_scores = (x[:, SLICES.team_scores] / 250.0).clamp(-4.0, 4.0)
        team_bags_raw = x[:, SLICES.team_bags].clamp(0.0, 99.0)
        team_bags = team_bags_raw / 10.0
        bag_distance = (10.0 - (team_bags_raw % 10.0)) / 10.0

        bid_kind = x[:, SLICES.bid_kind]
        bid_value = x[:, SLICES.bid_value]
        normal_bid_value = torch.where(bid_kind == 1, bid_value, torch.zeros_like(bid_value))
        team_bid_totals = torch.stack(
            [normal_bid_value[:, 0] + normal_bid_value[:, 2], normal_bid_value[:, 1] + normal_bid_value[:, 3]],
            dim=1,
        ) / 13.0
        tricks_taken = x[:, SLICES.tricks_taken]
        team_tricks = torch.stack(
            [tricks_taken[:, 0] + tricks_taken[:, 2], tricks_taken[:, 1] + tricks_taken[:, 3]],
            dim=1,
        ) / 13.0

        dense = torch.cat(
            [
                current_rel,
                dealer_rel,
                leader_rel,
                trick_number,
                turn_index,
                hand_number,
                spades_broken,
                team_scores,
                team_bags,
                bag_distance,
                team_bid_totals,
                team_tricks,
            ],
            dim=1,
        )
        token = self.phase_embed(phase) + self.global_proj(dense)
        return token.unsqueeze(1)

    def _card_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        own = x[:, SLICES.own_hand_mask]
        played = x[:, SLICES.played_cards_mask]
        mask = x[:, SLICES.action_mask]
        legal_cards = mask[:, :CARD_TOKEN_COUNT]

        current_trick_cards = x[:, SLICES.current_trick_cards_by_order]
        current_card_ids = _unknown_index(current_trick_cards, NUM_CARDS)
        in_current = torch.zeros(batch, CARD_TOKEN_COUNT, device=x.device, dtype=x.dtype)
        valid_current = current_card_ids < NUM_CARDS
        in_current.scatter_(1, current_card_ids.clamp(max=NUM_CARDS - 1), valid_current.float())
        current_winning_card = self._current_winning_card(current_trick_cards)
        winning = torch.zeros(batch, CARD_TOKEN_COUNT, device=x.device, dtype=x.dtype)
        valid_winner = current_winning_card >= 0
        winning.scatter_(1, current_winning_card.clamp(min=0).unsqueeze(1), valid_winner.float().unsqueeze(1))

        history_seat = x[:, SLICES.play_history_seat]
        history_trick = x[:, SLICES.play_history_trick]
        history_pos = x[:, SLICES.play_history_pos]
        history_led = x[:, SLICES.play_history_led_suit]
        history_followed = x[:, SLICES.play_history_followed_suit]

        known_status = torch.stack(
            [
                own,
                played,
                in_current,
                ((own + played + in_current) <= 0).float(),
            ],
            dim=-1,
        )
        card_static = torch.stack(
            [
                self.card_ranks.float() / 12.0,
                self.card_suits.float() / 3.0,
                self.card_is_spade,
            ],
            dim=-1,
        ).unsqueeze(0).expand(batch, -1, -1)

        dense = torch.cat(
            [
                card_static,
                known_status,
                legal_cards.unsqueeze(-1),
                _normalized_unknown(history_seat, 3.0),
                _normalized_unknown(history_trick, 12.0),
                _normalized_unknown(history_pos, 3.0),
                _normalized_unknown(history_led, 3.0),
                _normalized_unknown(history_followed, 1.0),
                winning.unsqueeze(-1),
            ],
            dim=-1,
        )
        card_ids = self.card_ids.to(x.device)
        token = (
            self.card_id_embed(card_ids).unsqueeze(0)
            + self.rank_embed(self.card_ranks.to(x.device)).unsqueeze(0)
            + self.suit_embed(self.card_suits.to(x.device)).unsqueeze(0)
            + self.card_proj(dense)
        )
        return token, current_winning_card

    def _current_winning_card(self, current_trick_cards: torch.Tensor) -> torch.Tensor:
        card_ids = _as_long(current_trick_cards)
        valid = card_ids >= 0
        safe_cards = card_ids.clamp(min=0, max=51)
        suits = safe_cards // 13
        ranks = safe_cards % 13
        first_valid = valid[:, 0]
        led_suit = suits[:, 0]
        is_spade = suits == 3
        follows_led = suits == led_suit.unsqueeze(1)
        priority = torch.where(is_spade, torch.full_like(ranks, 2), torch.zeros_like(ranks))
        priority = torch.where(follows_led & ~is_spade, torch.ones_like(priority), priority)
        score = priority.float() * 100.0 + ranks.float()
        score = score.masked_fill(~valid, -1.0e9)
        winner_idx = score.argmax(dim=1)
        winner_card = safe_cards.gather(1, winner_idx.unsqueeze(1)).squeeze(1)
        return torch.where(first_valid, winner_card, torch.full_like(winner_card, -1))

    def _event_tokens(self, x: torch.Tensor, current_winning_card: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        bid_seat = x[:, SLICES.bid_history_seat]
        bid_kind = x[:, SLICES.bid_history_kind]
        bid_value = x[:, SLICES.bid_history_value]
        bid_active = (bid_seat >= 0).float()
        bid_dense = torch.cat(
            [
                bid_active.unsqueeze(-1),
                _normalized_unknown(bid_value, 13.0),
                torch.zeros(batch, BID_EVENT_COUNT, 10, device=x.device),
            ],
            dim=-1,
        )
        bid_tokens = (
            self.event_kind_embed(torch.ones(BID_EVENT_COUNT, dtype=torch.long, device=x.device)).unsqueeze(0)
            + self.seat_embed(_unknown_index(bid_seat, 4))
            + self.bid_kind_embed(_unknown_index(bid_kind, 4).clamp(max=3))
            + self.event_proj(bid_dense)
        )

        history_seat = x[:, SLICES.play_history_seat]
        history_trick = x[:, SLICES.play_history_trick]
        history_pos = x[:, SLICES.play_history_pos]
        history_led = x[:, SLICES.play_history_led_suit]
        history_followed = x[:, SLICES.play_history_followed_suit]
        played = history_trick >= 0
        order = torch.where(played, history_trick * 4.0 + history_pos, torch.full_like(history_trick, 10_000.0))
        sorted_idx = order.argsort(dim=1)
        gather = sorted_idx
        card_ids = self.card_ids.to(x.device).unsqueeze(0).expand(batch, -1).gather(1, gather)
        seat = history_seat.gather(1, gather)
        trick = history_trick.gather(1, gather)
        pos = history_pos.gather(1, gather)
        led = history_led.gather(1, gather)
        followed = history_followed.gather(1, gather)
        active = played.gather(1, gather).float()
        winner_by_trick = x[:, SLICES.trick_winner_by_trick]
        winner = winner_by_trick.gather(1, trick.clamp(min=0, max=12).long())
        current_win = (card_ids == current_winning_card.unsqueeze(1)).float()
        play_dense = torch.cat(
            [
                active.unsqueeze(-1),
                (card_ids.float() / 51.0).unsqueeze(-1),
                _normalized_unknown(trick, 12.0),
                _normalized_unknown(pos, 3.0),
                _normalized_unknown(led, 3.0),
                _normalized_unknown(followed, 1.0),
                _normalized_unknown(winner, 3.0),
                current_win.unsqueeze(-1),
            ],
            dim=-1,
        )
        play_tokens = (
            self.event_kind_embed(torch.full((PLAY_EVENT_COUNT,), 2, dtype=torch.long, device=x.device)).unsqueeze(0)
            + self.card_id_embed(card_ids)
            + self.seat_embed(_unknown_index(seat, 4))
            + self.event_proj(play_dense)
        )
        return torch.cat([bid_tokens, play_tokens], dim=1)

    def _seat_tokens(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        seat_ids = self.seat_ids.to(x.device)
        scalars = x[:, SLICES.scalars]
        current_rel = _as_long(scalars[:, 1]).clamp(min=-1, max=3)
        dealer_rel = _as_long(scalars[:, 2]).clamp(min=-1, max=3)
        leader_rel = _as_long(scalars[:, 3]).clamp(min=-1, max=3)

        bid_kind = x[:, SLICES.bid_kind]
        bid_value = x[:, SLICES.bid_value]
        tricks_taken = x[:, SLICES.tricks_taken]
        cards_remaining = x[:, SLICES.cards_remaining_by_rel_seat]
        voids = x[:, SLICES.void_suits_by_rel_seat].reshape(batch, 4, 4)
        roles = torch.tensor([0, 2, 1, 2], dtype=torch.long, device=x.device)
        nil_status = torch.stack([(bid_kind == 2).float(), (bid_kind == 3).float()], dim=-1)
        flags = torch.stack(
            [
                seat_ids.unsqueeze(0).expand(batch, -1) == current_rel.unsqueeze(1),
                seat_ids.unsqueeze(0).expand(batch, -1) == dealer_rel.unsqueeze(1),
                seat_ids.unsqueeze(0).expand(batch, -1) == leader_rel.unsqueeze(1),
            ],
            dim=-1,
        ).float()
        dense = torch.cat(
            [
                (seat_ids.float() / 3.0).reshape(1, 4, 1).expand(batch, -1, -1),
                flags,
                (bid_value / 13.0).unsqueeze(-1),
                (tricks_taken / 13.0).unsqueeze(-1),
                (cards_remaining / 13.0).unsqueeze(-1),
                voids.float(),
                nil_status,
                torch.zeros(batch, 4, 8, device=x.device),
            ],
            dim=-1,
        )
        return (
            self.seat_embed(seat_ids).unsqueeze(0)
            + self.role_embed(roles).unsqueeze(0)
            + self.bid_kind_embed(_unknown_index(bid_kind, 4).clamp(max=3))
            + self.seat_proj(dense)
        )


class SpadesTransformerPolicy(nn.Module):
    """Transformer policy with card/event/seat tokens and action masking."""

    def __init__(
        self,
        obs_size: int = FLAT_OBSERVATION_SIZE,
        action_size: int = ACTION_SPACE_SIZE,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_size: int = 1024,
        dropout: float = 0.05,
    ):
        super().__init__()
        if obs_size != FLAT_OBSERVATION_SIZE:
            raise ValueError(f"SpadesTransformerPolicy expects obs_size={FLAT_OBSERVATION_SIZE}")
        if action_size != ACTION_SPACE_SIZE:
            raise ValueError(f"SpadesTransformerPolicy expects action_size={ACTION_SPACE_SIZE}")
        self.obs_size = obs_size
        self.action_size = action_size
        self.d_model = d_model
        self.tokenizer = FlatObservationTokenizer(d_model=d_model, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(d_model)

        self.play_query = nn.Linear(d_model, d_model)
        self.play_card_score = nn.Linear(d_model, 1)
        self.bid_policy_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 15))
        self.bid_q_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 15))
        self.value_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.margin_value_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.hidden_owner_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 4))
        self.void_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 4))

    def initial_state(self, batch_size: int, device: str | torch.device):
        return ()

    def forward_eval(self, observations: torch.Tensor, state=()):
        heads = self.forward_heads(observations, apply_mask=True)
        return heads["logits"], heads["value"], state

    def forward(self, observations: torch.Tensor):
        batch, horizon = observations.shape[:2]
        flat_obs = observations.reshape(batch * horizon, observations.shape[-1])
        heads = self.forward_heads(flat_obs, apply_mask=True)
        return heads["logits"], heads["value"].reshape(batch, horizon)

    def forward_heads(self, observations: torch.Tensor, apply_mask: bool = True) -> dict[str, Any]:
        x = observations.float()
        tokenized = self.tokenizer(x)
        encoded = self.final_norm(self.transformer(tokenized["tokens"]))
        global_repr = encoded[:, 0]
        card_repr = encoded[:, tokenized["card_token_slice"]]
        seat_repr = encoded[:, tokenized["seat_token_slice"]]

        query = self.play_query(global_repr).unsqueeze(1)
        play_logits = (query * card_repr).sum(dim=-1) / math.sqrt(self.d_model)
        play_logits = play_logits + self.play_card_score(card_repr).squeeze(-1)
        bid_logits = self.bid_policy_head(global_repr)
        logits = torch.cat([play_logits, bid_logits], dim=1)
        if apply_mask:
            logits = self._apply_action_mask(logits, x)
        return {
            "logits": logits,
            "value": self.value_head(global_repr),
            "bid_q": self.bid_q_head(global_repr),
            "margin_value": self.margin_value_head(global_repr),
            "hidden_owner_logits": self.hidden_owner_head(card_repr),
            "void_logits": self.void_head(seat_repr),
        }

    def _apply_action_mask(self, logits: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        mask = observations[:, SLICES.action_mask] > 0.5
        fallback = torch.zeros_like(mask)
        fallback[:, 0] = True
        mask = torch.where(mask.any(dim=1, keepdim=True), mask, fallback)
        return logits.masked_fill(~mask, -1.0e9)


def load_transformer_state(policy: SpadesTransformerPolicy, path: str, device: str | torch.device) -> None:
    state_dict = torch.load(path, map_location=device)
    try:
        policy.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint '{path}' is incompatible with SpadesTransformerPolicy. "
            "Pre-transformer MLP checkpoints cannot be loaded on this branch."
        ) from exc
