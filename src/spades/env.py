"""Core turn-based Spades+ environment."""

from copy import deepcopy
from typing import Any

import gymnasium
import numpy as np

from spades.actions import (
    ACTION_SPACE_SIZE,
    BLIND_NIL_ACTION,
    NIL_ACTION,
    bid_action_to_bid,
)
from spades.cards import NUM_CARDS
from spades.config import SpadesPlusConfig
from spades.observations import build_observation
from spades.rules import (
    legal_card_ids,
    first_trick_leader,
    game_result,
    player_team,
    resolve_trick_winner,
    should_break_spades,
)
from spades.scoring import HandScore, score_hand
from spades.state import Bid, Event, Phase, TrickPlay


class IllegalActionError(ValueError):
    pass


class SpadesPlusEnv:
    def __init__(self, config: SpadesPlusConfig | None = None):
        self.config = config or SpadesPlusConfig()
        self.action_space = gymnasium.spaces.Discrete(ACTION_SPACE_SIZE)
        self.rng = np.random.default_rng()
        self.history: list[Event] = []
        self.winning_team: int | None = None
        self._last_hand_score: HandScore | None = None
        self.reset()

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.history = []
        self.winning_team = None
        self._last_hand_score = None
        self.hand_number = 0
        self.team_scores = np.zeros(2, dtype=np.int32)
        self.team_bags = np.zeros(2, dtype=np.int32)
        if self.config.first_dealer is None:
            self.dealer = int(self.rng.integers(0, 4))
        else:
            self.dealer = self.config.first_dealer
        self._start_hand()
        return self.observe()

    def current_player(self) -> int:
        return self._current_player

    def observe(self, player: int | None = None) -> dict[str, Any]:
        return build_observation(self, self._current_player if player is None else player)

    def legal_actions(self) -> list[int]:
        if self.phase == Phase.BIDDING:
            actions = list(range(52, 65))
            if self.config.nil_enabled:
                actions.append(NIL_ACTION)
            if self.config.blind_nil_enabled and self.config.blind_nil_policy == "always":
                actions.append(BLIND_NIL_ACTION)
            return actions
        if self.phase == Phase.PLAYING:
            return legal_card_ids(self, self._current_player)
        return []

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        for action in self.legal_actions():
            mask[action] = True
        return mask

    def step(self, action: int):
        legal = self.legal_actions()
        if action not in legal:
            raise IllegalActionError(
                f"Player {self._current_player} attempted action {action}, "
                f"but legal actions are {legal}"
            )

        rewards = np.zeros(4, dtype=np.float32)
        terminated = False
        truncated = False
        info: dict[str, Any] = {
            "phase": self.phase,
            "current_player": self._current_player,
            "legal_actions": legal,
        }

        if self.phase == Phase.BIDDING:
            self._step_bid(action)
        elif self.phase == Phase.PLAYING:
            step_rewards, terminated, info_update = self._step_play(action)
            rewards += step_rewards
            info.update(info_update)
        else:
            raise IllegalActionError(f"Cannot step while phase is {self.phase}")

        if terminated:
            info.update(
                {
                    "game_completed": True,
                    "winning_team": self.winning_team,
                    "final_team_scores": self.team_scores.copy(),
                    "final_team_bags": self.team_bags.copy(),
                }
            )

        obs = {} if terminated else self.observe()
        return obs, rewards, terminated, truncated, info

    def get_debug_state(self) -> dict[str, Any]:
        return {
            "hands": deepcopy(self.hands),
            "dealer": self.dealer,
            "current_player": self._current_player,
            "leader": self.leader,
            "phase": self.phase,
            "team_scores": self.team_scores.copy(),
            "team_bags": self.team_bags.copy(),
            "player_bids": deepcopy(self.player_bids),
            "player_tricks": self.player_tricks.copy(),
            "current_trick": list(self.current_trick),
            "played_cards": set(self.played_cards),
            "history": list(self.history),
            "spades_broken": self.spades_broken,
            "hand_number": self.hand_number,
            "trick_number": self.trick_number,
        }

    def set_hands(self, hands: list[list[int]]) -> None:
        self._require_debug()
        if len(hands) != 4:
            raise ValueError("Expected four hands")
        normalized = [sorted(map(int, hand)) for hand in hands]
        if any(len(hand) != 13 for hand in normalized):
            raise ValueError("Each hand must contain 13 cards")
        cards = [card for hand in normalized for card in hand]
        if sorted(cards) != list(range(NUM_CARDS)):
            raise ValueError("Hands must partition all 52 unique card ids")
        self.hands = normalized

    def set_scores(self, team_scores, team_bags) -> None:
        self._require_debug()
        scores = np.asarray(team_scores, dtype=np.int32)
        bags = np.asarray(team_bags, dtype=np.int32)
        if scores.shape != (2,) or bags.shape != (2,):
            raise ValueError("team_scores and team_bags must have shape (2,)")
        if np.any(bags < 0):
            raise ValueError("team_bags must be nonnegative")
        self.team_scores = scores.copy()
        self.team_bags = bags.copy()

    def set_player_bids(self, bids: list[Bid]) -> None:
        self._require_debug()
        if len(bids) != 4 or not all(isinstance(bid, Bid) for bid in bids):
            raise ValueError("Expected four Bid objects")
        self.player_bids = list(bids)
        self.bid_count = 4

    def start_playing_phase(self) -> None:
        self._require_debug()
        if any(bid is None for bid in self.player_bids):
            raise ValueError("All player bids must be set before starting play")
        self.phase = Phase.PLAYING
        self.leader = first_trick_leader(self.dealer)
        self._current_player = self.leader
        self.current_trick = []
        self.played_cards = set()
        self.player_tricks = np.zeros(4, dtype=np.int8)
        self.trick_number = 0
        self.turn_index_within_trick = 0
        self.spades_broken = False

    def _require_debug(self) -> None:
        if not self.config.debug:
            raise RuntimeError("Debug setup APIs require SpadesPlusConfig(debug=True)")

    def _start_hand(self) -> None:
        deck = np.arange(NUM_CARDS, dtype=np.int16)
        self.rng.shuffle(deck)
        self.hands = [sorted(map(int, deck[i * 13 : (i + 1) * 13])) for i in range(4)]
        self.player_bids: list[Bid | None] = [None, None, None, None]
        self.player_tricks = np.zeros(4, dtype=np.int8)
        self.current_trick: list[TrickPlay] = []
        self.played_cards: set[int] = set()
        self.bid_count = 0
        self.trick_number = 0
        self.turn_index_within_trick = 0
        self.spades_broken = False
        self.phase = Phase.BIDDING
        self.leader = first_trick_leader(self.dealer)
        self._current_player = self.leader

    def _step_bid(self, action: int) -> None:
        bid = bid_action_to_bid(action)
        player = self._current_player
        self.player_bids[player] = bid
        self.bid_count += 1
        self.history.append(Event(type="bid", player=player, action=action, bid=bid))

        if self.bid_count == 4:
            self.phase = Phase.PLAYING
            self._current_player = self.leader
            self.turn_index_within_trick = 0
        else:
            self._current_player = (self._current_player + 1) % 4

    def _step_play(self, action: int) -> tuple[np.ndarray, bool, dict[str, Any]]:
        rewards = np.zeros(4, dtype=np.float32)
        info: dict[str, Any] = {}
        player = self._current_player
        self.hands[player].remove(action)
        if should_break_spades(self.current_trick, action):
            self.spades_broken = True
        self.current_trick.append(TrickPlay(player, action))
        self.played_cards.add(action)
        self.history.append(
            Event(
                type="play",
                player=player,
                action=action,
                card_id=action,
                trick_index=self.trick_number,
            )
        )

        if len(self.current_trick) < 4:
            self.turn_index_within_trick += 1
            self._current_player = (self._current_player + 1) % 4
            return rewards, False, info

        completed_trick = list(self.current_trick)
        winner = resolve_trick_winner(completed_trick)
        self.player_tricks[winner] += 1
        self.history.append(
            Event(
                type="trick_end",
                trick_winner=winner,
                hand_index=self.hand_number,
                trick_index=self.trick_number,
            )
        )
        self.current_trick = []
        self.trick_number += 1
        self.turn_index_within_trick = 0
        self.leader = winner
        self._current_player = winner
        info.update(
            {
                "trick_completed": True,
                "trick_winner": winner,
                "trick_cards": [(play.player, play.card_id) for play in completed_trick],
                "player_tricks": self.player_tricks.copy(),
            }
        )

        if self.trick_number < 13:
            return rewards, False, info

        hand_rewards, terminated, hand_info = self._finish_hand()
        rewards += hand_rewards
        info.update(hand_info)
        return rewards, terminated, info

    def _finish_hand(self) -> tuple[np.ndarray, bool, dict[str, Any]]:
        bids = [bid for bid in self.player_bids if bid is not None]
        if len(bids) != 4:
            raise RuntimeError("Cannot score a hand before all players bid")

        result = score_hand(
            bids,
            self.player_tricks,
            self.team_scores,
            self.team_bags,
            self.config,
        )
        self._last_hand_score = result
        self.team_scores = result.team_scores.copy()
        self.team_bags = result.team_bags.copy()

        rewards = np.zeros(4, dtype=np.float32)
        for player in range(4):
            rewards[player] = float(result.team_score_delta[player_team(player)])

        terminated = self._update_game_over()
        if terminated and self.config.terminal_win_reward_enabled:
            for player in range(4):
                if player_team(player) == self.winning_team:
                    rewards[player] += self.config.terminal_win_reward
                else:
                    rewards[player] -= self.config.terminal_win_reward

        self.history.append(Event(type="hand_end", hand_index=self.hand_number))
        info = {
            "hand_completed": True,
            "player_bids": bids,
            "player_tricks": self.player_tricks.copy(),
            "player_score_delta": result.player_score_delta.copy(),
            "bags_gained_by_player": result.bags_gained_by_player.copy(),
            "team_score_delta": result.team_score_delta.copy(),
            "team_scores": self.team_scores.copy(),
            "team_bags": self.team_bags.copy(),
            "made_contract": result.made_contract.copy(),
            "nil_success": result.nil_success.copy(),
            "official_reward": rewards.copy(),
            "shaped_reward": np.zeros(4, dtype=np.float32),
            "total_reward": rewards.copy(),
        }

        if terminated:
            self.phase = Phase.GAME_OVER
            return rewards, True, info

        self.hand_number += 1
        if self.config.dealer_rotates:
            self.dealer = (self.dealer + 1) % 4
        self._start_hand()
        return rewards, False, info

    def _update_game_over(self) -> bool:
        terminated, winner = game_result(self.team_scores, self.config.target_score)
        self.winning_team = winner
        return terminated
