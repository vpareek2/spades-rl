"""Simple baseline bots for smoke tests and debugging."""

from typing import Protocol

import numpy as np

from spades.actions import BLIND_NIL_ACTION, NIL_ACTION


class Bot(Protocol):
    def act(self, obs: dict, legal_actions: list[int], rng=None) -> int: ...


def _rng_choice(rng, legal_actions: list[int]) -> int:
    if rng is None:
        rng = np.random.default_rng()
    return int(rng.choice(legal_actions))


class RandomLegalBot:
    def act(self, obs: dict, legal_actions: list[int], rng=None) -> int:
        return _rng_choice(rng, legal_actions)


class SimpleBidBot:
    def estimate_bid(self, hand_mask) -> int:
        hand = np.flatnonzero(hand_mask)
        spades = sum(card >= 39 for card in hand)
        high_cards = sum(card % 13 >= 10 for card in hand)
        return min(13, max(1, high_cards // 2 + spades // 3))

    def act(self, obs: dict, legal_actions: list[int], rng=None) -> int:
        if legal_actions and legal_actions[0] >= 52:
            bid = self.estimate_bid(obs["own_hand_mask"])
            action = bid + 51
            return action if action in legal_actions else legal_actions[0]
        return _rng_choice(rng, legal_actions)


class LowestLegalCardBot(SimpleBidBot):
    def act(self, obs: dict, legal_actions: list[int], rng=None) -> int:
        if legal_actions and legal_actions[0] < 52:
            return min(legal_actions)
        return super().act(obs, legal_actions, rng)


class HighestLegalCardBot(SimpleBidBot):
    def act(self, obs: dict, legal_actions: list[int], rng=None) -> int:
        if legal_actions and legal_actions[0] < 52:
            return max(legal_actions)
        return super().act(obs, legal_actions, rng)


class ConservativeBidBot(SimpleBidBot):
    """Avoids nil/blind nil for bounded random smoke rollouts."""

    def act(self, obs: dict, legal_actions: list[int], rng=None) -> int:
        non_nil = [
            action for action in legal_actions if action not in {NIL_ACTION, BLIND_NIL_ACTION}
        ]
        if non_nil and non_nil[0] >= 52:
            return super().act(obs, non_nil, rng)
        return super().act(obs, legal_actions, rng)
