import numpy as np
import pytest

from spades.config import SpadesPlusConfig
from spades.env import SpadesPlusEnv
from spades.state import Bid, BidKind


@pytest.fixture
def spade_sweep_hands():
    return [
        list(range(39, 52)),
        list(range(0, 13)),
        list(range(13, 26)),
        list(range(26, 39)),
    ]


@pytest.fixture
def configured_debug_env(spade_sweep_hands):
    def make(team_scores=(0, 0), team_bags=(0, 0), target_score=250):
        env = SpadesPlusEnv(
            SpadesPlusConfig(first_dealer=3, target_score=target_score, debug=True)
        )
        env.reset(seed=1)
        env.set_hands(spade_sweep_hands)
        env.set_scores(np.array(team_scores), np.array(team_bags))
        env.set_player_bids([Bid(BidKind.NORMAL, 1) for _ in range(4)])
        env.start_playing_phase()
        return env

    return make


@pytest.fixture
def play_spade_sweep_hand():
    def play(env: SpadesPlusEnv):
        final_info = {}
        final_rewards = None
        final_terminated = False
        while env.phase.value == "playing":
            action = min(env.legal_actions())
            _, final_rewards, final_terminated, _, final_info = env.step(action)
        return final_rewards, final_terminated, final_info

    return play
