import numpy as np
import pytest

from spades.actions import ACTION_SPACE_SIZE, BLIND_NIL_ACTION, NIL_ACTION
from spades.config import SpadesPlusConfig
from spades.env import IllegalActionError, SpadesPlusEnv
from spades.observations import FLAT_OBSERVATION_SIZE, flatten_observation
from spades.rules import game_result
from spades.state import BID_BLIND_NIL, BID_NIL, BID_NORMAL, Phase


def bid_four_players(env: SpadesPlusEnv, actions=(52, 53, NIL_ACTION, BLIND_NIL_ACTION)):
    for action in actions:
        env.step(action)


def play_first_legal_trick(env: SpadesPlusEnv):
    info = {}
    for _ in range(4):
        action = env.legal_actions()[0]
        _, _, _, _, info = env.step(action)
        if info.get("trick_completed"):
            return info
    return info


def test_deal_is_complete_and_seeded():
    env = SpadesPlusEnv(SpadesPlusConfig(first_dealer=3))
    env.reset(seed=123)
    debug = env.get_debug_state()
    cards = [card for hand in debug["hands"] for card in hand]
    assert [len(hand) for hand in debug["hands"]] == [13, 13, 13, 13]
    assert len(cards) == 52
    assert len(set(cards)) == 52

    env2 = SpadesPlusEnv(SpadesPlusConfig(first_dealer=3))
    env2.reset(seed=123)
    assert env2.get_debug_state()["hands"] == debug["hands"]


def test_bidding_order_and_transition_to_playing():
    env = SpadesPlusEnv(SpadesPlusConfig(first_dealer=3))
    env.reset(seed=7)
    assert env.current_player() == 0
    assert env.leader == 0
    assert env.phase == Phase.BIDDING
    assert env.legal_actions() == list(range(52, 67))
    assert env.action_mask().shape == (ACTION_SPACE_SIZE,)
    assert np.flatnonzero(env.action_mask()).tolist() == list(range(52, 67))

    env.step(52)
    assert env.current_player() == 1
    env.step(53)
    assert env.current_player() == 2
    env.step(NIL_ACTION)
    assert env.current_player() == 3
    env.step(BLIND_NIL_ACTION)
    assert env.phase == Phase.PLAYING
    assert env.current_player() == 0
    assert env.leader == 0

    debug = env.get_debug_state()
    assert [bid.kind for bid in debug["player_bids"] if bid is not None]


def test_config_masks_nil_and_blind_nil():
    env = SpadesPlusEnv(
        SpadesPlusConfig(first_dealer=0, nil_enabled=False, blind_nil_enabled=False)
    )
    env.reset(seed=1)
    assert NIL_ACTION not in env.legal_actions()
    assert BLIND_NIL_ACTION not in env.legal_actions()


def test_invalid_v1_config_rejected():
    with pytest.raises(ValueError):
        SpadesPlusConfig(illegal_action_mode="ignore")
    with pytest.raises(ValueError):
        SpadesPlusConfig(blind_nil_policy="behind_100")


def test_observation_is_partial_and_seat_relative():
    env = SpadesPlusEnv(SpadesPlusConfig(first_dealer=1))
    obs = env.reset(seed=11)
    acting = env.current_player()
    debug = env.get_debug_state()
    assert obs["current_player_rel"] == 0
    assert obs["dealer_rel"] == 3
    assert obs["leader_rel"] == 0
    assert obs["own_hand_mask"].sum() == 13
    for card in debug["hands"][acting]:
        assert obs["own_hand_mask"][card]
    hidden_cards = set(debug["hands"][(acting + 1) % 4])
    assert not any(obs["own_hand_mask"][card] for card in hidden_cards)
    assert obs["action_mask"].shape == (67,)
    flat = flatten_observation(obs)
    assert flat.shape == (FLAT_OBSERVATION_SIZE,)
    assert flat.dtype == np.float32


def test_absolute_observation_mode():
    env = SpadesPlusEnv(SpadesPlusConfig(first_dealer=1, seat_relative_observation=False))
    obs = env.reset(seed=11)
    assert obs["current_player_rel"] == env.current_player()
    assert obs["dealer_rel"] == 1
    assert obs["leader_rel"] == 2


def test_bids_visible_by_relative_seat_after_made():
    env = SpadesPlusEnv(SpadesPlusConfig(first_dealer=3))
    env.reset(seed=5)
    env.step(52)
    obs_for_p1 = env.observe(1)
    assert obs_for_p1["bid_kind"][3] == BID_NORMAL
    assert obs_for_p1["bid_value"][3] == 1
    env.step(NIL_ACTION)
    obs_for_p2 = env.observe(2)
    assert obs_for_p2["bid_kind"][3] == BID_NIL
    env.step(BLIND_NIL_ACTION)
    obs_for_p3 = env.observe(3)
    assert obs_for_p3["bid_kind"][3] == BID_BLIND_NIL


def test_play_mask_trick_completion_and_played_card_observation():
    env = SpadesPlusEnv(SpadesPlusConfig(first_dealer=3))
    env.reset(seed=42)
    bid_four_players(env, (52, 52, 52, 52))
    first_player = env.current_player()
    legal = env.legal_actions()
    assert all(action < 52 for action in legal)
    assert np.flatnonzero(env.action_mask()).tolist() == legal
    first_card = legal[0]
    env.step(first_card)
    obs = env.observe(env.current_player())
    assert obs["played_cards_mask"][first_card]
    info = play_first_legal_trick(env)
    assert info["trick_completed"]
    assert env.get_debug_state()["current_trick"] == []
    assert env.current_player() == info["trick_winner"]
    assert first_player in range(4)


def test_illegal_actions_raise():
    env = SpadesPlusEnv(SpadesPlusConfig(first_dealer=0))
    env.reset(seed=1)
    with pytest.raises(IllegalActionError):
        env.step(0)
    bid_four_players(env, (52, 52, 52, 52))
    with pytest.raises(IllegalActionError):
        env.step(52)


def test_game_end_logic_public_helper():
    assert game_result([249, 10], 250) == (False, None)
    assert game_result([250, 10], 250) == (True, 0)
    assert game_result([260, 270], 250) == (True, 1)
    assert game_result([260, 260], 250) == (False, None)


def test_deterministic_trajectory_with_same_actions():
    actions = [52, 53, 54, 55]
    traces = []
    for _ in range(2):
        env = SpadesPlusEnv(SpadesPlusConfig(first_dealer=2))
        env.reset(seed=99)
        trace = []
        for action in actions:
            env.step(action)
            trace.append((env.phase.value, env.current_player()))
        for _ in range(8):
            action = env.legal_actions()[0]
            _, rewards, terminated, _, info = env.step(action)
            trace.append((action, tuple(rewards.tolist()), terminated, info.get("trick_winner")))
        traces.append(trace)
    assert traces[0] == traces[1]


def test_debug_state_returns_copies_and_history_events():
    env = SpadesPlusEnv(SpadesPlusConfig(first_dealer=0))
    env.reset(seed=3)
    debug = env.get_debug_state()
    debug["hands"][env.current_player()].clear()
    assert len(env.get_debug_state()["hands"][env.current_player()]) == 13
    env.step(52)
    assert env.get_debug_state()["history"][-1].type == "bid"
