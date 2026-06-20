import subprocess
import sys

import numpy as np
import pytest

from spades.config import SpadesPlusConfig
from spades.env import SpadesPlusEnv
from spades.state import Bid, BidKind, Phase


def test_debug_setup_apis_are_guarded(spade_sweep_hands):
    env = SpadesPlusEnv(SpadesPlusConfig(debug=False))
    with pytest.raises(RuntimeError):
        env.set_hands(spade_sweep_hands)


def test_set_hands_validation(spade_sweep_hands):
    env = SpadesPlusEnv(SpadesPlusConfig(debug=True))
    env.reset(seed=1)
    env.set_hands(spade_sweep_hands)
    assert env.get_debug_state()["hands"] == spade_sweep_hands

    duplicate = [hand[:] for hand in spade_sweep_hands]
    duplicate[0][0] = duplicate[1][0]
    with pytest.raises(ValueError):
        env.set_hands(duplicate)

    wrong_size = [hand[:] for hand in spade_sweep_hands]
    wrong_size[0] = wrong_size[0][:-1]
    with pytest.raises(ValueError):
        env.set_hands(wrong_size)


def test_set_scores_and_bids_validation(spade_sweep_hands):
    env = SpadesPlusEnv(SpadesPlusConfig(first_dealer=3, debug=True))
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.set_scores([0, 0, 0], [0, 0])
    with pytest.raises(ValueError):
        env.set_scores([0, 0], [-1, 0])
    with pytest.raises(ValueError):
        env.set_player_bids([Bid(BidKind.NORMAL, 1)])
    env.set_player_bids([Bid(BidKind.NORMAL, 1) for _ in range(4)])
    env.set_hands(spade_sweep_hands)
    env.start_playing_phase()
    assert env.phase == Phase.PLAYING
    assert env.current_player() == 0


def test_controlled_full_hand_scores_and_rotates(configured_debug_env, play_spade_sweep_hand):
    env = configured_debug_env()
    rewards, terminated, info = play_spade_sweep_hand(env)
    assert not terminated
    assert info["hand_completed"]
    assert info["player_tricks"].tolist() == [13, 0, 0, 0]
    assert info["team_score_delta"].tolist() == [-88, -20]
    assert rewards.tolist() == [-88, -20, -88, -20]
    assert info["team_scores"].tolist() == [-88, -20]
    assert env.phase == Phase.BIDDING
    assert env.dealer == 0
    assert env.current_player() == 1


def test_terminal_win_bonus_on_controlled_hand(configured_debug_env, play_spade_sweep_hand):
    env = configured_debug_env(team_scores=(338, 0), target_score=250)
    env.config.terminal_win_reward_enabled = True
    env.config.terminal_win_reward = 1.5
    rewards, terminated, info = play_spade_sweep_hand(env)
    assert terminated
    assert info["game_completed"] if "game_completed" in info else True
    assert env.winning_team == 0
    assert env.phase == Phase.GAME_OVER
    np.testing.assert_allclose(rewards, np.array([-86.5, -21.5, -86.5, -21.5]))


def test_both_teams_reach_target_higher_score_wins(
    configured_debug_env, play_spade_sweep_hand
):
    env = configured_debug_env(team_scores=(340, 270), target_score=250)
    _, terminated, _ = play_spade_sweep_hand(env)
    assert terminated
    assert env.team_scores.tolist() == [252, 250]
    assert env.winning_team == 0


def test_tied_at_target_continues(configured_debug_env, play_spade_sweep_hand):
    env = configured_debug_env(team_scores=(340, 272), target_score=250)
    _, terminated, info = play_spade_sweep_hand(env)
    assert not terminated
    assert info["team_scores"].tolist() == [252, 252]
    assert env.winning_team is None
    assert env.phase == Phase.BIDDING


def test_history_contains_full_hand_events(configured_debug_env, play_spade_sweep_hand):
    env = configured_debug_env()
    play_spade_sweep_hand(env)
    event_types = [event.type for event in env.get_debug_state()["history"]]
    assert event_types.count("play") == 52
    assert event_types.count("trick_end") == 13
    assert event_types[-1] == "hand_end"


def test_scripted_repl_smoke():
    proc = subprocess.run(
        [sys.executable, "-m", "spades.repl"],
        input="legal\nbid 1\nscores\nquit\n",
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "Spades REPL" in proc.stdout
    assert "scores=" in proc.stdout


def test_repl_illegal_command_reports_error():
    proc = subprocess.run(
        [sys.executable, "-m", "spades.repl"],
        input="play 0\nquit\n",
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "error:" in proc.stdout
