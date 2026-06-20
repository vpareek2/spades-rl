import numpy as np
from gymnasium.utils.env_checker import check_env
from pettingzoo.test import api_test

from spades.bots import ConservativeBidBot, HighestLegalCardBot, LowestLegalCardBot
from spades.config import SpadesPlusConfig
from spades.env import SpadesPlusEnv
from spades.observations import FLAT_OBSERVATION_SIZE
from spades.rollout import play_match
from spades.wrappers.gymnasium_env import SpadesGymnasiumEnv
from spades.wrappers.pettingzoo_aec import SpadesAECEnv


def test_bot_match_terminates_deterministically():
    bots = [ConservativeBidBot(), LowestLegalCardBot(), ConservativeBidBot(), HighestLegalCardBot()]
    config = SpadesPlusConfig(first_dealer=0, target_score=60)
    env1 = SpadesPlusEnv(config)
    env2 = SpadesPlusEnv(config)

    result1 = play_match(env1, bots, seed=123, max_steps=2000)
    result2 = play_match(env2, bots, seed=123, max_steps=2000)

    assert result1["terminated"]
    assert result1["steps"] < 2000
    assert result1["winning_team"] in {0, 1}
    assert result1["team_scores"].tolist() == result2["team_scores"].tolist()
    assert result1["winning_team"] == result2["winning_team"]


def test_gymnasium_wrapper_flat_obs_and_mask():
    env = SpadesGymnasiumEnv(SpadesPlusConfig(first_dealer=0))
    obs, info = env.reset(seed=1)
    assert obs.shape == (FLAT_OBSERVATION_SIZE,)
    assert info["action_mask"].shape == (67,)
    action = int(np.flatnonzero(info["action_mask"])[0])
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (FLAT_OBSERVATION_SIZE,)
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated
    assert info["all_rewards"].shape == (4,)


def test_pettingzoo_aec_wrapper_smoke():
    env = SpadesAECEnv(SpadesPlusConfig(first_dealer=0))
    env.reset(seed=1)
    assert env.agent_selection == "player_1"
    obs = env.observe(env.agent_selection)
    assert obs.shape == (FLAT_OBSERVATION_SIZE,)
    action = 52
    env.step(action)
    assert env.agent_selection == "player_2"
    assert set(env.rewards) == set(env.possible_agents)


def test_gymnasium_wrapper_passes_env_checker():
    check_env(SpadesGymnasiumEnv(SpadesPlusConfig(first_dealer=0)), skip_render_check=True)


def test_pettingzoo_wrapper_passes_api_checker():
    api_test(
        SpadesAECEnv(SpadesPlusConfig(first_dealer=0)),
        num_cycles=200,
        verbose_progress=False,
    )
