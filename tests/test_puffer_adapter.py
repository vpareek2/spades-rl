import ctypes

import numpy as np
import torch

from spades.actions import ACTION_SPACE_SIZE
from spades.observations import FLAT_OBSERVATION_SIZE
from spades.puffer import MASK_OFFSET, MaskedSpadesPolicy, SpadesPufferConfig, SpadesPufferVecEnv


def test_puffer_vec_env_cpu_step_shapes_and_masks():
    vec = SpadesPufferVecEnv(SpadesPufferConfig(num_envs=2, seed=3))

    assert vec.total_agents == 8
    assert vec._obs.shape == (8, FLAT_OBSERVATION_SIZE)
    assert vec._rewards.shape == (8,)
    assert vec._terminals.shape == (8,)

    active_masks = 0
    inactive_masks = 0
    actions = np.zeros((vec.total_agents, 1), dtype=np.float32)
    for env_idx, env in enumerate(vec.envs):
        active = env.current_player()
        for player in range(4):
            slot = env_idx * 4 + player
            mask = vec._obs[slot, MASK_OFFSET:]
            if player == active:
                active_masks += 1
                legal = np.flatnonzero(mask)
                assert legal.tolist() == env.legal_actions()
                actions[slot, 0] = legal[0]
            else:
                inactive_masks += 1
                assert np.flatnonzero(mask).tolist() == [0]

    assert active_masks == 2
    assert inactive_masks == 6

    ptr = actions.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    vec.cpu_step(ctypes.addressof(ptr.contents))

    assert vec._obs.shape == (8, FLAT_OBSERVATION_SIZE)
    assert np.isfinite(vec._rewards).all()
    assert np.isfinite(vec._terminals).all()
    assert vec.illegal_actions == 0


def test_masked_spades_policy_masks_invalid_logits():
    policy = MaskedSpadesPolicy(FLAT_OBSERVATION_SIZE, ACTION_SPACE_SIZE, hidden_size=32, num_layers=1)
    obs = torch.zeros(2, FLAT_OBSERVATION_SIZE)
    obs[0, MASK_OFFSET + 52] = 1.0
    obs[0, MASK_OFFSET + 65] = 1.0
    obs[1, MASK_OFFSET + 0] = 1.0

    logits, values, state = policy.forward_eval(obs, ())

    assert state == ()
    assert logits.shape == (2, ACTION_SPACE_SIZE)
    assert values.shape == (2, 1)
    assert logits[0, 52] > -1.0e8
    assert logits[0, 65] > -1.0e8
    assert logits[0, 0] < -1.0e8
    assert logits[1, 0] > -1.0e8
    assert logits[1, 52] < -1.0e8
