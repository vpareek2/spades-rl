import ctypes
from argparse import Namespace

import numpy as np
import pytest
import torch

from spades.actions import ACTION_SPACE_SIZE
from spades.observations import FLAT_OBSERVATION_SIZE
from spades.policy import (
    CARD_TOKEN_COUNT,
    EVENT_TOKEN_COUNT,
    SEAT_TOKEN_COUNT,
    TOKEN_COUNT,
    FlatObservationTokenizer,
    SpadesTransformerPolicy,
    load_transformer_state,
)
from spades.puffer import (
    MASK_OFFSET,
    _decision_phase_weights,
    SpadesPufferConfig,
    SpadesPufferVecEnv,
    attach_log_history,
    bid_anchor_losses,
    freeze_bid_heads,
    pretrain_bidding,
)


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


def test_puffer_vec_env_can_limit_bidding_curriculum():
    vec = SpadesPufferVecEnv(
        SpadesPufferConfig(
            num_envs=1,
            seed=7,
            max_normal_bid=5,
            nil_enabled=False,
            blind_nil_enabled=False,
        )
    )

    env = vec.envs[0]
    active = env.current_player()
    for player in range(4):
        slot = player
        mask = vec._obs[slot, MASK_OFFSET:]
        if player == active:
            assert np.flatnonzero(mask).tolist() == [52, 53, 54, 55, 56]
        else:
            assert np.flatnonzero(mask).tolist() == [0]


def test_transformer_tokenizer_shapes():
    vec = SpadesPufferVecEnv(SpadesPufferConfig(num_envs=1, seed=9))
    tokenizer = FlatObservationTokenizer(d_model=32, dropout=0.0)
    obs = torch.from_numpy(vec._obs[:1])

    tokenized = tokenizer(obs)

    assert tokenized["tokens"].shape == (1, TOKEN_COUNT, 32)
    assert tokenized["card_token_slice"].stop - tokenized["card_token_slice"].start == CARD_TOKEN_COUNT
    assert tokenized["seat_token_slice"].stop - tokenized["seat_token_slice"].start == SEAT_TOKEN_COUNT
    assert TOKEN_COUNT == 1 + CARD_TOKEN_COUNT + EVENT_TOKEN_COUNT + SEAT_TOKEN_COUNT


def test_spades_transformer_policy_masks_invalid_logits_and_exposes_heads():
    policy = SpadesTransformerPolicy(
        FLAT_OBSERVATION_SIZE,
        ACTION_SPACE_SIZE,
        d_model=32,
        num_layers=1,
        num_heads=4,
        ffn_size=64,
        dropout=0.0,
    )
    obs = torch.zeros(2, FLAT_OBSERVATION_SIZE)
    obs[0, MASK_OFFSET + 52] = 1.0
    obs[0, MASK_OFFSET + 65] = 1.0
    obs[1, MASK_OFFSET + 0] = 1.0

    logits, values, state = policy.forward_eval(obs, ())
    heads = policy.forward_heads(obs)

    assert state == ()
    assert logits.shape == (2, ACTION_SPACE_SIZE)
    assert values.shape == (2, 1)
    assert heads["bid_q"].shape == (2, 15)
    assert heads["margin_value"].shape == (2, 1)
    assert heads["hidden_owner_logits"].shape == (2, 52, 4)
    assert heads["void_logits"].shape == (2, 4, 4)
    assert logits[0, 52] > -1.0e8
    assert logits[0, 65] > -1.0e8
    assert logits[0, 0] < -1.0e8
    assert logits[1, 0] > -1.0e8
    assert logits[1, 52] < -1.0e8


def test_phase_weights_can_ignore_inactive_and_bid_rows():
    obs = torch.zeros(4, FLAT_OBSERVATION_SIZE)
    obs[0, 0] = 0.0
    obs[0, 1] = 0.0
    obs[1, 0] = 1.0
    obs[1, 1] = 0.0
    obs[2, 0] = 0.0
    obs[2, 1] = 2.0
    obs[3, 0] = 1.0
    obs[3, 1] = 3.0

    weights, active, bidding, playing = _decision_phase_weights(
        obs,
        active_only_loss=True,
        bid_ppo_weight=0.0,
        play_ppo_weight=1.0,
    )

    assert active.tolist() == [True, True, False, False]
    assert bidding.tolist() == [True, False, True, False]
    assert playing.tolist() == [False, True, False, True]
    assert weights.tolist() == [0.0, 1.0, 0.0, 0.0]


def test_freeze_bid_heads_keeps_other_modules_trainable():
    policy = SpadesTransformerPolicy(
        FLAT_OBSERVATION_SIZE,
        ACTION_SPACE_SIZE,
        d_model=32,
        num_layers=1,
        num_heads=4,
        ffn_size=64,
        dropout=0.0,
    )

    freeze_bid_heads(policy)

    assert not any(param.requires_grad for param in policy.bid_policy_head.parameters())
    assert not any(param.requires_grad for param in policy.bid_q_head.parameters())
    assert any(param.requires_grad for param in policy.transformer.parameters())
    assert any(param.requires_grad for param in policy.play_query.parameters())


def test_bid_anchor_loss_matches_identical_teacher_and_ignores_play_rows():
    policy = SpadesTransformerPolicy(
        FLAT_OBSERVATION_SIZE,
        ACTION_SPACE_SIZE,
        d_model=32,
        num_layers=1,
        num_heads=4,
        ffn_size=64,
        dropout=0.0,
    )
    teacher = SpadesTransformerPolicy(
        FLAT_OBSERVATION_SIZE,
        ACTION_SPACE_SIZE,
        d_model=32,
        num_layers=1,
        num_heads=4,
        ffn_size=64,
        dropout=0.0,
    )
    teacher.load_state_dict(policy.state_dict())
    policy.eval()
    teacher.eval()

    obs = torch.zeros(3, FLAT_OBSERVATION_SIZE)
    obs[0, 0] = 0.0
    obs[0, 1] = 0.0
    obs[0, MASK_OFFSET + 52 : MASK_OFFSET + 55] = 1.0
    obs[1, 0] = 1.0
    obs[1, 1] = 0.0
    obs[1, MASK_OFFSET: MASK_OFFSET + 3] = 1.0
    obs[2, 0] = 0.0
    obs[2, 1] = 1.0
    obs[2, MASK_OFFSET + 52 : MASK_OFFSET + 55] = 1.0

    policy_kl, q_anchor, rows = bid_anchor_losses(policy, teacher, obs, temperature=1.0)

    assert rows == 1
    assert float(policy_kl.item()) < 1e-6
    assert float(q_anchor.item()) < 1e-6


def test_attach_log_history_avoids_circular_final_log_reference():
    final = {"epoch": 2}
    history = [{"epoch": 1}, final]

    metrics = attach_log_history(final, history)

    assert metrics is not final
    assert metrics["history"][1] is not final
    assert metrics["history"][1] == {"epoch": 2}


def test_transformer_checkpoint_loader_rejects_mlp_state(tmp_path):
    checkpoint = tmp_path / "old_mlp.pt"
    torch.save({"encoder.0.weight": torch.zeros(1)}, checkpoint)
    policy = SpadesTransformerPolicy(
        FLAT_OBSERVATION_SIZE,
        ACTION_SPACE_SIZE,
        d_model=32,
        num_layers=1,
        num_heads=4,
        ffn_size=64,
        dropout=0.0,
    )

    with pytest.raises(RuntimeError, match="Pre-transformer MLP checkpoints"):
        load_transformer_state(policy, str(checkpoint), "cpu")


def test_puffer_bidding_pretrain_smoke(tmp_path):
    save_path = tmp_path / "pretrain.pt"
    metrics_path = tmp_path / "pretrain.json"

    metrics = pretrain_bidding(
        Namespace(
            samples=16,
            batch_size=8,
            learning_rate=0.001,
            d_model=32,
            transformer_layers=1,
            attention_heads=4,
            ffn_size=64,
            dropout=0.0,
            seed=5,
            max_normal_bid=5,
            nil=False,
            blind_nil=False,
            load_path="",
            save_path=str(save_path),
            metrics_path=str(metrics_path),
            cpu=True,
            freeze_encoder=False,
            log_interval=99,
        )
    )

    assert metrics["samples"] == 16
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert save_path.exists()
    assert metrics_path.exists()
