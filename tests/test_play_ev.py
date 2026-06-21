from argparse import Namespace

import numpy as np

from spades.actions import ACTION_SPACE_SIZE
from spades.eval import make_actor
from spades.observations import FLAT_OBSERVATION_SIZE
from spades.play_ev import (
    collect_play_states,
    evaluate_play_candidate_action,
    generate_play_ev_dataset,
    make_play_ev_env,
    replay_previous_plays,
    summarize_play_ev,
)
from spades.state import Phase


def _bot_actor(name: str = "bot:conservative"):
    return make_actor(
        name,
        label="test",
        max_normal_bid=13,
        d_model=32,
        transformer_layers=1,
        attention_heads=4,
        ffn_size=64,
        dropout=0.0,
        device="cpu",
    )


def test_reconstructed_play_env_matches_source_state():
    actor = _bot_actor()
    state = collect_play_states(
        count=5,
        seed=10,
        state_actor=actor,
        max_normal_bid=13,
        nil=False,
        blind_nil=False,
        progress=False,
    )[-1]

    env = make_play_ev_env(state=state, nil=False, blind_nil=False)

    assert env.phase == Phase.PLAYING
    assert env.current_player() == state.acting_player
    assert env.legal_actions() == state.legal_actions


def test_replay_previous_plays_rejects_hand_completion():
    actor = _bot_actor()
    state = collect_play_states(
        count=52,
        seed=11,
        state_actor=actor,
        max_normal_bid=13,
        nil=False,
        blind_nil=False,
        progress=False,
    )[-1]
    env = make_play_ev_env(state=state, nil=False, blind_nil=False)

    replay_previous_plays(env, [])
    assert env.phase == Phase.PLAYING


def test_play_candidate_evaluation_uses_rollout_samples():
    actor = _bot_actor()
    state = collect_play_states(
        count=1,
        seed=12,
        state_actor=actor,
        max_normal_bid=13,
        nil=False,
        blind_nil=False,
        progress=False,
    )[0]

    mean, std, visits = evaluate_play_candidate_action(
        state,
        state.legal_actions[0],
        actor,
        rollout_samples=2,
        max_normal_bid=13,
        nil=False,
        blind_nil=False,
        seed=123,
    )

    assert isinstance(mean, float)
    assert isinstance(std, float)
    assert visits == 2


def test_generate_play_ev_dataset_npz_shapes_and_summary(tmp_path):
    output = tmp_path / "play_ev.npz"

    summary = generate_play_ev_dataset(
        Namespace(
            states=4,
            rollout_samples=2,
            seed=123,
            state_actor="bot:conservative",
            rollout_actor="bot:conservative",
            max_normal_bid=13,
            nil=False,
            blind_nil=False,
            d_model=32,
            transformer_layers=1,
            attention_heads=4,
            ffn_size=64,
            dropout=0.0,
            device="cpu",
            progress=False,
            output=str(output),
        )
    )

    assert summary["rows"] == 4
    assert summary["valid_rows"] == 4
    assert output.exists()

    data = np.load(output, allow_pickle=False)
    assert data["observations"].shape == (4, FLAT_OBSERVATION_SIZE)
    assert data["observations"].dtype == np.float32
    assert data["legal_mask"].shape == (4, ACTION_SPACE_SIZE)
    assert data["evaluated_mask"].shape == (4, ACTION_SPACE_SIZE)
    assert data["q_values"].shape == (4, ACTION_SPACE_SIZE)
    assert data["q_std"].shape == (4, ACTION_SPACE_SIZE)
    assert data["visit_counts"].shape == (4, ACTION_SPACE_SIZE)
    assert data["best_action"].shape == (4,)
    assert data["acting_player"].shape == (4,)
    assert data["dealer"].shape == (4,)
    assert data["previous_bid_actions"].shape == (4, 4)
    assert data["previous_play_actions"].shape == (4, 52)
    assert data["evaluated_mask"][:, :52].any()
    assert not data["evaluated_mask"][:, 52:].any()

    reloaded_summary = summarize_play_ev(str(output))
    assert reloaded_summary["rows"] == 4
    assert reloaded_summary["metadata"]["rollout_samples"] == 2


def test_play_ev_cli_smoke(tmp_path, monkeypatch):
    output = tmp_path / "play_ev_cli.npz"
    monkeypatch.setattr(
        "sys.argv",
        [
            "spades-generate-play-ev",
            "--states",
            "2",
            "--rollout-samples",
            "1",
            "--no-nil",
            "--no-blind-nil",
            "--no-progress",
            "--output",
            str(output),
        ],
    )

    from spades.play_ev import generate_main, summarize_main

    generate_main()
    assert output.exists()

    monkeypatch.setattr("sys.argv", ["spades-summarize-play-ev", str(output)])
    summarize_main()
