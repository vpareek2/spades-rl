from argparse import Namespace

import numpy as np

from spades.actions import ACTION_SPACE_SIZE, NIL_ACTION
from spades.bid_ev import (
    collect_bidding_states,
    evaluate_candidate_action,
    generate_bid_ev_dataset,
    hidden_samples_for_state,
    make_ev_env,
    replay_previous_bids,
    summarize_bid_ev,
)
from spades.observations import FLAT_OBSERVATION_SIZE
from spades.state import Phase


def test_hidden_resampling_preserves_actor_hand_and_partitions_deck():
    state = collect_bidding_states(
        count=1,
        seed=10,
        state_bot_name="conservative",
        max_normal_bid=13,
        nil=True,
        blind_nil=True,
    )[0]

    samples = hidden_samples_for_state(state, samples=2, seed=99)

    for hands in samples:
        assert hands[state.acting_player] == state.hands[state.acting_player]
        cards = [card for hand in hands for card in hand]
        assert sorted(cards) == list(range(52))
        assert [len(hand) for hand in hands] == [13, 13, 13, 13]


def test_reconstructed_env_replays_previous_bids_to_source_player():
    state = collect_bidding_states(
        count=3,
        seed=20,
        state_bot_name="conservative",
        max_normal_bid=13,
        nil=True,
        blind_nil=True,
    )[2]
    hands = hidden_samples_for_state(state, samples=1, seed=7)[0]

    env = make_ev_env(
        dealer=state.dealer,
        hands=hands,
        team_scores=state.team_scores,
        team_bags=state.team_bags,
        nil=True,
        blind_nil=True,
    )
    replay_previous_bids(env, state.previous_bid_actions)

    assert env.phase == Phase.BIDDING
    assert env.current_player() == state.acting_player


def test_candidate_evaluation_uses_shared_hidden_samples():
    state = collect_bidding_states(
        count=1,
        seed=30,
        state_bot_name="conservative",
        max_normal_bid=13,
        nil=False,
        blind_nil=False,
    )[0]
    samples = hidden_samples_for_state(state, samples=2, seed=44)

    mean, std, visits = evaluate_candidate_action(
        state,
        state.legal_actions[0],
        samples,
        rollout_bot_name="conservative",
        max_normal_bid=13,
        nil=False,
        blind_nil=False,
        seed=123,
    )

    assert isinstance(mean, float)
    assert isinstance(std, float)
    assert visits == 2
    assert hidden_samples_for_state(state, samples=2, seed=44) == samples


def test_generate_bid_ev_dataset_npz_shapes_and_summary(tmp_path):
    output = tmp_path / "bid_ev.npz"

    summary = generate_bid_ev_dataset(
        Namespace(
            states=4,
            hidden_samples=2,
            seed=123,
            state_bot="conservative",
            rollout_bot="conservative",
            max_normal_bid=13,
            nil=False,
            blind_nil=False,
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
    assert not data["evaluated_mask"][:, NIL_ACTION].any()

    reloaded_summary = summarize_bid_ev(str(output))
    assert reloaded_summary["rows"] == 4
    assert reloaded_summary["metadata"]["hidden_samples"] == 2


def test_bid_ev_cli_smoke(tmp_path, monkeypatch):
    output = tmp_path / "bid_ev_cli.npz"
    monkeypatch.setattr(
        "sys.argv",
        [
            "spades-generate-bid-ev",
            "--states",
            "2",
            "--hidden-samples",
            "1",
            "--no-nil",
            "--no-blind-nil",
            "--output",
            str(output),
        ],
    )

    from spades.bid_ev import generate_main, summarize_main

    generate_main()
    assert output.exists()

    monkeypatch.setattr("sys.argv", ["spades-summarize-bid-ev", str(output)])
    summarize_main()
