from argparse import Namespace

import numpy as np
import torch

from spades.actions import ACTION_SPACE_SIZE
from spades.observations import FLAT_OBSERVATION_SIZE
from spades.play_ev_train import (
    PLAY_ACTION_COUNT,
    eval_play_ev_checkpoint,
    load_play_ev_dataset,
    train_play_ev,
)
from spades.policy import SpadesTransformerPolicy


def write_tiny_play_ev_npz(path, rows: int = 6) -> None:
    observations = np.zeros((rows, FLAT_OBSERVATION_SIZE), dtype=np.float32)
    legal_mask = np.zeros((rows, ACTION_SPACE_SIZE), dtype=np.bool_)
    evaluated_mask = np.zeros((rows, ACTION_SPACE_SIZE), dtype=np.bool_)
    q_values = np.zeros((rows, ACTION_SPACE_SIZE), dtype=np.float32)
    q_std = np.zeros((rows, ACTION_SPACE_SIZE), dtype=np.float32)
    visit_counts = np.zeros((rows, ACTION_SPACE_SIZE), dtype=np.int32)
    best_action = np.zeros(rows, dtype=np.int16)
    acting_player = np.zeros(rows, dtype=np.int8)
    dealer = np.zeros(rows, dtype=np.int8)
    previous_bid_actions = np.full((rows, 4), -1, dtype=np.int16)
    previous_play_actions = np.full((rows, 52), -1, dtype=np.int16)

    for row in range(rows):
        actions = [0, 13, 26]
        legal_mask[row, actions] = True
        evaluated_mask[row, actions] = True
        observations[row, -ACTION_SPACE_SIZE + np.asarray(actions)] = 1.0
        q_values[row, 0] = -5.0 + row
        q_values[row, 13] = 15.0 + row
        q_values[row, 26] = 2.0 + row
        q_std[row, actions] = 1.0
        visit_counts[row, actions] = 2
        best_action[row] = 13

    np.savez_compressed(
        path,
        observations=observations,
        legal_mask=legal_mask,
        evaluated_mask=evaluated_mask,
        q_values=q_values,
        q_std=q_std,
        visit_counts=visit_counts,
        best_action=best_action,
        acting_player=acting_player,
        dealer=dealer,
        previous_bid_actions=previous_bid_actions,
        previous_play_actions=previous_play_actions,
        metadata_json=np.asarray("{}"),
    )


def _train_args(dataset_path, save_path, metrics_path):
    return Namespace(
        dataset=str(dataset_path),
        load_path="",
        save_path=str(save_path),
        metrics_path=str(metrics_path),
        epochs=1,
        batch_size=3,
        learning_rate=1e-4,
        weight_decay=0.0,
        val_fraction=0.5,
        seed=0,
        policy_temperature=20.0,
        policy_weight=1.0,
        rank_weight=0.1,
        rank_gap=5.0,
        rank_margin=0.05,
        max_grad_norm=1.0,
        d_model=32,
        transformer_layers=1,
        attention_heads=4,
        ffn_size=64,
        dropout=0.0,
        cpu=True,
        amp=False,
        freeze_encoder=False,
        freeze_bid_heads=False,
        play_heads_only=True,
        progress=False,
        wandb=False,
        wandb_project="spades-rl",
        wandb_group="play-ev",
        wandb_run_name="",
        wandb_mode="disabled",
        wandb_artifact_name="",
    )


def test_load_play_ev_dataset_maps_card_actions(tmp_path):
    path = tmp_path / "tiny_play_ev.npz"
    write_tiny_play_ev_npz(path)

    dataset = load_play_ev_dataset(str(path))

    assert dataset.observations.shape == (6, FLAT_OBSERVATION_SIZE)
    assert dataset.q_values.shape == (6, PLAY_ACTION_COUNT)
    assert dataset.evaluated_mask[:, 13].all()
    assert dataset.best_action.tolist() == [13] * 6


def test_train_play_ev_writes_checkpoint_and_metrics(tmp_path):
    dataset_path = tmp_path / "tiny_play_ev.npz"
    save_path = tmp_path / "play_ev.pt"
    metrics_path = tmp_path / "metrics.json"
    write_tiny_play_ev_npz(dataset_path)

    metrics = train_play_ev(_train_args(dataset_path, save_path, metrics_path))

    assert save_path.exists()
    assert metrics_path.exists()
    assert metrics["rows"] == 6
    assert metrics["train_rows"] == 3
    assert metrics["val_rows"] == 3
    assert "policy_regret" in metrics["history"][0]["val"]


def test_play_ev_train_cli_smoke(tmp_path, monkeypatch):
    dataset_path = tmp_path / "tiny_play_ev_cli.npz"
    save_path = tmp_path / "play_ev_cli.pt"
    write_tiny_play_ev_npz(dataset_path, rows=4)
    monkeypatch.setattr(
        "sys.argv",
        [
            "spades-train-play-ev",
            "--dataset",
            str(dataset_path),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--d-model",
            "32",
            "--transformer-layers",
            "1",
            "--attention-heads",
            "4",
            "--ffn-size",
            "64",
            "--dropout",
            "0.0",
            "--cpu",
            "--play-heads-only",
            "--no-progress",
            "--wandb",
            "--wandb-mode",
            "disabled",
            "--save-path",
            str(save_path),
        ],
    )

    from spades.play_ev_train import main

    main()
    assert save_path.exists()


def test_eval_play_ev_checkpoint_writes_metrics(tmp_path):
    dataset_path = tmp_path / "tiny_play_ev_eval.npz"
    checkpoint_path = tmp_path / "policy.pt"
    output_path = tmp_path / "eval.json"
    write_tiny_play_ev_npz(dataset_path, rows=4)
    policy = SpadesTransformerPolicy(
        FLAT_OBSERVATION_SIZE,
        ACTION_SPACE_SIZE,
        d_model=32,
        num_layers=1,
        num_heads=4,
        ffn_size=64,
        dropout=0.0,
    )
    torch.save(policy.state_dict(), checkpoint_path)

    metrics = eval_play_ev_checkpoint(
        Namespace(
            checkpoint=str(checkpoint_path),
            dataset=str(dataset_path),
            batch_size=2,
            policy_temperature=20.0,
            policy_weight=1.0,
            rank_weight=0.1,
            rank_gap=5.0,
            rank_margin=0.05,
            d_model=32,
            transformer_layers=1,
            attention_heads=4,
            ffn_size=64,
            dropout=0.0,
            cpu=True,
            output=str(output_path),
        )
    )

    assert output_path.exists()
    assert metrics["rows"] == 4
    assert metrics["checkpoint"] == str(checkpoint_path)
    assert "policy_regret" in metrics


def test_play_ev_eval_cli_smoke(tmp_path, monkeypatch):
    dataset_path = tmp_path / "tiny_play_ev_eval_cli.npz"
    checkpoint_path = tmp_path / "policy_cli.pt"
    write_tiny_play_ev_npz(dataset_path, rows=4)
    policy = SpadesTransformerPolicy(
        FLAT_OBSERVATION_SIZE,
        ACTION_SPACE_SIZE,
        d_model=32,
        num_layers=1,
        num_heads=4,
        ffn_size=64,
        dropout=0.0,
    )
    torch.save(policy.state_dict(), checkpoint_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "spades-eval-play-ev",
            str(checkpoint_path),
            "--dataset",
            str(dataset_path),
            "--batch-size",
            "2",
            "--d-model",
            "32",
            "--transformer-layers",
            "1",
            "--attention-heads",
            "4",
            "--ffn-size",
            "64",
            "--dropout",
            "0.0",
            "--cpu",
        ],
    )

    from spades.play_ev_train import eval_main

    eval_main()
