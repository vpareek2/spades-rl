"""Supervised training for rollout-EV play labels."""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from spades.actions import ACTION_SPACE_SIZE
from spades.observations import FLAT_OBSERVATION_SIZE
from spades.policy import SpadesTransformerPolicy, load_transformer_state


PLAY_ACTION_COUNT = 52


@dataclass(frozen=True)
class PlayEVDataset:
    observations: torch.Tensor
    q_values: torch.Tensor
    evaluated_mask: torch.Tensor
    legal_mask: torch.Tensor
    best_action: torch.Tensor

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.observations,
            self.q_values,
            self.evaluated_mask,
            self.legal_mask,
            self.best_action,
        )


@dataclass
class PlayEVLossMetrics:
    loss: float
    policy_loss: float
    rank_loss: float
    policy_top1: float
    policy_regret: float
    mean_best_ev: float
    mean_policy_ev: float
    rows: int


def _play_slice(array: np.ndarray) -> np.ndarray:
    return array[:, :PLAY_ACTION_COUNT]


def load_play_ev_dataset(path: str) -> PlayEVDataset:
    data = np.load(path, allow_pickle=False)
    observations = np.asarray(data["observations"], dtype=np.float32)
    q_values = _play_slice(np.asarray(data["q_values"], dtype=np.float32))
    evaluated_mask = _play_slice(np.asarray(data["evaluated_mask"], dtype=np.bool_))
    legal_mask = _play_slice(np.asarray(data["legal_mask"], dtype=np.bool_))
    best_action = np.asarray(data["best_action"], dtype=np.int64)

    if observations.ndim != 2 or observations.shape[1] != FLAT_OBSERVATION_SIZE:
        raise ValueError(f"Expected observations with shape (N, {FLAT_OBSERVATION_SIZE})")
    if q_values.shape != (observations.shape[0], PLAY_ACTION_COUNT):
        raise ValueError(f"Expected q_values play slice with shape (N, {PLAY_ACTION_COUNT})")
    if evaluated_mask.shape != q_values.shape or legal_mask.shape != q_values.shape:
        raise ValueError("Mask shapes must match q_values play slice shape")

    valid_rows = evaluated_mask.any(axis=1) & (best_action >= 0) & (best_action < PLAY_ACTION_COUNT)
    if not np.any(valid_rows):
        raise ValueError("Dataset has no rows with evaluated play actions")

    return PlayEVDataset(
        observations=torch.from_numpy(observations[valid_rows]),
        q_values=torch.from_numpy(q_values[valid_rows]),
        evaluated_mask=torch.from_numpy(evaluated_mask[valid_rows]),
        legal_mask=torch.from_numpy(legal_mask[valid_rows]),
        best_action=torch.from_numpy(best_action[valid_rows]),
    )


def split_dataset(
    dataset: PlayEVDataset,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[PlayEVDataset, PlayEVDataset | None]:
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    rows = len(dataset.observations)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(rows, generator=generator)
    val_rows = int(round(rows * val_fraction))
    if rows > 1 and val_fraction > 0:
        val_rows = max(1, min(rows - 1, val_rows))
    train_idx = indices[val_rows:]
    val_idx = indices[:val_rows]

    def take(idx: torch.Tensor) -> PlayEVDataset:
        return PlayEVDataset(
            observations=dataset.observations[idx],
            q_values=dataset.q_values[idx],
            evaluated_mask=dataset.evaluated_mask[idx],
            legal_mask=dataset.legal_mask[idx],
            best_action=dataset.best_action[idx],
        )

    return take(train_idx), take(val_idx) if val_rows else None


def build_play_ev_policy(args: argparse.Namespace, device: str | torch.device) -> SpadesTransformerPolicy:
    return SpadesTransformerPolicy(
        obs_size=FLAT_OBSERVATION_SIZE,
        action_size=ACTION_SPACE_SIZE,
        d_model=args.d_model,
        num_layers=args.transformer_layers,
        num_heads=args.attention_heads,
        ffn_size=args.ffn_size,
        dropout=args.dropout,
    ).to(device)


def _masked_play_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, -1.0e9)


def _losses(
    policy: SpadesTransformerPolicy,
    batch: tuple[torch.Tensor, ...],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    observations, target_q, evaluated_mask, _legal_mask, _best_action = batch
    heads = policy.forward_heads(observations, apply_mask=False)
    play_logits = heads["logits"][:, :PLAY_ACTION_COUNT]
    evaluated_mask = evaluated_mask.bool()

    target_logits = _masked_play_logits(target_q / args.policy_temperature, evaluated_mask)
    target_probs = torch.softmax(target_logits, dim=1)
    log_probs = torch.log_softmax(_masked_play_logits(play_logits, evaluated_mask), dim=1)
    policy_loss = F.kl_div(log_probs, target_probs, reduction="batchmean")

    target_masked = _masked_play_logits(target_q, evaluated_mask)
    best_idx = target_masked.argmax(dim=1)
    best_pred = play_logits.gather(1, best_idx.unsqueeze(1))
    best_target = target_q.gather(1, best_idx.unsqueeze(1))
    worse_mask = evaluated_mask & ((best_target - target_q) >= args.rank_gap)
    pair_margin = best_pred - play_logits
    rank_losses = F.relu(args.rank_margin - pair_margin)[worse_mask]
    rank_loss = rank_losses.mean() if rank_losses.numel() else play_logits.sum() * 0.0

    total = args.policy_weight * policy_loss + args.rank_weight * rank_loss
    return total, {
        "policy_loss": policy_loss.detach(),
        "rank_loss": rank_loss.detach(),
    }


@torch.no_grad()
def evaluate_play_ev(
    policy: SpadesTransformerPolicy,
    dataset: PlayEVDataset,
    args: argparse.Namespace,
    device: str | torch.device,
) -> PlayEVLossMetrics:
    policy.eval()
    loader = make_loader(dataset, batch_size=args.batch_size, shuffle=False)
    totals = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "rank_loss": 0.0,
        "policy_top1": 0.0,
        "policy_regret": 0.0,
        "best_ev": 0.0,
        "policy_ev": 0.0,
    }
    rows = 0
    for cpu_batch in loader:
        batch = tuple(t.to(device) for t in cpu_batch)
        observations, target_q, evaluated_mask, _legal_mask, _best_action = batch
        loss, parts = _losses(policy, batch, args)
        heads = policy.forward_heads(observations, apply_mask=False)
        play_logits = heads["logits"][:, :PLAY_ACTION_COUNT]
        evaluated_mask = evaluated_mask.bool()
        masked_target_q = _masked_play_logits(target_q, evaluated_mask)
        masked_logits = _masked_play_logits(play_logits, evaluated_mask)

        best_idx = masked_target_q.argmax(dim=1)
        policy_idx = masked_logits.argmax(dim=1)
        best_q = target_q.gather(1, best_idx.unsqueeze(1)).squeeze(1)
        policy_q = target_q.gather(1, policy_idx.unsqueeze(1)).squeeze(1)

        batch_rows = observations.shape[0]
        rows += batch_rows
        totals["loss"] += float(loss.item()) * batch_rows
        totals["policy_loss"] += float(parts["policy_loss"].item()) * batch_rows
        totals["rank_loss"] += float(parts["rank_loss"].item()) * batch_rows
        totals["policy_top1"] += float((policy_idx == best_idx).sum().item())
        totals["policy_regret"] += float((best_q - policy_q).sum().item())
        totals["best_ev"] += float(best_q.sum().item())
        totals["policy_ev"] += float(policy_q.sum().item())

    if rows == 0:
        raise ValueError("Cannot evaluate empty dataset")
    return PlayEVLossMetrics(
        loss=totals["loss"] / rows,
        policy_loss=totals["policy_loss"] / rows,
        rank_loss=totals["rank_loss"] / rows,
        policy_top1=totals["policy_top1"] / rows,
        policy_regret=totals["policy_regret"] / rows,
        mean_best_ev=totals["best_ev"] / rows,
        mean_policy_ev=totals["policy_ev"] / rows,
        rows=rows,
    )


def make_loader(dataset: PlayEVDataset, *, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(*dataset.tensors()), batch_size=batch_size, shuffle=shuffle)


def _freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _configure_trainable(policy: SpadesTransformerPolicy, args: argparse.Namespace) -> list[nn.Parameter]:
    if args.play_heads_only:
        for param in policy.parameters():
            param.requires_grad = False
        for module in (policy.play_query, policy.play_card_score):
            for param in module.parameters():
                param.requires_grad = True
    else:
        if args.freeze_encoder:
            for module in (policy.tokenizer, policy.transformer, policy.final_norm):
                _freeze_module(module)
        if args.freeze_bid_heads:
            for module in (policy.bid_policy_head, policy.bid_q_head):
                _freeze_module(module)

    params = [param for param in policy.parameters() if param.requires_grad]
    if not params:
        raise ValueError("No trainable parameters")
    return params


def _wandb_config(args: argparse.Namespace, rows: dict[str, int], device: str, use_amp: bool) -> dict[str, Any]:
    keys = [
        "dataset",
        "load_path",
        "save_path",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "val_fraction",
        "seed",
        "policy_temperature",
        "policy_weight",
        "rank_weight",
        "rank_gap",
        "rank_margin",
        "max_grad_norm",
        "d_model",
        "transformer_layers",
        "attention_heads",
        "ffn_size",
        "dropout",
        "freeze_encoder",
        "freeze_bid_heads",
        "play_heads_only",
    ]
    config = {key: getattr(args, key) for key in keys}
    config.update(rows)
    config["device"] = device
    config["amp_enabled"] = use_amp
    return config


def _init_wandb(args: argparse.Namespace, config: dict[str, Any]):
    if not args.wandb:
        return None, None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B logging requested with --wandb, but wandb is not installed") from exc
    run = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group or None,
        name=args.wandb_run_name or None,
        mode=args.wandb_mode,
        config=config,
    )
    return wandb, run


def _flatten_epoch_metrics(epoch_metrics: dict[str, Any]) -> dict[str, float | int]:
    flat: dict[str, float | int] = {
        "epoch": int(epoch_metrics["epoch"]),
        "optimizer_loss": float(epoch_metrics["optimizer_loss"]),
    }
    for split in ("train", "val"):
        for key, value in epoch_metrics[split].items():
            if isinstance(value, int):
                flat[f"{split}/{key}"] = value
            else:
                flat[f"{split}/{key}"] = float(value)
    return flat


def _artifact_name(args: argparse.Namespace) -> str:
    if args.wandb_artifact_name:
        return args.wandb_artifact_name
    if args.save_path:
        return os.path.splitext(os.path.basename(args.save_path))[0]
    return "play_ev_transformer"


def train_play_ev(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.policy_temperature <= 0:
        raise ValueError("policy_temperature must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    use_amp = bool(args.amp and device == "cuda")

    dataset = load_play_ev_dataset(args.dataset)
    train_data, val_data = split_dataset(dataset, val_fraction=args.val_fraction, seed=args.seed)
    row_counts = {
        "rows": len(dataset.observations),
        "train_rows": len(train_data.observations),
        "val_rows": len(val_data.observations) if val_data is not None else 0,
    }
    wandb_module, wandb_run = _init_wandb(args, _wandb_config(args, row_counts, device, use_amp))
    policy = build_play_ev_policy(args, device)
    try:
        if args.load_path:
            load_transformer_state(policy, args.load_path, device)
        optimizer = torch.optim.AdamW(
            _configure_trainable(policy, args),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        train_loader = make_loader(train_data, batch_size=args.batch_size, shuffle=True)

        best_metric = float("inf")
        history: list[dict[str, Any]] = []
        started_at = time.time()
        epoch_iter = tqdm(range(1, args.epochs + 1), desc="Play EV training", disable=not args.progress)
        for epoch in epoch_iter:
            policy.train()
            running_loss = 0.0
            rows_seen = 0
            batch_iter = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{args.epochs}",
                leave=False,
                disable=not args.progress,
            )
            for cpu_batch in batch_iter:
                batch = tuple(t.to(device) for t in cpu_batch)
                optimizer.zero_grad(set_to_none=True)
                autocast_context = torch.amp.autocast("cuda") if use_amp else nullcontext()
                with autocast_context:
                    loss, _parts = _losses(policy, batch, args)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                batch_rows = batch[0].shape[0]
                rows_seen += batch_rows
                running_loss += float(loss.item()) * batch_rows
                batch_iter.set_postfix(loss=running_loss / max(rows_seen, 1))

            train_metrics = evaluate_play_ev(policy, train_data, args, device)
            val_metrics = evaluate_play_ev(policy, val_data, args, device) if val_data is not None else train_metrics
            epoch_metrics = {
                "epoch": epoch,
                "optimizer_loss": running_loss / max(rows_seen, 1),
                "train": asdict(train_metrics),
                "val": asdict(val_metrics),
            }
            history.append(epoch_metrics)
            if wandb_run is not None:
                wandb_run.log(_flatten_epoch_metrics(epoch_metrics), step=epoch)
            print(
                "epoch={epoch} train_loss={train_loss:.4f} val_regret={regret:.2f} "
                "val_ev={policy_ev:.2f}/{best_ev:.2f} val_top1={top1:.3f}".format(
                    epoch=epoch,
                    train_loss=train_metrics.loss,
                    regret=val_metrics.policy_regret,
                    policy_ev=val_metrics.mean_policy_ev,
                    best_ev=val_metrics.mean_best_ev,
                    top1=val_metrics.policy_top1,
                ),
                flush=True,
            )
            epoch_iter.set_postfix(
                val_regret=val_metrics.policy_regret,
                val_ev=val_metrics.mean_policy_ev,
                val_top1=val_metrics.policy_top1,
            )
            if val_metrics.policy_regret < best_metric:
                best_metric = val_metrics.policy_regret
                if args.save_path:
                    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
                    torch.save(policy.state_dict(), args.save_path)

        metrics = {
            "dataset": args.dataset,
            **row_counts,
            "best_val_policy_regret": best_metric,
            "model_path": args.save_path,
            "loaded_model_path": args.load_path,
            "completed_at": time.time(),
            "duration_seconds": time.time() - started_at,
            "history": history,
        }
        if args.metrics_path:
            os.makedirs(os.path.dirname(args.metrics_path) or ".", exist_ok=True)
            with open(args.metrics_path, "w") as f:
                json.dump(metrics, f, indent=2, sort_keys=True)
        if wandb_run is not None:
            wandb_run.summary["best_val_policy_regret"] = best_metric
            wandb_run.summary["duration_seconds"] = metrics["duration_seconds"]
            if args.save_path and os.path.exists(args.save_path):
                artifact = wandb_module.Artifact(
                    _artifact_name(args),
                    type="model",
                    metadata={"best_val_policy_regret": best_metric, **row_counts},
                )
                artifact.add_file(args.save_path)
                if args.metrics_path and os.path.exists(args.metrics_path):
                    artifact.add_file(args.metrics_path)
                wandb_run.log_artifact(artifact)
        return metrics
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def eval_play_ev_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.policy_temperature <= 0:
        raise ValueError("policy_temperature must be positive")

    dataset = load_play_ev_dataset(args.dataset)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    policy = build_play_ev_policy(args, device)
    load_transformer_state(policy, args.checkpoint, device)
    metrics = asdict(evaluate_play_ev(policy, dataset, args, device))
    metrics.update(
        {
            "checkpoint": args.checkpoint,
            "dataset": args.dataset,
            "rows": int(len(dataset.observations)),
        }
    )
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
            f.write("\n")
    return metrics


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Spades transformer play policy from rollout-EV labels")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--load-path", type=str, default="")
    parser.add_argument("--save-path", type=str, default="checkpoints/play_ev_transformer.pt")
    parser.add_argument("--metrics-path", type=str, default="")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-temperature", type=float, default=20.0)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--rank-weight", type=float, default=0.1)
    parser.add_argument("--rank-gap", type=float, default=5.0)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=6)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--ffn-size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--freeze-encoder", action="store_true", default=False)
    parser.add_argument("--freeze-bid-heads", action="store_true", default=False)
    parser.add_argument("--play-heads-only", action="store_true", default=False)
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-progress", action="store_false", dest="progress")
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb-project", type=str, default="spades-rl")
    parser.add_argument("--wandb-group", type=str, default="play-ev")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-artifact-name", type=str, default="")
    return parser


def make_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on rollout-EV play labels")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--policy-temperature", type=float, default=20.0)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--rank-weight", type=float, default=0.1)
    parser.add_argument("--rank-gap", type=float, default=5.0)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=6)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--ffn-size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--output", type=str, default="")
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    metrics = train_play_ev(args)
    print(json.dumps(metrics, indent=2, sort_keys=True))


def eval_main() -> None:
    parser = make_eval_parser()
    args = parser.parse_args()
    metrics = eval_play_ev_checkpoint(args)
    print(json.dumps(metrics, indent=2, sort_keys=True))
