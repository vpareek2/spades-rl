"""Supervised training for rollout-EV bidding labels."""

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

from spades.actions import ACTION_SPACE_SIZE, BLIND_NIL_ACTION, NIL_ACTION
from spades.observations import FLAT_OBSERVATION_SIZE
from spades.policy import SpadesTransformerPolicy, load_transformer_state


BID_ACTION_START = 52
BID_ACTION_COUNT = 15


@dataclass(frozen=True)
class BidEVDataset:
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
class BidEVLossMetrics:
    loss: float
    q_loss: float
    policy_loss: float
    rank_loss: float
    q_mae: float
    q_rmse: float
    policy_top1: float
    q_top1: float
    policy_regret: float
    q_regret: float
    nil_rate: float
    blind_nil_rate: float
    rows: int


def _bid_slice(array: np.ndarray) -> np.ndarray:
    return array[:, BID_ACTION_START : BID_ACTION_START + BID_ACTION_COUNT]


def load_bid_ev_dataset(path: str) -> BidEVDataset:
    data = np.load(path, allow_pickle=False)
    observations = np.asarray(data["observations"], dtype=np.float32)
    q_values = _bid_slice(np.asarray(data["q_values"], dtype=np.float32))
    evaluated_mask = _bid_slice(np.asarray(data["evaluated_mask"], dtype=np.bool_))
    legal_mask = _bid_slice(np.asarray(data["legal_mask"], dtype=np.bool_))
    best_action = np.asarray(data["best_action"], dtype=np.int64) - BID_ACTION_START

    if observations.ndim != 2 or observations.shape[1] != FLAT_OBSERVATION_SIZE:
        raise ValueError(f"Expected observations with shape (N, {FLAT_OBSERVATION_SIZE})")
    if q_values.shape != (observations.shape[0], BID_ACTION_COUNT):
        raise ValueError(f"Expected q_values bid slice with shape (N, {BID_ACTION_COUNT})")
    if evaluated_mask.shape != q_values.shape or legal_mask.shape != q_values.shape:
        raise ValueError("Mask shapes must match q_values bid slice shape")

    valid_rows = evaluated_mask.any(axis=1) & (best_action >= 0) & (best_action < BID_ACTION_COUNT)
    if not np.any(valid_rows):
        raise ValueError("Dataset has no rows with evaluated bid actions")

    return BidEVDataset(
        observations=torch.from_numpy(observations[valid_rows]),
        q_values=torch.from_numpy(q_values[valid_rows]),
        evaluated_mask=torch.from_numpy(evaluated_mask[valid_rows]),
        legal_mask=torch.from_numpy(legal_mask[valid_rows]),
        best_action=torch.from_numpy(best_action[valid_rows]),
    )


def split_dataset(
    dataset: BidEVDataset,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[BidEVDataset, BidEVDataset | None]:
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

    def take(idx: torch.Tensor) -> BidEVDataset:
        return BidEVDataset(
            observations=dataset.observations[idx],
            q_values=dataset.q_values[idx],
            evaluated_mask=dataset.evaluated_mask[idx],
            legal_mask=dataset.legal_mask[idx],
            best_action=dataset.best_action[idx],
        )

    return take(train_idx), take(val_idx) if val_rows else None


def build_bid_ev_policy(args: argparse.Namespace, device: str | torch.device) -> SpadesTransformerPolicy:
    return SpadesTransformerPolicy(
        obs_size=FLAT_OBSERVATION_SIZE,
        action_size=ACTION_SPACE_SIZE,
        d_model=args.d_model,
        num_layers=args.transformer_layers,
        num_heads=args.attention_heads,
        ffn_size=args.ffn_size,
        dropout=args.dropout,
    ).to(device)


def _masked_bid_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, -1.0e9)


def _losses(
    policy: SpadesTransformerPolicy,
    batch: tuple[torch.Tensor, ...],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    observations, target_q, evaluated_mask, _legal_mask, _best_action = batch
    heads = policy.forward_heads(observations, apply_mask=False)
    pred_q = heads["bid_q"]
    bid_logits = heads["logits"][:, BID_ACTION_START : BID_ACTION_START + BID_ACTION_COUNT]
    evaluated_mask = evaluated_mask.bool()

    scaled_pred_q = pred_q / args.q_scale
    scaled_target_q = target_q / args.q_scale
    q_loss = F.smooth_l1_loss(scaled_pred_q[evaluated_mask], scaled_target_q[evaluated_mask])

    target_logits = _masked_bid_logits(target_q / args.policy_temperature, evaluated_mask)
    target_probs = torch.softmax(target_logits, dim=1)
    log_probs = torch.log_softmax(_masked_bid_logits(bid_logits, evaluated_mask), dim=1)
    policy_loss = F.kl_div(log_probs, target_probs, reduction="batchmean")

    target_masked = _masked_bid_logits(target_q, evaluated_mask)
    best_idx = target_masked.argmax(dim=1)
    best_pred = pred_q.gather(1, best_idx.unsqueeze(1))
    best_target = target_q.gather(1, best_idx.unsqueeze(1))
    worse_mask = evaluated_mask & ((best_target - target_q) >= args.rank_gap)
    pair_margin = best_pred - pred_q
    rank_losses = F.relu(args.rank_margin - pair_margin)[worse_mask]
    rank_loss = rank_losses.mean() if rank_losses.numel() else pred_q.sum() * 0.0

    total = args.q_weight * q_loss + args.policy_weight * policy_loss + args.rank_weight * rank_loss
    return total, {
        "q_loss": q_loss.detach(),
        "policy_loss": policy_loss.detach(),
        "rank_loss": rank_loss.detach(),
    }


@torch.no_grad()
def evaluate_bid_ev(
    policy: SpadesTransformerPolicy,
    dataset: BidEVDataset,
    args: argparse.Namespace,
    device: str | torch.device,
) -> BidEVLossMetrics:
    policy.eval()
    loader = make_loader(dataset, batch_size=args.batch_size, shuffle=False)
    totals = {
        "loss": 0.0,
        "q_loss": 0.0,
        "policy_loss": 0.0,
        "rank_loss": 0.0,
        "abs_error": 0.0,
        "sq_error": 0.0,
        "policy_top1": 0.0,
        "q_top1": 0.0,
        "policy_regret": 0.0,
        "q_regret": 0.0,
        "nil": 0.0,
        "blind_nil": 0.0,
    }
    rows = 0
    q_points = 0
    for cpu_batch in loader:
        batch = tuple(t.to(device) for t in cpu_batch)
        observations, target_q, evaluated_mask, _legal_mask, _best_action = batch
        loss, parts = _losses(policy, batch, args)
        heads = policy.forward_heads(observations, apply_mask=False)
        pred_q = heads["bid_q"]
        bid_logits = heads["logits"][:, BID_ACTION_START : BID_ACTION_START + BID_ACTION_COUNT]
        evaluated_mask = evaluated_mask.bool()
        masked_target_q = _masked_bid_logits(target_q, evaluated_mask)
        masked_pred_q = _masked_bid_logits(pred_q, evaluated_mask)
        masked_logits = _masked_bid_logits(bid_logits, evaluated_mask)

        best_idx = masked_target_q.argmax(dim=1)
        policy_idx = masked_logits.argmax(dim=1)
        q_idx = masked_pred_q.argmax(dim=1)
        best_q = target_q.gather(1, best_idx.unsqueeze(1)).squeeze(1)
        policy_q = target_q.gather(1, policy_idx.unsqueeze(1)).squeeze(1)
        q_head_q = target_q.gather(1, q_idx.unsqueeze(1)).squeeze(1)

        diff = pred_q[evaluated_mask] - target_q[evaluated_mask]
        batch_rows = observations.shape[0]
        rows += batch_rows
        q_points += int(evaluated_mask.sum().item())
        totals["loss"] += float(loss.item()) * batch_rows
        totals["q_loss"] += float(parts["q_loss"].item()) * batch_rows
        totals["policy_loss"] += float(parts["policy_loss"].item()) * batch_rows
        totals["rank_loss"] += float(parts["rank_loss"].item()) * batch_rows
        totals["abs_error"] += float(diff.abs().sum().item())
        totals["sq_error"] += float((diff * diff).sum().item())
        totals["policy_top1"] += float((policy_idx == best_idx).sum().item())
        totals["q_top1"] += float((q_idx == best_idx).sum().item())
        totals["policy_regret"] += float((best_q - policy_q).sum().item())
        totals["q_regret"] += float((best_q - q_head_q).sum().item())
        totals["nil"] += float((policy_idx == (NIL_ACTION - BID_ACTION_START)).sum().item())
        totals["blind_nil"] += float((policy_idx == (BLIND_NIL_ACTION - BID_ACTION_START)).sum().item())

    if rows == 0:
        raise ValueError("Cannot evaluate empty dataset")
    q_points = max(q_points, 1)
    return BidEVLossMetrics(
        loss=totals["loss"] / rows,
        q_loss=totals["q_loss"] / rows,
        policy_loss=totals["policy_loss"] / rows,
        rank_loss=totals["rank_loss"] / rows,
        q_mae=totals["abs_error"] / q_points,
        q_rmse=(totals["sq_error"] / q_points) ** 0.5,
        policy_top1=totals["policy_top1"] / rows,
        q_top1=totals["q_top1"] / rows,
        policy_regret=totals["policy_regret"] / rows,
        q_regret=totals["q_regret"] / rows,
        nil_rate=totals["nil"] / rows,
        blind_nil_rate=totals["blind_nil"] / rows,
        rows=rows,
    )


def make_loader(dataset: BidEVDataset, *, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(*dataset.tensors()), batch_size=batch_size, shuffle=shuffle)


def _configure_trainable(policy: SpadesTransformerPolicy, freeze_encoder: bool) -> list[nn.Parameter]:
    if freeze_encoder:
        for module in (policy.tokenizer, policy.transformer, policy.final_norm):
            for param in module.parameters():
                param.requires_grad = False
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
        "q_scale",
        "policy_temperature",
        "q_weight",
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
    return "bid_ev_transformer"


def train_bid_ev(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.q_scale <= 0:
        raise ValueError("q_scale must be positive")
    if args.policy_temperature <= 0:
        raise ValueError("policy_temperature must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    use_amp = bool(args.amp and device == "cuda")

    dataset = load_bid_ev_dataset(args.dataset)
    train_data, val_data = split_dataset(dataset, val_fraction=args.val_fraction, seed=args.seed)
    row_counts = {
        "rows": len(dataset.observations),
        "train_rows": len(train_data.observations),
        "val_rows": len(val_data.observations) if val_data is not None else 0,
    }
    wandb_module, wandb_run = _init_wandb(args, _wandb_config(args, row_counts, device, use_amp))
    policy = build_bid_ev_policy(args, device)
    try:
        if args.load_path:
            load_transformer_state(policy, args.load_path, device)
        optimizer = torch.optim.AdamW(
            _configure_trainable(policy, args.freeze_encoder),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        train_loader = make_loader(train_data, batch_size=args.batch_size, shuffle=True)

        best_metric = float("inf")
        history: list[dict[str, Any]] = []
        started_at = time.time()
        epoch_iter = tqdm(range(1, args.epochs + 1), desc="EV training", disable=not args.progress)
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

            train_metrics = evaluate_bid_ev(policy, train_data, args, device)
            val_metrics = evaluate_bid_ev(policy, val_data, args, device) if val_data is not None else train_metrics
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
                "val_q_mae={q_mae:.2f} val_top1={top1:.3f}".format(
                    epoch=epoch,
                    train_loss=train_metrics.loss,
                    regret=val_metrics.policy_regret,
                    q_mae=val_metrics.q_mae,
                    top1=val_metrics.policy_top1,
                ),
                flush=True,
            )
            epoch_iter.set_postfix(
                val_regret=val_metrics.policy_regret,
                val_q_mae=val_metrics.q_mae,
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


def eval_bid_ev_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.q_scale <= 0:
        raise ValueError("q_scale must be positive")
    if args.policy_temperature <= 0:
        raise ValueError("policy_temperature must be positive")

    dataset = load_bid_ev_dataset(args.dataset)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    policy = build_bid_ev_policy(args, device)
    load_transformer_state(policy, args.checkpoint, device)
    metrics = asdict(evaluate_bid_ev(policy, dataset, args, device))
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
    parser = argparse.ArgumentParser(description="Train Spades transformer bidding heads from rollout-EV labels")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--load-path", type=str, default="")
    parser.add_argument("--save-path", type=str, default="checkpoints/bid_ev_transformer.pt")
    parser.add_argument("--metrics-path", type=str, default="")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--q-scale", type=float, default=100.0)
    parser.add_argument("--policy-temperature", type=float, default=25.0)
    parser.add_argument("--q-weight", type=float, default=1.0)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--rank-weight", type=float, default=0.1)
    parser.add_argument("--rank-gap", type=float, default=10.0)
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
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-progress", action="store_false", dest="progress")
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb-project", type=str, default="spades-rl")
    parser.add_argument("--wandb-group", type=str, default="bid-ev")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-artifact-name", type=str, default="")
    return parser


def make_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on rollout-EV bidding labels")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--q-scale", type=float, default=100.0)
    parser.add_argument("--policy-temperature", type=float, default=25.0)
    parser.add_argument("--q-weight", type=float, default=1.0)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--rank-weight", type=float, default=0.1)
    parser.add_argument("--rank-gap", type=float, default=10.0)
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
    metrics = train_bid_ev(args)
    print(json.dumps(metrics, indent=2, sort_keys=True))


def eval_main() -> None:
    parser = make_eval_parser()
    args = parser.parse_args()
    metrics = eval_bid_ev_checkpoint(args)
    print(json.dumps(metrics, indent=2, sort_keys=True))
