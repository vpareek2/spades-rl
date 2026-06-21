#!/usr/bin/env bash
set -euo pipefail

UV_BIN="${UV_BIN:-uv}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:?set BASE_CHECKPOINT to a checkpoint path}"
RUN_NAME="${RUN_NAME:-play_ev_team_aware}"
STATES="${STATES:-10000}"
ROLLOUT_SAMPLES="${ROLLOUT_SAMPLES:-4}"
SEED="${SEED:-5201}"
OPPONENT_ACTOR="${OPPONENT_ACTOR:-bot:conservative}"
WEIGHTS="${WEIGHTS:-30.0}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-512}"
D_MODEL="${D_MODEL:-256}"
TRANSFORMER_LAYERS="${TRANSFORMER_LAYERS:-6}"
ATTENTION_HEADS="${ATTENTION_HEADS:-8}"
FFN_SIZE="${FFN_SIZE:-1024}"
DROPOUT="${DROPOUT:-0.05}"
DEVICE="${DEVICE:-cuda}"
DUPLICATE_HANDS="${DUPLICATE_HANDS:-512}"
DATASET="${DATASET:-data/${RUN_NAME}.npz}"
LOGDIR="${LOGDIR:-logs/${RUN_NAME}}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-spades-rl}"
WANDB_GROUP="${WANDB_GROUP:-${RUN_NAME}}"
WANDB_MODE="${WANDB_MODE:-online}"

mkdir -p "$LOGDIR" "$(dirname "$DATASET")" checkpoints
exec > >(tee -a "$LOGDIR/pipeline.log") 2>&1

echo "=== ${RUN_NAME} start $(date -Is) ==="
echo "base checkpoint: ${BASE_CHECKPOINT}"
echo "dataset: ${DATASET}"
echo "states=${STATES} rollout_samples=${ROLLOUT_SAMPLES} seed=${SEED}"
echo "device: ${DEVICE}"
echo "weights: ${WEIGHTS}"

echo "=== ${RUN_NAME} generate $(date -Is) ==="
"$UV_BIN" run spades-generate-play-ev \
  --states "$STATES" \
  --rollout-samples "$ROLLOUT_SAMPLES" \
  --seed "$SEED" \
  --state-ally-actor "checkpoint:${BASE_CHECKPOINT}" \
  --state-opponent-actor "$OPPONENT_ACTOR" \
  --rollout-ally-actor "checkpoint:${BASE_CHECKPOINT}" \
  --rollout-opponent-actor "$OPPONENT_ACTOR" \
  --device "$DEVICE" \
  --output "$DATASET" \
  --progress

echo "=== ${RUN_NAME} summarize $(date -Is) ==="
"$UV_BIN" run spades-summarize-play-ev "$DATASET" | tee "$LOGDIR/dataset_summary.json"

echo "=== ${RUN_NAME} base eval $(date -Is) ==="
"$UV_BIN" run spades-eval-play-ev "$BASE_CHECKPOINT" \
  --dataset "$DATASET" \
  --d-model "$D_MODEL" \
  --transformer-layers "$TRANSFORMER_LAYERS" \
  --attention-heads "$ATTENTION_HEADS" \
  --ffn-size "$FFN_SIZE" \
  --dropout "$DROPOUT" \
  --output "$LOGDIR/eval_base.json"

for weight in $WEIGHTS; do
  tag="${weight//./p}"
  checkpoint="checkpoints/${RUN_NAME}_anchor_w${tag}.pt"
  wandb_args=()
  if [[ "$WANDB" == "1" ]]; then
    wandb_args=(
      --wandb
      --wandb-project "$WANDB_PROJECT"
      --wandb-group "$WANDB_GROUP"
      --wandb-run-name "${RUN_NAME}_w${tag}"
      --wandb-mode "$WANDB_MODE"
    )
  fi

  echo "=== ${RUN_NAME} train anchor w=${weight} $(date -Is) ==="
  "$UV_BIN" run spades-train-play-ev \
    --dataset "$DATASET" \
    --load-path "$BASE_CHECKPOINT" \
    --save-path "$checkpoint" \
    --metrics-path "$LOGDIR/train_anchor_w${tag}.json" \
    --device cuda \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --d-model "$D_MODEL" \
    --transformer-layers "$TRANSFORMER_LAYERS" \
    --attention-heads "$ATTENTION_HEADS" \
    --ffn-size "$FFN_SIZE" \
    --dropout "$DROPOUT" \
    --play-heads-only \
    --behavior-anchor-path "$BASE_CHECKPOINT" \
    --behavior-anchor-weight "$weight" \
    "${wandb_args[@]}"

  echo "=== ${RUN_NAME} eval anchor w=${weight} $(date -Is) ==="
  "$UV_BIN" run spades-eval-play-ev "$checkpoint" \
    --dataset "$DATASET" \
    --d-model "$D_MODEL" \
    --transformer-layers "$TRANSFORMER_LAYERS" \
    --attention-heads "$ATTENTION_HEADS" \
    --ffn-size "$FFN_SIZE" \
    --dropout "$DROPOUT" \
    --behavior-anchor-path "$BASE_CHECKPOINT" \
    --behavior-anchor-weight "$weight" \
    --output "$LOGDIR/eval_anchor_w${tag}.json"

  echo "=== ${RUN_NAME} duplicate${DUPLICATE_HANDS} anchor w=${weight} $(date -Is) ==="
  "$UV_BIN" run spades-eval-duplicate \
    --policy-a "checkpoint:${checkpoint}" \
    --policy-b "$OPPONENT_ACTOR" \
    --hands "$DUPLICATE_HANDS" \
    --d-model "$D_MODEL" \
    --transformer-layers "$TRANSFORMER_LAYERS" \
    --attention-heads "$ATTENTION_HEADS" \
    --ffn-size "$FFN_SIZE" \
    --dropout "$DROPOUT" \
    --output "$LOGDIR/duplicate_anchor_w${tag}_${DUPLICATE_HANDS}.json"
done

python3 - "$LOGDIR" "$DUPLICATE_HANDS" <<'PY'
import json
import pathlib
import sys

logdir = pathlib.Path(sys.argv[1])
hands = sys.argv[2]
rows = []
for duplicate_path in sorted(logdir.glob(f"duplicate_anchor_w*_{hands}.json")):
    tag = duplicate_path.name.removeprefix("duplicate_anchor_w").removesuffix(f"_{hands}.json")
    eval_path = logdir / f"eval_anchor_w{tag}.json"
    train_path = logdir / f"train_anchor_w{tag}.json"
    duplicate = json.loads(duplicate_path.read_text())
    eval_metrics = json.loads(eval_path.read_text()) if eval_path.exists() else {}
    train_metrics = json.loads(train_path.read_text()) if train_path.exists() else {}
    rows.append(
        {
            "tag": tag,
            "model_path": train_metrics.get("model_path", ""),
            "best_val_policy_regret": train_metrics.get("best_val_policy_regret"),
            "policy_regret": eval_metrics.get("policy_regret"),
            "policy_top1": eval_metrics.get("policy_top1"),
            "behavior_anchor_loss": eval_metrics.get("behavior_anchor_loss"),
            "duplicate_hands": int(hands),
            "duplicate_raw": duplicate.get("duplicate_margin_raw_mean"),
            "duplicate_ci95": duplicate.get("duplicate_margin_ci95"),
            "contract_failures": duplicate.get("contract_failures"),
        }
    )
rows.sort(key=lambda row: row["duplicate_raw"] if row["duplicate_raw"] is not None else -999999, reverse=True)
summary = {"rows": rows}
(logdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "=== ${RUN_NAME} done $(date -Is) ==="
