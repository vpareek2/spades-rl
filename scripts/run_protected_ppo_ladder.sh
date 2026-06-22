#!/usr/bin/env bash
set -euo pipefail

UV_BIN="${UV_BIN:-uv}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:?set BASE_CHECKPOINT to a checkpoint path}"
RUN_NAME="${RUN_NAME:-ppo_protected_ladder}"
OPPONENT_ACTOR="${OPPONENT_ACTOR:-bot:conservative}"

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-524288}"
CHECKPOINT_INTERVAL_STEPS="${CHECKPOINT_INTERVAL_STEPS:-65536}"
NUM_ENVS="${NUM_ENVS:-64}"
HORIZON="${HORIZON:-64}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-4096}"
LEARNING_RATE="${LEARNING_RATE:-0.00003}"
CLIP_COEF="${CLIP_COEF:-0.1}"
VF_CLIP_COEF="${VF_CLIP_COEF:-0.1}"
ENT_COEF="${ENT_COEF:-0.002}"
VF_COEF="${VF_COEF:-2.0}"
GAMMA="${GAMMA:-0.995}"
GAE_LAMBDA="${GAE_LAMBDA:-0.90}"
REPLAY_RATIO="${REPLAY_RATIO:-1.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.5}"

BID_PPO_WEIGHT="${BID_PPO_WEIGHT:-0.0}"
PLAY_PPO_WEIGHT="${PLAY_PPO_WEIGHT:-1.0}"
FREEZE_BID_HEADS="${FREEZE_BID_HEADS:-1}"
BID_ANCHOR_WEIGHT="${BID_ANCHOR_WEIGHT:-1.0}"
BID_Q_ANCHOR_WEIGHT="${BID_Q_ANCHOR_WEIGHT:-0.1}"
BID_ANCHOR_TEMPERATURE="${BID_ANCHOR_TEMPERATURE:-1.0}"
PLAY_ANCHOR_WEIGHT="${PLAY_ANCHOR_WEIGHT:-1.0}"
PLAY_ANCHOR_TEMPERATURE="${PLAY_ANCHOR_TEMPERATURE:-1.0}"
ACTIVE_ONLY_LOSS="${ACTIVE_ONLY_LOSS:-1}"
ANNEAL_LR="${ANNEAL_LR:-0}"

D_MODEL="${D_MODEL:-256}"
TRANSFORMER_LAYERS="${TRANSFORMER_LAYERS:-6}"
ATTENTION_HEADS="${ATTENTION_HEADS:-8}"
FFN_SIZE="${FFN_SIZE:-1024}"
DROPOUT="${DROPOUT:-0.05}"
SEED="${SEED:-6201}"

DUPLICATE_HANDS="${DUPLICATE_HANDS:-512}"
CONFIRM_HANDS="${CONFIRM_HANDS:-2048}"
CONFIRM_THRESHOLD_RAW="${CONFIRM_THRESHOLD_RAW:-10.8}"
AUTO_CONFIRM="${AUTO_CONFIRM:-1}"
EVAL_BASE="${EVAL_BASE:-1}"
EVAL_SEED="${EVAL_SEED:-123}"

LOGDIR="${LOGDIR:-logs/${RUN_NAME}}"
CHECKPOINT_PREFIX="${CHECKPOINT_PREFIX:-checkpoints/${RUN_NAME}}"
SAVE_PATH="${SAVE_PATH:-checkpoints/${RUN_NAME}_final.pt}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-spades-rl}"
WANDB_GROUP="${WANDB_GROUP:-${RUN_NAME}}"
WANDB_MODE="${WANDB_MODE:-online}"

mkdir -p "$LOGDIR" "$(dirname "$SAVE_PATH")"
exec > >(tee -a "$LOGDIR/pipeline.log") 2>&1

echo "=== ${RUN_NAME} start $(date -Is) ==="
echo "base checkpoint: ${BASE_CHECKPOINT}"
echo "save path: ${SAVE_PATH}"
echo "checkpoint prefix: ${CHECKPOINT_PREFIX}"
echo "timesteps=${TOTAL_TIMESTEPS} interval=${CHECKPOINT_INTERVAL_STEPS} seed=${SEED}"
echo "ppo weights: bid=${BID_PPO_WEIGHT} play=${PLAY_PPO_WEIGHT}"
echo "anchors: bid=${BID_ANCHOR_WEIGHT} bid_q=${BID_Q_ANCHOR_WEIGHT} play=${PLAY_ANCHOR_WEIGHT}"

train_args=(
  "$UV_BIN" run spades-train-puffer
  --load-path "$BASE_CHECKPOINT"
  --save-path "$SAVE_PATH"
  --metrics-path "$LOGDIR/train.json"
  --checkpoint-prefix "$CHECKPOINT_PREFIX"
  --checkpoint-interval-steps "$CHECKPOINT_INTERVAL_STEPS"
  --total-timesteps "$TOTAL_TIMESTEPS"
  --num-envs "$NUM_ENVS"
  --horizon "$HORIZON"
  --minibatch-size "$MINIBATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --clip-coef "$CLIP_COEF"
  --vf-clip-coef "$VF_CLIP_COEF"
  --ent-coef "$ENT_COEF"
  --vf-coef "$VF_COEF"
  --gamma "$GAMMA"
  --gae-lambda "$GAE_LAMBDA"
  --replay-ratio "$REPLAY_RATIO"
  --max-grad-norm "$MAX_GRAD_NORM"
  --bid-ppo-weight "$BID_PPO_WEIGHT"
  --play-ppo-weight "$PLAY_PPO_WEIGHT"
  --bid-anchor-path "$BASE_CHECKPOINT"
  --bid-anchor-weight "$BID_ANCHOR_WEIGHT"
  --bid-q-anchor-weight "$BID_Q_ANCHOR_WEIGHT"
  --bid-anchor-temperature "$BID_ANCHOR_TEMPERATURE"
  --play-anchor-path "$BASE_CHECKPOINT"
  --play-anchor-weight "$PLAY_ANCHOR_WEIGHT"
  --play-anchor-temperature "$PLAY_ANCHOR_TEMPERATURE"
  --d-model "$D_MODEL"
  --transformer-layers "$TRANSFORMER_LAYERS"
  --attention-heads "$ATTENTION_HEADS"
  --ffn-size "$FFN_SIZE"
  --dropout "$DROPOUT"
  --seed "$SEED"
)

if [[ "$FREEZE_BID_HEADS" == "1" ]]; then
  train_args+=(--freeze-bid-heads)
fi
if [[ "$ACTIVE_ONLY_LOSS" == "1" ]]; then
  train_args+=(--active-only-loss)
fi
if [[ "$ANNEAL_LR" == "1" ]]; then
  train_args+=(--anneal-lr)
fi
if [[ "$WANDB" == "1" ]]; then
  train_args+=(
    --wandb
    --wandb-project "$WANDB_PROJECT"
    --wandb-group "$WANDB_GROUP"
    --wandb-run-name "$RUN_NAME"
    --wandb-mode "$WANDB_MODE"
  )
fi

echo "=== ${RUN_NAME} train $(date -Is) ==="
"${train_args[@]}"

eval_duplicate() {
  local policy="$1"
  local output="$2"
  local hands="$3"
  echo "=== ${RUN_NAME} duplicate ${hands} ${policy} $(date -Is) ==="
  "$UV_BIN" run spades-eval-duplicate \
    --policy-a "$policy" \
    --policy-b "$OPPONENT_ACTOR" \
    --hands "$hands" \
    --seed "$EVAL_SEED" \
    --d-model "$D_MODEL" \
    --transformer-layers "$TRANSFORMER_LAYERS" \
    --attention-heads "$ATTENTION_HEADS" \
    --ffn-size "$FFN_SIZE" \
    --dropout "$DROPOUT" \
    --output "$output"
}

if [[ "$EVAL_BASE" == "1" ]]; then
  eval_duplicate "checkpoint:${BASE_CHECKPOINT}" "$LOGDIR/duplicate_base_${DUPLICATE_HANDS}.json" "$DUPLICATE_HANDS"
fi

checkpoint_files=()
while IFS= read -r checkpoint; do
  checkpoint_files+=("$checkpoint")
done < <(find "$(dirname "$CHECKPOINT_PREFIX")" -maxdepth 1 -type f -name "$(basename "$CHECKPOINT_PREFIX")_*.pt" | sort)

if [[ -f "$SAVE_PATH" ]]; then
  checkpoint_files+=("$SAVE_PATH")
fi

for checkpoint in "${checkpoint_files[@]}"; do
  stem="$(basename "$checkpoint" .pt)"
  tag="${stem#$(basename "$CHECKPOINT_PREFIX")_}"
  eval_duplicate "checkpoint:${checkpoint}" "$LOGDIR/duplicate_${tag}_${DUPLICATE_HANDS}.json" "$DUPLICATE_HANDS"
done

python3 - "$LOGDIR" "$RUN_NAME" "$DUPLICATE_HANDS" "$BASE_CHECKPOINT" "$CHECKPOINT_PREFIX" "$SAVE_PATH" <<'PY'
import json
import pathlib
import re
import sys

logdir = pathlib.Path(sys.argv[1])
run_name = sys.argv[2]
hands = sys.argv[3]
base_checkpoint = sys.argv[4]
checkpoint_prefix = pathlib.Path(sys.argv[5])
save_path = pathlib.Path(sys.argv[6])
prefix_name = checkpoint_prefix.name

rows = []
for path in sorted(logdir.glob(f"duplicate_*_{hands}.json")):
    tag = path.name.removeprefix("duplicate_").removesuffix(f"_{hands}.json")
    metrics = json.loads(path.read_text())
    if tag == "base":
        model_path = base_checkpoint
        kind = "base"
        step = 0
    elif tag == "final":
        model_path = str(save_path)
        kind = "checkpoint"
        step = None
    else:
        model_path = str(checkpoint_prefix.with_name(f"{prefix_name}_{tag}.pt"))
        kind = "checkpoint"
        match = re.fullmatch(r"\d+", tag)
        step = int(tag) if match else None
    rows.append(
        {
            "tag": tag,
            "kind": kind,
            "step": step,
            "model_path": model_path,
            "duplicate_hands": int(hands),
            "duplicate_raw": metrics.get("duplicate_margin_raw_mean"),
            "duplicate_ci95": metrics.get("duplicate_margin_ci95"),
            "contract_failures": metrics.get("contract_failures"),
            "bid_counts": metrics.get("bid_counts"),
        }
    )

rows.sort(key=lambda row: row["duplicate_raw"] if row["duplicate_raw"] is not None else -999999, reverse=True)
summary = {"run_name": run_name, "rows": rows}
(logdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

if [[ "$AUTO_CONFIRM" == "1" ]]; then
  set +e
  candidate_json="$(python3 - "$LOGDIR/summary.json" "$CONFIRM_THRESHOLD_RAW" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
threshold = float(sys.argv[2])
rows = summary.get("rows", [])
for row in rows:
    if row.get("kind") != "checkpoint":
        continue
    raw = row.get("duplicate_raw")
    if raw is None:
        continue
    print(json.dumps(row, sort_keys=True))
    raise SystemExit(0 if float(raw) >= threshold else 3)
print("{}")
raise SystemExit(2)
PY
)"
  candidate_status=$?
  set -e
  echo "confirm candidate: ${candidate_json}"
  if [[ "$candidate_status" == "0" ]]; then
    confirm_tag="$(CANDIDATE_JSON="$candidate_json" python3 - <<'PY'
import json
import os
print(json.loads(os.environ["CANDIDATE_JSON"]).get("tag", "best"))
PY
)"
    confirm_model="$(CANDIDATE_JSON="$candidate_json" python3 - <<'PY'
import json
import os
print(json.loads(os.environ["CANDIDATE_JSON"]).get("model_path", ""))
PY
)"
    eval_duplicate "checkpoint:${confirm_model}" "$LOGDIR/duplicate_${confirm_tag}_${CONFIRM_HANDS}.json" "$CONFIRM_HANDS"
  elif [[ "$candidate_status" == "3" ]]; then
    echo "best checkpoint below confirm threshold ${CONFIRM_THRESHOLD_RAW}; skipping ${CONFIRM_HANDS}-hand confirmation"
  else
    echo "no checkpoint candidate available for confirmation"
  fi
fi

echo "=== ${RUN_NAME} done $(date -Is) ==="
