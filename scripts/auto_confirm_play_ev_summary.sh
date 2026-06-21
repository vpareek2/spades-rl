#!/usr/bin/env bash
set -euo pipefail

UV_BIN="${UV_BIN:-uv}"
RUN_NAME="${RUN_NAME:?set RUN_NAME to the pipeline run name}"
LOGDIR="${LOGDIR:-logs/${RUN_NAME}}"
SUMMARY_PATH="${SUMMARY_PATH:-${LOGDIR}/summary.json}"
CONFIRM_THRESHOLD_RAW="${CONFIRM_THRESHOLD_RAW:-10.5}"
POLL_SECONDS="${POLL_SECONDS:-300}"
OPPONENT_ACTOR="${OPPONENT_ACTOR:-bot:conservative}"
D_MODEL="${D_MODEL:-256}"
TRANSFORMER_LAYERS="${TRANSFORMER_LAYERS:-6}"
ATTENTION_HEADS="${ATTENTION_HEADS:-8}"
FFN_SIZE="${FFN_SIZE:-1024}"
DROPOUT="${DROPOUT:-0.05}"
CONFIRM_HANDS="${CONFIRM_HANDS:-2048}"

mkdir -p "$LOGDIR"

echo "=== auto-confirm ${RUN_NAME} start $(date -Is) ==="
echo "summary: ${SUMMARY_PATH}"
echo "threshold raw: ${CONFIRM_THRESHOLD_RAW}"

while [[ ! -f "$SUMMARY_PATH" ]]; do
  echo "$(date -Is) waiting for ${SUMMARY_PATH}"
  sleep "$POLL_SECONDS"
done

set +e
candidate_json="$(python3 - "$SUMMARY_PATH" "$CONFIRM_THRESHOLD_RAW" <<'PY'
import json
import sys

summary_path = sys.argv[1]
threshold = float(sys.argv[2])
summary = json.load(open(summary_path))
rows = summary.get("rows", [])
if not rows:
    print("{}")
    raise SystemExit(2)
best = rows[0]
raw = best.get("duplicate_raw")
if raw is None:
    print(json.dumps(best, sort_keys=True))
    raise SystemExit(2)
print(json.dumps(best, sort_keys=True))
if float(raw) < threshold:
    raise SystemExit(3)
PY
)"
candidate_status=$?
set -e

echo "best candidate: ${candidate_json}"
if [[ "$candidate_status" == "3" ]]; then
  echo "best candidate below threshold; skipping ${CONFIRM_HANDS}-hand confirmation"
  exit 0
fi
if [[ "$candidate_status" != "0" ]]; then
  echo "could not parse a confirmable candidate"
  exit "$candidate_status"
fi

model_path="$(CANDIDATE_JSON="$candidate_json" python3 - <<'PY'
import json
import os

print(json.loads(os.environ["CANDIDATE_JSON"]).get("model_path", ""))
PY
)"
tag="$(CANDIDATE_JSON="$candidate_json" python3 - <<'PY'
import json
import os

print(json.loads(os.environ["CANDIDATE_JSON"]).get("tag", "best"))
PY
)"

if [[ -z "$model_path" ]]; then
  echo "candidate had no model_path"
  exit 2
fi

output="${LOGDIR}/duplicate_anchor_w${tag}_${CONFIRM_HANDS}.json"
echo "=== auto-confirm ${RUN_NAME} w=${tag} ${CONFIRM_HANDS} hands $(date -Is) ==="
"$UV_BIN" run spades-eval-duplicate \
  --policy-a "checkpoint:${model_path}" \
  --policy-b "$OPPONENT_ACTOR" \
  --hands "$CONFIRM_HANDS" \
  --d-model "$D_MODEL" \
  --transformer-layers "$TRANSFORMER_LAYERS" \
  --attention-heads "$ATTENTION_HEADS" \
  --ffn-size "$FFN_SIZE" \
  --dropout "$DROPOUT" \
  --output "$output"

echo "=== auto-confirm ${RUN_NAME} done $(date -Is) ==="
