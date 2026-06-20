# Spades Puffer Checkpoint

This directory contains the best checkpoint from the first PufferLib training
pass on the A6000 cloud instance.

## Files

- `checkpoint.pt`: masked Puffer policy weights.
- `pretrain_metrics.json`: supervised bidding bootstrap metrics.
- `eval_curriculum_16384_cpu.json`: greedy eval with normal bids capped at 5,
  nil disabled, and blind nil disabled.
- `eval_full_4096_cpu.json`: greedy eval with full normal bids and nil/blind
  nil enabled.

## Result

- Curriculum eval: `mean=-0.010721435770392418` over 16,384 hands, `0`
  illegal actions.
- Full-action eval: `mean=-0.025358887389302254` over 4,096 hands, `0`
  illegal actions.

The checkpoint SHA-256 is:

```text
7d0c0ea796ea4e1b1ebb0c8c9a15f962ff8a55c3e03ef3ceeaea0267fabb255e
```

Re-run the curriculum eval:

```bash
uv run spades-eval-puffer artifacts/spades_curriculum_25m_bidbot/checkpoint.pt \
  --hands 16384 \
  --max-normal-bid 5 \
  --no-nil \
  --no-blind-nil \
  --cpu
```

Re-run the full-action eval:

```bash
uv run spades-eval-puffer artifacts/spades_curriculum_25m_bidbot/checkpoint.pt \
  --hands 4096 \
  --max-normal-bid 13 \
  --nil \
  --blind-nil \
  --cpu
```
