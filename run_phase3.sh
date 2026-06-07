#!/usr/bin/env bash
# =============================================================================
# Phase-3: push toward the optimal driver — corner SMOOTHLY + fast, no crash.
# Two levers, both measured against the Phase-2 winner `timeattack`:
#   (1) random-start curriculum: reset training episodes at a random track
#       position so EVERY corner is practised equally (not just the first one).
#       Eval / reported metrics stay at idx=0 -> comparable to Phase-1/2.
#   (2) steering-smoothness penalty λ (|Δsteer|): sweep to find the value that
#       keeps lap time / success while lowering jerk (the λ selection process).
#
#   bash run_phase3.sh                    # run AFTER run_phase2.sh
#
#  * Reported on the idx=0 eval dump ({tag}_eval0.csv), never on training
#    rollouts (random-start) — see §9-9.  λ=0 = curriculum-only baseline.
#  * ~15 h on the 6-core + 0.20-GPU server (4 λ × 3 seed × 500k SAC).
# =============================================================================
cd "$(dirname "$0")"

SEEDS="0 1 2"
STEPS=500000
LAMBDAS="0.0 0.02 0.05 0.1"        # λ=0 isolates the random-start curriculum

run () {  # echo + timestamp + continue-on-error
  echo ">>> [$(date +%H:%M:%S)] $*"
  python train.py "$@" || echo "!!! FAILED: $*"
}

# --- idx=0 eval baseline: Phase-2 winner (no retrain, just evaluate) ----------
for SEED in $SEEDS; do
  BEST="results/sac_timeattack_seed${SEED}_best/best_model.zip"
  if [ -f "$BEST" ]; then
    run --algo sac --reward-preset timeattack --eval-only --init-from "$BEST" --seed $SEED
  else
    echo "!!! $BEST not found — run run_phase2.sh first (baseline skipped)"
  fi
done

# --- curriculum + smoothness λ sweep (random start, from scratch) ------------
for SEED in $SEEDS; do
  for LAM in $LAMBDAS; do
    run --algo sac --reward-preset timeattack --steer-pen $LAM --random-start \
        --timesteps $STEPS --seed $SEED
  done
done

# --- figures -----------------------------------------------------------------
echo ">>> [$(date +%H:%M:%S)] visualize"
python visualize.py || echo "!!! viz failed"

echo ">>> [$(date +%H:%M:%S)] DONE"
