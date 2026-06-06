#!/usr/bin/env bash
# =============================================================================
# Phase-2: reward-design study toward the TRUE objective — finish the 3 laps
# within the time limit, as fast as possible.  Run AFTER run_all.sh (the
# fine-tune step warm-starts from results/sac_racing_seed0_best).
#
#   bash run_phase2.sh
#
#  * Judged on the invariant metric (lap time / success% / crash%), never on
#    reward — the presets use different reward scales.
#  * `racing` (from run_all.sh) is the comparison baseline for the sweep.
#  * ~15-17 h on the 6-core + 0.20-GPU server (SAC ~1.5 h / 500k run).
# =============================================================================
cd "$(dirname "$0")"

SEEDS="0 1 2"
STEPS=500000
FT_STEPS=200000
FT_LR=1e-4
BASE="results/sac_racing_seed0_best/best_model.zip"   # warm-start source
FT_PRESET="${FT_PRESET:-timeattack}"                  # set to the sweep winner before 2b

run () {  # echo + timestamp + continue-on-error
  echo ">>> [$(date +%H:%M:%S)] $*"
  python train.py "$@" || echo "!!! FAILED: $*"
}

# --- 2a) Reward-design sweep (SAC, from scratch) -----------------------------
#   isolate the levers: dense pace / terminal fast-finish bonus / both
for SEED in $SEEDS; do
  for PRESET in timeattack_dense timeattack_finish timeattack; do
    run --algo sac --reward-preset $PRESET --timesteps $STEPS --seed $SEED
  done
done

# --- 2b) Warm-start fine-tune of the best base policy toward time-attack ------
#   cheap: continues from `racing` best at low LR.  Set FT_PRESET to the winner.
if [ -f "$BASE" ]; then
  for SEED in $SEEDS; do
    run --algo sac --reward-preset $FT_PRESET --init-from "$BASE" \
        --learning-rate $FT_LR --timesteps $FT_STEPS --seed $SEED
  done
else
  echo "!!! base model $BASE not found — skip fine-tune (run run_all.sh first)"
fi

# --- 2c) Figures -------------------------------------------------------------
echo ">>> [$(date +%H:%M:%S)] visualize"
python visualize.py || echo "!!! viz failed"

echo ">>> [$(date +%H:%M:%S)] DONE"
