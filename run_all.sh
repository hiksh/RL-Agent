#!/usr/bin/env bash
# =============================================================================
# Full Deep-RL training matrix for F1DriverEnv  (run on the GPU server).
#
#   pip install -r requirements.txt        # first time only
#   bash run_all.sh                         # runs the whole matrix, then plots
#
# Notes
#  * Sequential by design (single 6-core + 0.20-GPU allocation). Each run prints
#    its own fps so you can re-estimate total time on the server after run #1.
#  * Continue-on-error: a failed run is logged and skipped, partial results kept.
#  * Knobs below: cut SEEDS to "0 1 2" or drop WX to shorten.
# =============================================================================
cd "$(dirname "$0")"

SEEDS="0 1 2 3 4"          # 5 seeds for the main comparison
REWARD="racing"           # reward preset that solves the driving task
WX=""                     # forced wet starts re-break driving (timid collapse);
                          #   weather/pit still occurs naturally at lap boundaries.
                          #   set WX="--random-weather" only for a wet-strategy demo.
PPO_ENVS=6                # match server core count

run () {  # echo + timestamp + continue-on-error
  echo ">>> [$(date +%H:%M:%S)] $*"
  python train.py "$@" || echo "!!! FAILED: $*"
}

# --- 1) Main comparison ------------------------------------------------------
#   value-based (DQN) + policy-based (PPO) + your-solution (SAC) + TD3
for SEED in $SEEDS; do
  run --algo dqn --reward-preset $REWARD $WX --timesteps 500000  --seed $SEED
  run --algo ppo --reward-preset $REWARD $WX --timesteps 1000000 --n-envs $PPO_ENVS --seed $SEED
  run --algo sac --reward-preset $REWARD $WX --timesteps 500000  --seed $SEED
  run --algo td3 --reward-preset $REWARD $WX --timesteps 500000  --seed $SEED
done

# --- 2) Reward-shaping ablation (SAC) ---------------------------------------
#   shows reward-design validity; 'racing' is already covered by the main runs
for PRESET in baseline no_shaping aggressive; do
  run --algo sac --reward-preset $PRESET $WX --timesteps 500000 --seed 0
done

# --- 3) Sensor ablation ------------------------------------------------------
#   does the raycast look-ahead help?  (SAC racing, raycast masked off)
run --algo sac --reward-preset $REWARD $WX --no-raycast --timesteps 500000 --seed 0

# --- 4) Figures --------------------------------------------------------------
echo ">>> [$(date +%H:%M:%S)] visualize"
python visualize.py || echo "!!! viz failed"

echo ">>> [$(date +%H:%M:%S)] DONE"
