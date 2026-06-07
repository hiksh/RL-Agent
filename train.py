"""
train.py — Deep-RL training pipeline for F1DriverEnv (Stable-Baselines3).

Algorithms
  dqn : value-based baseline   (on DiscretizedF1Driver, 21 discrete actions)
  ppo : policy-based baseline  (continuous Box(4))
  sac : your solution          (continuous Box(4), off-policy actor-critic)

Examples
  python train.py --algo all --smoke              # quick pipeline check (CPU ok)
  python train.py --algo sac --timesteps 1000000  # full run (GPU server)
  python train.py --algo ppo --n-envs 8 --seed 0

Outputs (results/):
  <algo>_seed<seed>.zip          final model
  <algo>_seed<seed>_best/        best model by eval (EvalCallback)
  ckpt/<algo>_seed<seed>_*.zip   periodic checkpoints
  tb/                            tensorboard logs
"""

import os
import csv
import argparse
import numpy as np
import torch

from stable_baselines3 import DQN, PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

from env import F1DriverEnv
from wrappers import DiscretizedF1Driver

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

ALGOS = {
    "dqn": dict(cls=DQN, discretized=True),
    "ppo": dict(cls=PPO, discretized=False),
    "sac": dict(cls=SAC, discretized=False),
    "td3": dict(cls=TD3, discretized=False),
}
DEFAULT_STEPS = {"dqn": 500_000, "ppo": 1_000_000, "sac": 500_000, "td3": 500_000}

# Reward-shaping variants for the ablation study (passed as F1DriverEnv kwargs).
# 'baseline' = env defaults; others probe the design's sensitivity.
REWARD_PRESETS = {
    "baseline":   {},
    "no_shaping": dict(overheat_pen=0.0, slip_pen=0.0, time_pen=0.0),  # only progress/crash/finish
    "aggressive": dict(crash_pen=200.0, speed_reward=0.02),            # less timid, reward speed
    "racing":     dict(crash_pen=100.0, speed_reward=0.05),           # strong anti-stop (escape "brake-and-park")
    # Phase-2 time-attack study (vs `racing` baseline) — target the true objective:
    # finish the 3 laps within the time limit, as fast as possible.
    "timeattack_dense":  dict(crash_pen=100.0, speed_reward=0.10, time_pen=0.06),  # stronger dense pace pressure
    "timeattack_finish": dict(crash_pen=100.0, speed_reward=0.05, finish_time_bonus=300.0),  # terminal fast-finish bonus
    "timeattack":        dict(crash_pen=100.0, speed_reward=0.10, time_pen=0.06, finish_time_bonus=300.0),  # both
}


class EpisodeMetrics(BaseCallback):
    """Logs per-episode racing metrics (success/crash/lap/speed) to a CSV so the
    report's curves and comparisons can be regenerated from one training run."""
    FIELDS = ["timestep", "ep_reward", "ep_len", "laps", "success",
              "crashed", "mean_speed", "overheat_frac", "progress_m"]

    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path
        self._f = self._w = None

    def _on_training_start(self):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        self._f = open(self.csv_path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.FIELDS)
        self._w.writeheader()

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "episode" in info:
                self._w.writerow({
                    "timestep": self.num_timesteps,
                    "ep_reward": round(float(info["episode"]["r"]), 2),
                    "ep_len": int(info["episode"]["l"]),
                    "laps": int(info.get("laps", 0)),
                    "success": int(info.get("success", False)),
                    "crashed": int(info.get("crashed", False)),
                    "mean_speed": round(float(info.get("mean_speed", 0.0)), 3),
                    "overheat_frac": round(float(info.get("overheat_frac", 0.0)), 4),
                    "progress_m": round(float(info.get("progress_m", 0.0)), 1),
                })
                self._f.flush()
        return True

    def _on_training_end(self):
        if self._f:
            self._f.close()


def make_env(discretized, n_laps, env_kwargs):
    """Picklable env factory (SB3 SubprocVecEnv serialises via cloudpickle)."""
    def _init():
        env = F1DriverEnv(n_laps=n_laps, **env_kwargs)
        return DiscretizedF1Driver(env) if discretized else env
    return _init


def build_model(algo, venv, device, seed, tb):
    common = dict(policy="MlpPolicy", env=venv, verbose=1, seed=seed,
                  device=device, tensorboard_log=tb, gamma=0.99)
    if algo == "dqn":
        return DQN(**common, learning_rate=1e-3, buffer_size=200_000,
                   learning_starts=5_000, batch_size=128, train_freq=4,
                   target_update_interval=2_000, exploration_fraction=0.3,
                   exploration_final_eps=0.05)
    if algo == "ppo":
        return PPO(**common, n_steps=1024, batch_size=256, n_epochs=10,
                   gae_lambda=0.95, ent_coef=0.0, learning_rate=3e-4,
                   policy_kwargs=dict(net_arch=[256, 256]))
    if algo == "sac":
        return SAC(**common, learning_rate=3e-4, buffer_size=300_000,
                   learning_starts=10_000, batch_size=256, tau=0.005,
                   train_freq=1, gradient_steps=1, ent_coef="auto",
                   policy_kwargs=dict(net_arch=[256, 256]))
    if algo == "td3":
        n_act = venv.action_space.shape[0]
        noise = NormalActionNoise(mean=np.zeros(n_act), sigma=0.1 * np.ones(n_act))
        return TD3(**common, learning_rate=3e-4, buffer_size=300_000,
                   learning_starts=10_000, batch_size=256, tau=0.005,
                   train_freq=1, gradient_steps=1, action_noise=noise,
                   policy_kwargs=dict(net_arch=[256, 256]))
    raise ValueError(algo)


def train_one(algo, args, device):
    spec = ALGOS[algo]
    disc = spec["discretized"]
    preset = args.reward_preset
    env_kwargs = dict(REWARD_PRESETS[preset])
    if args.no_raycast:
        env_kwargs["use_raycast"] = False
    if args.random_weather:
        env_kwargs["random_weather"] = True

    tag = algo + ("" if preset == "baseline" else f"_{preset}")
    tag += ("_noray" if args.no_raycast else "") + ("_wx" if args.random_weather else "")
    tag += ("_ft" if args.init_from else "")          # warm-started fine-tune run
    tag += f"_seed{args.seed}"

    steps = args.timesteps or (3_000 if args.smoke else DEFAULT_STEPS[algo])
    n_envs = 2 if args.smoke else (args.n_envs if algo == "ppo" else 1)
    vec_cls = DummyVecEnv if (args.smoke or n_envs == 1) else SubprocVecEnv

    print(f"\n=== {algo.upper()}  |  steps={steps:,}  n_envs={n_envs}  device={device}  "
          f"reward={preset}  raycast={not args.no_raycast}  random_weather={args.random_weather} ===")

    venv = make_vec_env(make_env(disc, args.n_laps, env_kwargs), n_envs=n_envs,
                        seed=args.seed, vec_env_cls=vec_cls)
    eval_env = make_vec_env(make_env(disc, args.n_laps, env_kwargs), n_envs=1,
                            seed=args.seed + 1000, vec_env_cls=DummyVecEnv)

    callbacks = [
        EvalCallback(eval_env, best_model_save_path=os.path.join(RESULTS, f"{tag}_best"),
                     log_path=os.path.join(RESULTS, "eval"),
                     eval_freq=max(args.eval_freq // n_envs, 1),
                     n_eval_episodes=10, deterministic=True, verbose=0),
        CheckpointCallback(save_freq=max(args.save_freq // n_envs, 1),
                           save_path=os.path.join(RESULTS, "ckpt"),
                           name_prefix=tag, verbose=0),
        EpisodeMetrics(os.path.join(RESULTS, f"{tag}_metrics.csv")),
    ]

    tb = os.path.join(RESULTS, "tb")
    if args.init_from:                                # warm-start fine-tuning
        co = {"learning_rate": args.learning_rate,
              "lr_schedule": lambda _: args.learning_rate} if args.learning_rate else {}
        print(f"fine-tuning from {args.init_from}" + (f"  lr={args.learning_rate}" if co else ""))
        model = spec["cls"].load(args.init_from, env=venv, device=device,
                                 tensorboard_log=tb, custom_objects=co)
        model.set_random_seed(args.seed)              # .load() restores base RNG; reseed per run
    else:
        model = build_model(algo, venv, device, args.seed, tb)
    model.learn(total_timesteps=steps, progress_bar=not args.smoke,
                tb_log_name=tag, callback=callbacks)

    out = os.path.join(RESULTS, f"{tag}.zip")
    model.save(out)
    print(f"saved -> {out}")
    venv.close(); eval_env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=list(ALGOS) + ["all"], default="all")
    p.add_argument("--timesteps", type=int, default=0, help="0 = per-algo default")
    p.add_argument("--n-envs", type=int, default=8, help="parallel envs (PPO only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-laps", type=int, default=3)
    p.add_argument("--eval-freq", type=int, default=25_000)
    p.add_argument("--save-freq", type=int, default=100_000)
    p.add_argument("--reward-preset", choices=list(REWARD_PRESETS), default="baseline",
                   help="reward-shaping variant for ablation (see REWARD_PRESETS)")
    p.add_argument("--init-from", default="",
                   help="warm-start from a model .zip (fine-tuning); run is tagged _ft")
    p.add_argument("--learning-rate", type=float, default=0.0,
                   help="override LR (use with --init-from for low-LR fine-tune)")
    p.add_argument("--no-raycast", action="store_true", help="mask raycast obs (sensor ablation)")
    p.add_argument("--random-weather", action="store_true",
                   help="random wet start each episode (activate tire/pit strategy)")
    p.add_argument("--smoke", action="store_true", help="tiny run to verify the pipeline")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"device={device}  torch={torch.__version__}")

    algos = list(ALGOS) if args.algo == "all" else [args.algo]
    for algo in algos:
        train_one(algo, args, device)


if __name__ == "__main__":
    main()
