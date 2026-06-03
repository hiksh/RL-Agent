"""
Action wrappers for F1DriverEnv.

DiscretizedF1Driver maps the continuous Box(4) action onto a small discrete set
so value-based methods that require a `Discrete` action space (DQN and variants)
can serve as the value-based baseline on the *same* environment used by the
continuous methods (PPO, SAC) — keeping the comparison apples-to-apples.
"""

import itertools
import numpy as np
import gymnasium as gym
from gymnasium import spaces


def _build_action_table():
    """Curated discrete action set -> Box(4) [steer, pedal, ers_raw, pit_raw].

    ers_raw/pit_raw are in [-1,1] (the env maps ers_raw->[0,1] and treats
    pit_raw>0.5 as a pit request)."""
    steers = [-1.0, -0.5, 0.0, 0.5, 1.0]
    table = []
    # steer x {brake, coast, throttle}, no ERS, no pit
    for s, p in itertools.product(steers, [-1.0, 0.0, 1.0]):
        table.append([s, p, -1.0, -1.0])
    # throttle + full ERS deploy (overtake/accelerate), no pit
    for s in steers:
        table.append([s, 1.0, 1.0, -1.0])
    # pit request (drive straight, coast, request pit)
    table.append([0.0, 0.0, -1.0, 1.0])
    return np.array(table, dtype=np.float32)


class DiscretizedF1Driver(gym.ActionWrapper):
    ACTION_TABLE = _build_action_table()   # shape (K, 4)

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(len(self.ACTION_TABLE))

    def action(self, act):
        return self.ACTION_TABLE[int(act)]


if __name__ == "__main__":
    from env import F1DriverEnv
    env = DiscretizedF1Driver(F1DriverEnv())
    print(f"Discrete actions: {env.action_space.n}")
    from stable_baselines3.common.env_checker import check_env
    check_env(env, warn=True)
    obs, _ = env.reset(seed=0)
    for _ in range(50):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc:
            break
    print("DiscretizedF1Driver check_env: PASSED")
