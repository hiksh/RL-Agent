"""
F1DriverEnv — 2-D continuous in-car driver environment (gymnasium API).

Pivot from Proj-01's discrete pit-wall strategist to a continuous driver that
steers, throttles/brakes, deploys ERS, and decides pit-stops on a faithfully
reconstructed 2-D track (see build_track.py -> assets/track.npz).

Observation : flat Box vector (vehicle + 5 raycasts + weather), see _get_obs.
Action      : Box(4) = [steering, pedal, ers_deploy, pit_signal]
              pit_signal > PIT_THRESH requests a pit; the stop executes on the
              next start/finish crossing.  A single Box keeps the env compatible
              with SB3 SAC/DDPG/TD3/PPO (none accept Dict/Tuple actions).

Run `python env.py` for an env_checker + random rollout + short SB3 smoke test.
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

HERE       = os.path.dirname(os.path.abspath(__file__))
TRACK_NPZ  = os.path.join(HERE, "assets", "track.npz")

# ── Physics / dynamics constants ───────────────────────────────────────────────
DT          = 0.25      # s per step
MAX_SPEED   = 90.0      # m/s
ACCEL       = 14.0      # m/s^2  throttle
BRAKE       = 28.0      # m/s^2  full brake
ERS_BOOST   = 10.0      # m/s^2  extra accel at full deploy
DRAG        = 0.015     # linear drag per step
MAX_YAW     = 1.2       # rad/s  max yaw rate
YAW_VREF    = 18.0      # m/s    speed at which steering reaches full authority

# ERS battery
ERS_USE     = 0.6       # drain per second at full deploy
ERS_REGEN   = 0.25      # recharge per second while braking

# Tire temperature (0..1), dynamics scaled 10x for the micro-race
TIRE_SCALE  = 10.0
TIRE_AMB    = 0.15
TIRE_HEAT   = 0.10
TIRE_COOL   = 0.06
TEMP_OPT_LO, TEMP_OPT_HI = 0.40, 0.75
TEMP_OVERHEAT            = 0.85

# Weather / wetness (0..1), also scaled 10x; transitions at lap boundaries
WET_RATE    = 0.05      # per second drift toward target (×TIRE_SCALE applied)
P_DRY2RAIN  = 0.30
P_RAIN2DRY  = 0.40
WET_TIRE_THRESH = 0.45  # wetness above which Dry tires start slipping

# Raycast sensor
N_RAYS      = 5
RAY_ANGLES  = np.deg2rad([-70, -35, 0, 35, 70])
MAX_RAY     = 120.0     # m, normalisation cap
RAY_WIN     = (-4, 28)  # boundary-segment index window (relative to car idx)

# Reward shaping
PROGRESS_SCALE = 0.05   # reward per metre of forward progress
TIME_PEN       = 0.03   # per-step cost -> rewards finishing fast
OVERHEAT_PEN   = 0.5
SLIP_PEN       = 0.3
PIT_PEN        = 30.0
CRASH_PEN      = 500.0
COMPLETE_BONUS = 200.0
PIT_THRESH     = 0.5

# Compounds
DRY = 0; INTER = 1


def _load_track():
    if not os.path.exists(TRACK_NPZ):
        raise FileNotFoundError(f"{TRACK_NPZ} missing — run `python build_track.py` first.")
    d = np.load(TRACK_NPZ)
    return d["centerline"], d["left"], d["right"], d["half_width"], float(d["length_m"])


class F1DriverEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, n_laps=3, max_steps=None, seed=None,
                 progress_scale=PROGRESS_SCALE, time_pen=TIME_PEN,
                 overheat_pen=OVERHEAT_PEN, slip_pen=SLIP_PEN, pit_pen=PIT_PEN,
                 crash_pen=CRASH_PEN, complete_bonus=COMPLETE_BONUS,
                 speed_reward=0.0, finish_time_bonus=0.0, steer_pen=0.0,
                 use_raycast=True, random_weather=False, random_start=False):
        super().__init__()
        # reward weights (kwargs default to the module constants -> baseline
        # behaviour unchanged; override per-instance for ablation studies)
        self.progress_scale = progress_scale; self.time_pen = time_pen
        self.overheat_pen = overheat_pen;     self.slip_pen = slip_pen
        self.pit_pen = pit_pen;               self.crash_pen = crash_pen
        self.complete_bonus = complete_bonus; self.speed_reward = speed_reward
        self.finish_time_bonus = finish_time_bonus  # extra terminal reward scaled by how early the race finished
        self.use_raycast = use_raycast            # False -> mask ray obs (ablation)
        self.random_weather = random_weather      # True  -> random wet start (activate pit strategy)
        self.steer_pen = steer_pen                # penalise |Δsteer| per step -> smoother steering (Phase-3)
        self.random_start = random_start          # True  -> reset at a random track idx (training-only curriculum)
        cl, left, right, half_w, length_m = _load_track()
        self.cl       = cl.astype(np.float64)
        self.left     = left.astype(np.float64)
        self.right    = right.astype(np.float64)
        self.half_w   = half_w.astype(np.float64)
        self.length_m = length_m
        self.N        = len(cl)

        # arc length + tangents/normals along the centreline
        d = np.linalg.norm(np.roll(self.cl, -1, 0) - self.cl, axis=1)
        self.s = np.concatenate([[0], np.cumsum(d)[:-1]])
        tang = np.roll(self.cl, -1, 0) - np.roll(self.cl, 1, 0)
        tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)
        self.tang = tang
        self.normal = np.column_stack([-tang[:, 1], tang[:, 0]])

        self.n_laps    = n_laps
        self.max_steps = max_steps or n_laps * 500

        # boundary segments as (P, D) for ray intersection
        self._segP = np.vstack([self.left, self.right])
        segB = np.vstack([np.roll(self.left, -1, 0), np.roll(self.right, -1, 0)])
        self._segD = segB - self._segP

        # symmetric [-1,1] action space (best practice for SB3 / SAC);
        # ers_deploy is mapped to [0,1] internally in step().
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.obs_dim = 1 + 2 + 1 + 1 + 1 + 1 + N_RAYS + 1 + 1   # = 14
        self.observation_space = spaces.Box(
            low=-3.0, high=3.0, shape=(self.obs_dim,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ helpers
    def _nearest_idx(self, forward_only=True):
        lo, hi = (-3, 14) if forward_only else (-self.N // 4, self.N // 4)
        cand = (self.idx + np.arange(lo, hi)) % self.N
        d = np.linalg.norm(self.cl[cand] - self.pos, axis=1)
        return int(cand[int(np.argmin(d))])

    def _lateral_offset(self):
        return float(np.dot(self.pos - self.cl[self.idx], self.normal[self.idx]))

    def _heading_err(self):
        tx, ty = self.tang[self.idx]
        hx, hy = np.cos(self.heading), np.sin(self.heading)
        cos = hx * tx + hy * ty
        sin = hx * ty - hy * tx          # signed
        return sin, cos

    def _raycast(self):
        """Distance from car to track boundary along each ray (normalised 0..1)."""
        w0 = (self.idx + RAY_WIN[0]) % self.N
        win = (w0 + np.arange(RAY_WIN[1] - RAY_WIN[0])) % self.N
        idxs = np.concatenate([win, win + self.N])     # left + right segments
        P = self._segP[idxs]; D = self._segD[idxs]
        out = np.empty(N_RAYS, dtype=np.float32)
        for k, a in enumerate(RAY_ANGLES):
            ang = self.heading + a
            r = np.array([np.cos(ang), np.sin(ang)])
            # solve pos + t*r = P + u*D  ->  t,u ; keep t>0, 0<=u<=1
            denom = r[0] * (-D[:, 1]) - r[1] * (-D[:, 0])
            ok = np.abs(denom) > 1e-9
            diff = P - self.pos
            t = np.where(ok, (diff[:, 0] * (-D[:, 1]) - diff[:, 1] * (-D[:, 0])) / denom, np.inf)
            u = np.where(ok, (r[0] * diff[:, 1] - r[1] * diff[:, 0]) / denom, np.inf)
            valid = ok & (t > 0) & (u >= 0) & (u <= 1)
            dist = np.min(t[valid]) if np.any(valid) else MAX_RAY
            out[k] = min(dist, MAX_RAY) / MAX_RAY
        return out

    def _get_obs(self):
        sin, cos = self._heading_err()
        lat = self._lateral_offset() / max(self.half_w[self.idx], 1e-3)
        rays = self._rays if self.use_raycast else np.ones(N_RAYS, dtype=np.float32)
        obs = np.array([
            self.speed / MAX_SPEED,
            sin, cos,
            np.clip(lat, -2, 2),
            float(self.compound),
            self.temp,
            self.ers,
            *rays,
            self.wetness,
            self.rain_prob,
        ], dtype=np.float32)
        return np.clip(obs, -3.0, 3.0)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        start = int(self._rng.integers(self.N)) if self.random_start else 0
        self.idx      = start
        self.pos      = self.cl[start].copy()
        self.heading  = float(np.arctan2(self.tang[start, 1], self.tang[start, 0]))
        self.speed    = 0.0
        self.compound = DRY
        self.temp     = TIRE_AMB
        self.ers      = 0.8
        if self.random_weather and self._rng.random() < 0.4:
            self.raining = True
            self.wetness = float(self._rng.uniform(0.5, 1.0))
        else:
            self.raining = False
            self.wetness = 0.0
        self.rain_prob = (1 - P_RAIN2DRY) if self.raining else P_DRY2RAIN
        self.pit_requested = False

        self.laps     = 0
        self.steps    = 0
        self.progress = float(self.s[start])   # arc-length at the (possibly random) start
        self._prev_progress = self.progress
        self._speed_sum     = 0.0    # for episode mean-speed metric
        self._prev_steer    = 0.0
        self._jerk_sum      = 0.0    # sum of |Δsteer| -> smoothness metric
        self._overheat_steps = 0
        self._rays    = self._raycast()
        return self._get_obs(), {}

    def step(self, action):
        steer, pedal, ers_raw, pit_signal = [float(x) for x in action]
        ers_deploy = (ers_raw + 1.0) / 2.0       # map [-1,1] -> [0,1]
        self.steps += 1

        # grip loss when running Dry tires on a wet track
        slip = self.wetness > WET_TIRE_THRESH and self.compound == DRY
        grip = 0.6 if slip else 1.0

        # ── longitudinal dynamics ──
        if pedal >= 0:
            a = pedal * ACCEL + ers_deploy * ERS_BOOST * (self.ers > 0)
        else:
            a = pedal * BRAKE
        self.speed += a * grip * DT
        self.speed -= DRAG * self.speed
        self.speed = float(np.clip(self.speed, 0.0, MAX_SPEED))

        # ── ERS battery ──
        self.ers -= ers_deploy * ERS_USE * DT * (pedal >= 0)
        if pedal < 0:
            self.ers += ERS_REGEN * DT
        self.ers = float(np.clip(self.ers, 0.0, 1.0))

        # ── steering / heading (speed-scaled authority) ──
        authority = min(1.0, self.speed / YAW_VREF)
        self.heading += steer * MAX_YAW * authority * grip * DT
        self.heading = float((self.heading + np.pi) % (2 * np.pi) - np.pi)

        # ── integrate position ──
        self.pos = self.pos + self.speed * DT * np.array([np.cos(self.heading), np.sin(self.heading)])

        # ── tire temperature (heat from hard driving, cooling toward ambient) ──
        load = abs(pedal) * (self.speed / MAX_SPEED) + 0.5 * abs(steer)
        self.temp += (TIRE_HEAT * load - TIRE_COOL * (self.temp - TIRE_AMB)) * DT * TIRE_SCALE
        self.temp = float(np.clip(self.temp, 0.0, 1.0))

        # ── wetness drift toward current weather target ──
        target = 1.0 if self.raining else 0.0
        self.wetness += (target - self.wetness) * WET_RATE * DT * TIRE_SCALE
        self.wetness = float(np.clip(self.wetness, 0.0, 1.0))

        # ── track progress & lap / pit handling ──
        prev_idx = self.idx
        self.idx = self._nearest_idx()
        if pit_signal > PIT_THRESH:
            self.pit_requested = True

        crossed = prev_idx > 0.8 * self.N and self.idx < 0.2 * self.N
        reward = 0.0
        if crossed:
            self.laps += 1
            self.raining = (self._rng.random() < (P_DRY2RAIN if not self.raining else 1 - P_RAIN2DRY))
            self.rain_prob = P_DRY2RAIN if not self.raining else 1 - P_RAIN2DRY
            if self.pit_requested:
                self.compound = INTER if self.wetness > WET_TIRE_THRESH else DRY
                self.temp = TIRE_AMB
                self.pit_requested = False
                reward -= self.pit_pen

        # continuous progress in metres (project pos onto the current segment so
        # the dense reward is smooth even when the car moves < one waypoint/step)
        nxt = (self.idx + 1) % self.N
        seg = self.cl[nxt] - self.cl[self.idx]
        seglen = float(np.linalg.norm(seg))
        frac = np.clip(np.dot(self.pos - self.cl[self.idx], seg) / (seglen ** 2 + 1e-9), 0.0, 1.0)
        self.progress = self.laps * self.length_m + self.s[self.idx] + frac * seglen
        d_prog = self.progress - self._prev_progress
        if d_prog < -self.length_m / 2:      # guard against idx wrap glitch
            d_prog = 0.0
        self._prev_progress = self.progress
        reward += (self.progress_scale * max(d_prog, 0.0) - self.time_pen
                   + self.speed_reward * (self.speed / MAX_SPEED))

        # steering-smoothness penalty (|Δsteer| -> discourage jerky inputs)
        dsteer = abs(steer - self._prev_steer)
        self._prev_steer = steer
        self._jerk_sum += dsteer
        reward -= self.steer_pen * dsteer

        # penalties
        if self.temp > TEMP_OVERHEAT:
            reward -= self.overheat_pen
            self._overheat_steps += 1
        if slip:
            reward -= self.slip_pen
        self._speed_sum += self.speed

        # ── sensors / termination ──
        self._rays = self._raycast()
        lateral = abs(self._lateral_offset())
        off_track = lateral > self.half_w[self.idx]

        terminated = False
        if off_track:
            reward -= self.crash_pen
            terminated = True
        elif self.laps >= self.n_laps:
            reward += self.complete_bonus
            # reward finishing fast: full bonus for an instant lap, ->0 at the time limit
            reward += self.finish_time_bonus * max(1.0 - self.steps / self.max_steps, 0.0)
            terminated = True

        truncated = self.steps >= self.max_steps
        success = (not off_track) and (self.laps >= self.n_laps)
        info = {"laps": self.laps, "progress_m": self.progress,
                "speed": self.speed, "off_track": off_track,
                "crashed": bool(off_track), "success": bool(success),
                "mean_speed": self._speed_sum / max(self.steps, 1),
                "overheat_frac": self._overheat_steps / max(self.steps, 1),
                "jerk": self._jerk_sum / max(self.steps, 1)}
        return self._get_obs(), float(reward), terminated, truncated, info


# ── Self-test ───────────────────────────────────────────────────────────────────
def _main():
    env = F1DriverEnv(n_laps=3)
    print(f"obs_dim={env.obs_dim}  action={env.action_space.shape}  "
          f"track N={env.N}  lap={env.length_m:.0f} m  max_steps={env.max_steps}")

    from stable_baselines3.common.env_checker import check_env
    check_env(env, warn=True)
    print("check_env: PASSED")

    # random rollout
    obs, _ = env.reset(seed=0)
    total, steps = 0.0, 0
    term = trunc = False
    while not (term or trunc):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total += r; steps += 1
    print(f"random rollout: steps={steps} reward={total:.1f} "
          f"laps={info['laps']} off_track={info['off_track']}")

    # SB3 smoke test (PPO, tiny)
    from stable_baselines3 import PPO
    model = PPO("MlpPolicy", env, n_steps=256, batch_size=64, verbose=0)
    model.learn(total_timesteps=512)
    print("SB3 PPO smoke test: PASSED")


if __name__ == "__main__":
    _main()
