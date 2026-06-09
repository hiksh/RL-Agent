"""
baseline.py — non-RL control-theory baseline (pure-pursuit) for the F1 driver.

A geometric path tracker: steer toward a look-ahead point on the centreline
(pure pursuit, `delta ~ sin(alpha)/L_d`), with a physics-based corner-speed cap
(`v_max = MAX_YAW/kappa`) P-controlled on the pedal.  It deliberately ignores the
strategic layer (ERS/pit held neutral, no tire-temp management) — that gap is
exactly where RL earns its keep (it learns to deploy ERS for faster laps).

Compares pure-pursuit vs the trained RL look-ahead policy on the SAME axis
(idx=0, timeattack preset, deterministic policies, 3 seeds × dry/wet, 300 ep),
writing per-seed CSVs (eval_idx0 schema) + a summary figure.  Run: python baseline.py
"""
import os, csv, argparse
import numpy as np
from env import F1DriverEnv, MAX_YAW, YAW_VREF, BRAKE

HERE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIELDS  = ["timestep", "ep_reward", "ep_len", "laps", "success",
           "crashed", "mean_speed", "overheat_frac", "progress_m", "jerk"]
DT, N_LAPS = 0.25, 3

# match the RL Phase-3 comparison axis (train.py REWARD_PRESETS["timeattack"])
PRESET = dict(crash_pen=100.0, speed_reward=0.10, time_pen=0.06, finish_time_bonus=300.0)
# pure-pursuit gains chosen by tune() on the dry track (seed 0)
BEST_PARAMS = dict(ld_base=8.0, v_cap=55.0, margin=0.85, kp_v=0.6)
RL_MODEL    = "results/sac_timeattack_rs_sp0.0_la_seed{s}_best/best_model.zip"
SEEDS       = (0, 1, 2)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _max_curvature(env, dist_ahead):
    """Max |dθ/ds| of the centreline over the next dist_ahead metres (1/m)."""
    n = max(int(dist_ahead / env.length_m * env.N), 2)
    idxs = (env.idx + np.arange(n + 1)) % env.N
    th = np.unwrap(np.arctan2(env.tang[idxs, 1], env.tang[idxs, 0]))
    ds = np.diff(env.s[idxs]) % env.length_m + 1e-6
    return float(np.max(np.abs(np.diff(th) / ds)))


def pursuit_action(env, ld_base, v_cap, margin, kp_v):
    """Pure-pursuit steering + physics-based corner-speed P-control -> Box(4)."""
    ld = float(np.clip(ld_base + 0.5 * env.speed, 8.0, 32.0))   # speed-adaptive look-ahead
    j = int(np.searchsorted(env.s, (env.s[env.idx] + ld) % env.length_m) % env.N)
    vec = env.cl[j] - env.pos
    dist = float(np.linalg.norm(vec)) + 1e-6
    alpha = _wrap(np.arctan2(vec[1], vec[0]) - env.heading)

    authority = max(min(1.0, env.speed / YAW_VREF), 0.05)        # invert env yaw model
    omega_des = 2.0 * env.speed * np.sin(alpha) / dist
    steer = np.clip(omega_des / (MAX_YAW * authority), -1.0, 1.0)

    kappa = _max_curvature(env, max(40.0, 0.5 * env.speed ** 2 / BRAKE))
    v_target = min(v_cap, MAX_YAW / (kappa + 1e-3) * margin)     # corner speed from yaw limit
    pedal = np.clip(kp_v * (v_target - env.speed), -1.0, 1.0)

    return np.array([steer, pedal, -1.0, -1.0], dtype=np.float32)  # ERS/pit neutral


def _row(ep_r, ep_len, info):
    return {"timestep": 0, "ep_reward": round(ep_r, 2), "ep_len": ep_len,
            "laps": int(info.get("laps", 0)), "success": int(info.get("success", False)),
            "crashed": int(info.get("crashed", False)),
            "mean_speed": round(float(info.get("mean_speed", 0.0)), 3),
            "overheat_frac": round(float(info.get("overheat_frac", 0.0)), 4),
            "progress_m": round(float(info.get("progress_m", 0.0)), 1),
            "jerk": round(float(info.get("jerk", 0.0)), 4)}


def rollout(env, params, n_episodes, csv_path=None):
    rows = []
    for _ in range(n_episodes):
        env.reset(); done = trunc = False; ep_r = 0.0; info = {}
        while not (done or trunc):
            _, r, done, trunc, info = env.step(pursuit_action(env, **params))
            ep_r += float(r)
        rows.append(_row(ep_r, env.steps, info))
    if csv_path:
        _write(csv_path, rows)
    return rows


def eval_rl(model, env, n_episodes, csv_path=None):
    rows = []
    for _ in range(n_episodes):
        obs, _ = env.reset(); done = trunc = False; ep_r = 0.0; info = {}
        while not (done or trunc):
            act, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, info = env.step(act); ep_r += float(r)
        rows.append(_row(ep_r, env.steps, info))
    if csv_path:
        _write(csv_path, rows)
    return rows


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)


# ── tuning (run once to pick BEST_PARAMS) ─────────────────────────────────────
def tune(seed=0, n_episodes=40):
    env = F1DriverEnv(n_laps=N_LAPS, seed=seed, random_weather=False, **PRESET)
    best, best_key = None, None
    for ld_base in (4.0, 8.0, 14.0):
        for v_cap in (55.0, 70.0, 90.0):
            for margin in (0.7, 0.85, 1.0):
                p = dict(ld_base=ld_base, v_cap=v_cap, margin=margin, kp_v=0.6)
                rows = rollout(env, p, n_episodes)
                succ = np.mean([r["success"] for r in rows])
                crash = np.mean([r["crashed"] for r in rows])
                if best is None or (succ, -crash) > best_key:
                    best, best_key, best_params = (succ, crash), (succ, -crash), p
                print(f"  {p} -> success={succ:.2f} crash={crash:.2f}")
    print(f"BEST: {best_params}  (success={best[0]:.2f} crash={best[1]:.2f})")
    return best_params


# ── aggregation / reporting ───────────────────────────────────────────────────
CONFIGS = [  # (label, method, wet, csv_template)
    ("Pursuit\ndry", "pursuit", False, "pursuit_dry_seed{s}_eval0.csv"),
    ("RL-LA\ndry",   "rl",      False, "sac_timeattack_la_seed{s}_eval0.csv"),
    ("Pursuit\nwet", "pursuit", True,  "pursuit_wet_seed{s}_eval0.csv"),
    ("RL-LA\nwet",   "rl",      True,  "sac_timeattack_wx_la_seed{s}_eval0.csv"),
]


def _agg(tmpl):
    """Per-seed means -> (mean, std) across seeds for success / crash / laptime / overheat."""
    succ, crash, lap, oh = [], [], [], []
    for s in SEEDS:
        rs = [{k: float(v) for k, v in r.items()}
              for r in csv.DictReader(open(os.path.join(RESULTS, tmpl.format(s=s))))]
        succ.append(np.mean([r["success"] for r in rs]))
        crash.append(np.mean([r["crashed"] for r in rs]))
        oh.append(np.mean([r["overheat_frac"] for r in rs]))
        lt = [r["ep_len"] * DT / N_LAPS for r in rs if r["success"] == 1]
        if lt:
            lap.append(np.mean(lt))
    ms = lambda x: (float(np.mean(x)), float(np.std(x))) if x else (float("nan"), 0.0)
    return dict(success=ms(succ), crash=ms(crash), laptime=ms(lap), overheat=ms(oh))


def report():
    print(f"\n{'config':14s} {'lap(s)':>13} {'completion':>13} {'crash':>13} {'overheat':>13}")
    rows = {}
    for label, _, _, tmpl in CONFIGS:
        a = rows.setdefault(label.replace(chr(10), " "), _agg(tmpl))
        f = lambda k, sc=1: f"{a[k][0]*sc:.2f}±{a[k][1]*sc:.2f}"
        lp = "—" if a["laptime"][0] != a["laptime"][0] else f"{a['laptime'][0]:.1f}±{a['laptime'][1]:.1f}"
        print(f"{label.replace(chr(10),' '):14s} {lp:>13} {f('success'):>13} {f('crash'):>13} {f('overheat'):>13}")
    return rows


def figure():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    aggs = [_agg(tmpl) for *_, tmpl in CONFIGS]
    labels = [c[0] for c in CONFIGS]
    colors = ["#c44", "#28a", "#c44", "#28a"]
    panels = [("Completion rate ↑", "success", 1.0, "{:.2f}"),
              ("Lap time (s) ↓  [time-attack objective]", "laptime", None, "{:.0f}"),
              ("Crash rate ↓", "crash", 1.0, "{:.2f}")]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
    for a, (title, key, ymax, fmt) in zip(ax, panels):
        means = [g[key][0] for g in aggs]; stds = [g[key][1] for g in aggs]
        a.bar(labels, means, yerr=stds, color=colors, capsize=4)
        a.set_title(title); a.grid(axis="y", alpha=0.3)
        if ymax: a.set_ylim(0, ymax)
        off = (ymax or np.nanmax(means)) * 0.03
        for i, (m, sd) in enumerate(zip(means, stds)):
            if m == m: a.text(i, m + sd + off, fmt.format(m), ha="center", fontsize=9)
    fig.suptitle("Control-theory baseline (pure-pursuit) vs RL — why RL?  (3 seeds, deterministic)",
                 fontweight="bold")
    fig.tight_layout()
    out = os.path.join(RESULTS, "viz", "baseline_compare.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120); print(f"Saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--tune", action="store_true", help="re-run the pure-pursuit gain grid")
    ap.add_argument("--figure-only", action="store_true", help="rebuild figure/table from CSVs")
    args = ap.parse_args()

    if args.figure_only:
        report(); figure(); return
    if args.tune:
        tune(); return

    from stable_baselines3 import SAC
    for label, method, wet, tmpl in CONFIGS:
        for s in SEEDS:
            path = os.path.join(RESULTS, tmpl.format(s=s))
            if method == "pursuit":
                env = F1DriverEnv(n_laps=N_LAPS, seed=s, random_weather=wet, **PRESET)
                rollout(env, BEST_PARAMS, args.episodes, path)
            else:
                model = SAC.load(RL_MODEL.format(s=s), device="cpu")
                env = F1DriverEnv(n_laps=N_LAPS, seed=s, random_weather=wet,
                                  lookahead=True, **PRESET)
                eval_rl(model, env, args.episodes, path)
            print(f"  wrote {os.path.basename(path)}")
    report(); figure()


if __name__ == "__main__":
    main()
