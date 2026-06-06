"""
visualize.py — figures & animations for the F1DriverEnv Deep-RL results.

Consumes results/ artifacts produced by run_all.sh (tags like
`<algo>_racing_seed<n>`, `sac_no_shaping_seed0`, `sac_racing_noray_seed0`, ...)
and produces report figures into results/viz/.

    python visualize.py        # auto-detects whatever models / metric CSVs exist

Produces (when the matching runs exist):
  track_map.png                 track layout
  traj_<algo>.png               racing line of each main algorithm (speed-coloured)
  drive_sac.gif                 animated lap with raycasts + telemetry
  learning_curves.png           main comparison, mean±std over seeds
  metric_comparison.png         success / crash / reward / lap-time bars
  reward_ablation.png           SAC across reward presets (design validity)
  raycast_ablation.png          SAC racing with vs without the raycast sensor
  eval_curve.png                EvalCallback curve (if present)
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.animation import FuncAnimation

from env import (F1DriverEnv, RAY_ANGLES, MAX_RAY, MAX_SPEED, TEMP_OVERHEAT, DRY, INTER, DT)
from wrappers import DiscretizedF1Driver

HERE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
VIZ     = os.path.join(RESULTS, "viz")
TRACK   = os.path.join(HERE, "assets", "track.npz")

ALGO_CLS    = {}
ALGO_COLORS = {"dqn": "#1E88E5", "ppo": "#F5A623", "sac": "#D0021B", "td3": "#7ED321"}
ALGO_NAMES  = {"dqn": "DQN", "ppo": "PPO", "sac": "SAC", "td3": "TD3"}
COMP_NAMES  = {DRY: "Dry", INTER: "Inter"}
N_LAPS_DISPLAY = 3


def _algo_cls(algo):
    if not ALGO_CLS:
        from stable_baselines3 import DQN, PPO, SAC, TD3
        ALGO_CLS.update(dqn=DQN, ppo=PPO, sac=SAC, td3=TD3)
    return ALGO_CLS[algo]


# ── Track drawing ────────────────────────────────────────────────────────────
def _track():
    """Centerline, the half-width the env uses for off-track checks, and the unit
    normal at the start line. (raw left/right contours have corner spikes; we render
    from the centerline + half_width instead.)"""
    d = np.load(TRACK)
    cl = d["centerline"].astype(float); hw = d["half_width"].astype(float)
    tang = np.roll(cl, -1, 0) - np.roll(cl, 1, 0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
    return cl, hw, nrm

def _road_path(cl, hw, extra, nseg=22):
    """Buffer tube around the centerline (= the env's drivable region, lateral ≤
    half_width) as ONE filled path: union of per-vertex disks + per-segment capsules
    (all CCW so nonzero-winding fills the union). Never self-intersects into notches
    at hairpins, unlike offsetting the edges directly; one artist keeps it fast."""
    th = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
    unit = np.column_stack([np.cos(th), np.sin(th)])
    V, C = [], []
    def add(poly):
        V.append(poly); V.append(np.zeros((1, 2)))           # +dummy vertex for CLOSEPOLY
        C.extend([Path.MOVETO] + [Path.LINETO] * (len(poly) - 1) + [Path.CLOSEPOLY])
    for i, (x, y) in enumerate(cl):
        add(np.array([x, y]) + (hw[i] + extra) * unit)        # disk
    for i in range(len(cl)):
        a = cl[i]; b = cl[(i + 1) % len(cl)]; t = b - a; L = float(np.hypot(*t))
        if L < 1e-6:
            continue
        n = np.array([-t[1], t[0]]) / L * (hw[i] + extra)
        add(np.array([a - n, b - n, b + n, a + n]))           # capsule (CCW)
    return Path(np.concatenate(V), C)

def draw_track(ax):
    cl, hw, nrm = _track()
    # white kerb underlay (slightly wider) then asphalt on top -> uniform clean border
    ax.add_patch(PathPatch(_road_path(cl, hw, 4.0), facecolor="#f2f2f2", edgecolor="none", zorder=1))
    ax.add_patch(PathPatch(_road_path(cl, hw, 0.0), facecolor="#34383f", edgecolor="none", zorder=2))
    ax.autoscale_view()
    # subtle dashed centerline (kept faint so overlaid racing lines stand out)
    clc = np.vstack([cl, cl[0]])
    ax.plot(clc[:, 0], clc[:, 1], color="#FFD54F", lw=1.0, alpha=0.4,
            dashes=(5, 7), zorder=3)
    # start/finish line (perpendicular) + marker
    s, n = cl[0], nrm[0] * hw[0]
    ax.plot([s[0]-n[0], s[0]+n[0]], [s[1]-n[1], s[1]+n[1]],
            color="#00E676", lw=3.0, zorder=5, solid_capstyle="round")
    ax.plot(s[0], s[1], "o", color="#00E676", ms=9, zorder=6,
            markeredgecolor="white", markeredgewidth=1.4)
    ax.set_aspect("equal"); ax.axis("off"); ax.margins(0.03)

def plot_track_map():
    fig, ax = plt.subplots(figsize=(11, 8)); fig.patch.set_facecolor("white")
    draw_track(ax); ax.set_title("F1DriverEnv — Reconstructed 2-D Track", fontsize=13)
    _save(fig, os.path.join(VIZ, "track_map.png"))


# ── Rollout a trained policy (records true env state) ────────────────────────
def rollout(model, algo, n_laps=3, seed=7, deterministic=True):
    base = F1DriverEnv(n_laps=n_laps)
    env  = DiscretizedF1Driver(base) if algo == "dqn" else base
    obs, _ = env.reset(seed=seed)
    frames, cum, term, trunc = [], 0.0, False, False
    while not (term or trunc):
        a, _ = model.predict(obs, deterministic=deterministic)
        cont = (DiscretizedF1Driver.ACTION_TABLE[int(a)] if algo == "dqn"
                else np.asarray(a, dtype=float))
        obs, r, term, trunc, info = env.step(a); cum += r
        frames.append(dict(x=base.pos[0], y=base.pos[1], speed=base.speed,
                           heading=base.heading, rays=base._rays.copy(),
                           steer=float(cont[0]), pedal=float(cont[1]),
                           ers=base.ers, temp=base.temp, compound=int(base.compound),
                           wetness=base.wetness, lap=base.laps, reward=r, cum=cum,
                           crashed=bool(info["crashed"]), success=bool(info["success"])))
    return frames

def plot_trajectory(frames, algo, save):
    xs = np.array([f["x"] for f in frames]); ys = np.array([f["y"] for f in frames])
    sp = np.array([f["speed"] for f in frames])
    fig, ax = plt.subplots(figsize=(11, 8)); fig.patch.set_facecolor("white")
    draw_track(ax)
    pts = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="turbo", lw=2.6, zorder=4)
    lc.set_array(sp); lc.set_clim(0, MAX_SPEED); ax.add_collection(lc)
    if frames[-1]["crashed"]:
        ax.plot(xs[-1], ys[-1], "X", color="red", ms=15, zorder=7,
                markeredgecolor="white", markeredgewidth=1.4)
    fig.colorbar(lc, ax=ax, fraction=0.04, pad=0.02).set_label("Speed (m/s)")
    res = "completed" if frames[-1]["success"] else ("crashed" if frames[-1]["crashed"] else "truncated")
    ax.set_title(f"{ALGO_NAMES.get(algo, algo)} racing line — laps={frames[-1]['lap']}, "
                 f"reward={frames[-1]['cum']:.0f}, {res}", fontsize=12)
    _save(fig, save)

def animate_drive(frames, algo, save, stride=2, fps=20):
    fr = frames[::stride]
    fig = plt.figure(figsize=(15, 8), facecolor="white")
    ax_t = fig.add_axes([0.01, 0.03, 0.62, 0.94]); ax_i = fig.add_axes([0.66, 0.06, 0.32, 0.88])
    draw_track(ax_t)
    car,   = ax_t.plot([], [], "o", ms=11, color="#FF1744", zorder=8,
                       markeredgecolor="white", markeredgewidth=1.3)
    trail, = ax_t.plot([], [], "-", color="#FF8A80", lw=1.4, alpha=0.8, zorder=5)
    rays_l = [ax_t.plot([], [], "-", color="#00E5FF", lw=1.0, alpha=0.8, zorder=6)[0] for _ in RAY_ANGLES]

    def panel(f):
        ax_i.cla(); ax_i.set_xlim(0, 1); ax_i.set_ylim(0, 1); ax_i.axis("off")
        ax_i.add_patch(plt.Rectangle((0, 0.92), 1, 0.08, color="#2c2c2c"))
        ax_i.text(0.5, 0.96, f"{ALGO_NAMES.get(algo, algo)}  TELEMETRY", ha="center",
                  va="center", color="white", fontsize=12, fontweight="bold")
        def bar(y, lab, val, vmax, col, txt):
            ax_i.text(0.0, y, lab, fontsize=9, color="#444", va="center")
            ax_i.add_patch(plt.Rectangle((0.34, y-0.02), 0.5, 0.04, color="#e2e2e2"))
            ax_i.add_patch(plt.Rectangle((0.34, y-0.02), 0.5*np.clip(val/vmax, 0, 1), 0.04, color=col))
            ax_i.text(0.86, y, txt, fontsize=8.5, color=col, va="center", fontweight="bold")
        bar(0.82, "Speed", f["speed"], MAX_SPEED, "#1565C0", f"{f['speed']:.0f} m/s")
        bar(0.74, "ERS", f["ers"], 1.0, "#2E7D32", f"{f['ers']*100:.0f}%")
        tcol = "#C62828" if f["temp"] > TEMP_OVERHEAT else "#F9A825"
        bar(0.66, "Tire °", f["temp"], 1.0, tcol, "OVERHEAT" if f["temp"] > TEMP_OVERHEAT else f"{f['temp']:.2f}")
        bar(0.58, "Wetness", f["wetness"], 1.0, "#0277BD", f"{f['wetness']*100:.0f}%")
        ax_i.text(0.0, 0.48, "Compound", fontsize=9, color="#444", va="center")
        ax_i.text(0.34, 0.48, COMP_NAMES[f["compound"]], fontsize=10, fontweight="bold",
                  color="#E53935" if f["compound"] == DRY else "#1E88E5", va="center")
        ax_i.text(0.0, 0.40, "Lap", fontsize=9, color="#444", va="center")
        ax_i.text(0.34, 0.40, f"{f['lap']+1} / {N_LAPS_DISPLAY}", fontsize=10, fontweight="bold", va="center")
        ax_i.text(0.0, 0.28, "Steer", fontsize=9, color="#444", va="center")
        ax_i.add_patch(plt.Rectangle((0.34, 0.26), 0.5, 0.04, color="#e2e2e2"))
        ax_i.add_patch(plt.Rectangle((0.59, 0.26), 0.25*f["steer"], 0.04, color="#6A1B9A"))
        ax_i.text(0.0, 0.20, "Pedal", fontsize=9, color="#444", va="center")
        ax_i.add_patch(plt.Rectangle((0.34, 0.18), 0.5, 0.04, color="#e2e2e2"))
        ax_i.add_patch(plt.Rectangle((0.59, 0.18), 0.25*f["pedal"], 0.04,
                                     color="#2E7D32" if f["pedal"] >= 0 else "#C62828"))
        ax_i.text(0.0, 0.08, "Reward", fontsize=9, color="#444", va="center")
        ax_i.text(0.34, 0.08, f"{f['cum']:.0f}", fontsize=11, fontweight="bold", va="center")

    def update(i):
        f = fr[i]; car.set_data([f["x"]], [f["y"]])
        trail.set_data([g["x"] for g in fr[:i+1]], [g["y"] for g in fr[:i+1]])
        for k, ang in enumerate(RAY_ANGLES):
            a = f["heading"]+ang; d = f["rays"][k]*MAX_RAY
            rays_l[k].set_data([f["x"], f["x"]+d*np.cos(a)], [f["y"], f["y"]+d*np.sin(a)])
        panel(f); return [car, trail, *rays_l]

    FuncAnimation(fig, update, frames=len(fr), interval=1000//fps, blit=False).save(
        save, writer="pillow", fps=fps); plt.close()
    print(f"Saved: {save}")


# ── Metrics loading / aggregation across seeds ───────────────────────────────
def _smooth(x, w):
    w = min(w, len(x)); return np.convolve(x, np.ones(w)/w, mode="valid") if w > 0 else x

def _seed_csvs(tag):
    """All per-seed metric CSVs for a tag (everything except the _seed<n> suffix)."""
    paths = sorted(glob.glob(os.path.join(RESULTS, f"{tag}_seed*_metrics.csv")))
    return [np.genfromtxt(p, delimiter=",", names=True) for p in paths]

def _final(arrays, key, last=300):
    vals = [np.atleast_1d(a[key]).astype(float)[-last:].mean() for a in arrays if a.size]
    return (np.mean(vals), np.std(vals)) if vals else (np.nan, 0.0)

def _laptime(arrays, last=300):
    """Mean lap time in SECONDS among successful episodes (the true race metric).
    ep_len is steps for the whole race -> seconds = steps*DT, per lap /N_LAPS."""
    out = []
    for a in arrays:
        s = np.atleast_1d(a["success"])[-last:].astype(bool)
        ln = np.atleast_1d(a["ep_len"])[-last:]
        if s.any():
            out.append(ln[s].mean() * DT / N_LAPS_DISPLAY)
    return (np.mean(out), np.std(out)) if out else (np.nan, 0.0)


# ── Comparison figures ───────────────────────────────────────────────────────
def plot_learning_curves(groups, save, w=100):
    """groups: list of (label, tag, color). x-axis is environment timestep so
    algorithms with very different episode counts (DQN crashes early -> thousands
    of short episodes) stay comparable."""
    fig, ax = plt.subplots(figsize=(11, 4.5)); fig.patch.set_facecolor("white")
    any_data = False
    for label, tag, color in groups:
        arrays = [a for a in _seed_csvs(tag) if a.size]
        ys = [_smooth(np.atleast_1d(a["ep_reward"]).astype(float), w) for a in arrays]
        xs = [np.atleast_1d(a["timestep"]).astype(float)[w-1:w-1+len(y)]
              for a, y in zip(arrays, ys)]
        pairs = [(x, y) for x, y in zip(xs, ys) if len(y)]
        if not pairs:
            continue
        any_data = True
        n = min(len(y) for _, y in pairs)
        x = np.vstack([p[0][:n] for p in pairs]).mean(0)
        M = np.vstack([p[1][:n] for p in pairs]); m, s = M.mean(0), M.std(0)
        ax.plot(x, m, color=color, lw=1.8, label=label)
        ax.fill_between(x, m-s, m+s, color=color, alpha=0.15)
    if not any_data:
        plt.close(fig); return
    ax.set_xlabel("Timestep"); ax.set_ylabel("Episode reward (smoothed)")
    ax.set_title("Learning Curves (mean ± std over seeds)"); ax.legend(); ax.grid(alpha=0.3)
    _save(fig, save)

def plot_bar_comparison(groups, save, title):
    """4-panel: success% / crash% / reward / steps-to-finish."""
    data = [(lab, _seed_csvs(tag), col) for lab, tag, col in groups]
    data = [d for d in data if d[1]]
    if not data:
        return
    labels = [d[0] for d in data]; cols = [d[2] for d in data]
    succ = [(_final(a, "success")[0]*100, _final(a, "success")[1]*100) for _, a, _ in data]
    crash = [(_final(a, "crashed")[0]*100, _final(a, "crashed")[1]*100) for _, a, _ in data]
    rew = [_final(a, "ep_reward") for _, a, _ in data]
    lap = [_laptime(a) for _, a, _ in data]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8)); fig.patch.set_facecolor("white")
    for a, vals, ttl, fmt in [(ax[0,0], lap, "Mean lap time (s) ↓  [true objective]", "{:.1f}"),
                              (ax[0,1], succ, "Success rate (%) ↑", "{:.0f}"),
                              (ax[1,0], crash, "Crash rate (%) ↓", "{:.0f}"),
                              (ax[1,1], rew, "Mean episode reward (surrogate)", "{:.0f}")]:
        m = [v[0] for v in vals]; e = [v[1] for v in vals]
        bars = a.bar(labels, m, yerr=e, capsize=4, color=cols, alpha=0.85, edgecolor="white")
        for b, v in zip(bars, m):
            if np.isfinite(v):
                a.text(b.get_x()+b.get_width()/2, v, fmt.format(v), ha="center",
                       va="bottom", fontsize=9, fontweight="bold")
        a.set_title(ttl); a.grid(axis="y", alpha=0.3)
    fig.suptitle(title, fontsize=13)
    _save(fig, save)

def plot_eval_curve():
    p = os.path.join(RESULTS, "eval", "evaluations.npz")
    if not os.path.exists(p):
        return
    d = np.load(p); ts = d["timesteps"]; res = d["results"]
    fig, ax = plt.subplots(figsize=(10, 4)); fig.patch.set_facecolor("white")
    ax.plot(ts, res.mean(1), color="#D0021B", lw=1.8)
    ax.fill_between(ts, res.mean(1)-res.std(1), res.mean(1)+res.std(1), alpha=0.2, color="#D0021B")
    ax.set_xlabel("Timestep"); ax.set_ylabel("Eval reward"); ax.grid(alpha=0.3)
    ax.set_title("Evaluation reward (last run)")
    _save(fig, os.path.join(VIZ, "eval_curve.png"))


# ── helpers / main ───────────────────────────────────────────────────────────
def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {path}")

def _best_model(tag_seed):
    for c in (os.path.join(RESULTS, f"{tag_seed}_best", "best_model.zip"),
              os.path.join(RESULTS, f"{tag_seed}.zip")):
        if os.path.exists(c):
            return c
    return None

MAIN = [("DQN", "dqn_racing", "#1E88E5"), ("PPO", "ppo_racing", "#F5A623"),
        ("SAC", "sac_racing", "#D0021B"), ("TD3", "td3_racing", "#7ED321")]
REWARD_ABL = [("baseline", "sac", "#9E9E9E"), ("no_shaping", "sac_no_shaping", "#90CAF9"),
              ("aggressive", "sac_aggressive", "#FFB74D"), ("racing", "sac_racing", "#D0021B")]
RAYCAST_ABL = [("raycast ON", "sac_racing", "#2E7D32"), ("raycast OFF", "sac_racing_noray", "#C62828")]
# Phase-2 time-attack reward study (vs the `racing` baseline) + warm-start fine-tune.
TIMEATTACK  = [("racing", "sac_racing", "#9E9E9E"), ("ta_dense", "sac_timeattack_dense", "#90CAF9"),
               ("ta_finish", "sac_timeattack_finish", "#FFB74D"), ("timeattack", "sac_timeattack", "#D0021B")]
FINETUNE    = [("racing (base)", "sac_racing", "#9E9E9E"), ("timeattack FT", "sac_timeattack_ft", "#D0021B")]


def main():
    os.makedirs(VIZ, exist_ok=True)
    plot_track_map()

    # rollouts (best model, seed 0) for each main algorithm that exists
    for name, tag, _ in MAIN:
        algo = tag.split("_")[0]
        mp = _best_model(f"{tag}_seed0")
        if not mp:
            continue
        print(f"[{algo}] rollout {os.path.relpath(mp, HERE)}")
        frames = rollout(_algo_cls(algo).load(mp, device="cpu"), algo)
        plot_trajectory(frames, algo, os.path.join(VIZ, f"traj_{algo}.png"))
        if algo == "sac":                       # animate the headline method only
            animate_drive(frames, algo, os.path.join(VIZ, "drive_sac.gif"), stride=4)

    plot_learning_curves(MAIN, os.path.join(VIZ, "learning_curves.png"))
    plot_bar_comparison(MAIN, os.path.join(VIZ, "metric_comparison.png"),
                        "Algorithm Comparison (mean ± std over seeds, last 300 eps)")
    plot_bar_comparison(REWARD_ABL, os.path.join(VIZ, "reward_ablation.png"),
                        "SAC Reward-shaping Ablation")
    plot_bar_comparison(RAYCAST_ABL, os.path.join(VIZ, "raycast_ablation.png"),
                        "SAC Raycast Sensor Ablation")
    plot_bar_comparison(TIMEATTACK, os.path.join(VIZ, "timeattack_ablation.png"),
                        "SAC Time-attack Reward Study (true objective)")
    plot_bar_comparison(FINETUNE, os.path.join(VIZ, "finetune_compare.png"),
                        "SAC racing -> time-attack fine-tune")
    plot_eval_curve()
    print("\nDone. Figures in results/viz/")


if __name__ == "__main__":
    main()
