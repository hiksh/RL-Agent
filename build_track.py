"""
build_track.py  —  Extract a drivable 2-D centreline from the F1 track image.

Pipeline:  track.webp  ->  assets/track.npz   (centreline / boundaries / width)
                       ->  assets/track_preview.png  (visual sanity check)

The track is a closed black band (an annulus).  We extract its outer and inner
contours and take their mid-line as the racing centreline; the gap between the
two contours gives the local track width.  Contours from skimage are already
ordered closed loops, so no fragile graph-walking is needed.

Re-run this whenever the source image is edited.  The environment (env.py)
loads only the .npz artifact, so geometry can be iterated on freely without
touching the env or training code.

    python build_track.py [path/to/track.webp]

Tunable parameters are grouped in the CONFIG block below.
"""

import os
import sys
import warnings
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize, closing, disk
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.measure import find_contours

warnings.filterwarnings("ignore", category=FutureWarning)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMG = os.path.join(HERE, "assets", "track.webp")
OUT_NPZ     = os.path.join(HERE, "assets", "track.npz")
OUT_PREVIEW = os.path.join(HERE, "assets", "track_preview.png")

DARK_MAX        = 70      # pixel is "track band" if max(R,G,B) < DARK_MAX
CLOSE_RADIUS    = 4       # morphological closing radius (bridges colour-stripe gaps)
HOLE_AREA       = 4000    # fill enclosed holes up to this area (corner circles, stripes)
MIN_OBJECT      = 2000    # drop connected components smaller than this (labels, legend)
N_WAYPOINTS     = 400     # resampled centreline resolution
SMOOTH_PASSES   = 4       # moving-average smoothing passes on the centreline
TARGET_LENGTH_M = 5000.0  # scale the lap so total centreline length == this (metres)
WIDTH_LOW_PCT   = 10      # clamp half-width below this percentile (avoid pinch points)
WIDTH_HIGH_PCT  = 80      # clamp half-width above this percentile (kills corner balloons)
WIDTH_SMOOTH    = 8       # moving-average passes on half-width
# ───────────────────────────────────────────────────────────────────────────────


def load_band_mask(img_path):
    """Binary mask of the black track band (largest connected component)."""
    img = Image.open(img_path).convert("RGB")
    arr = np.asarray(img)
    band = arr.max(axis=2) < DARK_MAX

    band = closing(band, disk(CLOSE_RADIUS))
    band = remove_small_holes(band, area_threshold=HOLE_AREA)
    band = remove_small_objects(band, min_size=MIN_OBJECT)

    lbl, n = ndimage.label(band)
    if n == 0:
        raise RuntimeError("No track band found — lower DARK_MAX or check the image.")
    sizes = ndimage.sum(np.ones_like(lbl), lbl, index=range(1, n + 1))
    band = lbl == (1 + int(np.argmax(sizes)))
    return arr, band


def _resample_closed(loop, n):
    """Uniform arc-length resample of a closed polyline (rows,cols) to n points."""
    if np.allclose(loop[0], loop[-1]):          # drop duplicate closing vertex
        loop = loop[:-1]
    closed = np.vstack([loop, loop[0]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    targets = np.linspace(0, s[-1], n, endpoint=False)
    out = np.empty((n, 2))
    for d in range(2):
        out[:, d] = np.interp(targets, s, closed[:, d])
    return out


def centerline_from_contours(band, n):
    """Mid-line between the band's outer and inner contours, plus half-width."""
    contours = find_contours(band.astype(float), 0.5)
    if len(contours) < 2:
        raise RuntimeError("Expected an inner+outer track boundary; got "
                           f"{len(contours)} contour(s). Check mask parameters.")
    contours.sort(key=len, reverse=True)
    outer = _resample_closed(contours[0], n)
    inner = _resample_closed(contours[1], 4 * n)        # fine for nearest matching

    tree = cKDTree(inner)
    dist, idx = tree.query(outer)
    centre   = (outer + inner[idx]) / 2.0
    half_w   = dist / 2.0
    return centre, half_w


def smooth_closed(pts, passes):
    """Periodic 3-tap moving average (works for (N,2) points or (N,) scalars)."""
    for _ in range(passes):
        pts = (np.roll(pts, 1, axis=0) + pts + np.roll(pts, -1, axis=0)) / 3.0
    return pts


def clean_width(half_w):
    """Clamp half-width to a percentile band, then smooth — removes the
    rounded 'balloons' the nearest-contour match produces at tight corners."""
    lo = np.percentile(half_w, WIDTH_LOW_PCT)
    hi = np.percentile(half_w, WIDTH_HIGH_PCT)
    return smooth_closed(np.clip(half_w, lo, hi), WIDTH_SMOOTH)


def main():
    img_path = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG)
    print(f"Source image : {img_path}")

    arr, band = load_band_mask(img_path)
    skel = skeletonize(band)                              # for preview overlay only

    cl_px, half_w_px = centerline_from_contours(band, N_WAYPOINTS)
    cl_px = smooth_closed(cl_px, SMOOTH_PASSES)
    half_w_px = clean_width(half_w_px)
    print(f"Centreline   : {len(cl_px)} waypoints")

    # image (row, col) -> world (x, y) with y up; scale so lap == TARGET_LENGTH_M
    world = np.column_stack([cl_px[:, 1], band.shape[0] - cl_px[:, 0]])
    seg = np.linalg.norm(np.diff(np.vstack([world, world[0]]), axis=0), axis=1)
    scale = TARGET_LENGTH_M / seg.sum()
    centerline = world * scale
    half_width = half_w_px * scale

    # left/right boundaries from centreline normals
    tang = np.roll(centerline, -1, axis=0) - np.roll(centerline, 1, axis=0)
    tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)
    normal = np.column_stack([-tang[:, 1], tang[:, 0]])
    left  = centerline + normal * half_width[:, None]
    right = centerline - normal * half_width[:, None]

    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez(OUT_NPZ,
             centerline=centerline.astype(np.float32),
             left=left.astype(np.float32),
             right=right.astype(np.float32),
             half_width=half_width.astype(np.float32),
             length_m=np.float32(TARGET_LENGTH_M),
             scale=np.float32(scale))
    print(f"Lap length   : {TARGET_LENGTH_M:.0f} m   |  mean width: {2*half_width.mean():.1f} m "
          f"(min {2*half_width.min():.1f}, max {2*half_width.max():.1f})")
    print(f"Saved        : {OUT_NPZ}")

    _save_preview(arr, band, skel, cl_px, centerline, left, right)


def _save_preview(arr, band, skel, cl_px, centerline, left, right):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor("white")

    axes[0, 0].imshow(arr);               axes[0, 0].set_title("1. Source image")
    axes[0, 1].imshow(band, cmap="gray")
    ys, xs = np.nonzero(skel)
    axes[0, 1].plot(xs, ys, ".", color="#00E5FF", ms=0.5)
    axes[0, 1].set_title("2. Track band mask + skeleton")

    axes[1, 0].imshow(arr, alpha=0.35)
    axes[1, 0].plot(cl_px[:, 1], cl_px[:, 0], "-", color="#00E5FF", lw=2)
    axes[1, 0].plot(cl_px[0, 1], cl_px[0, 0], "o", color="red", ms=8)
    axes[1, 0].set_title("3. Extracted centreline (over image)")

    ax = axes[1, 1]
    ax.plot(centerline[:, 0], centerline[:, 1], "-", color="#1a1a1a", lw=1.5, label="centreline")
    ax.plot(left[:, 0],  left[:, 1],  "-", color="#1E88E5", lw=0.8, label="left edge")
    ax.plot(right[:, 0], right[:, 1], "-", color="#D0021B", lw=0.8, label="right edge")
    ax.plot(centerline[0, 0], centerline[0, 1], "o", color="green", ms=9, label="start")
    ax.set_aspect("equal"); ax.legend(fontsize=8)
    ax.set_title("4. World-space track (metres)")

    for a in axes[0]:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_PREVIEW, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Preview      : {OUT_PREVIEW}")


if __name__ == "__main__":
    main()
