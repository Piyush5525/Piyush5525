"""Shared helpers for the animated GitHub-profile banner generator."""
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from scipy import ndimage

GRID_W, GRID_H = 300, 340


def load_portrait_crop(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    target_ratio = GRID_W / GRID_H
    top = int(h * 0.080)
    crop_h = int(w / target_ratio)
    if top + crop_h > h:
        crop_h = h - top
    box = (0, top, w, top + crop_h)
    return im.crop(box)


def to_processed_gray(crop_im):
    """Resize to the dot grid, then apply the exact tone pipeline from the spec."""
    gray = crop_im.convert("L").resize((GRID_W, GRID_H), Image.LANCZOS)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    gray = ImageEnhance.Contrast(gray).enhance(1.3)
    return gray


def foreground_mask(crop_im):
    """Segment the subject out of a flat backdrop: threshold on colour distance
    from the estimated background, close, fill holes, keep the largest blob."""
    small = crop_im.resize((GRID_W, GRID_H), Image.LANCZOS)
    arr = np.array(small).astype(np.float32)
    border = np.concatenate([
        arr[0:8, :, :].reshape(-1, 3),
        arr[:, 0:8, :].reshape(-1, 3),
        arr[:, -8:, :].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(arr - bg, axis=2)
    mask = dist > 28
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5)), iterations=2)
    mask = ndimage.binary_fill_holes(mask)
    labeled, n = ndimage.label(mask)
    if n > 0:
        sizes = ndimage.sum(mask, labeled, range(1, n + 1))
        biggest = np.argmax(sizes) + 1
        mask = labeled == biggest
    mask = ndimage.binary_erosion(mask, iterations=1)
    return mask


def floyd_steinberg_serpentine(gray_img):
    """1-bit Floyd-Steinberg dither in serpentine (boustrophedon) order.
    Returns a boolean array, True = ink (dot drawn)."""
    arr = np.array(gray_img).astype(np.float64)
    h, w = arr.shape
    out = np.zeros((h, w), dtype=bool)
    for y in range(h):
        left_to_right = (y % 2 == 0)
        xs = range(w) if left_to_right else range(w - 1, -1, -1)
        step = 1 if left_to_right else -1
        for x in xs:
            old = arr[y, x]
            new = 0.0 if old < 128 else 255.0
            out[y, x] = new < 128
            err = old - new
            nx = x + step
            if 0 <= nx < w:
                arr[y, nx] += err * 7 / 16
            if y + 1 < h:
                if 0 <= x - step < w:
                    arr[y + 1, x - step] += err * 3 / 16
                arr[y + 1, x] += err * 5 / 16
                if 0 <= nx < w:
                    arr[y + 1, nx] += err * 1 / 16
    return out


def dots_to_runs(dot_mask):
    """Run-length encode each row's consecutive True cells -> list of (y, x0, length)."""
    h, w = dot_mask.shape
    runs = []
    for y in range(h):
        row = dot_mask[y]
        x = 0
        while x < w:
            if row[x]:
                x0 = x
                while x < w and row[x]:
                    x += 1
                runs.append((y, x0, x - x0))
            else:
                x += 1
    return runs


def runs_to_path_d(runs, cell, dot_size, ox=0.0, oy=0.0, merge_threshold=2):
    """Build a single <path> `d` string. Isolated ink cells (run length below
    merge_threshold) are drawn as small gapped squares -- this is what reads
    as halftone dot texture in detailed/dithered areas. Longer runs (a solid
    swath like a dark sweater) are drawn as one edge-to-edge bar instead of
    one square per cell -- visually equivalent once cells are packed solid,
    but avoids emitting tens of thousands of near-touching path commands."""
    parts = []
    pad = (cell - dot_size) / 2.0
    for (y, x0, length) in runs:
        py = oy + y * cell + pad
        if length < merge_threshold:
            for i in range(length):
                px = ox + (x0 + i) * cell + pad
                parts.append(f"M{px:.2f} {py:.2f}h{dot_size:.2f}v{dot_size:.2f}h{-dot_size:.2f}Z")
        else:
            px = ox + x0 * cell
            w = length * cell
            parts.append(f"M{px:.2f} {oy+y*cell:.2f}h{w:.2f}v{cell:.2f}h{-w:.2f}Z")
    return "".join(parts)


def sample_points_from_glyph(png_path, n_points, box=1.0, seed=0):
    """Rasterized black-on-white glyph -> n_points sample locations inside the
    ink, normalised to [0, box]. Used for the traveler morph targets."""
    im = Image.open(png_path).convert("L")
    arr = np.array(im)
    ink_y, ink_x = np.where(arr < 128)
    if len(ink_x) == 0:
        raise ValueError(f"no ink found in {png_path}")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ink_x), size=n_points)
    xs = ink_x[idx].astype(np.float64) + rng.uniform(-0.4, 0.4, n_points)
    ys = ink_y[idx].astype(np.float64) + rng.uniform(-0.4, 0.4, n_points)
    xs = xs / arr.shape[1] * box
    ys = ys / arr.shape[0] * box
    return np.stack([xs, ys], axis=1)


def evenness_metric(groups, grid_w, grid_h, cells_per_group_positions):
    """Rough measure of how spatially scattered each group is (lower = more even
    spread per group, i.e. good interleaving). Returns mean normalized spatial
    variance shortfall vs. a fully-random reference."""
    variances = []
    ref = (grid_w ** 2 + grid_h ** 2) / 12.0
    for pts in cells_per_group_positions:
        if len(pts) < 2:
            continue
        v = np.var(pts[:, 0]) + np.var(pts[:, 1])
        variances.append(v / ref)
    return 1.0 - float(np.mean(variances)) if variances else 1.0
