#!/usr/bin/env python3
"""Remove cyan/white backdrop, crop, and fit topbar icons to a standard canvas."""
from PIL import Image
import numpy as np
import sys

OUT_W, OUT_H = 71, 80  # same canvas as topbar/shop.png
PAD = 1


def remove_bg_rgba(arr):
    rgb = arr[:, :, :3].astype(np.float32)
    corners = np.array([
        rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1],
        rgb[0, rgb.shape[1] // 2], rgb[-1, rgb.shape[1] // 2],
    ])
    key = corners.mean(axis=0)
    dist = np.linalg.norm(rgb - key, axis=2)
    white = np.linalg.norm(rgb - (255.0, 255.0, 255.0), axis=2)
    d = np.minimum(dist, white)
    thr, soft = 38.0, 22.0
    return np.clip((d - thr) / soft * 255.0, 0, 255).astype(np.uint8)


def crop_alpha(im, alpha_thr=12):
    a = np.array(im.split()[-1])
    ys, xs = np.where(a > alpha_thr)
    if len(xs) == 0:
        return im
    pad = max(2, int(min(im.size) * 0.01))
    x0 = max(0, xs.min() - pad)
    y0 = max(0, ys.min() - pad)
    x1 = min(im.width, xs.max() + 1 + pad)
    y1 = min(im.height, ys.max() + 1 + pad)
    return im.crop((x0, y0, x1, y1))


def fit_canvas(im):
    max_w = OUT_W - PAD * 2
    max_h = OUT_H - PAD * 2
    im.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    x = (OUT_W - im.width) // 2
    y = (OUT_H - im.height) // 2
    canvas.paste(im, (x, y), im)
    return canvas


def process(src, dst):
    im = Image.open(src).convert("RGBA")
    if max(im.size) > 512:
        im = im.resize((512, 512), Image.Resampling.LANCZOS)
    arr = np.array(im, dtype=np.uint8)
    arr[:, :, 3] = np.minimum(arr[:, :, 3], remove_bg_rgba(arr.astype(np.float32)))
    im = Image.fromarray(arr, "RGBA")
    im = crop_alpha(im)
    im = fit_canvas(im)
    im.save(dst, "PNG", optimize=True)
    a = np.array(im.split()[-1])
    ys, xs = np.where(a > 12)
    cw, ch = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    print(dst, f"content {cw}x{ch} in {OUT_W}x{OUT_H}")


def normalize_existing(src, dst):
    """Re-fit an icon that already has transparency (no cyan key)."""
    im = Image.open(src).convert("RGBA")
    if max(im.size) > 512:
        im = im.resize((512, 512), Image.Resampling.LANCZOS)
    im = crop_alpha(im)
    im = fit_canvas(im)
    im.save(dst, "PNG", optimize=True)
    print(dst, "normalized", OUT_W, OUT_H)


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--normalize":
        normalize_existing(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 3:
        process(sys.argv[1], sys.argv[2])
    else:
        print("Usage: key-topbar-icon.py <src> <dst>  OR  --normalize <src> <dst>")
        sys.exit(1)
