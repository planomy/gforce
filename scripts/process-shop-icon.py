#!/usr/bin/env python3
"""Remove export backdrop + cyan fringe from pet shop item art."""
from PIL import Image
import numpy as np
from collections import deque
import sys

MAX_DIM = 320


def flood_edge_connected(candidate):
    h, w = candidate.shape
    visited = np.zeros((h, w), dtype=bool)
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if candidate[y, x] and not visited[y, x]:
                visited[y, x] = True
                q.append((x, y))
    for y in range(h):
        for x in (0, w - 1):
            if candidate[y, x] and not visited[y, x]:
                visited[y, x] = True
                q.append((x, y))
    while q:
        x, y = q.popleft()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < w and 0 <= ny < h and candidate[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                q.append((nx, ny))
    return visited


def strip_cyan_fringe(arr):
    r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
    cyan = (b.astype(np.int16) > g.astype(np.int16) + 10) & (b > 110) & (g > 70) & (r < 115)
    yellow = (r > 145) & (g > 115) & (b < 200)
    cyan &= ~yellow
    opaque = a > 20
    edge = opaque & (
        (~np.roll(opaque, 1, 0)) | (~np.roll(opaque, -1, 0))
        | (~np.roll(opaque, 1, 1)) | (~np.roll(opaque, -1, 1))
    )
    fringe = cyan & (edge | (a < 235))
    arr[fringe, 3] = (arr[fringe, 3].astype(np.float32) * 0.06).astype(np.uint8)
    return arr


def crop_baked_title_band(im):
    """Drop export title text baked into the bottom of shop art."""
    arr = np.array(im)
    h, w = arr.shape[0], arr.shape[1]
    a = arr[:, :, 3]
    rgb = arr[:, :, :3]
    title_top = None
    for y in range(h - 1, int(h * 0.35), -1):
        white = (
            (rgb[y, :, 0] > 198)
            & (rgb[y, :, 1] > 198)
            & (rgb[y, :, 2] > 198)
            & (a[y] > 100)
        ).sum()
        if white > w * 0.1:
            title_top = y
    if title_top is not None:
        arr = arr[: max(1, title_top - 4)]
    return Image.fromarray(arr, 'RGBA')


def crop_alpha(im, alpha_thr=10):
    a = np.array(im.split()[-1])
    ys, xs = np.where(a > alpha_thr)
    if len(xs) == 0:
        return im
    pad = max(6, int(min(im.size) * 0.02))
    return im.crop((
        max(0, xs.min() - pad), max(0, ys.min() - pad),
        min(im.width, xs.max() + 1 + pad), min(im.height, ys.max() + 1 + pad),
    ))


def process(src, dst):
    im = Image.open(src).convert('RGBA')
    im = crop_baked_title_band(im)
    arr = np.array(im, dtype=np.uint8)
    rgb = arr[:, :, :3].astype(np.float32)
    corners = np.array([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]], dtype=np.float32)
    key = corners.mean(axis=0)
    dist = np.linalg.norm(rgb - key, axis=2)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    is_green_screen = key[1] > key[0] + 30 and key[1] > key[2] + 30 and key[1] > 120
    is_cyan_screen = key[2] > 180 and key[1] > 140 and key[0] < 90
    if is_green_screen:
        # Strong green excess only — keeps yellow moon off the key mask.
        excess_g = g.astype(np.int16) - np.maximum(r, b)
        candidate = excess_g > 75
    elif is_cyan_screen:
        candidate = (dist < 42) | ((b > 185) & (g > 145) & (r < 100))
    else:
        candidate = dist < 40
    candidate |= (r < 12) & (g < 12) & (b < 12)
    bg = flood_edge_connected(candidate)
    arr[bg, 3] = 0
    arr = strip_cyan_fringe(arr)
    im = Image.fromarray(arr, 'RGBA')
    im = crop_alpha(im)
    w, h = im.size
    scale = min(1.0, MAX_DIM / max(w, h))
    if scale < 1.0:
        im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    im.save(dst, 'PNG', optimize=True)
    print(dst, im.size)


if __name__ == '__main__':
    process(sys.argv[1], sys.argv[2])
