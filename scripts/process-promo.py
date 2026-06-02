#!/usr/bin/env python3
"""Remove outer backdrop from wide promo banners (edge flood only — keeps glows)."""
from PIL import Image
import numpy as np
from collections import deque
import sys

MAX_W = 1100


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


def trim_transparent_border(im, max_strip=48):
    a = np.array(im.split()[-1])
    h, w = a.shape
    top = bottom = left = right = 0
    while top < max_strip and top < h and (a[top] == 0).all():
        top += 1
    while bottom < max_strip and bottom < h and (a[h - 1 - bottom] == 0).all():
        bottom += 1
    while left < max_strip and left < w and (a[:, left] == 0).all():
        left += 1
    while right < max_strip and right < w and (a[:, w - 1 - right] == 0).all():
        right += 1
    if top or bottom or left or right:
        im = im.crop((left, top, w - right, h - bottom))
    return im


def near_mask(mask, radius=1):
    out = mask.copy()
    for _ in range(radius):
        n = out.copy()
        n[1:, :] |= out[:-1, :]
        n[:-1, :] |= out[1:, :]
        n[:, 1:] |= out[:, :-1]
        n[:, :-1] |= out[:, 1:]
        out = n
    return out


def defringe_cyan(out, bg_mask):
    rgb = out[:, :, :3]
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    a = out[:, :, 3]
    cyan = (b > 160) & (g > 120) & (r < 120)
    edge = near_mask(bg_mask, radius=2) & (~bg_mask)
    fringe = edge & cyan
    # Aggressively fade cyan halos hugging transparent edges.
    out[fringe, 3] = (out[fringe, 3].astype(np.float32) * 0.12).astype(np.uint8)
    return out


def remove_outer_bg(im, tol=34):
    """Only remove background connected to image edges (never interior glows)."""
    rgb = np.array(im)[:, :, :3]
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    corners = np.array([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]], dtype=np.float32)
    key = corners.mean(axis=0)
    dist = np.linalg.norm(rgb.astype(np.float32) - key, axis=2)
    candidate = dist < tol
    # Cyan export backdrop (only when corners read as cyan).
    if key[2] > 180 and key[1] > 140 and key[0] < 90:
        candidate |= (b > 190) & (g > 150) & (r < 100)
    # Black letterbox margins on dark exports.
    candidate |= (r < 14) & (g < 14) & (b < 14)
    bg = flood_edge_connected(candidate)
    out = np.array(im)
    out[bg, 3] = 0
    out = defringe_cyan(out, bg)
    return Image.fromarray(out)


def process(src, dst):
    im = Image.open(src).convert("RGBA")
    im = remove_outer_bg(im)
    im = trim_transparent_border(im)
    if im.width > MAX_W:
        h = int(im.height * (MAX_W / im.width))
        im = im.resize((MAX_W, h), Image.Resampling.LANCZOS)
    im.save(dst, "PNG", optimize=True)
    a = np.array(im.split()[-1])
    print(dst, im.size, f"opaque {round(100 * (a > 20).sum() / a.size, 1)}%")


if __name__ == "__main__":
    process(sys.argv[1], sys.argv[2])
