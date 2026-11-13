import math

import numpy as np

from .term import pal_array, rgb
from .theory import TAU
from .world import WORLD, wrng, wseed


# ════════════════════════════════════════════════════════════════════════════
#  Pixel painting helpers (all numpy, coordinates in pixels)
# ════════════════════════════════════════════════════════════════════════════
_GRIDS = {}


def grid(H, W):
    g = _GRIDS.get((H, W))
    if g is None:
        yy, xx = np.mgrid[0:H, 0:W]
        g = _GRIDS[(H, W)] = (yy.astype(np.float32), xx.astype(np.float32))
    return g


def _c(col):
    return np.asarray(col, np.float32)


def vgrad(n, stops):
    """n colours down a gradient; stops = [(pos 0…1, (r, g, b)), …]."""
    pos = np.linspace(0, 1, max(1, n))
    ps = [p for p, _ in stops]
    return np.stack([np.interp(pos, ps, [c[i] for _, c in stops]) for i in range(3)], -1).astype(np.float32)


def fill_grad(P, y0, y1, stops):
    y0, y1 = max(0, int(y0)), min(P.shape[0], int(y1))
    if y1 > y0:
        P[y0:y1] = vgrad(y1 - y0, stops)[:, None, :]


def grad_at(H, y0, y1, stops):
    """(H, 3) gradient that runs from row y0 to y1 and clamps outside."""
    pos = np.clip((np.arange(H) - y0) / max(1, y1 - y0), 0, 1)
    ps = [p for p, _ in stops]
    return np.stack([np.interp(pos, ps, [c[i] for _, c in stops]) for i in range(3)], -1).astype(np.float32)


def _box(P, x0, y0, x1, y1):
    H, W = P.shape[:2]
    x0, y0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
    x1, y1 = min(W, int(math.ceil(x1))), min(H, int(math.ceil(y1)))
    if x0 >= x1 or y0 >= y1:
        return None
    yy, xx = grid(H, W)
    return P[y0:y1, x0:x1], yy[y0:y1, x0:x1], xx[y0:y1, x0:x1]


def glow(P, cx, cy, r, col, k=1.0, ry=None):
    """Additive soft light."""
    ry = ry or r
    b = _box(P, cx - 2.6 * r, cy - 2.6 * ry, cx + 2.6 * r + 1, cy + 2.6 * ry + 1)
    if b is None or k <= 0:
        return
    v, yy, xx = b
    a = np.exp(-((xx - cx) / r) ** 2 - ((yy - cy) / ry) ** 2) * k
    v += a[..., None] * _c(col)


def disc(P, cx, cy, r, col, a=1.0, ry=None):
    ry = ry or r
    b = _box(P, cx - r - 1, cy - ry - 1, cx + r + 2, cy + ry + 2)
    if b is None:
        return
    v, yy, xx = b
    d = np.sqrt(((xx - cx) / r) ** 2 + ((yy - cy) / ry) ** 2)
    m = np.clip((1 - d) * min(r, ry) + 0.5, 0, 1) * a
    v += (_c(col) - v) * m[..., None]


def poly(P, pts, col, a=1.0):
    """Fill a polygon (pixel centres inside, even–odd rule)."""
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    b = _box(P, min(xs), min(ys), max(xs) + 1, max(ys) + 1)
    if b is None:
        return None
    v, yy, xx = b
    py, px = yy + 0.5, xx + 0.5
    inside = np.zeros(v.shape[:2], bool)
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if y1 == y2:
            continue
        cond = (y1 > py) != (y2 > py)
        xint = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cond & (px < xint)
    v += (_c(col) - v) * (inside * a)[..., None]
    return inside


def line(P, x0, y0, x1, y1, col, wid=1.0, a=1.0):
    """Anti-aliased thick line; wid may be (start, end) for a taper."""
    w0, w1 = wid if isinstance(wid, tuple) else (wid, wid)
    m = max(w0, w1)
    b = _box(P, min(x0, x1) - m - 1, min(y0, y1) - m - 1, max(x0, x1) + m + 2, max(y0, y1) + m + 2)
    if b is None:
        return
    v, yy, xx = b
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy or 1e-6
    t = np.clip(((xx + 0.5 - x0) * dx + (yy + 0.5 - y0) * dy) / L2, 0, 1)
    d = np.hypot(xx + 0.5 - (x0 + t * dx), yy + 0.5 - (y0 + t * dy))
    k = np.clip((w0 + (w1 - w0) * t) / 2 + 0.5 - d, 0, 1) * a
    v += (_c(col) - v) * k[..., None]


def rrect(H, W, x, y, cw, ch, r):
    """Boolean mask of a rounded rectangle."""
    yy, xx = grid(H, W)
    dx = np.maximum(np.maximum(x + r - xx, xx - (x + cw - 1 - r)), 0)
    dy = np.maximum(np.maximum(y + r - yy, yy - (y + ch - 1 - r)), 0)
    return (xx >= x) & (xx < x + cw) & (yy >= y) & (yy < y + ch) & (dx * dx + dy * dy <= r * r + 0.5)


def blur1(A, r, axis, wrap=False):
    r = int(r)
    if r < 1:
        return A
    n = A.shape[axis]
    pw = [(0, 0)] * A.ndim
    pw[axis] = (r + 1, r)
    c = np.cumsum(np.pad(A, pw, mode="wrap" if wrap else "edge"), axis=axis, dtype=np.float32)
    hi, lo = [slice(None)] * A.ndim, [slice(None)] * A.ndim
    hi[axis], lo[axis] = slice(2 * r + 1, 2 * r + 1 + n), slice(0, n)
    return (c[tuple(hi)] - c[tuple(lo)]) / (2 * r + 1)


def blur(A, r, wrap=False):
    return blur1(blur1(A, r, 0, wrap), r, 1, wrap)


# ── the world seed: every layout, texture and generated shape derives from it ──
def noise_tex(H, W, scale, seed, wrap=True):
    """Smooth fractal noise, roughly zero-mean / unit-std, tileable."""
    rng = np.random.default_rng(wseed("noise", seed))
    out, amp = np.zeros((H, W), np.float32), 1.0
    for o in range(3):
        s = max(1, int(scale / 2 ** o))
        n = rng.random((H, W)).astype(np.float32)
        n = blur(blur(n, s, wrap), s, wrap)
        out += amp * (n - n.mean()) / (n.std() + 1e-6)
        amp *= 0.5
    return out / (out.std() + 1e-6)


def vnoise3(x, y, z, seed):
    """Smooth 3D value noise in [0, 1]. Sampled on a sphere it has no seams and no pinched poles."""
    xi, yi, zi = np.floor(x), np.floor(y), np.floor(z)
    fx, fy, fz = ((f * f * (3 - 2 * f)) for f in (x - xi, y - yi, z - zi))
    xi, yi, zi = xi.astype(np.int64), yi.astype(np.int64), zi.astype(np.int64)
    s = wseed("vn3", seed)
    out = 0.0
    with np.errstate(over="ignore"):
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    n = ((xi + dx) * 73856093) ^ ((yi + dy) * 19349663) ^ ((zi + dz) * 83492791) ^ s
                    n = (n ^ (n >> 13)) * 1274126177
                    v = ((n ^ (n >> 16)) & 0xFFFF) / 65535.0
                    out = out + v * (fx if dx else 1 - fx) * (fy if dy else 1 - fy) * (fz if dz else 1 - fz)
    return out


def fbm3(x, y, z, seed, octaves=5, freq=2.0):
    """Fractal 3D noise, zero-mean / unit-std."""
    out, amp, tot = 0.0, 1.0, 0.0
    for o in range(octaves):
        out = out + amp * vnoise3(x * freq + o * 17.3, y * freq - o * 9.1, z * freq + o * 5.7, (seed, o))
        tot += amp
        amp *= 0.5
        freq *= 2.03
    out = out / tot
    return ((out - out.mean()) / (out.std() + 1e-6)).astype(np.float32)


def ramp(v, stops):
    """Map values through a colour gradient; stops = [(pos, (r, g, b)), …] → (..., 3)."""
    ps = [p for p, _ in stops]
    return np.stack([np.interp(v, ps, [c[i] for _, c in stops]) for i in range(3)], -1).astype(np.float32)


def grain_tex(H, W, seed):
    """Wood grain: noise stretched along x."""
    rng = np.random.default_rng(wseed("grain", seed))
    n = rng.random((H, W)).astype(np.float32)
    n = blur1(blur1(n, 9, 1), 9, 1)
    n = blur1(n, 1, 0)
    return (n - n.mean()) / (n.std() + 1e-6)


def hash01(xx, yy, seed=0):
    seed = (seed + WORLD["seed"] * 7919) % 1000003            # keep the products below int64
    return (((xx.astype(np.int64) * 73856093) ^ (yy.astype(np.int64) * 19349663 + seed * 83492791)) % 997) / 997.0


def fog_rows(P, y0, y1, col, amt):
    """Haze that thickens downward from row y0 to y1."""
    y1 = min(P.shape[0], int(y1))
    if y1 <= 0:
        return
    a = (np.clip((np.arange(y1) - y0) / max(1, y1 - y0), 0, 1) ** 1.4 * amt)[:, None, None]
    P[:y1] += (_c(col) - P[:y1]) * a


def make_stars(rng, ymax, w, n):
    r = np.random.default_rng(rng.randrange(1 << 30))
    n = max(4, n)
    return (r.integers(0, max(1, ymax), n), r.integers(0, w, n), r.uniform(0, TAU, n),
            r.uniform(0.25, 1.0, n) ** 2, r.uniform(0.6, 2.2, n))


def draw_stars(P, stars, t, dim=1.0):
    ys, xs, ph, br, sp = stars
    tw = br * (0.55 + 0.45 * np.sin(t * sp + ph)) * dim
    col = _c((236, 234, 255))
    P[ys, xs] = np.maximum(P[ys, xs], col * tw[:, None])
    big = br > 0.7
    H, W = P.shape[:2]
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        y2, x2 = np.clip(ys[big] + dy, 0, H - 1), np.clip(xs[big] + dx, 0, W - 1)
        P[y2, x2] += col * (tw[big] * 0.28)[:, None]


_CRATERS = {}


def craters():
    """This world's moon: 4–8 craters scattered over the disc (biggest first)."""
    k = WORLD["seed"]
    if k not in _CRATERS:
        r = wrng("craters")
        out = []
        for _ in range(r.randint(4, 8)):
            a, d = r.uniform(0, TAU), math.sqrt(r.random()) * 0.6
            out.append((math.cos(a) * d, math.sin(a) * d, r.uniform(0.08, 0.26)))
        _CRATERS[k] = sorted(out, key=lambda c: -c[2])
    return _CRATERS[k]


def moon(P, cx, cy, r, dim=1.0, haze=1.0, tint=(252, 240, 214)):
    glow(P, cx, cy, r * 4.5, (70, 58, 120), 0.32 * haze * dim)
    glow(P, cx, cy, r * 1.9, (170, 160, 210), 0.32 * haze * dim)
    b = _box(P, cx - r - 1, cy - r - 1, cx + r + 2, cy + r + 2)
    if b is None:
        return
    v, yy, xx = b
    d = np.hypot(xx - cx, yy - cy) / r
    m = np.clip((1 - d) * r + 0.5, 0, 1)
    shade = np.clip(0.8 + 0.25 * (-(xx - cx) - (yy - cy)) / (r * 1.4), 0.55, 1.08) * (1 - 0.22 * np.clip(d, 0, 1) ** 3)
    for ox, oy, cr in craters():
        dd = np.hypot(xx - (cx + ox * r), yy - (cy + oy * r)) / max(0.6, cr * r)
        shade = shade * (1 - 0.14 * np.clip(1.2 - dd, 0, 1))
    col = _c(tint) * shade[..., None] * dim
    v += (col - v) * m[..., None]


def flame_colours(u):
    return np.stack([np.interp(u, [0, 0.3, 0.62, 1.0], [255, 255, 255, 180]),
                     np.interp(u, [0, 0.3, 0.62, 1.0], [250, 210, 124, 40]),
                     np.interp(u, [0, 0.3, 0.62, 1.0], [214, 112, 44, 34])], -1)


class Canvas:
    """Character overlay drawn on top of the pixels (text, particles, braille)."""

    def __init__(self, h, w):
        self.h, self.w = h, w
        self.ch = [[" "] * w for _ in range(h)]
        self.co = [[0] * w for _ in range(h)]

    def put(self, y, x, c, col):
        if 0 <= y < self.h and 0 <= x < self.w:
            self.ch[y][x] = c
            self.co[y][x] = col if isinstance(col, int) else rgb(*col)

    def text(self, y, x, s, col):
        if not isinstance(col, int):
            col = rgb(*col)
        for i, c in enumerate(s):
            self.put(y, x + i, c, col)


def compose(P, cv):
    """Pixels + overlay → per-cell (char, fg, bg)."""
    np.clip(P, 0, 255, out=P)
    top, bot = P[0::2], P[1::2]
    chs = np.array(cv.ch, dtype="<U1")
    m = chs != " "
    m3 = m[..., None]
    fg = np.where(m3, pal_array()[np.array(cv.co)], top)
    bg = np.where(m3, (top + bot) * 0.5, bot)
    flat = ~m & (np.abs(top - bot).sum(-1) < 4)
    ch = np.where(m, chs, np.where(flat, " ", "▀"))
    fg[flat] = bg[flat]
    return ch, fg, bg


def ridge1d(n, scale, seed):
    """Smooth, tileable 1-D noise in 0…1 (hill lines, hedges)."""
    rng = np.random.default_rng(wseed("ridge", seed))
    out, amp = np.zeros(n, np.float32), 1.0
    for o in range(3):
        s = max(1, int(scale / 2 ** o))
        r = rng.random(n).astype(np.float32)
        r = blur1(blur1(r, s, 0, True), s, 0, True)
        out += amp * (r - r.mean()) / (r.std() + 1e-6)
        amp *= 0.5
    return (out - out.min()) / (np.ptp(out) + 1e-6)


def rect(P, x0, y0, x1, y1, col, a=1.0):
    """Clipped, optionally translucent filled rectangle."""
    b = _box(P, x0, y0, x1, y1)
    if b is not None:
        v = b[0]
        v += (_c(col) - v) * a


def mix(c1, c2, k):
    return tuple(float(a + (b - a) * k) for a, b in zip(c1, c2))
