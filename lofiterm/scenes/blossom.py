import math
import random
import time

import numpy as np

from ..paint import _c, disc, fill_grad, fog_rows, glow, grad_at, grid, hash01, line, make_stars, noise_tex, poly, rect, vgrad
from ..world import wrng, wseed
from ..term import rgb
from ..theory import TAU


BLOSSOM = [  # sakura, white plum, red maple, snowy, wisteria, moonlit sakura
    dict(tones=[(255, 226, 238), (255, 186, 216), (238, 142, 186), (184, 94, 142), (128, 62, 108)],
         sky=[(14, 10, 38), (54, 34, 92), (150, 90, 150), (222, 142, 162)], lantern=((255, 120, 70), (236, 90, 70)),
         ground=[(78, 108, 96), (24, 42, 44)], flowers=[(255, 226, 238), (255, 246, 200), (236, 236, 255)]),
    dict(tones=[(255, 255, 255), (236, 236, 248), (206, 206, 230), (160, 160, 196), (110, 110, 150)],
         sky=[(4, 6, 24), (18, 24, 64), (50, 60, 110), (90, 100, 150)], lantern=((255, 210, 120), (240, 170, 90)),
         ground=[(48, 72, 82), (12, 22, 32)], flowers=[(236, 240, 255), (200, 210, 250)]),
    dict(tones=[(255, 210, 120), (255, 160, 70), (230, 100, 50), (180, 60, 40), (120, 36, 30)],
         sky=[(50, 40, 90), (140, 80, 120), (240, 140, 110), (255, 200, 140)], lantern=((255, 120, 70), (220, 70, 50)),
         ground=[(168, 124, 86), (66, 44, 38)], flowers=[(255, 214, 120), (255, 160, 90), (240, 230, 200)]),
    dict(tones=[(255, 255, 255), (230, 238, 255), (200, 214, 240), (150, 166, 206), (100, 116, 160)],
         sky=[(20, 26, 50), (50, 60, 96), (110, 120, 160), (170, 178, 210)], lantern=((255, 200, 120), (240, 150, 90)),
         ground=[(214, 220, 240), (132, 142, 176)], flowers=[]),
    dict(tones=[(240, 200, 255), (210, 150, 255), (170, 100, 240), (120, 60, 200), (80, 30, 150)],
         sky=[(10, 2, 30), (40, 10, 70), (120, 30, 120), (220, 70, 150)], lantern=((80, 255, 230), (40, 200, 220)),
         ground=[(62, 46, 104), (20, 12, 40)], flowers=[(120, 255, 230), (240, 170, 255), (255, 240, 140)]),
    dict(tones=[(255, 236, 244), (255, 204, 226), (240, 164, 200), (190, 110, 156), (130, 70, 118)],
         sky=[(2, 14, 24), (10, 40, 60), (40, 100, 110), (120, 170, 160)], lantern=((255, 220, 130), (240, 180, 90)),
         ground=[(56, 110, 90), (14, 38, 36)], flowers=[(255, 236, 244), (255, 240, 170), (200, 255, 230)]),
    dict(tones=[(255, 240, 246), (255, 206, 226), (250, 170, 206), (220, 120, 170), (160, 80, 130)],
         sky=[(80, 150, 235), (130, 190, 245), (190, 225, 252), (230, 244, 255)], lantern=((255, 140, 90), (240, 100, 80)),
         ground=[(150, 200, 110), (66, 124, 58)], flowers=[(255, 255, 255), (255, 226, 90), (255, 170, 210)]),
    dict(tones=[(255, 232, 240), (255, 178, 206), (246, 120, 160), (200, 70, 120), (130, 40, 90)],      # tokyo: deep sakura
         sky=[(16, 10, 44), (70, 34, 96), (190, 90, 140), (250, 160, 150)], lantern=((240, 60, 50), (210, 40, 40)),
         ground=[(70, 96, 90), (22, 36, 42)], flowers=[(255, 226, 238), (255, 120, 150), (255, 246, 210)]),
    dict(tones=[(255, 236, 200), (250, 206, 150), (226, 160, 110), (170, 104, 80), (110, 60, 56)],      # jazz: magnolia, amber
         sky=[(10, 6, 16), (40, 22, 40), (110, 60, 66), (200, 120, 90)], lantern=((255, 190, 90), (240, 150, 60)),
         ground=[(70, 70, 56), (26, 24, 26)], flowers=[(255, 236, 200), (240, 200, 255)]),
    dict(tones=[(255, 252, 246), (250, 230, 236), (236, 196, 210), (184, 140, 160), (110, 80, 100)],    # english: apple blossom, moor dusk
         sky=[(10, 12, 22), (36, 40, 62), (96, 100, 120), (176, 168, 160)], lantern=((255, 210, 130), (240, 170, 80)),
         ground=[(60, 84, 56), (22, 30, 26)], flowers=[(255, 250, 240), (250, 220, 90)])
]


def grow_tree(r, x, base, H, w, s, f=1.0):
    """Grow a cherry tree by recursive branching from (x, base) in pixels. A trunk splits into two or three
    limbs, each limb into branches and those into twigs: every branch bends a little along its length and
    tapers, children share their parent's thickness (the 'pipe model'), and the low limbs splay outwards for
    a cherry's wide crown. Returns the wood as segments (x0, y0, x1, y1, w0, w1, depth) and the buds,
    (x, y, size), where blossom clusters grow: every twig tip, and along the youngest branches."""
    segs, buds = [], []
    maxd = 4 + (s > 1.35)
    lo, hi = 3.0, w - 3.0
    top = H * 0.05
    reach = min(w * 0.22, base * 0.7) * f                # how far the crown may spread either side

    floor = [base - base * 0.22]                         # branches may not dip below this (set by the trunk)

    def grow(x, y, ang, L, wd, d):
        n = max(2, int(L / 2.5))
        bend = r.uniform(-0.1, 0.1) if d else r.uniform(-0.03, 0.03)   # trunks only bend gently
        w0 = wd
        for i in range(n):
            ang += bend + r.uniform(-0.07, 0.07)
            if d >= 2:
                ang += 0.04 * math.cos(ang)               # the outer branches arch over and droop a little
            ang = min(max(ang, -math.pi - 0.25), 0.25)    # …but never hang down towards the ground
            if d >= 1 and abs(x - x_root) > reach:        # reached the crown's edge: it ends in blossom
                buds.append((x, y, r.uniform(3.0, 4.6) * s))
                return
            x1, y1 = x + math.cos(ang) * L / n, y + math.sin(ang) * L / n
            w1 = wd * (1 - 0.3 * (i + 1) / n)
            if d >= 1 and y1 > floor[0]:                  # too low: the branch just ends
                return
            if not (lo <= x1 <= hi) or y1 < top:          # don't grow off the screen
                buds.append((x, y, r.uniform(2.6, 4.2) * s))
                return
            segs.append((x, y, x1, y1, w0, w1, d))
            if d >= maxd - 2 or (d >= 2 and i == n - 1):    # blossom all along the young wood
                buds.append((x1, y1, r.uniform(2.2, 3.4) * s))
            x, y, w0 = x1, y1, w1
        if d == 0:
            floor[0] = max(floor[0], y + 1.5)
        if d >= maxd or L < 2.4 * s:
            buds.append((x, y, r.uniform(3.0, 4.8) * s))
            return
        nc = 3 if r.random() < (0.45 if d == 0 else 0.25) else 2
        spread = r.uniform(0.45, 0.7) if d <= 1 else r.uniform(0.35, 0.6)
        for k in range(nc):
            a = ang + spread * (-1 + 2 * k / (nc - 1)) + r.uniform(-0.15, 0.15)
            if d <= 1:                                   # low limbs splay out: a crown wider than it is tall
                a = -math.pi / 2 + (a + math.pi / 2) * 1.2
                a = min(max(a, -math.pi + 0.5), -0.5)    # …but leave the trunk at least ~30° above level
            cl = L * (r.uniform(0.75, 0.88) if d == 0 else r.uniform(0.62, 0.74))
            grow(x, y, a, cl, max(0.7, w0 * (1 / nc) ** 0.5 * 1.05), d + 1)

    lean = r.uniform(-0.15, 0.15)
    x_root = x
    grow(x, base, -math.pi / 2 + lean, base * r.uniform(0.22, 0.26) * f, r.uniform(5.4, 6.6) * s, 0)   # a stout old trunk under a wide crown
    return segs, buds


class BlossomScene:
    # ── cherry blossom ─────────────────────────────────────────────────────
    def _blossom_pagoda(self, P, b, H, w, gp, mx, hill, kind="pagoda"):
        """A distant five-storey pagoda in the clearest stretch of sky: away from the tree's crown, Mount
        Fuji, the moon and the stone lantern. It is drawn before the back hill and stands on the lowest point
        of the hill under its whole width, so the hill always covers its foot (it never hangs over a slope).
        Hazy like the hill around it, with a few windows lit and rings on its spire."""
        def pick(r):
            xs = np.arange(int(w * 0.08), int(w * 0.92))
            busy = [(c0 - 6, c1 + 6) for _, c0, c1 in b["crowns"]] + [(b["fuji"] - w * 0.12, b["fuji"] + w * 0.12),
                                                                      (b["stone"] - 6, b["stone"] + 8), (mx - 5, mx + 5)]
            gap = np.full(len(xs), 1e9)
            for a0, a1 in busy:
                gap = np.minimum(gap, np.where(xs < a0, a0 - xs, np.where(xs > a1, xs - a1, -1.0)))
            best = xs[gap >= gap.max() - 2]
            return int(r.choice(list(best)))
        px = self.layout("pagoda", H, w, pick, round(b["fuji"]))
        s = max(9.0, min(H * 0.24, gp * 0.36))
        span = int(math.ceil(s * 0.4)) + 1                           # half the widest roof
        foot = b["hills"][0][max(0, px - span):px + span + 1]
        base = float(foot.max()) + max(1.0, s * 0.06)                # below the ridge everywhere under it
        hz = _c(hill)
        body, roof = hz * 0.5, hz * 0.36
        if kind == "spire":                                       # a village church instead (old English)
            self.spire(P, px, base, s * 1.1, body, lit=(255, 206, 130))
            rect(P, px + s * 0.12, base - s * 0.22, px + s * 0.5, base, body)          # the nave
            poly(P, [(px + s * 0.1, base - s * 0.22), (px + s * 0.31, base - s * 0.34), (px + s * 0.52, base - s * 0.22)], roof)
            for k in range(3):
                P[int(base - s * 0.12), int(px + s * (0.2 + 0.1 * k)) % w] = (255, 200, 130)
            return
        th = s * 0.78 / 5
        for k in range(5):                                        # a lit window on some storeys
            if (k * 7 + px) % 3:
                y = int(base - k * th - th * 0.35)
                if 0 <= y < gp:
                    P[y, px] = (255, 196, 130)
        top = base - 5 * th
        for k in range(3):                                        # the sōrin's rings
            y = top - (s - 5 * th) * (0.3 + 0.2 * k)
            rect(P, px - 1, y, px + 2, y + 0.6, roof)

    @staticmethod
    def _fuji(V, H, w, gp, cx, mx, snowy, raining):
        """Mount Fuji behind the hills, as a colour layer and its coverage (rows 0…gp): long concave slopes
        up to a broad flat summit with a little crater notch, a snowcap that runs down the gullies in
        streaks, one flank lit by the moon (or the sun) and the other in shade, fading into the haze."""
        yy, xx = grid(H, w)
        yy, xx = yy[:gp], xx[:gp]
        hgt = gp * 0.5
        half = min(w * 0.45, hgt * 2.7)
        dx = xx - cx
        u = np.abs(dx) / half
        elev = np.clip((1 - u) / 0.95, 0, 1) ** 1.8                     # flat between u = 0 and 0.05
        elev = elev - 0.02 * np.exp(-(dx / (half * 0.02)) ** 2)        # the crater
        ridge = gp - hgt * elev
        cover = np.clip(yy - ridge + 0.5, 0, 1) * (u < 1)
        e = np.clip((gp - yy) / hgt, 0, 1)                              # 0 at the foot … 1 at the summit
        sky = [_c(c) for c in V["sky"]]
        lit = np.clip(0.5 + dx * (1 if mx >= cx else -1) / 2.5, 0, 1)   # the flank facing the moon is lit
        rock_d = sky[1] * 0.55 + _c((16, 16, 30)) * 0.45
        rock_l = sky[2] * 0.55 + sky[1] * 0.35
        col = rock_d[None, None] * (1 - lit[..., None]) + rock_l[None, None] * lit[..., None]
        col *= (0.94 + 0.06 * np.sin(dx * 1.9 + np.sqrt(np.abs(dx)) * 3))[..., None]   # gullies
        g = dx / max(1.0, half) * 40                                     # gullies, spaced by the mountain's size
        line_ = (0.56 - (0.22 if snowy else 0.0) + 0.1 * np.abs(np.sin(g * 0.5)) ** 0.7 + 0.05 * np.sin(g * 1.3 + 1.0)
                 - 0.05)
        snow = np.clip((e - line_) / 0.035, 0, 1)
        white = _c((238, 242, 255)) * 0.8 + sky[3] * 0.2
        shade = white * 0.62 + sky[1] * 0.3
        sc = shade[None, None] * (1 - lit[..., None]) + white[None, None] * lit[..., None]
        col = col * (1 - snow[..., None]) + sc * snow[..., None]
        skyrow = grad_at(H, 0, gp, list(zip((0, 0.5, 0.85, 1), V["sky"])))[:gp][:, None, :]
        haze = 0.18 + 0.5 * (1 - e) ** 1.6 + (0.2 if raining else 0.0)  # aerial perspective, thickest at the foot
        col = col * (1 - haze[..., None]) + skyrow * haze[..., None]
        return col.astype(np.float32), cover.astype(np.float32)

    @staticmethod
    def _meadow(V, H, w, gp, b, mx):
        """The ground (rows gp…H) as a meadow: hazy far away and darker up close, grass blades that are fine in
        the distance and long in front, light from the moon (or sun) pooling below it, the trees' soft
        shadows, scattered little flowers and a stepping-stone path from the stone lantern towards us."""
        yy, xx = grid(H, w)
        Y, X = yy[gp:] - gp, xx[gp:]
        gh = H - gp
        v = Y / max(1, gh - 1)                                          # 0 at the horizon … 1 at the front
        far, near = _c(V["ground"][0]), _c(V["ground"][1])
        G = far * (1 - v[..., None] ** 0.7) + near * v[..., None] ** 0.7
        L = 1.0 + 4.5 * v                                               # blade length grows towards us
        ph = hash01(X, X * 0 + 11, 5) * L
        pos = ((Y + ph) % L) / L                                        # 0 at a blade's tip
        if V["flowers"]:                                                # grass
            blade = (1.32 - 0.6 * pos) * (0.8 + 0.4 * hash01(X, ((Y + ph) // L).astype(int), 9))
            G *= (1 + (blade - 1) * (0.45 + 0.55 * v))[..., None]
            tint = hash01(X // 3, Y // 2, 13)                          # patches of lusher and drier grass
            G *= (0.9 + 0.2 * tint)[..., None]
        else:                                                           # snow: soft drifts, lit on their tops
            dr = np.sin(X * 0.11 + np.sin(X * 0.031) * 3 + Y * 0.9)
            G *= (0.94 + 0.07 * dr)[..., None]
        light = np.exp(-((X - mx) / (w * 0.35)) ** 2) * (1 - v) ** 1.5    # moonlight pooling near the horizon
        G += light[..., None] * (_c(V["sky"][3]) * 0.22)
        for tx, c0, c1 in b["crowns"]:                                  # soft shadows under the trees
            cxs, rxs = (c0 + c1) / 2 + (tx - mx) * 0.08, (c1 - c0) * 0.42 + 2
            sh = np.exp(-(((X - cxs) / rxs) ** 2 + ((Y - gh * 0.25) / (gh * 0.3)) ** 2) * 1.6)
            G *= (1 - 0.45 * sh)[..., None]
        if V["flowers"]:
            fl = hash01(X, Y, 21) < 0.012 * (0.4 + v)
            pick = (hash01(X, Y, 22) * len(V["flowers"])).astype(int)
            G[fl] = np.array(V["flowers"], np.float32)[pick[fl]] * (0.75 + 0.25 * v[fl])[:, None]
        else:                                                           # snow: a glitter of frost instead
            fl = hash01(X, Y, 21) < 0.02
            G[fl] = np.minimum(255, G[fl] * 1.25 + 20)
        # stepping stones from the lantern towards the front, getting bigger as they come closer
        r = wrng("path", w, H)
        sx, bend = b["stone"] + 1.5, r.uniform(-0.12, 0.12) * w
        stone = _c(V["ground"][0]) * 0.35 + _c((150, 146, 160)) * 0.65
        n = max(3, int(gh / 3))
        for i in range(n):
            u = (i + 0.7) / n
            cx, cy = sx + bend * u * u + r.uniform(-0.8, 0.8), gh * u
            rx = 1.2 + 4.2 * u * b["sc"]
            disc(G, cx + 0.4, cy + 0.6, rx, near * 0.6, ry=max(0.6, rx * 0.32))          # its shadow
            disc(G, cx, cy, rx, stone * (0.8 + 0.25 * u), ry=max(0.6, rx * 0.32))
            disc(G, cx - rx * 0.2, cy - rx * 0.08, rx * 0.55, np.minimum(255, stone * 1.15), ry=max(0.5, rx * 0.15))
        return G

    def v_blossom(self, cv, P, h, w):
        e = self.eng
        t = time.time()
        H = P.shape[0]
        level = float(self.rms.mean())
        gy = h - max(4, int(h * 0.17))                   # ground row (cells)
        gp = 2 * gy
        b = getattr(self, "blossom", None)
        V = self.var(BLOSSOM)
        if b is None or b["size"] != (h, w, e.station_idx):
            rng = wrng("blossom", h, w)
            nrng = np.random.default_rng(wseed("blossom", h, w))
            sc = max(1.0, min(2.2, H / 70))                  # bigger screens get a fuller tree
            if w <= 160:
                specs = [(int(w * rng.uniform(0.22, 0.34)), 1.0)]
                lx = int(w * rng.uniform(0.6, 0.68))          # the stone lantern
            else:                                            # wide: an open meadow, then a grove beyond it
                tx = int(min(w * 0.3, 40 + 34 * sc))
                specs = [(tx, 1.0)]
                lx = tx + int(34 * sc) + 10
                x = min(w - 3, tx + int(34 * sc) + 16 + 90) + 14 + int(14 * sc)
                while x < w - 14:
                    specs.append((x, rng.uniform(0.6, 0.85)))
                    x += rng.randint(50, 70) + int(26 * sc)
            N = noise_tex(H, w, 2, 5)
            yy, xx = grid(H, w)
            best = np.full((H, w), 9.0, np.float32)
            ccol = np.zeros((H, w, 3), np.float32)
            wood = np.zeros((H, w, 3), np.float32)           # branches, painted once (premultiplied)
            wa = np.zeros((H, w, 3), np.float32)             # and how much they cover
            tones = np.array(V["tones"], np.float32)
            snowy = e.st.get("weather") == "snow"
            lanterns, crowns = [], []
            for ti_, (tx, f) in enumerate(specs):
                reach = min(w * 0.22, gp * 0.7) * f + 8
                tx = int(min(max(tx, reach), w - reach))
                segs, buds = grow_tree(rng, tx, gp, H, w, sc * f, f)
                for x0, y0, x1, y1, w0, w1, d in sorted(segs, key=lambda q: q[6]):
                    line(wood, x0, y0, x1, y1, (58, 36, 40), (w0, w1))
                    line(wa, x0, y0, x1, y1, (1, 1, 1), (w0, w1))
                    if w0 > 1.3:                                 # bark lit from the upper left
                        o = w0 * 0.22
                        line(wood, x0 - o, y0, x1 - o, y1, (146, 96, 84), (w0 * 0.45, w1 * 0.45))
                    if snowy and d <= 3 and abs(x1 - x0) > abs(y1 - y0) * 0.5:
                        line(wood, x0, y0 - w0 * 0.4, x1, y1 - w1 * 0.4, (230, 236, 250), (w0 * 0.4, w1 * 0.4))
                trunk_w = segs[0][4] if segs else 3
                poly(wood, [(tx - trunk_w * 1.6, gp), (tx - trunk_w * 0.7, gp - trunk_w * 0.9),
                            (tx + trunk_w * 0.7, gp - trunk_w * 0.9), (tx + trunk_w * 1.6, gp)], (96, 62, 56))
                poly(wa, [(tx - trunk_w * 1.6, gp), (tx - trunk_w * 0.7, gp - trunk_w * 0.9),
                          (tx + trunk_w * 0.7, gp - trunk_w * 0.9), (tx + trunk_w * 1.6, gp)], (1, 1, 1))
                for px, py, rr in buds:                        # blossom clusters, lit from above
                    crx, cry = rr * rng.uniform(1.0, 1.35), rr * rng.uniform(0.75, 0.95)
                    y0, y1 = max(0, int(py - cry) - 2), min(gp, int(py + cry) + 3)
                    x0, x1 = max(0, int(px - crx) - 2), min(w, int(px + crx) + 3)
                    if y0 >= y1 or x0 >= x1:
                        continue
                    Yb, Xb = yy[y0:y1, x0:x1], xx[y0:y1, x0:x1]
                    d = ((Xb - px) / crx) ** 2 + ((Yb - py) / cry) ** 2 + 0.35 * N[y0:y1, x0:x1]
                    s_ = (Xb - px) / crx + 1.3 * (Yb - py) / cry + 0.5 * N[y0:y1, x0:x1] + 0.3 * d
                    col = tones[np.clip(np.digitize(s_, [-0.9, -0.2, 0.6, 1.2]), 0, 4)]
                    upd = (d < 1.0) & (d < best[y0:y1, x0:x1])
                    best[y0:y1, x0:x1][upd] = d[upd]
                    ccol[y0:y1, x0:x1][upd] = col[upd]
                if buds:
                    bx_ = [q[0] for q in buds]
                    crowns.append((tx, min(bx_), max(bx_)))
                hang = sorted((q for q in segs if 2 <= q[6] <= 3), key=lambda q: q[0] + q[2])
                nl_ = min(len(hang), 3 if ti_ == 0 else 1)          # lanterns hang from real branches, spread out
                for j in range(nl_):
                    seg = hang[int((j + rng.uniform(0.2, 0.8)) / nl_ * len(hang)) % len(hang)]
                    lanterns.append((int((seg[0] + seg[2]) / 2), int((seg[1] + seg[3]) / 2) + 2))
            cmask = best < 1.0
            spk = cmask & (nrng.random((H, w)) < 0.06)
            ccol[spk] = (255, 240, 248)
            cy_, cx_ = np.nonzero(cmask)
            xs = np.arange(w)
            hph = [rng.uniform(0, TAU) for _ in range(4)]
            hq = [rng.uniform(0.6, 1.5) for _ in range(4)]
            hills = [gp - (H * 0.16 + H * 0.06 * np.sin(xs * 0.06 * hq[0] + hph[0]) + H * 0.03 * np.sin(xs * 0.17 * hq[1] + hph[1])),
                     gp - (H * 0.09 + H * 0.05 * np.sin(xs * 0.09 * hq[2] + hph[2]) + H * 0.03 * np.sin(xs * 0.23 * hq[3] + hph[3]))]
            old_acc = b["acc"] if b is not None and len(b["acc"]) == w else np.zeros(w)
            b = self.blossom = dict(size=(h, w, e.station_idx), wood=wood, wa=wa[..., 0], cmask=cmask, ccol=ccol, sc=sc,
                                    crowns=crowns,
                                    cells=(cy_, cx_), lanterns=lanterns, stone=lx, hills=hills,
                                    fuji=(w * rng.uniform(0.7, 0.78) if w <= 160 else lx + w * rng.uniform(-0.03, 0.08)),
                                    stars=make_stars(rng, int(gp * 0.5), w, w * h // 40), petals=[],
                                    drops=[], splash=[], acc=old_acc, wind=0.1,
                                    moon=(rng.uniform(0.64, 0.92), rng.uniform(0.08, 0.2)),
                                    top=int(min(cy_.min() if len(cy_) else gp, np.nonzero(wa[..., 0].any(1))[0].min()
                                                 if wa.any() else gp)))
        acc = b["acc"]
        raining = e.wet
        yy, xx = grid(H, w)
        Y = yy[:gp]
        mx = int(w * b["moon"][0])
        # ── sky, moon, distant hills ──
        fill_grad(P, 0, gp, list(zip((0, 0.5, 0.85, 1), V["sky"])))
        if raining:
            P[:gp] = P[:gp] * 0.6 + _c((40, 34, 58)) * 0.4
        self.chord_wash(P, 0, gp, 0.045)                        # the sky leans toward each chord's colour
        self.stars(P, b["stars"], t, 0.3 if raining else 1.0)
        self.sky_body(P, mx, H * b["moon"][1], max(3.0, H * 0.065), dim=0.6 if raining else 1.0, tint=(255, 236, 226))
        fc, fa = self.tex("fuji", (H, w, gp, e.station_idx, round(b["fuji"]), mx, raining),
                          lambda: self._fuji(V, H, w, gp, b["fuji"], mx, e.st.get("weather") == "snow", raining))
        P[:gp] = P[:gp] * (1 - fa[..., None]) + fc * fa[..., None]
        s2 = _c(V["sky"][2])
        th = self.theme()
        for i, (hl, (c0, c1)) in enumerate(zip(b["hills"], ((s2 * 0.8, s2 * 0.66), (s2 * 0.52, s2 * 0.42)))):
            if i == 0 and th in ("japan", "english"):    # a far-off pagoda (or church) behind the back hill's ridge
                self._blossom_pagoda(P, b, H, w, gp, mx, c0, "pagoda" if th == "japan" else "spire")
            m = Y >= hl[None, :]
            g = vgrad(gp, [(0, c0), (1, c1)])
            P[:gp][m] = np.broadcast_to(g[:, None, :], (gp, w, 3))[m]
            fog_rows(P, gp * 0.6, gp, np.minimum(255, s2 * 1.15), 0.25)
        ev = self.rare(22.0)
        if ev:
            self._rare_blossom(P, ev, gp, w)
        # ── stone lantern, glowing on the beat ──
        sx = b["stone"]
        on = int(e.beat_pos) % 2 == 0
        stone = (128, 122, 140)
        P[gp - 8:gp - 7, sx - 1:sx + 4] = stone
        P[gp - 7:gp - 4, sx:sx + 3] = (98, 94, 112)
        P[gp - 6:gp - 5, sx + 1] = (255, 214, 140) if on else (230, 170, 110)
        P[gp - 4:gp - 1, sx + 1] = (110, 106, 124)
        P[gp - 1:gp, sx - 1:sx + 4] = stone
        poly(P, [(sx - 2, gp - 8), (sx + 1.5, gp - 10.5), (sx + 5, gp - 8)], (140, 134, 152))
        glow(P, sx + 1.5, gp - 5.5, 2.5 + 1.5 * self.kick, (255, 190, 110), (0.45 + 0.15 * on + 0.35 * self.kick) * (0.15 if self.noon() else 1.0))
        if th == "jazz":                                 # a bench under a street lamp, a sax left on it
            jx = int(w * 0.86) if w <= 160 else sx + 40
            if jx + 8 < w:
                line(P, jx + 8, gp, jx + 8, gp - 22, (40, 36, 44), wid=1.2)
                rect(P, jx + 6, gp - 24, jx + 11, gp - 21, (60, 54, 60))
                rect(P, jx + 7, gp - 22, jx + 10, gp - 21, (255, 224, 150))
                glow(P, jx + 8.5, gp - 20, 6, (255, 190, 100), 0.5 + 0.3 * self.bass, ry=8)
                rect(P, jx - 8, gp - 6, jx + 4, gp - 5, (120, 80, 56))
                rect(P, jx - 8, gp - 10, jx + 4, gp - 9, (120, 80, 56))
                for lx_ in (jx - 7, jx + 2):
                    rect(P, lx_, gp - 5, lx_ + 1, gp, (50, 40, 40))
                self.sax(P, jx - 5, gp - 15, 10)
                self.notes(cv, jx - 2, (gp - 16) // 2, (255, 214, 150), n=3, rise=6, seed=2)
        # ── the tree: wood, then blossoms, swaying together (more at the top) ──
        target = 0.1 + 0.15 * math.sin(t * 0.07) + (0.7 if self.bass > 0.75 else 0) + (0.25 if raining else 0) + 0.15 * self.energy
        b["wind"] += (target - b["wind"]) * 0.05          # bass hits send gusts through
        sway = b["wind"] * 1.6 * math.sin(t * 0.9)
        top, span = b["top"], max(1, gp - b["top"])
        cm, cc, wd_, wa_ = b["cmask"], b["ccol"], b["wood"], b["wa"]
        for y in range(top, gp):
            dx = int(round(sway * (1 - (y - top) / span) ** 1.5))
            a = np.roll(wa_[y], dx) if dx else wa_[y]
            if a.any():
                P[y] = P[y] * (1 - np.clip(a, 0, 1)[:, None]) + (np.roll(wd_[y], dx, axis=0) if dx else wd_[y])
            row = cm[y]
            if row.any():
                mrow, crow = (np.roll(row, dx), np.roll(cc[y], dx, axis=0)) if dx else (row, cc[y])
                P[y][mrow] = crow[mrow]
        if self.bass > 0.6 or self.hat > 0.5:                 # the blossoms glint on the hats
            cy_, cx_ = b["cells"]
            pick = np.random.random(len(cy_)) < 0.01 + 0.01 * (self.bass > 0.6)
            P[cy_[pick], np.clip(cx_[pick] + int(round(sway * 0.5)), 0, w - 1)] = (255, 246, 250)
        for lx, ly in b["lanterns"]:                      # paper lanterns under the blossoms
            dx = int(round(sway * 0.5))
            x = lx + dx
            if 0 <= x - 1 and x + 2 < w and ly + 5 < gp:
                P[ly - 2:ly, x] = (70, 50, 60)
                hot = int(e.beat_pos * 2 + lx) % 2
                body = _c(V["lantern"][0] if hot else V["lantern"][1])
                if self.noon():                           # unlit paper by day
                    body = body * 0.55 + _c((236, 226, 210)) * 0.45
                P[ly + 1:ly + 4, x - 1:x + 2] = body
                P[ly + 1:ly + 4, x + 1] = body * 0.7
                P[ly, x - 1:x + 2] = (220, 180, 90)
                P[ly + 4, x - 1:x + 2] = (220, 180, 90)
                if not self.noon():
                    glow(P, x + 0.5, ly + 2.5, 3.5, V["lantern"][0], (0.55 if hot else 0.3) + 0.25 * self.pluck)

        # ── ground with fallen petals ──
        lv, _ = self.band_levels(w)
        acc *= 0.9992                                    # old petals blow away slowly
        gh = H - gp
        P[gp:] = self.tex("bground", (H, w, gp, e.station_idx), lambda: self._meadow(V, H, w, gp, b, mx))
        if raining:                                        # wet ground: darker, with a sheen of sky
            P[gp:] = P[gp:] * 0.72 + _c(V["sky"][2]) * 0.08
        v = (np.arange(gh) / max(1, gh - 1))[:, None]
        pink = hash01(xx[gp:], yy[gp:], 3) < acc[None, :] * (0.8 - 0.5 * v)   # fallen petals, thinner up front
        P[gp:][pink] = np.where(hash01(xx[gp:], yy[gp:], 4)[pink][:, None] < 0.5,
                                _c(V["tones"][1]) * 0.95, _c(V["tones"][0]) * 0.9)
        grass = _c(V["ground"][0])
        cols = np.arange(w)
        tuft = hash01(cols, cols * 0 + 7, 8)                   # a ragged grassy edge: tufts of every height…
        for k in range(1, 5 if V["flowers"] else 3):       # …that grow taller with the spectrum
            mk = ((lv * 3 + tuft * 1.6) >= k) & (tuft > 0.25)
            P[gp - k][mk] = np.minimum(255, grass * (1.0 + 0.1 * k) * (0.85 + 0.3 * tuft[mk])[:, None])
        # ── falling petals: more when the music swells, flutter and drift ──
        n_new = (0.5 + level * 7 + (1.0 if raining else 0) + (8.0 if "change" in self.fired else 0)) * (w / 80)   # a puff on a chord change
        cy_, cx_ = b["cells"]
        if "note" in self.fired and len(cy_):              # each melody note shakes one bright petal loose, left to right with pitch
            j = int(np.argmin(np.abs(cx_ - (cx_.min() + self.pitch * (cx_.max() - cx_.min()))) + (cy_ % 7)))
            b["petals"].append([float(cx_[j]), cy_[j] / 2.0, 0.2, random.random() * TAU, rgb(255, 248, 252)])
        for _ in range(int(n_new) + (random.random() < n_new % 1)):
            if not len(cy_):
                break
            j = random.randrange(len(cy_))
            b["petals"].append([float(cx_[j]), cy_[j] / 2.0, random.uniform(0.12, 0.3) * (1.6 if raining else 1),
                                random.random() * TAU, rgb(*random.choice(V["tones"][:3] + V["tones"][1:2]))])
        alive = []
        for p in b["petals"]:
            p[0] += b["wind"] + 0.35 * math.sin(t * 2.2 + p[3])
            p[1] += p[2] * (0.7 + 0.3 * math.sin(t * 3 + p[3]))
            xi = int(p[0])
            if not (-1 <= xi <= w):
                continue
            if p[1] >= gy:                                  # lands: the petals pile up on the ground
                if 0 <= xi < w:
                    acc[xi] = min(1.0, acc[xi] + 0.035)
                continue
            alive.append(p)
            cv.put(int(p[1]), xi, "•·'`,"[int(t * 3 + p[3] * 2) % 5], p[4])
        b["petals"] = alive[-500:]
        self.sky = (int(w * 0.55), w - 1, 0, int(gy * 0.6))
        self.weather(b, cv, 1, w - 1, 0, gy)

    def _rare_blossom(self, P, ev, gp, w):
        """Rare event: a skein of red-crowned cranes flies over in a V, wings beating slowly out of step."""
        u, r = ev
        H = P.shape[0]
        f = r.choice((-1, 1))
        n = r.choice((3, 5, 5, 7))
        y0 = gp * r.uniform(0.12, 0.3)
        s = max(1.0, H / 70) * r.uniform(0.9, 1.2)
        span = w + 60 * s
        lead = (-20 * s + u * span) if f > 0 else (w + 20 * s - u * span)
        lead_y = y0 - u * gp * 0.06
        dim = 1.0 if self.noon() else 0.78
        white, black = _c((244, 242, 238)) * dim, _c((30, 28, 34))
        T = self.rclock
        for i in range(n):
            rank = (i + 1) // 2
            side = -1 if i % 2 else 1
            x = lead - f * rank * 6 * s
            y = lead_y + side * rank * 3.2 * s
            flap = math.sin(T * 3.2 + i * 0.9)
            line(P, x - f * 3 * s, y, x + f * 2.5 * s, y, white, wid=1.0)                    # body
            line(P, x + f * 2.5 * s, y, x + f * 4.6 * s, y - 0.3 * s, white, wid=0.7)       # neck
            if 0 <= x + f * 4.6 * s < w and 0 <= y < H:
                P[int(y), int(x + f * 4.6 * s)] = (220, 40, 50)                              # red crown
            line(P, x - f * 3 * s, y + 0.3, x - f * 5.5 * s, y + 0.7 * s, black, wid=0.6)    # legs trailing
            tip_y = y - flap * 3.2 * s
            line(P, x, y, x - f * 1.2 * s, tip_y, white, wid=1.0)                            # wings, black-tipped
            line(P, x - f * 1.2 * s, tip_y, x - f * 2.2 * s, tip_y - flap * 0.8 * s, black, wid=0.8)
