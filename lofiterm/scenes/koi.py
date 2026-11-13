import math
import random
import time

import numpy as np

from ..paint import _box, _c, disc, glow, grid, hash01, line, noise_tex, poly, rect
from ..world import wrng
from ..term import C_RAIN, C_WHITE
from ..theory import TAU


KOI_PROFILE = np.array([0.62, 0.92, 1.0, 0.96, 0.84, 0.68, 0.5, 0.34, 0.22])


# koi pond per station: water (shallow → deep), bank, moss, stones, bushes (shade, lit), lily pads, lotus,
# light glinting on the water, lanterns, bridge rails, things floating on the water, fireflies
KOI = [
    dict(water=[(70, 104, 140), (30, 52, 96), (10, 16, 44)], bank=[(50, 64, 72), (30, 40, 52)], moss=(74, 108, 86),
         stone=(124, 118, 140), bush=[(26, 52, 50), (66, 108, 92)], pad=(64, 120, 84), lotus=(255, 170, 206),
         glint=(255, 222, 212), lantern=(255, 176, 110), rail=(196, 64, 58),
         float=[(255, 198, 224), (255, 170, 206)], flies=0.0, fly=(255, 230, 150)),
    dict(water=[(34, 54, 96), (12, 24, 60), (3, 8, 26)], bank=[(26, 34, 44), (14, 20, 30)], moss=(40, 64, 60),
         stone=(86, 90, 110), bush=[(14, 30, 32), (40, 66, 72)], pad=(40, 88, 66), lotus=(240, 240, 255),
         glint=(220, 232, 255), lantern=(255, 210, 140), rail=(150, 50, 50),
         float=[(240, 240, 252), (214, 216, 236)], flies=0.4, fly=(220, 240, 255)),
    dict(water=[(110, 160, 140), (40, 100, 104), (14, 46, 60)], bank=[(126, 116, 64), (86, 78, 46)], moss=(150, 142, 68),
         stone=(178, 158, 134), bush=[(120, 58, 30), (224, 124, 52)], pad=(96, 150, 70), lotus=(255, 190, 200),
         glint=(255, 214, 150), lantern=(255, 170, 90), rail=(206, 70, 46),
         float=[(240, 120, 50), (220, 80, 40), (255, 180, 70)], flies=0.0, fly=(255, 230, 150)),
    dict(water=[(96, 122, 150), (44, 66, 100), (16, 26, 48)], bank=[(140, 150, 170), (104, 114, 138)], moss=(120, 140, 140),
         stone=(150, 154, 176), bush=[(40, 66, 70), (110, 140, 150)], pad=(70, 110, 96), lotus=(255, 255, 255),
         glint=(240, 246, 255), lantern=(255, 200, 130), rail=(180, 66, 66),
         float=None, flies=0.0, fly=(255, 230, 150)),
    dict(water=[(80, 30, 130), (30, 10, 80), (6, 2, 30)], bank=[(30, 18, 48), (16, 8, 30)], moss=(44, 32, 86),
         stone=(84, 66, 124), bush=[(20, 30, 60), (60, 40, 140)], pad=(30, 120, 110), lotus=(255, 80, 200),
         glint=(90, 255, 230), lantern=(255, 60, 190), rail=(255, 60, 190),
         float=None, flies=1.2, fly=(90, 255, 230)),
    dict(water=[(50, 120, 120), (14, 60, 72), (2, 20, 30)], bank=[(32, 58, 48), (16, 34, 30)], moss=(54, 96, 70),
         stone=(110, 128, 124), bush=[(16, 44, 36), (56, 110, 80)], pad=(56, 130, 84), lotus=(255, 214, 230),
         glint=(236, 255, 240), lantern=(255, 214, 120), rail=(190, 70, 56),
         float=[(255, 214, 232)], flies=2.0, fly=(220, 255, 140)),
    dict(water=[(112, 202, 212), (50, 150, 170), (20, 90, 120)], bank=[(142, 152, 110), (100, 110, 80)], moss=(122, 172, 90),
         stone=(192, 188, 178), bush=[(60, 122, 60), (132, 192, 90)], pad=(90, 172, 90), lotus=(255, 182, 212),
         glint=(255, 255, 240), lantern=(255, 202, 122), rail=(222, 70, 50),
         float=[(255, 232, 122), (255, 255, 255)], flies=0.0, fly=(255, 230, 150)),
    dict(water=[(60, 70, 130), (26, 30, 80), (8, 8, 34)], bank=[(60, 50, 70), (36, 30, 46)], moss=(80, 100, 86),     # tokyo
         stone=(130, 122, 140), bush=[(70, 30, 60), (230, 140, 180)], pad=(60, 116, 84), lotus=(255, 150, 190),
         glint=(255, 214, 230), lantern=(255, 150, 100), rail=(224, 56, 46),
         float=[(255, 200, 224), (255, 170, 206), (255, 236, 244)], flies=0.0, fly=(255, 230, 150)),
    dict(water=[(100, 84, 70), (48, 36, 40), (16, 12, 18)], bank=[(56, 48, 42), (34, 28, 28)], moss=(70, 80, 56),     # jazz
         stone=(120, 110, 100), bush=[(30, 40, 30), (110, 100, 60)], pad=(70, 100, 60), lotus=(255, 226, 170),
         glint=(255, 214, 150), lantern=(255, 180, 90), rail=(150, 110, 50),
         float=None, flies=1.4, fly=(255, 210, 120)),
    dict(water=[(80, 100, 96), (34, 48, 50), (10, 16, 20)], bank=[(70, 76, 66), (44, 50, 44)], moss=(76, 104, 66),     # english
         stone=(150, 146, 136), bush=[(30, 52, 36), (90, 120, 70)], pad=(70, 110, 64), lotus=(255, 246, 236),
         glint=(255, 226, 170), lantern=(255, 206, 130), rail=(70, 66, 60),
         float=[(255, 250, 244), (250, 226, 232)], flies=0.6, fly=(255, 230, 160))
]


class KoiScene:
    # ── koi pond (seen from above) ─────────────────────────────────────────
    @staticmethod
    def _koi_fish(rng, x, y, L):
        WH, RD, BK = (250, 246, 238), (232, 74, 44), (34, 30, 38)
        kind = rng.choice(["kohaku", "kohaku", "kohaku", "sanke", "sanke", "showa", "tancho", "ogon", "chagoi", "asagi"])
        base, patches = WH, []

        def blobs(col, n, rs=(0.08, 0.16), ru=(0.45, 0.85)):
            for _ in range(n):
                patches.append((rng.uniform(0.1, 0.75), rng.uniform(-0.35, 0.35), rng.uniform(*rs), rng.uniform(*ru), col))
        if kind == "kohaku":
            blobs(RD, rng.randint(2, 4))
        elif kind == "sanke":
            blobs(RD, rng.randint(2, 3))
            blobs(BK, rng.randint(3, 6), (0.03, 0.05), (0.2, 0.35))
        elif kind == "showa":
            base = BK
            blobs(RD, 3, (0.1, 0.16))
            blobs(WH, 2, (0.06, 0.12))
        elif kind == "tancho":
            patches.append((0.07, 0.0, 0.06, 0.55, RD))
        elif kind == "ogon":
            base = (255, 196, 84)
        elif kind == "chagoi":
            base = (180, 122, 76)
        else:                                                # asagi: blue-grey back, red flanks
            base = (118, 140, 174)
            patches += [(0.55, 1.0, 0.5, 0.45, RD), (0.55, -1.0, 0.5, 0.45, RD)]
        L *= rng.uniform(0.75, 1.15)
        a, n = rng.uniform(0, TAU), len(KOI_PROFILE)
        seg = L / (n - 1)
        pts = np.array([[x - math.cos(a) * seg * i, y - math.sin(a) * seg * i] for i in range(n)])
        return dict(pts=pts, a=a, L=L, seg=seg, turn=0.0, base=base, patches=patches, ogon=kind == "ogon",
                    long=rng.random() < 0.25, speed=rng.uniform(0.18, 0.3) * L / 14, phase=rng.uniform(0, TAU),
                    z=rng.uniform(0.2, 0.9), zt=rng.uniform(0.2, 0.9), boost=0.0, seed=rng.uniform(0, 9))

    def _koi_layout(self, h, w):
        """Pond shape, rocks, bushes, bridges, lily pads, lanterns and fish for one window size."""
        H = 2 * h
        rng = wrng("koi", h, w)
        yy, xx = grid(H, w)
        X, Y = xx + 0.5, yy + 0.5
        sc = max(0.8, min(1.8, H / 70))
        if w < 110:
            lobes = [(w * 0.5, H * 0.5, w * 0.4, H * 0.37)]
        else:                                                # wide: a chain of pools joined by bridges
            n = 2 if w < 230 else 3
            lobes = [(w * (i + 0.5) / n, H * (0.5 + rng.uniform(-0.04, 0.04)), w / n * 0.6, H * rng.uniform(0.34, 0.39))
                     for i in range(n)]
        N = noise_tex(H, w, 5, 31)
        F = (np.min([((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 for cx, cy, rx, ry in lobes], axis=0) + 0.1 * N).astype(np.float32)
        water = F < 1
        bw = max(5, int(round(5 * sc)))
        bridges = []
        for (ax, _, _, _), (bx, _, _, _) in zip(lobes, lobes[1:]):
            xb = int((ax + bx) / 2)
            ys = np.nonzero(water[:, xb])[0]
            if len(ys):
                bridges.append((xb, int(ys.min()), int(ys.max())))

        def near_bridge(x, y, pad):
            return any(abs(x - xb) < bw / 2 + pad and y0 - pad - 5 < y < y1 + pad + 6 for xb, y0, y1 in bridges)

        def scatter(mask, n, rmin, rmax, gap, taken, tries=600, spread=0.0):
            cy_, cx_ = np.nonzero(mask)
            out = []
            if not len(cy_):
                return out
            for _ in range(tries):
                if len(out) >= n:
                    break
                i = rng.randrange(len(cy_))
                x, y, r = cx_[i] + 0.5, cy_[i] + 0.5, rng.uniform(rmin, rmax)
                if near_bridge(x, y, r + 1) or any(math.hypot(x - a, y - b) < (r + c) * gap for a, b, c, *_ in taken + out):
                    continue
                if any(math.hypot(x - a, y - b) < spread for a, b, *_ in out):
                    continue
                out.append((x, y, r, rng.random()))
            return out
        rim = (F > 0.97) & (F < 1.1)
        stones = scatter(rim, int(rim.sum() / (5 * sc)), 1.8 * sc, 3.6 * sc, 0.95, [], tries=3000)
        stones = [s for s in stones if rng.random() > 0.2]   # gaps where the grass reaches the water
        edge_ok = (xx > 2) & (xx < w - 3) & (yy > 2) & (yy < H - 3)
        bushes = scatter((F > 1.9) & edge_ok, w // 22 + 2, 3.5 * sc, 6.5 * sc, 1.3, stones)
        lanterns = scatter((F > 1.15) & (F < 1.7) & edge_ok, len(lobes) + 1, 2.5 * sc, 2.5 * sc, 1.1, stones + bushes,
                           spread=w / (len(lobes) + 2))
        pads = scatter(F < 0.75, int(water.sum() / 650) + 2, 2.5 * sc, 5 * sc, 1.15, [])
        pads = [(x, y, r, q * TAU, rng.random() < 0.35) for x, y, r, q in pads]
        cx, cy, rx, ry = lobes[-1]
        moon_at = (cx + rx * 0.35, cy - ry * 0.4)
        L = max(8.0, min(20.0, H * 0.17))
        area = int(water.sum())
        safe = np.nonzero(F < 0.5)
        fish = []
        for _ in range(max(3, min(16, int(area / (L * L * 3.4))))):
            i = rng.randrange(len(safe[0]))
            fish.append(self._koi_fish(rng, safe[1][i] + 0.5, safe[0][i] + 0.5, L))
        flies = [[rng.uniform(0, w), rng.uniform(0, H), rng.uniform(0, TAU), rng.uniform(0.6, 1.6)] for _ in range(w // 18 + 3)]
        wy, wx = np.nonzero(F < 0.97)
        return dict(size=(h, w), F=F, D=np.clip((1 - F) * 2.2, 0, 1), N=N, N2=noise_tex(H, w, 2, 32), sc=sc,
                    CA=noise_tex(H, w, 5, 21) * 3.2, CB=noise_tex(H, w, 7, 22) * 3.2,
                    centers=[(a, b) for a, b, _, _ in lobes], bridges=bridges, bw=bw, stones=stones, bushes=bushes,
                    lanterns=lanterns, pads=pads, moon=moon_at, fish=fish, flies=flies, area=area, wpts=(wy, wx),
                    ripples=[], floating=[], flakes=[], splash=[], bubbles=[], hit=False, snow=0.0, style=None)

    def _koi_style(self, k, V):
        """Station colours baked into a static water base and a bank overlay (grass, rocks, bushes, bridges)."""
        F, D, N, N2 = k["F"], k["D"], k["N"], k["N2"]
        H, w = F.shape
        yy, xx = grid(H, w)
        stops = [(0, V["water"][0]), (0.45, V["water"][1]), (1, V["water"][2])]
        W0 = np.stack([np.interp(D, [p for p, _ in stops], [c[i] for _, c in stops]) for i in range(3)], -1).astype(np.float32)
        W0 *= (0.93 + 0.07 * N)[..., None]
        W0 *= (1 - 0.3 * np.clip((F - 0.82) / 0.18, 0, 1))[..., None]   # rocks shade the water at the rim
        g = np.clip(0.5 + 0.4 * N2 + 0.2 * N, 0, 1)[..., None]
        O = _c(V["bank"][1]) * (1 - g) + _c(V["bank"][0]) * g
        O += (_c(V["moss"]) - O) * (np.clip((N2 - 0.5) * 2, 0, 1) * 0.7)[..., None]
        blade = hash01(xx, yy, 7) < 0.08
        O[blade] *= 1.25
        O *= (1 - 0.3 * np.clip(1 - (F - 1) / 0.25, 0, 1) * (F >= 1))[..., None]   # damp soil by the water
        A = np.clip((F - 1) * 12 + 0.5, 0, 1)
        SH = np.zeros((H, w), np.float32)
        LIT = np.full((H, w), 0.5, np.float32)

        def box(cx, cy, r, pad=3):
            y0, y1 = max(0, int(cy - r - pad)), min(H, int(cy + r + pad) + 1)
            x0, x1 = max(0, int(cx - r - pad)), min(w, int(cx + r + pad) + 1)
            return (slice(y0, y1), slice(x0, x1)) if y0 < y1 and x0 < x1 else None

        def blob(cx, cy, rx, ry, rough, colfn, shadow=0.5, sx=1.2, sy=1.5):
            b = box(cx, cy, max(rx, ry))
            if b is None:
                return
            Xb, Yb, nb = xx[b] + 0.5, yy[b] + 0.5, N2[b]
            dx, dy = (Xb - cx) / rx, (Yb - cy) / ry
            d = np.sqrt(dx * dx + dy * dy) + rough * nb
            m = np.clip((1 - d) * min(rx, ry) + 0.5, 0, 1)
            ds = np.sqrt(((Xb - cx - sx) / rx) ** 2 + ((Yb - cy - sy) / ry) ** 2) + rough * nb
            SH[b] = np.maximum(SH[b], np.clip((1 - ds) * min(rx, ry) + 0.5, 0, 1) * shadow)
            lit = np.clip(0.5 - 0.45 * (dx + dy) + 0.2 * nb, 0, 1)      # light from the upper left
            col = colfn(lit, d, nb)
            O[b] += (col - O[b]) * m[..., None]
            A[b] = np.maximum(A[b], m)
            LIT[b] = LIT[b] * (1 - m) + lit * m

        stone = _c(V["stone"])
        for sx_, sy_, r, q in k["stones"]:
            def scol(lit, d, nb, q=q):
                c = stone * (0.8 + 0.3 * q) * (0.6 + 0.6 * lit)[..., None] * (1 - 0.25 * np.clip(d, 0, 1) ** 4)[..., None]
                if q > 0.6:                                  # mossy top
                    c += (_c(V["moss"]) - c) * (np.clip(lit * 1.4 - 0.5 + 0.4 * nb, 0, 1) * 0.6)[..., None]
                return c
            blob(sx_, sy_, r * 1.2, r, 0.12, scol)
        dark, light = _c(V["bush"][0]), _c(V["bush"][1])
        for bx_, by_, r, q in k["bushes"]:
            rr = wrng("bush", int(bx_ * 7 + by_))
            for j in range(4):
                ox, oy = (rr.uniform(-0.6, 0.6) * r, rr.uniform(-0.5, 0.5) * r) if j else (0, 0)
                def bcol(lit, d, nb):
                    return (dark + (light - dark) * np.clip(lit + 0.25 * nb, 0, 1)[..., None]) * (0.9 + 0.1 * nb)[..., None]
                blob(bx_ + ox, by_ + oy, r * rr.uniform(0.55, 0.8), r * rr.uniform(0.5, 0.75), 0.25, bcol, 0.6, 2.0, 2.5)
        wood, rail = _c((150, 106, 74)), _c(V["rail"])
        bw = k["bw"]
        for xb, wy0, wy1 in k["bridges"]:                    # little arched bridge, planks across
            x0, x1 = max(1, xb - bw // 2), min(w - 1, xb - bw // 2 + bw)
            y0, y1 = max(0, wy0 - 5), min(H, wy1 + 6)
            for y in range(y0, y1):
                arch = 0.8 + 0.3 * math.sin(math.pi * (y - y0) / max(1, y1 - y0 - 1))
                O[y, x0:x1] = wood * arch * (1.0 if (y // 2) % 2 else 0.88)
                O[y, x0] = O[y, x1 - 1] = rail * arch
                A[y, x0:x1] = 1
                LIT[y, x0:x1] = 0.7
                if (y - y0) % 6 == 0:                        # posts
                    O[y, [x0 - 1, x1]] = rail * arch * 0.75
                    A[y, [x0 - 1, x1]] = 1
            SH[min(H, y0 + 2):min(H, y1 + 2), x1:min(w, x1 + 3)] = 0.55
        k.update(W0=W0, O=O, A=A, SH=SH, LIT=LIT)

    @staticmethod
    def _koi_mask(H, W, pts, rads):
        """Soft mask of a body built from capsules along the spine, plus body coordinates
        s (0 at the nose → 1 at the tail) and u (−1…1 across the body)."""
        pad = rads.max() + 2
        x0, x1 = max(0, int(pts[:, 0].min() - pad)), min(W, int(pts[:, 0].max() + pad) + 1)
        y0, y1 = max(0, int(pts[:, 1].min() - pad)), min(H, int(pts[:, 1].max() + pad) + 1)
        if x0 >= x1 or y0 >= y1:
            return None
        yy, xx = grid(H, W)
        X, Y = xx[y0:y1, x0:x1] + 0.5, yy[y0:y1, x0:x1] + 0.5
        best = np.full(X.shape, 1e9, np.float32)
        S, U = np.zeros_like(best), np.zeros_like(best)
        n = len(pts)
        for i in range(n - 1):
            ax, ay = pts[i]
            dx, dy = pts[i + 1] - pts[i]
            tt = np.clip(((X - ax) * dx + (Y - ay) * dy) / (dx * dx + dy * dy or 1e-6), 0, 1)
            px, py = X - (ax + tt * dx), Y - (ay + tt * dy)
            d = np.hypot(px, py)
            r = rads[i] + (rads[i + 1] - rads[i]) * tt
            sd = d - r
            upd = sd < best
            best = np.where(upd, sd, best)
            S = np.where(upd, (i + tt) / (n - 1), S)
            U = np.where(upd, np.sign(dx * py - dy * px) * np.clip(d / np.maximum(r, 0.3), 0, 1), U)
        return (y0, y1, x0, x1), np.clip(0.5 - best, 0, 1), S, U, best

    def _koi_swim(self, k, f, pace, storm):
        F = k["F"]
        H, w = F.shape
        pts = f["pts"]
        f["boost"] *= 0.95
        spd = f["speed"] * pace * (1 + 2.2 * f["boost"])
        f["turn"] = max(-0.045, min(0.045, f["turn"] * 0.97 + random.gauss(0, 0.012)))
        a = f["a"] + f["turn"]
        hx, hy = pts[0]
        look = max(6.0, f["L"] * 0.9)
        lx, ly = int(hx + math.cos(a) * look), int(hy + math.sin(a) * look)
        ahead = F[ly, lx] if 0 <= lx < w and 0 <= ly < H else 9.0
        if ahead > 0.6:                                      # bank ahead: curve back toward open water
            cx, cy = min(k["centers"], key=lambda c: (c[0] - hx) ** 2 + (c[1] - hy) ** 2)
            diff = (math.atan2(cy - hy, cx - hx) - a + math.pi) % TAU - math.pi
            here = F[int(hy), int(hx)] if 0 <= int(hx) < w and 0 <= int(hy) < H else 9.0
            lim = 0.2 if here > 0.85 else 0.1
            a += max(-lim, min(lim, diff * 0.25))
            f["turn"] *= 0.6
        f["a"] = a
        pts[0] += (math.cos(a) * spd, math.sin(a) * spd)
        for i in range(1, len(pts)):                         # the body follows the head
            d = pts[i - 1] - pts[i]
            pts[i] = pts[i - 1] - d * (f["seg"] / (math.hypot(d[0], d[1]) or 1e-6))
        f["phase"] += 0.1 + spd * 0.6
        if random.random() < 0.004:
            f["zt"] = random.uniform(0.55, 0.95) if storm else random.uniform(0.05, 0.9)
        f["z"] += (f["zt"] - f["z"]) * 0.01
        if f["z"] < 0.14 and random.random() < 0.012:       # comes up for a gulp
            x, y = pts[0]
            k["ripples"].append([x, y, 0.0, 5 * k["sc"], 1.0])
            k["bubbles"].append([int(y) // 2, int(x), 10])

    def _koi_draw(self, P, f, V):
        H, W = P.shape[:2]
        p0 = f["pts"]
        L, n, z = f["L"], len(p0), f["z"]
        tang = np.empty_like(p0)
        tang[:-1] = p0[:-1] - p0[1:]
        tang[-1] = tang[-2]
        tang /= np.hypot(tang[:, 0], tang[:, 1])[:, None] + 1e-6
        nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
        i = np.arange(n)
        amp = L * 0.07 * (0.8 + 0.6 * min(1.0, f["boost"]))
        pts = p0 + nrm * (np.sin(f["phase"] - i * 0.75) * amp * (i / (n - 1)) ** 1.4)[:, None]   # swimming wave
        R = L * 0.18
        rads = R * KOI_PROFILE
        res = self._koi_mask(H, W, pts, rads)
        if res is None:
            return
        (y0, y1, x0, x1), al, S, U, best = res
        zt = 0.04 + 0.4 * z                                  # deeper fish fade into the water
        water = _c(V["water"][1])
        # shadow on the pond floor: further away the closer the fish swims to the surface
        o = int(round(1 + 3 * (1 - z)))
        sy0, sy1, sx0, sx1 = y0 + o, min(H, y1 + o), x0 + o, min(W, x1 + o)
        if sy0 < sy1 and sx0 < sx1:
            P[sy0:sy1, sx0:sx1] *= (1 - 0.4 * (1 - 0.5 * z) * al[:sy1 - sy0, :sx1 - sx0])[..., None]
        base = _c(f["base"])
        fin = (base * 0.55 + 110) * (1 - zt) + water * zt     # fins are pale and see-through
        long = 1.7 if f["long"] else 1.0

        def rot(v, ang):
            c, s = math.cos(ang), math.sin(ang)
            return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c])
        back = -tang[-1]
        wig = 0.45 * math.sin(f["phase"] - n * 0.75)
        Lt = L * 0.22 * long
        bx, by = pts[-1]
        t1, t2, tm = rot(back, 0.42 + wig), rot(back, -0.42 + wig), rot(back, wig)
        poly(P, [(bx - back[0], by - back[1]), (bx + t1[0] * Lt, by + t1[1] * Lt),
                 (bx + tm[0] * Lt * 0.55, by + tm[1] * Lt * 0.55), (bx + t2[0] * Lt, by + t2[1] * Lt)], fin, a=0.5)
        pa = pts[2] * 0.6 + pts[1] * 0.4
        flap = 0.4 * math.sin(f["phase"] * 0.6)
        for side in (1, -1):                                 # pectoral fins paddle
            root = pa + nrm[2] * side * rads[2] * 0.6
            tip = root + rot(-tang[2], -side * (1.0 + flap)) * L * 0.15 * long
            line(P, root[0], root[1], tip[0], tip[1], fin, (max(1.0, R * 0.6), 0.5), a=0.45)
        col = np.broadcast_to(base, S.shape + (3,)).copy()
        for s0, u0, rs, ru, pc in f["patches"]:
            m = 1 - ((S - s0) / rs) ** 2 - ((U - u0) / ru) ** 2 + 0.25 * np.sin(S * 37 + U * 5 + f["seed"])
            col += (_c(pc) - col) * np.clip(m * 3, 0, 1)[..., None]
        col *= (1.1 - 0.4 * U * U)[..., None]
        col += (np.exp(-((U + 0.3) / 0.22) ** 2) * (1 - S) * (60 if f["ogon"] else 28))[..., None]
        col *= (1 - 0.18 * np.clip(best + 1.2, 0, 1))[..., None]
        col = col * (1 - zt) + water * zt
        v = P[y0:y1, x0:x1]
        v += (col - v) * al[..., None]
        if R >= 2:
            for side in (1, -1):
                ex, ey = pts[0] + tang[0] * rads[0] * 0.3 + nrm[0] * side * rads[0] * 0.75
                if 0 <= int(ey) < H and 0 <= int(ex) < W:
                    P[int(ey), int(ex)] *= 0.35 + 0.5 * zt

    def _koi_lights(self, P, k, th, t, sc):
        H, w = P.shape[:2]
        L = k.get("lights")
        if L is None or k.get("lights_th") != th:
            r = wrng("koilights", w, H)
            wy, wx = k["wpts"]
            L = k["lights"] = [[wx[j] + 0.5, wy[j] + 0.5, r.random()] for j in (r.randrange(len(wy)) for _ in range(max(4, w // 14)))]
            k["lights_th"] = th
        F = k["F"]
        for fp in L:
            if not self.eng.paused:
                cx, cy = min(k["centers"], key=lambda c: (c[0] - fp[0]) ** 2 + (c[1] - fp[1]) ** 2)
                dx, dy = fp[0] - cx, fp[1] - cy
                n_ = math.hypot(dx, dy) or 1.0
                nx_, ny_ = fp[0] - dy / n_ * 0.02, fp[1] + dx / n_ * 0.02
                xi, yi = int(nx_), int(ny_)
                if 0 <= xi < w and 0 <= yi < H and F[yi, xi] < 0.9:
                    fp[0], fp[1] = nx_, ny_
            x, y = fp[0], fp[1]
            fl = 0.85 + 0.15 * math.sin(t * 9 + fp[2] * 40)
            if th == "japan":
                s = max(1.0, 1.2 * sc)
                glow(P, x, y, 5 * sc, (255, 150, 90), 0.35 * fl)
                rect(P, x - s, y - s, x + s + 1, y + s + 1, (250, 236, 210))
                rect(P, x - s, y - s, x + s + 1, y - s + 1, (200, 50, 46))
            else:
                glow(P, x, y, 4 * sc, (255, 180, 80), (0.3 + 0.25 * self.bass) * fl)
                disc(P, x, y, max(1.0, 1.1 * sc), (230, 220, 200))
                P[int(y) % H, int(x) % w] = (255, 230, 150)

    def v_koi(self, cv, P, h, w):
        e = self.eng
        t = time.time()
        H = P.shape[0]
        level = float(self.rms.mean())
        k = getattr(self, "koi", None)
        if k is None or k["size"] != (h, w):
            k = self.koi = self._koi_layout(h, w)
        V = self.var(KOI)
        if k["style"] != e.station_idx:
            self._koi_style(k, V)
            k["style"], k["snow"] = e.station_idx, 0.0
        F, sc = k["F"], k["sc"]
        wet = e.wet
        glint = _c(V["glint"])
        # ── water: depth, dancing caustics, the moon's reflection ──
        P[:] = k["W0"]
        c = np.sin(k["CA"] + t * 0.9) + np.sin(k["CB"] - t * 0.75)
        cau = np.exp(-c * c * 4) * (1 - 0.6 * k["D"])
        P += cau[..., None] * glint * ((0.05 + 0.25 * level + 0.05 * self.bass + 0.1 * self.kick) * (0.5 if wet else 1.0))
        P += (k["D"] < 0.5)[..., None] * self.tint * (0.03 * (0.5 + self.change))    # the shallows pick up the chord
        mx, my = k["moon"]
        rr, dim = max(2.0, H * 0.045), 0.45 if wet else 1.0
        glow(P, mx, my, rr * 2.6, V["glint"], 0.1 * dim, ry=rr * 1.6)
        glow(P, mx, my, rr, V["glint"], 0.35 * dim, ry=rr * 0.7)
        b = _box(P, mx - rr * 3, my - rr * 2, mx + rr * 3, my + rr * 2)
        if b is not None:
            v, yb, xb = b
            sp = (hash01(xb, yb, int(t * 5) % 1000) < 0.14) & (((xb - mx) / (rr * 3)) ** 2 + ((yb - my) / (rr * 2)) ** 2 < 1)
            v[sp] += glint * 0.55 * dim
        # ── fish: wander, dart on a bass hit, rise for a gulp, sink in a storm ──
        if (self.bass > 0.8 and not k["hit"]) or ("kick" in self.fired and random.random() < 0.2):
            f = random.choice(k["fish"])
            f["boost"], f["zt"] = 1.0, 0.05
        k["hit"] = self.bass > 0.8
        if "note" in self.fired or "pluck" in self.fired and random.random() < 0.3:   # a drop for every note, across by pitch
            wy, wx = k["wpts"]
            p_ = self.pitch if "note" in self.fired else self.ppitch
            near = np.nonzero(np.abs(wx - p_ * (w - 1)) < 4)[0]
            if len(near):
                j = int(near[random.randrange(len(near))])
                k["ripples"].append([wx[j] + 0.5, wy[j] + 0.5, 0.0, random.uniform(4.0, 6.0) * sc, 1.0])
        pace = 0.55 + 1.8 * level
        for f in k["fish"]:
            self._koi_swim(k, f, pace, e.storm_on)
        for f in sorted(k["fish"], key=lambda f: -f["z"]):
            self._koi_draw(P, f, V)
        # ── rain rings ──
        snowy = e.st.get("weather") == "snow"
        inten = (1.0 if e.rain_on else 0.0) + (1.4 if e.storm_on else 0.0)
        if not snowy and inten:
            wy, wx = k["wpts"]
            spawn = inten * k["area"] / 2500 * (0.6 + 3 * level)
            for _ in range(int(spawn) + (random.random() < spawn % 1)):
                j = random.randrange(len(wy))
                k["ripples"].append([wx[j] + 0.5, wy[j] + 0.5, 0.0, random.uniform(2.5, 4.5) * sc, 0.7])
                if random.random() < 0.5:
                    k["splash"].append([int(wy[j]) // 2, int(wx[j]), 2])
        keep = []
        for r in k["ripples"]:
            r[2] += 0.3
            if r[2] >= r[3]:
                continue
            keep.append(r)
            x, y, rad, mr, st = r
            b = _box(P, x - rad - 3, y - rad - 3, x + rad + 4, y + rad + 4)
            if b is None:
                continue
            v, yb, xb = b
            d = np.hypot(xb + 0.5 - x, yb + 0.5 - y)
            fade = (1 - rad / mr) * st
            v += (np.exp(-((d - rad) / 0.8) ** 2) * fade * 0.5)[..., None] * glint
            v *= (1 - 0.25 * np.exp(-((d - rad + 1.4) / 0.8) ** 2) * fade)[..., None]
        k["ripples"] = keep[-160:]
        # ── lily pads and lotus flowers (they bloom brighter with the bass) ──
        pad = _c(V["pad"])
        for i, (x, y, r, rot0, flower) in enumerate(k["pads"]):
            disc(P, x + 1, y + 1.2, r, (0, 0, 0), a=0.35)
            b = _box(P, x - r - 1, y - r - 1, x + r + 2, y + r + 2)
            if b is None:
                continue
            v, yb, xb = b
            dx, dy = xb + 0.5 - x, yb + 0.5 - y
            d = np.hypot(dx, dy)
            ang = (np.arctan2(dy, dx) - rot0 - 0.06 * math.sin(t * 0.3 + i) + math.pi) % TAU - math.pi
            m = np.clip(r - d + 0.5, 0, 1) * ~((np.abs(ang) < 0.35) & (d > r * 0.15))
            shade = (0.85 + 0.25 * (-dx - dy) / (r * 1.4)) * (0.93 + 0.07 * np.cos(ang * 7)) * np.where(d > r - 1, 0.8, 1.0)
            v += (pad * shade[..., None] - v) * m[..., None]
            if flower:
                pr = (1.4 + 0.4 * self.bass + 0.3 * self.kick) * sc
                for j in range(6):
                    pa_ = j * TAU / 6 + rot0
                    disc(P, x + math.cos(pa_) * pr * 0.7, y + math.sin(pa_) * pr * 0.7, max(0.9, pr * 0.55), V["lotus"])
                disc(P, x, y, max(0.7, pr * 0.4), (255, 226, 120))
                glow(P, x, y, 2.5 * sc, V["lotus"], 0.08 + 0.2 * self.bass + 0.2 * self.note)
        # ── petals / leaves drifting on a slow current ──
        if V["float"] and random.random() < 0.03 * (w / 80) * (1 + 4 * level):
            wy, wx = k["wpts"]
            j = random.randrange(len(wy))
            k["floating"].append([wx[j] + 0.5, wy[j] + 0.5, random.randint(300, 900), random.choice(V["float"]), random.random()])
        keep = []
        for fp in k["floating"]:
            cx, cy = min(k["centers"], key=lambda c: (c[0] - fp[0]) ** 2 + (c[1] - fp[1]) ** 2)
            dx, dy = fp[0] - cx, fp[1] - cy
            n_ = math.hypot(dx, dy) or 1.0
            fp[0] += -dy / n_ * 0.03 + 0.01 * math.sin(t + fp[4] * 9)
            fp[1] += dx / n_ * 0.03
            fp[2] -= 1
            xi, yi = int(fp[0]), int(fp[1])
            if fp[2] <= 0 or not (0 <= xi < w - 1 and 0 <= yi < H) or F[yi, xi] > 0.95:
                continue
            keep.append(fp)
            a_ = min(1.0, fp[2] / 60)
            P[yi, xi] += (_c(fp[3]) - P[yi, xi]) * a_
            if e.station_idx == 2:                           # maple leaves are bigger
                P[yi, xi + 1] += (_c(fp[3]) * 0.75 - P[yi, xi + 1]) * a_
        k["floating"] = keep[-90:]
        th = self.theme()
        if th:                                           # floating lights: paper lanterns, or candles in glass
            self._koi_lights(P, k, th, t, sc)
        # ── the bank, rocks, bushes and bridges on top (snow piles up on Snowfall Tape) ──
        O = k["O"]
        if snowy:
            k["snow"] = min(1.0, k["snow"] + 0.0004 * (1 + inten))
            S = np.clip(k["snow"] * 1.8 - 0.8 + 0.35 * k["N2"] + 0.5 * (k["LIT"] - 0.5), 0, 1) * 0.92
            O = O + (_c((236, 240, 252)) - O) * S[..., None]
        P += (O - P) * k["A"][..., None]
        P *= (1 - 0.45 * k["SH"])[..., None]
        # ── stone lanterns: glow on the beat, warmer with the fireplace on ──
        stone = _c(V["stone"])
        for i, (x, y, _, q) in enumerate(k["lanterns"]):
            on = (int(e.beat_pos) + i) % 2 == 0
            heat = (0.3 + 0.1 * on + 0.2 * self.kick + (0.35 if e.fire_on else 0)) * (0.9 + 0.1 * math.sin(t * 13 + q * 20))
            if self.noon():                                  # unlit by day
                heat *= 0.12
            glow(P, x, y, 7 * sc, V["lantern"], heat * 0.6)
            glow(P, x, y, 3 * sc, V["lantern"], heat * 1.2)
            s = max(2, int(round(1.8 * sc)))
            xi, yi = int(x), int(y)
            P[max(0, yi - s):yi + s, max(0, xi - s):xi + s] = stone * 1.05
            P[max(0, yi - s):yi, max(0, xi - s):xi] = stone * 1.25
            P[max(0, yi - s):yi + s, xi + s:xi + s + 1] *= 0.6
            P[yi + s:yi + s + 1, max(0, xi - s):xi + s + 1] *= 0.6
            P[yi - 1:yi + 1, xi - 1:xi + 1] = _c(V["lantern"]) * (0.7 + 0.5 * heat)   # light under the roof
        # ── fireflies over the water ──
        nf = int(len(k["flies"]) * min(1.0, V["flies"] / 2) + 0.5)
        for fl in k["flies"][:nf]:
            fl[0] = (fl[0] + 0.25 * math.sin(t * 0.5 * fl[3] + fl[2])) % w
            fl[1] = (fl[1] + 0.2 * math.cos(t * 0.4 * fl[3] + fl[2] * 2)) % H
            blink = min(1.0, max(0.0, math.sin(t * fl[3] + fl[2])) ** 3 + 0.4 * self.hat)
            if blink > 0.05:
                glow(P, fl[0], fl[1], 1.8, V["fly"], 0.6 * blink)
                P[int(fl[1]), int(fl[0])] += _c(V["fly"]) * blink
        ev = self.rare(20.0)
        if ev:
            self._rare_koi(P, ev, k, w)
        if th == "jazz" and k["lanterns"]:                # someone's radio by the first lantern
            x, y = k["lanterns"][0][:2]
            self.notes(cv, int(x), int(y) // 2 - 1, (255, 214, 150), n=3, rise=4, seed=8)
        # ── overlay: snowflakes, rain splashes, bubbles ──
        if snowy:
            spawn = (0.6 + inten) * w / 30 * (0.4 + 4 * level) * 0.4
            for _ in range(int(spawn) + (random.random() < spawn % 1)):
                k["flakes"].append([random.uniform(0, w), random.uniform(0, h), random.randint(40, 140), random.random()])
        keep = []
        for fk in k["flakes"]:
            fk[0] += 0.12 + 0.1 * math.sin(t * 1.5 + fk[3] * 9)
            fk[1] += 0.06
            fk[2] -= 1
            if fk[2] > 0:
                keep.append(fk)
                cv.put(int(fk[1]), int(fk[0]), "*" if fk[3] > 0.6 else "·", C_WHITE)
        k["flakes"] = keep[-400:]
        for lst, chars, col in ((k["splash"], "˙·", C_RAIN), (k["bubbles"], "∘°°°°", V["glint"])):
            keep = []
            for sp in lst:
                cy_, cx_ = sp[0], sp[1]
                if 0 <= cy_ < h and 0 <= cx_ < w and F[min(H - 1, 2 * cy_), cx_] < 1:
                    cv.put(cy_, cx_, chars[min(len(chars) - 1, sp[2] // 2)], col)
                sp[2] -= 1
                if sp[2] > 0:
                    keep.append(sp)
            lst[:] = keep

    def _rare_koi(self, P, ev, k, w):
        """Rare event: a dragonfly zips in, hovers over the pond in a few spots, dips to touch the water
        (leaving a ring) and darts away. Its shadow follows it across the water."""
        u, r = ev
        H = P.shape[0]
        wy, wx = k["wpts"]
        spots = [(float(wx[j]), float(wy[j])) for j in (r.randrange(len(wy)) for _ in range(3))]
        side = r.choice((-1, 1))
        a0 = (-6.0 if side > 0 else w + 6.0, r.uniform(0.1, 0.9) * H)
        a1 = (w + 6.0 if side > 0 else -6.0, r.uniform(0.1, 0.9) * H)
        pts = [a0] + spots + [a1]
        body = r.choice([(60, 170, 220), (220, 76, 60), (90, 200, 120)])
        fly, hover = 1.0, 2.6                                            # time units for each leg
        legs = len(pts) - 1
        total = legs * fly + len(spots) * hover
        tt = u * total
        T = self.rclock
        dip = 0.0
        for i in range(legs):
            if tt < fly:
                q = tt / fly
                q = q * q * (3 - 2 * q)
                (x0, y0), (x1, y1) = pts[i], pts[i + 1]
                x, y, head = x0 + (x1 - x0) * q, y0 + (y1 - y0) * q, math.atan2(y1 - y0, x1 - x0)
                break
            tt -= fly
            if i < len(spots) and tt < hover:
                x, y = spots[i]
                x += math.sin(T * 2.3) * 1.2
                y += math.sin(T * 3.1) * 0.8
                (x0, y0), (x1, y1) = pts[i], pts[i + 1]
                head = math.atan2(y1 - y0, x1 - x0) + 0.3 * math.sin(T * 0.9)
                q = tt / hover
                if 0.7 < q < 0.85:                                       # dip to the water
                    dip = math.sin(math.pi * (q - 0.7) / 0.15)
                    if q > 0.77 and not self.rare_mem.get(("dip", i)):
                        self.rare_mem[("dip", i)] = True
                        k["ripples"].append([x, y, 0.0, 5.0 * k["sc"], 0.9])
                break
            if i < len(spots):
                tt -= hover
        else:
            return
        alt = 3.0 * (1 - dip)                                           # height above the water → shadow offset
        c, s = math.cos(head), math.sin(head)
        sc = max(1.5, k["sc"] * 1.5)

        def bug(ox, oy, col_b, col_w, a):
            for wo, wl in ((0.6, 3.2), (-0.4, 2.8)):                    # two pairs of wings, a blur of beats
                fl = 0.7 + 0.3 * math.sin(T * 60 + wo)
                for sd in (-1, 1):
                    line(P, ox + c * wo * sc, oy + s * wo * sc, ox + (c * wo - s * sd * wl * fl) * sc,
                         oy + (s * wo + c * sd * wl * fl) * sc, col_w, wid=1.0, a=a * 0.65)
            line(P, ox - c * 3.5 * sc, oy - s * 3.5 * sc, ox + c * 1.4 * sc, oy + s * 1.4 * sc, col_b, wid=(0.8, 1.2), a=a)
            disc(P, ox + c * 1.8 * sc, oy + s * 1.8 * sc, 0.9 * sc, col_b, a=a)
        bug(x + alt * 0.8, y + alt * 1.4, (0, 0, 0), (0, 0, 0), 0.3)         # shadow
        bug(x, y, body, (220, 232, 255), 1.0)
