import math
import random
import time

import numpy as np

from ..paint import (
    _c, blur, blur1, disc, fill_grad, fog_rows, glow, grain_tex, grid, hash01, line, make_stars, mix, poly,
    noise_tex, rect, ridge1d, rrect, vgrad)
from ..world import wrng, wseed
from ..theory import TAU


# night train per station: sky (top → horizon), far hills, near hills, ground, lights in the landscape,
# carriage wall (top, bottom), seat, curtain, wall lamp; city = a skyline slides by instead of hills
TRAIN = [
    dict(sky=[(10, 9, 32), (40, 30, 78), (126, 74, 122)], far=(58, 48, 98), near=(30, 26, 56), ground=(18, 15, 34),
         lights=(255, 206, 140), wall=[(70, 48, 62), (40, 26, 38)], seat=(128, 58, 78), curtain=(178, 98, 126), lamp=(255, 200, 150)),
    dict(sky=[(3, 4, 16), (12, 16, 44), (38, 42, 86)], far=(32, 36, 72), near=(16, 20, 42), ground=(10, 12, 26),
         lights=(255, 220, 160), wall=[(34, 44, 58), (18, 24, 34)], seat=(54, 72, 116), curtain=(84, 100, 146), lamp=(255, 214, 160)),
    dict(sky=[(72, 64, 132), (226, 124, 120), (255, 196, 124)], far=(176, 104, 122), near=(112, 70, 80), ground=(72, 46, 48),
         lights=(255, 232, 170), wall=[(128, 86, 60), (84, 54, 40)], seat=(180, 98, 60), curtain=(236, 174, 104), lamp=(255, 222, 170), sun=True),
    dict(sky=[(26, 32, 58), (72, 82, 116), (146, 156, 186)], far=(124, 134, 168), near=(96, 106, 136), ground=(206, 212, 230),
         lights=(255, 214, 150), wall=[(74, 78, 100), (46, 48, 66)], seat=(92, 112, 154), curtain=(154, 174, 214), lamp=(255, 216, 170)),
    dict(sky=[(10, 2, 30), (52, 10, 84), (168, 32, 124)], far=(74, 22, 104), near=(36, 10, 62), ground=(16, 6, 30),
         lights=(80, 255, 230), wall=[(44, 22, 64), (24, 12, 40)], seat=(206, 44, 154), curtain=(124, 62, 206), lamp=(255, 96, 206), city=True),
    dict(sky=[(2, 14, 24), (10, 46, 62), (52, 114, 112)], far=(30, 82, 86), near=(16, 48, 52), ground=(8, 26, 30),
         lights=(255, 228, 140), wall=[(32, 54, 48), (18, 34, 30)], seat=(62, 122, 98), curtain=(112, 172, 142), lamp=(255, 228, 160)),
    dict(sky=[(64, 136, 228), (128, 188, 244), (206, 232, 255)], far=(132, 170, 200), near=(92, 152, 90), ground=(72, 124, 60),
         lights=(255, 240, 200), wall=[(156, 122, 90), (112, 86, 64)], seat=(60, 130, 172), curtain=(252, 222, 122), lamp=(255, 240, 210)),
    dict(sky=[(18, 12, 44), (90, 46, 110), (230, 130, 150)], far=(110, 66, 120), near=(46, 30, 62), ground=(24, 16, 34),        # tokyo
         lights=(255, 190, 150), wall=[(84, 50, 50), (52, 30, 32)], seat=(50, 70, 130), curtain=(200, 70, 80), lamp=(255, 200, 170)),
    dict(sky=[(6, 4, 12), (30, 18, 30), (90, 50, 50)], far=(50, 30, 40), near=(24, 14, 22), ground=(12, 8, 12),                 # jazz
         lights=(255, 196, 110), wall=[(70, 30, 30), (40, 16, 18)], seat=(130, 40, 46), curtain=(160, 110, 60), lamp=(255, 196, 120), city=True),
    dict(sky=[(10, 12, 20), (40, 46, 60), (110, 110, 116)], far=(60, 70, 72), near=(30, 40, 34), ground=(16, 20, 18),                 # english
         lights=(255, 210, 130), wall=[(50, 30, 30), (30, 18, 18)], seat=(40, 70, 56), curtain=(120, 40, 46), lamp=(255, 206, 130), city=False)
]


# broadleaf trees by the line (lit, shade) on the stations that have them; the rest pass pines
TRAIN_TREES = {2: ((206, 112, 50), (112, 52, 36)), 6: ((92, 160, 72), (44, 100, 52)), 7: ((206, 132, 172), (104, 58, 98)),
               9: ((54, 70, 46), (20, 30, 22))}


class TrainScene:
    # ── night train ────────────────────────────────────────────────────────
    def v_train(self, cv, P, h, w):
        """A night train: mountains, villages, lakes and telegraph poles slide past the carriage
        windows at their own speeds, and now and then the train dives into a tunnel. The carriage
        rocks on the rail joints in time with the beat and runs faster when the music is louder."""
        e = self.eng
        t = time.time()
        H = P.shape[0]
        V = self.var(TRAIN)
        level = float(self.rms.mean())
        snow = e.st.get("weather") == "snow"
        TW = 4096
        s = getattr(self, "train", None)
        if s is None or s["size"] != (h, w):
            rng = np.random.default_rng(wseed("train", h, w))
            houses = []
            for c in rng.integers(0, TW, 14):                 # villages: little clusters of houses
                for k in range(int(rng.integers(3, 8))):
                    houses.append((int(c + k * rng.integers(6, 10)) % TW, int(rng.integers(0, 4)), bool(rng.random() < 0.15)))
            lakes = [(int(a), int(rng.integers(60, 160))) for a in rng.integers(0, TW, 5)]
            trees, x = [], 0
            while x < TW - 30:                                # lineside trees: copses with open stretches between
                x += int(rng.integers(9, 26)) if rng.random() < 0.6 else int(rng.integers(70, 220))
                trees.append((x, float(rng.random()), float(rng.random())))
            mtn = ridge1d(TW, 200, 9) * 0.4 + ridge1d(TW, 48, 12) * 0.42 + ridge1d(TW, 12, 13) * 0.18   # peaks, not a plateau
            s = self.train = dict(size=(h, w), mtn=mtn, mslope=np.gradient(mtn), trees=trees,
                                  cloud=noise_tex(H, 1024, 7, 77), far=ridge1d(TW, 110, 1), mid=ridge1d(TW, 36, 2),
                                  bush=ridge1d(TW, 5, 3), blocks=rng.random(TW // 8 + 2), houses=houses, lakes=lakes,
                                  off=0.0, v=60.0, last=t, specks=[], SP=int(rng.integers(90, 140)),
                                  TP=int(rng.integers(3800, 6400)), mx=float(rng.uniform(0.55, 0.85)),
                                  stars=make_stars(wrng("trainstars", h, w), int(H * 0.42), w, w * h // 40))
        dt = min(0.1, max(0.0, t - s["last"]))
        s["last"] = t
        target = 0.0 if e.paused else 55 * (0.8 + min(2.5, level * 9))
        s["v"] += (target - s["v"]) * min(1.0, dt * 0.7)       # the train eases to a halt when paused
        s["off"] += s["v"] * dt
        off = s["off"]
        # ── layout: one, two or three windows along the carriage ──
        wy0, wy1 = max(5, int(H * 0.14)), int(H * 0.62)
        m = max(4, w // 14)
        npn = 1 if w < 96 else 2 if w < 170 else 3
        pil = 18
        pw_ = (w - 2 * m - (npn - 1) * pil) / npn
        panes = [(int(m + i * (pw_ + pil)), int(m + i * (pw_ + pil) + pw_)) for i in range(npn)]
        bp = e.beat_pos
        moving = min(1.0, s["v"] / 40)
        bump = sum(math.exp(-((bp - c) % 4) * 10) for c in (0, 0.28, 2, 2.28)) * moving
        oy = int(round(bump * 1.2 + 0.7 * self.kick * moving + math.sin(t * 1.9) * 0.5 * moving))
        hy = int(wy0 + (wy1 - wy0) * 0.6) + oy
        yy, xx = grid(H, w)
        cols = np.arange(w)
        # ── outside: sky ──
        fill_grad(P, 0, hy, list(zip((0, 0.6, 1), V["sky"])))
        if e.wet:
            P[:hy] = P[:hy] * 0.6 + _c(V["sky"][1]) * 0.25
        self.chord_wash(P, wy0, hy, 0.045)                       # the night outside takes each chord's colour
        self.stars(P, s["stars"], t, 0.25 if e.wet else 1.0)
        if not V.get("sun"):                                     # a star winks for every note of the melody
            self.note_sparks(P, "train", lambda p, n: (w * (0.06 + 0.88 * ((n * 0.618) % 1)), hy - 4 - p * (hy - wy0 - 6)),
                             (236, 240, 255), rise=0.0, drift=-3.0, life=1.1, r=0.7, k=0.4 if e.wet else 1.0)
        mxs, mys = w * s["mx"], wy0 + (hy - wy0) * 0.28
        if V.get("sun"):
            mxs, mys = w * 0.32, hy - 3
            glow(P, mxs, mys + 1, H * 0.35, (255, 150, 90), 0.5)
            disc(P, mxs, mys, max(3.0, H * 0.07), (255, 222, 152))
        else:
            self.sky_body(P, mxs, mys, max(2.5, H * 0.05), dim=0.6 if e.wet else 1.0)
        # clouds: long streaks that barely drift, heavier in the rain
        cl = s["cloud"][:hy][:, ((cols * 0.4 + off * 0.004).astype(int)) % 1024]
        span = max(1.0, hy - wy0)
        env = np.exp(-((yy[:hy] - (wy0 + span * 0.25)) / (span * 0.55)) ** 2)
        ca = np.clip((cl - (0.2 if e.wet else 0.9)) * 0.3, 0, 0.35) * env            # thin, soft-edged streaks
        under = np.clip(cl - np.roll(cl, -2, axis=0), -1, 1) * 0.5 + 0.5          # lit tops, shaded bellies
        ccol = _c(mix(V["sky"][1], V["sky"][2], 0.6))[None, None, :] * (0.6 + 0.35 * under)[..., None]
        P[:hy] += (ccol - P[:hy]) * ca[..., None]
        # mountains (barely move): faces lit from the moon's side, shaded behind, snowcaps, haze at the foot
        xm0 = (cols + off * 0.012).astype(int) % TW
        mt = s["mtn"][xm0]
        topm0 = hy - (hy - wy0) * (0.12 + 0.8 * mt)
        mm0 = (yy >= topm0[None, :]) & (yy < hy)
        below = yy - topm0[None, :]
        sgn = 1.0 if V.get("sun") else -1.0
        face = (sgn * s["mslope"][xm0][None, :] * 300 + 0.3 * np.sin(below * 0.55 + xm0[None, :] * 0.21)) > 0
        mlit, msh = _c(mix(V["far"], V["sky"][2], 0.22)) * 1.08, _c(mix(V["far"], V["sky"][0], 0.5)) * 0.85
        colm = np.where(face[..., None], mlit, msh)
        haze = 0.3 + 0.4 * np.clip(below / np.maximum(1.0, hy - topm0)[None, :], 0, 1)
        colm = colm + (P - colm) * haze[..., None]                    # aerial perspective: fade into the sky behind
        P[mm0] = colm[mm0]
        capd = np.clip((mt - (0.35 if snow else 0.52)) * (hy - wy0) * 0.6, 0, None) * (0.75 + 0.25 * np.sin(xm0 * 0.9))
        cap = mm0 & (below < capd[None, :])
        P[cap & face] = mix((244, 246, 252), V["sky"][2], 0.2)
        P[cap & ~face] = mix((176, 186, 216), V["sky"][2], 0.3)
        if self.theme() == "japan":                          # Fuji, so far off it hardly moves
            fx_ = (w * 0.62 + 80 - off * 0.012) % (w + 160) - 80
            fh_ = (hy - wy0) * 0.85
            body = mix(V["far"], (14, 8, 26), 0.25)
            poly(P, [(fx_ - fh_ * 2.2, hy), (fx_ - fh_ * 0.28, hy - fh_), (fx_ + fh_ * 0.28, hy - fh_), (fx_ + fh_ * 2.2, hy)], body)
            poly(P, [(fx_ - fh_ * 0.28, hy - fh_), (fx_ + fh_ * 0.28, hy - fh_), (fx_ + fh_ * 0.62, hy - fh_ * 0.72),
                     (fx_ + fh_ * 0.2, hy - fh_ * 0.8), (fx_ - fh_ * 0.1, hy - fh_ * 0.66), (fx_ - fh_ * 0.62, hy - fh_ * 0.74)],
                 (236, 228, 246))
        fog_rows(P, wy0, hy, V["sky"][2], 0.25)
        # far hills
        xf = (cols + off * 0.05).astype(int) % TW
        topf = hy - (2 + (hy - wy0) * 0.22 * s["far"][xf])
        wood = hash01(xf, xf * 0, 11)                        # a ragged treeline along the crest
        topf = topf - (wood < 0.45) - (wood < 0.15)
        mf = (yy >= topf[None, :]) & (yy < hy)
        kf = np.clip((yy - topf[None, :]) / 6.0, 0, 1)[..., None]
        hill0, hill1 = _c(mix(V["far"], V["near"], 0.35)), _c(mix(V["far"], V["near"], 0.7))    # nearer and darker than the peaks
        P[mf] = (hill0 * (1 - kf) + hill1 * kf)[mf]
        if self.theme() == "english":                        # castles on the far hills, drifting slowly by
            period = max(w * 1.5, 160.0)                     # spaced by the window, so one is usually in view
            for k in range(2):
                cx_ = (k * period + w * 0.3 - off * 0.05) % (period * 2) - 20
                if -20 < cx_ < w + 20:
                    xi = int(min(w - 1, max(0, cx_)))
                    cs = max(8.0, (hy - wy0) * 0.42)
                    cb = float(topf[xi]) - cs * 0.25                     # it stands on a mound of its own hill
                    poly(P, [(cx_ - cs * 1.9, hy), (cx_ - cs * 0.85, cb), (cx_ + cs * 0.85, cb), (cx_ + cs * 1.9, hy)], hill0)
                    self.castle(P, cx_, cb + 1, cs, mix(V["far"], V["near"], 0.8),
                                lit=V["lights"], t=t, seed=k)
        # mid layer: hills with villages, or a neon skyline
        xm = (cols + off * 0.2).astype(int) % TW
        if V.get("city"):
            topm = hy - 2 - (hy - wy0) * (0.12 + 0.42 * s["blocks"][xm // 8])
            mmm = (yy >= topm[None, :]) & (yy <= hy)
            P[mmm] = V["near"]
            wm = mmm & (xm[None, :] % 3 == 1) & (yy % 3 == 1) & (yy > topm[None, :] + 1) & (hash01(xm[None, :] + 0 * yy, yy, 5) < 0.4)
            P[wm] = _c(V["lights"]) * 0.85
            P += blur(wm.astype(np.float32), 1)[..., None] * _c(V["lights"]) * 0.35
        else:
            jp = self.theme() == "japan"
            topm = hy - 1 - (hy - wy0) * 0.13 * s["mid"][xm]
            P[(yy >= topm[None, :]) & (yy <= hy)] = V["near"]
            hc = _c(V["near"]) * 0.55
            for hx, var, steeple in s["houses"]:
                px = int((hx - off * 0.2) % TW)
                if px < w - 5:
                    base = int(topm[px]) + 1
                    hgt = 3 + var % 2
                    rect(P, px, base - hgt, px + 4 + var % 2, base, hc)
                    poly(P, [(px - 1, base - hgt), (px + 2, base - hgt - 2), (px + 5 + var % 2, base - hgt)], hc)
                    if steeple and jp:                       # a pagoda instead of a steeple
                        self.pagoda(P, px + 2, base, 12, hc, hc * 0.8, tiers=3)
                    elif steeple:
                        rect(P, px + 1, base - hgt - 6, px + 3, base - hgt, hc)
                        poly(P, [(px + 0.5, base - hgt - 6), (px + 2, base - hgt - 10), (px + 3.5, base - hgt - 6)], hc)
                    if var != 1:
                        wc = mix(V["lights"], (120, 140, 170), 0.6) if self.noon() else V["lights"]
                        P[base - 2, px + 1] = wc
                        if var == 3:
                            P[base - 2, px + 3] = wc
                        if not self.noon():
                            glow(P, px + 1.5, base - 2, 2.4, V["lights"], 0.42 + 0.25 * self.hat)
        P[hy + 1:H] = vgrad(H - hy - 1, [(0, V["near"]), (1, V["ground"])])[:, None, :]
        # a patchwork of fields, sliding faster the nearer they are; some are ploughed
        y1f = min(H, wy1 + 3)
        if y1f > hy + 1:
            Df = ((yy[hy + 1:y1f] - hy) / max(1.0, wy1 - hy)).astype(np.float32)
            XR = xx[hy + 1:y1f] + off * (0.2 + 0.8 * Df)
            tone = hash01(np.floor(XR / (14 + 60 * Df)), np.floor(Df * 4), 17)
            furrow = (np.floor(XR / (2 + 5 * Df)) % 2) * (tone > 0.55)
            P[hy + 1:y1f] *= ((0.95 + 0.5 * tone) * (1 - 0.1 * furrow))[..., None]
        if self.theme() == "japan":                          # shrine gates by the line, flicking past
            for k in range(12):
                gx = (k * TW / 12 + 60 - off * 0.35) % TW
                if gx < w + 12:
                    self.torii(P, gx - 6, hy + 3, min(16.0, (wy1 - hy) * 1.1))
        # lakes that mirror the sky and the moon: flat ovals lying in the fields, with ragged shores
        lake_h = max(3, int((wy1 - hy) * 0.4))
        xl = (cols + off * 0.35) % TW
        top_k, bot_k = np.full(w, 1e9, np.float32), np.full(w, -1e9, np.float32)
        for a, ln in s["lakes"]:
            u = ((xl - a) % TW) / ln
            inl = u < 1
            if not inl.any():
                continue
            half = np.sqrt(np.clip(1 - (2 * u - 1) ** 2, 0, 1)) ** 0.8         # 0 at the ends → 1 in the middle
            wob = 0.5 * np.sin(xl * 0.31 + a) + 0.35 * np.sin(xl * 0.83 + a * 2)
            mid = 1 + lake_h * 0.45
            t_ = mid - half * lake_h * 0.3 + wob * 0.4 * half                  # the far shore: foreshortened
            b_ = mid + half * lake_h * 0.55 + wob * half                       # the near shore
            ok = inl & (b_ - t_ > 0.6)
            top_k[ok], bot_k[ok] = np.minimum(top_k[ok], t_[ok]), np.maximum(bot_k[ok], b_[ok])
        if e.waves_on:                                       # the line runs along the coast: open sea to the horizon
            top_k[:], bot_k[:] = 1.0, 1.0 + max(2, int((wy1 - hy) * 0.6))
        span = int(np.ceil(bot_k.max())) + 1 if (bot_k > 0).any() else 0
        if span > 0:
            kk = np.arange(span, dtype=np.float32)[:, None]
            water = (kk >= np.floor(top_k)[None, :]) & (kk < np.ceil(bot_k)[None, :])
            ys = slice(hy, min(H, hy + span))
            water = water[:ys.stop - ys.start]
            refl = vgrad(span, [(0, mix(V["sky"][2], (10, 14, 40), 0.35)), (1, mix(V["sky"][1], (6, 8, 26), 0.5))])
            refl = refl[:ys.stop - ys.start]
            P[ys][water] = np.broadcast_to(refl[:, None, :], water.shape + (3,))[water]
            shore = water & ~np.roll(water, -1, axis=0)                        # a dark lip along the near bank
            P[ys][shore] = mix(V["near"], (0, 0, 0), 0.3)
            glint = (240, 232, 204) if not V.get("sun") else (255, 200, 130)
            for k in range(1, ys.stop - ys.start):
                n = 1 + k // 2
                xs = (mxs + np.random.randn(n) * (1 + k * 0.8)).astype(int)
                xs = xs[(xs >= 0) & (xs < w)]
                xs = xs[water[k, xs] & ~shore[k, xs]]
                P[hy + k, xs] = glint
        # trees by the line: copses that swing past between the fields and the hedge
        if not V.get("city"):
            leaf = TRAIN_TREES.get(e.station_idx)
            base = int(wy1 - (wy1 - hy) * 0.2)
            plit, psh = mix(V["near"], V["sky"][2], 0.3), tuple(_c(V["near"]) * 0.55)
            for tx_, sz, u_ in s["trees"]:
                px = (tx_ - off * 0.55) % TW
                if px > TW - 16:
                    px -= TW
                if px < w + 16:
                    hg = int((wy1 - hy) * (1.5 + 1.2 * sz))
                    if leaf and u_ < 0.6:
                        self.broadleaf(P, int(px), base, int(hg * 0.85), leaf[0], leaf[1], snow=0.6 if snow else 0.0,
                                       trunk=(30, 24, 30), seed=int(tx_))
                    else:
                        self.pine(P, int(px), base, hg, plit, psh, snow=0.7 if snow else 0.0, trunk=(30, 24, 30),
                                  seed=int(tx_), fat=0.8 + 0.4 * u_)
        # hedges whipping by (motion-blurred)
        xb = (cols + off).astype(int) % TW
        topb = wy1 + 2 - (wy1 - hy) * (0.2 + 0.45 * s["bush"][xb])
        mb = blur1((yy >= topb[None, :]).astype(np.float32), 2, 1)
        P += (_c(V["near"]) * 0.4 - P) * mb[..., None]
        # telegraph poles and sagging wires
        SP = s["SP"]
        ph = off % SP
        light = 0.0
        for px in np.arange(-ph, w + SP, SP):
            line(P, px, wy0 - 6, px, H, (14, 12, 22), wid=2, a=0.85)
            line(P, px - 3, wy0 + 2, px + 3, wy0 + 2, (14, 12, 22), wid=1, a=0.85)
            if int(round((px + off) / SP)) % 3 == 0 and not self.noon():
                glow(P, px, wy0 + 4, 5, V["lights"], 0.6 + 0.3 * self.snare)
                light += math.exp(-((px - w / 2) / (w * 0.35)) ** 2)
        for k in range(2):
            u = ((cols + ph) % SP) / SP
            yw = (wy0 + 2 + k * 3 + oy + 7 * 4 * u * (1 - u)).astype(int)
            ok = (yw >= 0) & (yw < H)
            P[yw[ok], cols[ok]] = P[yw[ok], cols[ok]] * 0.6 + _c((14, 12, 22)) * 0.4
        ev = self.rare(7.0)
        if ev:
            light += self._rare_train(P, ev, wy0, wy1, hy, w, V)
        # tunnels: the view goes dark, tunnel lamps streak past, the carriage reflects in the glass
        TP, TL = s["TP"], 760
        tw_ = (cols + off) % TP
        inside = tw_ > TP - TL
        dark = 0.0
        if inside.any():
            P[:, inside] = vgrad(H, [(0, (18, 16, 22)), (1, (8, 7, 10))])[:, None, :]
            for lx in np.arange(-(off % 46), w + 46, 46):
                li = int(lx)
                if 0 <= li < w and inside[li]:
                    line(P, lx - 6, wy0 + 5, lx + 1, wy0 + 5, (255, 170, 90), wid=1.2, a=0.9)
                    glow(P, lx - 2, wy0 + 5, 4, (255, 150, 70), 0.4 + 0.35 * self.kick)
            edges = np.nonzero(np.diff(inside.astype(int)))[0]
            for ex in edges:                                 # stone portals
                rect(P, ex - 2, 0, ex + 2, H, (70, 64, 66))
                rect(P, ex - 2, 0, ex - 1, H, (96, 90, 92))
            mid = inside[panes[0][0]:panes[-1][1]]
            dark = float(mid.mean()) if len(mid) else 0.0
            light += 0.6 * dark * (0.5 + 0.5 * math.sin(off / 46 * TAU))
        # rain streaks / snow flecks on the glass, swept back by the speed
        inten = (1.0 if e.rain_on else 0.0) + (1.3 if e.storm_on else 0.0) + (0.6 if snow else 0.0)
        sp = s["specks"]
        nn = inten * dt * w * 1.2 * (1 - dark)
        for _ in range(int(nn) + (random.random() < nn % 1)):
            x0, x1 = random.choice(panes)
            sp.append([random.uniform(x0, x1 + 40), random.uniform(wy0 - 6, wy1), random.uniform(0.6, 1.3)])
        vx, vy = -(s["v"] * 0.9 + 20), (5 if snow else 16)
        for q in sp:
            q[0] += vx * dt * q[2]
            q[1] += vy * dt * q[2]
        sp[:] = [q for q in sp if q[0] > -6 and q[1] < wy1 + 2][-500:]
        if sp:
            arr = np.array(sp)
            col = _c((240, 244, 255) if snow else (176, 196, 236))
            for i in range(1 if snow else 4):
                xi = (arr[:, 0] + i).astype(int)
                yi = (arr[:, 1] - i * 0.35).astype(int)
                ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < H)
                P[yi[ok], xi[ok]] += (col - P[yi[ok], xi[ok]]) * (0.7 - i * 0.12)
        self.sky = (panes[0][0], panes[-1][1], wy0 // 2 + 1, max(wy0 // 2 + 2, hy // 2))

        # ── the carriage around the windows ──
        wall0, wall1 = _c(V["wall"][0]), _c(V["wall"][1])

        def build():
            Q = np.zeros((H, w, 3), np.float32)
            fill_grad(Q, 0, H, [(0, V["wall"][0]), (1, V["wall"][1])])
            g = grain_tex(H, w, 3)                           # wood panelling
            Q *= (0.95 + 0.05 * g)[..., None]
            Q[:, (np.arange(w) % 7) == 0] *= 0.85
            win = np.zeros((H, w), bool)
            fr = np.zeros((H, w), bool)
            for x0, x1 in panes:
                win |= rrect(H, w, x0, wy0, x1 - x0, wy1 - wy0, 3)
                fr |= rrect(H, w, x0 - 2, wy0 - 2, x1 - x0 + 4, wy1 - wy0 + 4, 4)
            fr &= ~win
            Q[fr] = np.clip(wall0 * 1.5 + 22, 0, 255)
            Q[fr & (yy < wy0)] += 18
            fill_grad(Q, wy1 + 4, H, [(0, wall1 * 0.9), (1, wall1 * 0.65)])
            for x0 in range(2, w - 8, 18):                   # wainscot panels below the windows
                if wy1 + 7 < H - 4:
                    msk = np.zeros((H, w), bool)
                    msk[wy1 + 7:H - 4, x0:x0 + 14] = True
                    inner = np.zeros((H, w), bool)
                    inner[wy1 + 8:H - 5, x0 + 1:x0 + 13] = True
                    Q[msk & ~inner] = wall1 * 0.7
                    Q[msk & ~inner & (yy == wy1 + 7)] = wall0 * 1.2
            rect(Q, m - 5, wy1 + 2, w - m + 5, wy1 + 4, np.clip(wall0 * 1.7 + 16, 0, 255))     # sill
            if H > 26:                                       # patterned carpet at the bottom
                rows = slice(H - 3, H)
                Q[rows] = _c(V["seat"]) * 0.45
                Q[rows][((np.arange(w)[None, :] + np.arange(3)[:, None] * 2) % 6) == 0] = _c(V["seat"]) * 0.75
            ry = max(1, wy0 - 7)                             # luggage rack with bags on it
            rect(Q, m - 3, ry, w - m + 3, ry + 1, (168, 156, 130))
            rect(Q, m - 3, ry + 2, w - m + 3, ry + 3, (120, 110, 96))
            if ry > 4:
                rng = wrng("luggage", w)
                x = m + 2
                while x < w - m - 12:
                    kind = rng.random()
                    if kind < 0.35:
                        rect(Q, x, ry - 4, x + 11, ry, rng.choice([(126, 84, 60), (90, 110, 140), (150, 70, 70)]))
                        rect(Q, x + 4, ry - 5, x + 7, ry - 4, (80, 60, 50))
                        rect(Q, x, ry - 2, x + 11, ry - 1, (60, 46, 40))
                        x += 14
                    elif kind < 0.6:
                        disc(Q, x + 5, ry - 2, 5, rng.choice([(70, 96, 80), (120, 90, 130)]), ry=2.2)
                        x += 12
                    elif kind < 0.75:
                        rect(Q, x, ry - 3, x + 6, ry, (200, 170, 120))
                        rect(Q, x, ry - 3, x + 6, ry - 2, (150, 60, 60))
                        x += 9
                    else:
                        x += rng.randint(6, 16)
            return Q, win
        Q, win = self.tex("train", (h, w, e.station_idx), build)
        I = Q.copy()
        lamps = [(a[1] + b[0]) / 2 for a, b in zip(panes, panes[1:])] or [m - 3, w - m + 2]
        lamp_k = 0.6 if self.noon() else 1.0
        for lx in lamps:                                     # brass sconces with fabric shades
            glow(I, lx + 0.5, wy0 + 7, 7, V["lamp"], 0.4 * lamp_k * (0.85 + 0.15 * self.bass + 0.15 * self.kick), ry=6)
            rect(I, lx - 0.5, wy0 + 6, lx + 1.5, wy0 + 10, (190, 160, 90))
            poly(I, [(lx - 2.5, wy0 + 6), (lx - 1, wy0 + 2), (lx + 2, wy0 + 2), (lx + 3.5, wy0 + 6)], mix(V["lamp"], (255, 255, 255), 0.25))
        if light:
            glow(I, w / 2, wy1 + 2, w * 0.5, V["lights"], 0.16 * light, ry=H * 0.3)
        P[win] = P[win] * 0.9 + _c(V["lamp"]) * 0.025 + I[win] * 0.3 * dark     # in a tunnel, the glass turns mirror
        P[~win] = I[~win]
        # curtains tied back at each window, with tassels
        for x0, x1 in panes:
            for side in (-1, 1):
                ex = x0 if side < 0 else x1
                tie = wy0 + (wy1 - wy0) * 0.55
                pts = [(ex - 4 * side, wy0 - 3), (ex + 5 * side, wy0 - 3), (ex + 2 * side, tie),
                       (ex + 4 * side, wy1 + 1), (ex - 4 * side, wy1 + 1)]
                poly(P, pts, V["curtain"])
                line(P, ex + 1 * side, wy0 - 2, ex + 1 * side, wy1, _c(V["curtain"]) * 0.75, wid=1)
                line(P, ex - 2 * side, wy0 - 2, ex - 2 * side, wy1, _c(V["curtain"]) * 0.85, wid=1)
                rect(P, ex + side - 1, tie, ex + 2 * side + 2, tie + 1, (230, 200, 140))
                rect(P, ex + 2 * side, tie + 1, ex + 2 * side + 1, tie + 3 + math.sin(t * 2 + side) * 0.5, (230, 200, 140))
        # high-backed seats with lace covers on the headrests
        if w >= 60:
            seat = _c(V["seat"])
            for side in (-1, 1):
                sw_ = int(w * 0.17)
                sx = -6 if side < 0 else w - sw_
                st0 = int(H * 0.68)
                msk = rrect(H, w, sx, st0, sw_ + 6, H, 4)
                pat = 0.88 + 0.12 * (((xx[msk] + yy[msk]) % 4 == 0) | ((xx[msk] - yy[msk]) % 4 == 0))
                P[msk] = seat[None, :] * pat[:, None] * (0.8 + 0.3 * np.clip((yy[msk] - st0) / 6, 0, 1))[:, None]
                cx0, cx1 = (sx + 4, sx + sw_ + 1) if side < 0 else (sx + 3, sx + sw_ - 1)
                rect(P, cx0, st0, cx1, st0 + 3, (236, 232, 222), a=0.92)
                lace = (np.arange(int(cx0), int(cx1)) % 2 == 0)
                P[min(H - 1, st0 + 3), np.arange(int(cx0), int(cx1))[lace]] = (236, 232, 222)
        # the table: banker's lamp, tea that trembles on the rail joints, a book and a ticket
        tx = w // 2
        ty = wy1 + 7
        if ty + 3 < H:
            rect(P, tx - 16, ty, tx + 16, ty + 2, (118, 82, 58))
            rect(P, tx - 16, ty + 2, tx + 16, ty + 3, (70, 48, 36))
            rect(P, tx - 1, ty + 3, tx + 1, H, (44, 34, 30))
            lkx = tx - 12                                    # banker's lamp
            glow(P, lkx + 2, ty - 1, 12, (255, 220, 150), (0.25 if not self.noon() else 0.1) * (0.9 + 0.2 * self.bass), ry=4)
            rect(P, lkx, ty - 1, lkx + 5, ty, (190, 160, 90))
            rect(P, lkx + 2, ty - 4, lkx + 3, ty - 1, (190, 160, 90))
            poly(P, [(lkx - 1, ty - 4), (lkx + 1, ty - 6), (lkx + 4, ty - 6), (lkx + 6, ty - 4)], (40, 120, 80))
            rect(P, lkx, ty - 4, lkx + 6, ty - 3.5, (255, 230, 160))
            rect(P, tx - 5, ty - 2, tx + 2, ty, (96, 70, 140))                 # paperback
            rect(P, tx - 4, ty - 2, tx + 1, ty - 1, (236, 226, 200))
            rect(P, tx + 10, ty - 1, tx + 15, ty, (240, 228, 196))            # ticket
            rect(P, tx + 10, ty - 1, tx + 11, ty, (200, 70, 70))
            rect(P, tx + 4, ty - 5, tx + 9, ty, (236, 230, 220))              # mug
            P[max(0, ty - 4):ty - 2, min(w - 1, tx + 9)] = (236, 230, 220)
            if self.theme() == "jazz":                       # a coupe glass on the table, the radio leaking tunes
                gx = tx - 6
                poly(P, [(gx - 2.5, ty - 6), (gx + 2.5, ty - 6), (gx, ty - 3.5)], (230, 236, 246), a=0.7)
                poly(P, [(gx - 1.8, ty - 5.7), (gx + 1.8, ty - 5.7), (gx, ty - 4)], (230, 150, 60))
                rect(P, gx - 0.3, ty - 3.5, gx + 0.7, ty - 0.5, (220, 226, 236))
                rect(P, gx - 1.5, ty - 1, gx + 2, ty, (220, 226, 236))
                self.notes(cv, tx + 13, (ty - 6) // 2, V["lights"], n=3, rise=4, seed=9)
            tea = mix((130, 80, 46), (200, 150, 100), min(1.0, bump + self.kick))     # the tea shivers on every kick
            rect(P, tx + 5, ty - 5, tx + 8, ty - 4, tea)
            for i in range(3):
                cy = (ty - 7) // 2 - i
                if cy >= 0:
                    cv.put(cy, int(tx + 6 + math.sin(t * 1.4 + i * 1.9) * 1.2), "~" if (i + int(t * 2)) % 2 else "˜",
                           mix((200, 196, 210), V["wall"][0], i / 3))

    def _rare_train(self, P, ev, wy0, wy1, hy, w, V):
        """Rare event: a train on the other line rushes past the other way: a headlight swelling out of
        the dark, then a long blur of lit windows filling the glass for a few seconds. Returns how much
        its light spills into the carriage."""
        u, r = ev
        H = P.shape[0]
        day = self.noon()
        top = int(hy - (wy1 - wy0) * 0.28)
        L = w * r.uniform(2.5, 4.0)
        if u < 0.2:                                                     # the headlight, swelling
            k = u / 0.2
            if not day:
                glow(P, w * 0.92, hy - 2, 2 + 10 * k * k, (255, 246, 220), 0.25 + 0.6 * k * k)
            return 0.2 * k * k
        k = (u - 0.2) / 0.65
        head = w + 4 - k * (w + L + 8)                                   # its front edge, moving left
        a, b = int(max(0, head)), int(min(w, head + L))
        if a >= b:
            return 0.0
        body = _c(r.choice([(196, 200, 210), (60, 70, 96), (150, 40, 50), (40, 90, 80)])) * (0.9 if day else 0.35)
        P[top:H, a:b] = body
        P[top, a:b] = body * 1.4 + 20
        bw0, bw1 = top + max(2, (H - top) // 7), top + max(4, (H - top) // 3)
        xs = np.arange(a, b)
        rel = xs - head
        lit = ((rel % 26) > 4) & ((rel % 104) > 8)                      # windows smeared into a band; gaps between cars
        winc = _c((150, 170, 200)) if day else _c(V["lights"])
        flick = 0.75 + 0.25 * np.random.random(len(xs))
        P[bw0:bw1, a:b] += (winc * flick[:, None] - P[bw0:bw1, a:b]) * (lit * 0.9)[None, :, None]
        P[bw1 + 1:bw1 + 2, a:b] = body * 0.6
        gap = (rel % 104) <= 2
        P[top:H, a:b][:, gap] *= 0.3
        if head > 0 and not day:
            glow(P, head, hy - 2, 8, (255, 246, 220), 0.6)
        return 0.0 if day else 0.9 * (b - a) / w
