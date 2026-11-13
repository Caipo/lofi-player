import math
import random
import time

import numpy as np

from ..paint import (
    _box, _c, blur1, disc, draw_stars, fill_grad, glow, grid, line, make_stars, mix, noise_tex, poly, rect, vgrad)
from ..world import wrng, wseed
from ..term import C_WHITE
from ..theory import TAU
from .shared import BOKEH


OCEAN = [   # night, midnight, sunset, winter, neon, teal moonlight
    dict(sky=[(8, 9, 30), (34, 32, 80), (98, 80, 140)], sea=[(58, 62, 118), (30, 40, 92), (6, 12, 34)], spark=(255, 228, 170), moon=(252, 240, 214), stars=1.0, sun=False),
    dict(sky=[(2, 3, 14), (10, 14, 44), (30, 36, 80)], sea=[(30, 36, 80), (12, 20, 56), (2, 6, 20)], spark=(220, 230, 255), moon=(230, 236, 255), stars=1.2, sun=False),
    dict(sky=[(60, 50, 120), (200, 110, 130), (255, 170, 100)], sea=[(220, 140, 110), (110, 70, 110), (30, 24, 60)], spark=(255, 200, 120), moon=(255, 214, 150), stars=0.0, sun=True),
    dict(sky=[(40, 46, 70), (90, 100, 130), (150, 160, 186)], sea=[(110, 120, 150), (50, 60, 90), (16, 20, 36)], spark=(230, 236, 255), moon=(250, 250, 255), stars=0.3, sun=False),
    dict(sky=[(10, 2, 30), (60, 10, 90), (200, 40, 140)], sea=[(150, 30, 130), (40, 20, 90), (6, 4, 30)], spark=(90, 255, 230), moon=(200, 255, 250), stars=0.6, sun=False),
    dict(sky=[(2, 14, 22), (10, 44, 60), (40, 110, 110)], sea=[(40, 110, 110), (12, 50, 64), (2, 14, 24)], spark=(220, 255, 230), moon=(236, 255, 240), stars=1.0, sun=False),
    dict(sky=[(64, 136, 228), (128, 188, 244), (200, 230, 255)], sea=[(96, 186, 222), (40, 122, 190), (14, 60, 120)], spark=(255, 255, 240), moon=(255, 250, 230), stars=0.0, sun=False),
    dict(sky=[(20, 14, 50), (110, 60, 120), (246, 150, 150)], sea=[(200, 120, 140), (70, 50, 110), (14, 12, 44)], spark=(255, 210, 220), moon=(255, 236, 240), stars=0.6, sun=False),
    dict(sky=[(8, 6, 16), (36, 24, 40), (96, 60, 60)], sea=[(90, 60, 56), (30, 22, 34), (6, 4, 12)], spark=(255, 200, 120), moon=(255, 226, 170), stars=0.8, sun=False),
    dict(sky=[(10, 12, 22), (40, 46, 62), (110, 112, 120)], sea=[(70, 84, 90), (26, 36, 46), (6, 10, 16)], spark=(255, 220, 160), moon=(240, 236, 220), stars=0.4, sun=False)
]


class OceanScene:
    # ── ocean ──────────────────────────────────────────────────────────────
    def v_ocean(self, cv, P, h, w):
        e = self.eng
        t = time.time()
        H = P.shape[0]
        level = float(self.rms.mean())
        raining = e.wet
        hp = max(6, int(H * 0.42))                   # horizon (pixels)
        o = getattr(self, "ocean", None)
        if o is None or o["size"] != (h, w):
            rng = wrng("ocean", h, w)
            gulls = [[rng.uniform(0, w), rng.uniform(2, max(3, hp / 2 - 3)), rng.uniform(0.06, 0.15), rng.random() * TAU]
                     for _ in range(rng.randint(2, 5))]
            left = rng.random() < 0.6                        # lighthouse on one side, the moon over the other
            ix_ = w * (rng.uniform(0.06, 0.2) if left else rng.uniform(0.8, 0.93))
            isl = []                                         # the island: rocky slopes up to a flat top
            for sd in (-1, 1):
                wd_, n_ = rng.uniform(6, 13), rng.randint(2, 4)
                side = [(ix_ + sd * wd_, hp + 1)]
                for j in range(1, n_ + 1):
                    u = j / (n_ + 1)
                    side.append((ix_ + sd * (wd_ - u * (wd_ - 2.5)), hp + 1 - 6 * u ** rng.uniform(0.6, 1.4) + rng.uniform(-0.8, 0.8)))
                side.append((ix_ + sd * 2.5, hp - 5))
                isl += side if sd < 0 else side[::-1]
            sails = [((246, 228, 198), (220, 200, 176)), ((255, 238, 236), (230, 200, 200)),
                     ((236, 226, 180), (210, 196, 150)), ((206, 226, 246), (172, 196, 226))]
            o = self.ocean = dict(size=(h, w), stars=make_stars(rng, hp - 3, w, w * hp // 14), gulls=gulls,
                                  clouds=noise_tex(hp, w, 10, 21), drops=[], splash=[], wake=[],
                                  ix=int(ix_), island=isl, mxf=rng.uniform(0.45, 0.85) if left else rng.uniform(0.15, 0.55),
                                  tower=rng.randint(10, 15), band=rng.choice((2, 3, 3, 4)),
                                  stripe=rng.choice([(214, 70, 82), (60, 96, 180), (46, 46, 60), (232, 160, 60), (70, 150, 110)]),
                                  bx=rng.uniform(0.34, 0.5), sail=rng.choice(sails), flag=rng.choice(BOKEH),
                                  hull=rng.choice([(96, 58, 44), (150, 60, 56), (128, 84, 52), (70, 110, 90)]))
        # how rough the sea is follows the tempo: slow stations get a long glassy swell, fast ones a short
        # choppy sea with whitecaps. Phases accumulate, so changing station eases the pace instead of jumping
        rough = float(np.clip((e.st["bpm"] - 60) / 48, 0, 1))
        o.setdefault("rough", rough)
        o["rough"] += (rough - o["rough"]) * 0.03
        rough = o["rough"]
        dt = min(0.1, max(0.0, t - o.get("last", t)))
        o["last"] = t
        pace = 0.0 if e.paused else 1.0
        o["ph"] = o.get("ph", 0.0) + dt * (0.55 + 1.1 * rough) * (0.25 + 0.75 * pace)
        ph = o["ph"]
        mx = int(w * o["mxf"])
        my = hp * 0.4
        mr = max(3.0, H * 0.065)
        yy, xx = grid(H, w)
        # ── sky ──
        V = self.var(OCEAN)
        fill_grad(P, 0, hp, list(zip((0, 0.6, 1), V["sky"])))
        if raining:
            P[:hp] = P[:hp] * 0.5 + _c((26, 28, 44)) * 0.5
        self.chord_wash(P, hp // 2, hp, 0.05)                  # the horizon glow takes the chord's colour
        draw_stars(P, o["stars"], t, (0.25 if raining else 1.0) * V["stars"])
        if not V["sun"]:                                       # a star glints for every melody note, high notes higher
            self.note_sparks(P, "ocean", lambda p, n: (w * (0.08 + 0.84 * ((n * 0.618) % 1)), hp * (0.62 - 0.5 * p)),
                             (236, 240, 255), rise=0.0, life=1.2, r=0.7, k=0.4 if raining else 1.0)
        if V["sun"]:                                     # a sunset sinking into the sea
            my = hp - mr * 1.3
            glow(P, mx, my, mr * 6, (255, 140, 90), 0.45 * (0.5 if raining else 1))
            disc(P, mx, my, mr * 1.5, V["moon"])
        else:
            self.sky_body(P, mx, my, mr, dim=0.5 if raining else 1.0, tint=V["moon"])
        n = np.roll(o["clouds"], int(t * (3 if raining else 1.2)) % w, axis=1)
        a = np.clip((n - (-0.2 if raining else 0.55)) * 0.9, 0, 0.92) * (1 if raining or not V["sun"] else 0.5)
        lit_ = np.exp(-((xx[:hp] - mx) ** 2 + (yy[:hp] - my) ** 2) / (mr * 7) ** 2)
        ccol = (_c((44, 44, 70)) if raining else _c(V["sky"][1]) * 0.75 + _c(V["sky"][2]) * 0.2) \
            + lit_[..., None] * (_c(V["moon"]) * 0.6)
        P[:hp] += (ccol - P[:hp]) * a[..., None]
        th = self.theme()
        fx = w * (0.3 if o["ix"] > w / 2 else 0.7)
        if th == "japan":                                # a snowy cone on the horizon
            fh = hp * 0.5
            body = mix(V["sky"][1], (20, 14, 40), 0.55)
            poly(P, [(fx - fh * 2.6, hp), (fx - fh * 0.3, hp - fh), (fx + fh * 0.3, hp - fh), (fx + fh * 2.6, hp)], body)
            poly(P, [(fx - fh * 0.3, hp - fh), (fx + fh * 0.3, hp - fh), (fx + fh * 0.75, hp - fh * 0.7), (fx + fh * 0.35, hp - fh * 0.76),
                     (fx, hp - fh * 0.62), (fx - fh * 0.4, hp - fh * 0.74), (fx - fh * 0.8, hp - fh * 0.68)], mix(V["moon"], body, 0.25))
        elif th == "english":                            # a chalk headland running in from the edge, a castle on its brow
            left = fx < w / 2
            ch_ = max(5.0, hp * 0.26)
            length = min(w * 0.42, ch_ * 9)
            face = _c(mix(V["moon"], V["sky"][1], 0.55))
            turf = _c(mix(V["sky"][0], (46, 62, 46), 0.6))
            shade = _c(mix(V["sky"][0], (20, 20, 30), 0.5))
            tip = length if left else w - length           # where the land ends in a sheer cliff

            def u_at(x_):                                  # 0 at the screen edge → 1 at the cliff
                return (x_ / max(1.0, tip)) if left else ((w - 1 - x_) / max(1.0, w - tip))

            def ground(x_):                                # downs rolling up to the brow
                rise = 0.55 + 0.45 * math.sin(min(1.0, u_at(x_) * 1.25) * math.pi / 2)
                return hp - ch_ * rise * (0.94 + 0.06 * math.sin(x_ * 0.4))
            cs = ch_ * 1.05
            bx = tip - (1 if left else -1) * ch_ * 2.4      # the castle sits back from the cliff edge
            fw = cs * 0.95                                 # half its footprint (curtain wall and towers)
            base_ = min(ground(x_) for x_ in np.arange(bx - fw, bx + fw + 1))   # level with the highest ground under it
            for x_ in range(w):
                u = u_at(x_)
                if u > 1:
                    continue
                top_ = ground(x_)
                d = abs(x_ + 0.5 - bx)
                if d < fw * 1.9:                           # the castle mound: a level platform, banks sloping away
                    k = np.clip((fw * 1.9 - d) / (fw * 0.9), 0, 1)
                    top_ = min(top_, base_ * k + top_ * (1 - k))
                y0_ = int(top_)
                n = hp - y0_
                if n <= 0:
                    continue
                if u > 0.82:                               # the last stretch is the white cliff face, seen side-on
                    k = (u - 0.82) / 0.18
                    streak = 0.85 + 0.15 * ((x_ * 7) % 5 > 1)
                    P[y0_:hp, x_] = face * streak * (1.0 - 0.25 * k) * np.linspace(1.0, 0.65, n)[:, None]
                    P[y0_:y0_ + 1, x_] = turf
                else:
                    P[y0_:hp, x_] = turf * np.linspace(1.0, 0.55, n)[:, None] + shade * 0.15
            self.castle(P, bx, int(base_) + 1, cs, mix(V["sky"][0], (24, 24, 34), 0.5), lit=(150, 116, 74), t=t)   # far off: dim windows
            glow(P, bx, base_ - ch_ * 0.4, ch_ * 1.2, V["moon"], 0.05)
        if not raining:
            for g in o["gulls"]:
                g[0] = (g[0] + g[2]) % (w + 4)
                gy_ = g[1] + 0.4 * math.sin(t * 0.8 + g[3])
                cv.put(int(gy_), int(g[0]), "v" if int(t * 4 + g[3]) % 2 else "ˇ", C_WHITE)
        # ── the sea: streaky wave texture in perspective + broken moonlight path ──
        sea = H - hp
        P[hp:] = vgrad(sea, list(zip((0, 0.08, 1), V["sea"])))[:, None, :]
        TW = 512
        tex = self.tex("seatex", sea, lambda: (lambda n: (n - n.mean()) / (n.std() + 1e-6))(
            blur1(blur1(np.random.default_rng(wseed("sea")).random((sea, TW)).astype(np.float32), 5, 1, True), 5, 1, True)))
        surf_up = 0.8 if e.waves_on else 0.0
        amp = 0.55 + 0.45 * rough + 0.4 * min(1.5, 2.5 * level + 0.8 * self.bass) + 0.35 * surf_up
        kk = (np.arange(sea) / sea)[:, None]
        scale = (0.35 + 1.5 * kk) * (1.5 - 0.6 * rough)                 # calm water: longer, smoother streaks
        u = ((xx[hp:] / scale + ph * 2.2 * (2 + 7 * kk) * np.where(np.arange(sea) % 2, 1, -0.6)[:, None]) % TW).astype(int)
        v = tex[np.arange(sea)[:, None], u] * amp
        S = P[hp:]
        S += np.clip((v - 1.0) / 0.7, 0, 1)[..., None] * _c((48, 100, 128)) * (0.45 + 0.8 * kk[..., None])
        cap = 2.25 - 0.75 * rough - 0.3 * surf_up                       # whitecaps only once it gets choppy
        S += np.clip((v - cap) * 2.0, 0, 1)[..., None] * _c((130, 140, 150)) * (0.35 + 0.6 * surf_up + 0.5 * kk[..., None])
        S *= (1 - 0.3 * np.clip(-v - 0.8, 0, 1))[..., None]
        X = xx[hp:]
        path = 1.5 + kk * 11
        near = np.clip(1 - np.abs(X - mx) / path, 0, 1)
        S += (near * np.exp(-kk * 1.5) * 0.3)[..., None] * _c(V["spark"]) * 0.5 * (0.4 if raining else 1)
        sparkle = near * np.clip(v - 0.2, 0, 1.2)
        S += sparkle[..., None] * _c(V["spark"]) * (0.45 if raining else 1.0) * (0.8 + 0.6 * self.hat)   # the hats glitter on it
        P[hp] += (50, 40, 70)
        if th == "japan":                                # a shrine gate standing in the shallows, mirrored
            gx = max((w * f_ for f_ in (0.1, 0.32, 0.55, 0.78, 0.92)),
                     key=lambda x_: min(abs(x_ - o["ix"]), abs(x_ - w * o["bx"]) * 0.8))
            gs = max(8.0, sea * 0.36)
            gb = hp + sea * 0.2
            self.torii(P, gx, gb, gs)
            b = _box(P, gx - gs, gb, gx + gs, gb + gs * 0.6)
            if b is not None:
                vb, by_, bx_ = b
                src = np.clip((2 * gb - by_ - 1).astype(int), 0, H - 1)
                wob = np.round(np.sin(by_ * 1.3 + t * 2) * 0.8).astype(int)
                vb += (P[src, np.clip(bx_.astype(int) + wob, 0, w - 1)] - vb) * 0.35
        elif th == "jazz":                               # a showboat on the horizon, every window lit
            sx = (w * 0.15 + t * 0.6) % (w + 40) - 20 if not e.paused else o.setdefault("sbx", w * 0.3)
            o["sbx"] = sx
            rect(P, sx - 12, hp - 3, sx + 12, hp + 1, (30, 20, 26))
            rect(P, sx - 9, hp - 6, sx + 8, hp - 3, (44, 30, 36))
            rect(P, sx - 3, hp - 10, sx - 1, hp - 6, (60, 30, 30))
            rect(P, sx + 2, hp - 10, sx + 4, hp - 6, (60, 30, 30))
            for k in range(-10, 11, 2):
                if (k * 7 + int(e.beat_pos * 2)) % 9:
                    P[int(hp - 2), int(sx + k) % w] = (255, 200, 120)
            for k in range(-8, 8, 3):
                P[int(hp - 5), int(sx + k) % w] = (255, 220, 150)
            glow(P, sx, hp - 3, 10, (255, 170, 80), 0.25 + self.bass * 0.3, ry=3)
            self.notes(cv, int(sx), (hp - 12) // 2, (255, 214, 150), n=3, rise=5, seed=5)
        ev = self.rare(18.0)
        if ev:
            self._rare_ocean(P, ev, hp, sea, w, o)
        # ── lighthouse on a little island, beam sweeping ──
        ix = o["ix"]
        poly(P, o["island"], (22, 22, 40))
        P[hp - 5, max(0, ix - 2):max(0, ix + 3)] = (52, 50, 80)
        for k in range(o["tower"]):
            y = hp - 6 - k
            wd = 2 if k > o["tower"] // 2 else 3
            col = (232, 230, 240) if (k // o["band"]) % 2 else o["stripe"]
            x0_ = ix - wd // 2
            if 0 <= x0_ and x0_ + wd <= w:
                P[y, x0_:x0_ + wd] = col
                P[y, x0_ + wd - 1] = _c(col) * 0.65
        ly = hp - 8 - o["tower"]
        P[ly + 1:ly + 3, ix - 2:ix + 3] = (40, 36, 54)
        poly(P, [(ix - 2, ly + 0.5), (ix + 0.5, ly - 2.5), (ix + 3, ly + 0.5)], (150, 50, 60))
        phi = t * 0.8
        cs, sn = math.cos(phi), math.sin(phi)
        day = self.noon()                                # lighthouses (and boat lanterns) are dark by day
        P[ly + 1, ix] = (200, 210, 220) if day else (255, 240, 190)
        if not day:
            glow(P, ix + 0.5, ly + 1.5, 2.5 + 3 * max(0, sn) + 2 * self.change, (255, 236, 180), 0.5 + 0.6 * max(0, sn) + 0.3 * self.change)
        if abs(cs) > 0.1 and not day:
            side = 1 if cs > 0 else -1
            Lb = w * 0.7 * abs(cs)
            b = _box(P, ix + (0 if side > 0 else -Lb), 0, ix + (Lb if side > 0 else 1), hp)
            if b is not None:
                vb, by_, bx_ = b
                d = (bx_ - ix) * side
                beam = (d > 0) & (np.abs(by_ - ly - 1) < 0.8 + d * 0.1)
                vb += (beam * np.clip(1 - d / Lb, 0, 1) * 0.35 * abs(cs) ** 0.5)[..., None] * _c((255, 236, 180))
        # ── the swell the boat rides on ──
        by = hp + sea * 0.38
        A = (0.7 + 2.6 * rough) * (0.6 + 0.6 * min(1.0, level * 4 + self.bass * 0.5)) \
            + (0.6 if raining else 0) + 1.6 * surf_up + 0.5 * self.energy
        kx = 0.1 + 0.14 * rough                                         # long swells when calm, short chop when not

        def surf(x):
            return by + A * np.sin(x * kx - ph * 2.4) + 0.35 * A * np.sin(x * kx * 2.4 + ph * 1.3)

        xs = np.arange(w, dtype=np.float32)
        sw = surf(xs)
        on = (yy >= sw[None, :] - 0.5) & (yy < sw[None, :] + 0.5)
        crest = np.broadcast_to((sw < by - A * 0.8)[None, :], on.shape)          # just the peaks
        P[on & ~crest] = P[on & ~crest] * 0.8 + _c((96, 170, 200)) * 0.2
        foam = 0.1 + 0.3 * rough + 0.35 * surf_up                       # foamy crests only on a rough sea
        P[on & crest] = P[on & crest] * (1 - foam) + _c((232, 242, 252)) * foam
        above = (yy >= sw[None, :] - 4) & (yy < sw[None, :] - 0.5)
        P[above] *= 0.88                                   # the back of the swell is in shadow
        Pw = P.copy()
        # ── boat: bobs on the surface and rolls with the wave slope ──
        bx = w * o["bx"] + w * 0.05 * math.sin(t * 0.05)
        hull_x = bx + np.linspace(-8, 9, 9)
        hull_y = surf(hull_x)
        yb = float(hull_y.mean()) - 1.4                                 # floats on the water under the whole hull
        slope = float(np.polyfit(hull_x - bx, hull_y, 1)[0])             # and pitches with it
        ang = max(-0.2, min(0.2, 0.8 * math.atan(slope) + 0.03 * math.sin(t * 2.3)))
        ca, sa = math.cos(ang), math.sin(ang)

        def R(pts):
            return [(bx + dx * ca - dy * sa, yb - 0.5 + dx * sa + dy * ca) for dx, dy in pts]
        beat_on = int(e.beat_pos * 2) % 2 == 0
        poly(P, R([(-9, -1), (10, -1), (8, 2.5), (-7, 2.5)]), o["hull"])
        poly(P, R([(-9, -1.2), (10, -1.2), (10, 0), (-9, 0)]), (156, 102, 72))
        poly(P, R([(-5, -4), (-1, -4), (-1, -1), (-5, -1)]), (140, 96, 74))
        (wx_, wy_), = R([(-3, -2.5)])
        if 0 <= int(wy_) < H and 0 <= int(wx_) < w:
            P[int(wy_), int(wx_)] = (255, 220, 140)
        (m0x, m0y), (m1x, m1y) = R([(1, -1), (1, -18)])
        line(P, m0x, m0y, m1x, m1y, (210, 200, 190), 1)
        poly(P, R([(2, -17), (2, -2), (11, -2)]), o["sail"][0])
        poly(P, R([(2, -17), (2, -2), (4.5, -2)]), o["sail"][1])
        poly(P, R([(0, -15), (0, -2), (-7, -2)]), mix(o["sail"][0], (214, 214, 234), 0.6))
        poly(P, R([(1, -18), (1, -20.5), (4.5, -19.3)]), o["flag"])
        (lx_, ly_), = R([(-8.5, -2)])
        if not day:
            glow(P, lx_, ly_, 1.6 + 1.2 * beat_on, (255, 210, 120), 0.9 if beat_on else 0.5)
        b = _box(P, bx - 12, yb - 22, bx + 13, yb + 6)
        if b is not None:                                  # water laps over the hull
            v_, by2, bx2 = b
            sub = (by2 > surf(bx2) + 0.6) & (by2 > yb + 1.0)            # only ever the bottom of the hull
            y0_, x0_ = int(by2[0, 0]), int(bx2[0, 0])
            ref = Pw[y0_:y0_ + v_.shape[0], x0_:x0_ + v_.shape[1]]
            v_[sub] = v_[sub] * 0.2 + ref[sub] * 0.8
        # foamy wake trailing behind
        if random.random() < 0.5 + level * 3:
            o["wake"].append([bx - 8, 14])
        alive = []
        for wk in o["wake"]:
            wk[0] -= 0.25
            wk[1] -= 1
            if wk[1] > 0 and 0 <= wk[0] < w:
                alive.append(wk)
                xi = int(wk[0])
                yi = int(round(float(sw[xi])))
                if 0 <= yi < H:
                    P[yi, xi] = P[yi, xi] * 0.4 + _c((236, 244, 255)) * 0.6 * (wk[1] / 14)
        o["wake"] = alive
        self.sky = (0, w, 0, hp // 2 - 2)
        self.weather(o, cv, 1, w - 1, 0, [int(round(float(s))) // 2 for s in sw])

    def _rare_ocean(self, P, ev, hp, sea, w, o):
        """Rare event: out past the boat a whale surfaces, blows a plume of spray, rolls its back under
        and lifts its tail flukes as it dives."""
        u, r = ev
        avoid = (w * o["bx"], o["ix"])
        xw = max((r.uniform(0.12, 0.9) * w for _ in range(6)), key=lambda x: min(abs(x - a) for a in avoid))
        yw = int(hp + sea * r.uniform(0.12, 0.22))                  # the water line where it surfaces
        s = max(5.0, min(20.0, w * 0.085)) * (0.75 + 0.5 * (yw - hp) / max(1, sea * 0.22))
        f = r.choice((-1, 1))
        day = self.noon()
        skin, top = ((58, 68, 92), (130, 144, 170)) if day else ((34, 38, 62), (112, 122, 166))
        T = u * 18.0
        Q = P[:yw]                                                  # only what's above the water line shows
        if 1.0 < T < 8.5:                                           # the back rolls up and over
            k = (T - 1.0) / 7.5
            rise = math.sin(math.pi * min(1.0, k * 1.6)) if k < 0.62 else 0.0
            if k >= 0.62:
                rise = max(0.0, math.sin(math.pi * k) * 0.6)
            cx = xw + f * s * (k - 0.5) * 1.6
            disc(Q, cx, yw + s * 0.55 - rise * s * 0.75, s * 1.2, skin, ry=s * 0.6)
            disc(Q, cx - s * 0.1, yw + s * 0.5 - rise * s * 0.75, s * 1.0, top, a=0.35, ry=s * 0.45)
            if 0.35 < k < 0.7:                                      # the little dorsal fin
                fx_ = cx - f * s * 0.5
                poly(Q, [(fx_ - f * s * 0.25, yw - rise * s * 0.1), (fx_ + f * s * 0.1, yw - rise * s * 0.45),
                         (fx_ + f * s * 0.25, yw - rise * s * 0.05)], skin)
            foam = 0.25 + 0.5 * rise
            line(P, cx - s * 1.4, yw, cx + s * 1.4, yw, (220, 232, 246), wid=1, a=foam)
        if 2.2 < T < 5.6:                                           # the blow: a plume that rises, drifts and falls
            a_ = T - 2.2
            n = 40
            rs = random.Random(r.random())
            for j in range(n):
                vx, vy = rs.uniform(-1.6, 1.6), rs.uniform(6, 12)
                age = a_ - j * 0.01
                if age <= 0:
                    continue
                x = xw + f * s * 0.3 + vx * age + 1.2 * age * age
                y = yw - s * 0.2 - vy * age + 4.0 * age * age
                if 0 <= int(y) < yw and 0 <= int(x) < w:
                    k = max(0.0, 1 - a_ / 3.4)
                    P[int(y), int(x)] += (_c((236, 244, 255)) - P[int(y), int(x)]) * 0.8 * k
            glow(P, xw + f * s * 0.3, yw - s * 0.2 - min(a_, 0.8) * 6, s * 0.8, (220, 232, 250), 0.25 * max(0.0, 1 - a_ / 2))
        if 10.0 < T < 15.0:                                         # flukes up, then slipping under
            k = (T - 10.0) / 5.0
            lift = math.sin(math.pi * min(1.0, k * 1.3)) * s * 1.5
            tx = xw + f * s * 1.6
            base = yw - lift
            line(Q, tx, yw + 1, tx, base + s * 0.25, skin, wid=max(1.0, s * 0.22))
            for sd in (-1, 1):
                poly(Q, [(tx, base + s * 0.3), (tx + sd * s * 0.9, base - s * 0.3), (tx + sd * s * 0.55, base + s * 0.15)], skin)
                line(Q, tx, base + s * 0.3, tx + sd * s * 0.9, base - s * 0.3, top, wid=0.8, a=0.7)   # moonlit edge
            for j in range(int(s * 1.2)):                           # water streaming off the flukes
                y = base + s * 0.3 + (j * 1.7 + T * 14) % max(1.0, lift)
                xj = tx + ((j * 37) % 7 - 3) * s * 0.12
                if 0 <= int(y) < yw:
                    P[int(y), int(xj) % w] += (_c((200, 220, 245)) - P[int(y), int(xj) % w]) * 0.5
            line(P, tx - s, yw, tx + s, yw, (220, 232, 246), wid=1, a=0.5)
        if T > 14.0:                                                # rings spreading where it went down
            k = (T - 14.0) / 4.0
            disc_r = s * (0.8 + 2.5 * k)
            b = math.exp(-k * 1.5) * 0.4
            for sd in (-1, 1):
                line(P, xw + f * s * 1.6 + sd * disc_r * 0.6, yw + 0.5, xw + f * s * 1.6 + sd * disc_r, yw + 0.5,
                     (200, 216, 240), wid=1, a=b)
