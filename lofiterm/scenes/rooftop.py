import math
import random
import time

import numpy as np

from ..paint import (
    _c, blur, disc, fill_grad, fog_rows, glow, grid, hash01, line, make_stars, mix, noise_tex, poly, rect,
    vgrad)
from ..world import wrng
from ..theory import TAU
from .shared import CITY, gen_cat


def cat_sitting(P, cx, base, g, col, rim, t, sc=1.0, drop=6):
    """A cat sitting with its back to us on a ledge whose top is pixel row `base`, looking out over the city.
    Its tail hangs down over the front of the wall (`drop` pixels tall) and sways. `rim` is light catching
    the side facing the sky, so a dark cat still reads against the skyline."""
    hr = max(2.2, g["head"] * sc * 1.05)
    bw = max(3.2, g["body"] * sc * 1.1)                   # haunch half-width
    bh = max(5.0, g["hunch"] * sc * 1.15)                 # haunches to shoulders
    turn = 0.6 * math.sin(t * 0.35)                       # it looks around now and then
    hy = base - bh - hr * 0.75
    shapes = [("disc", cx, base - bh * 0.32, bw, bh * 0.36),             # haunches
              ("disc", cx, base - bh * 0.62, bw * 0.78, bh * 0.4),       # back
              ("disc", cx + turn * 0.3, base - bh * 0.95, bw * 0.5, bh * 0.22),  # neck / shoulders
              ("disc", cx + turn, hy, hr, hr * 0.92)]                     # head
    ears = []
    for sd in (-1, 1):
        ex = cx + turn + sd * hr * g["ear_sp"] * 1.1
        tip = (ex + sd * (0.5 + g["ear_tilt"]) * sc, hy - hr * 0.5 - max(1.6, g["ear"] * sc * 0.85))
        ears.append([(ex - hr * 0.55, hy - hr * 0.25), tip, (ex + hr * 0.55, hy - hr * 0.25)])
    sway = math.sin(t * 1.3)
    tail = []                                             # out from the haunch, over the edge, down the wall, curl
    for j in range(11):
        u = j / 10
        x = cx + bw * 0.75 + 1.2 * sc * math.sin(u * math.pi * 0.6) + sway * u * u * 1.8 * sc
        y = base - 1 + u * drop * 0.9
        if u > 0.75:                                      # the tip curls
            x += (u - 0.75) * 6 * sc * (0.6 + 0.4 * sway)
            y -= (u - 0.75) ** 2 * 10 * sc
        tail.append((x, y))
    tw = max(1.2, 0.9 * sc)

    def draw(c, ox, oy):
        for _, x, y, rx, ry in shapes:
            disc(P, x + ox, y + oy, rx, c, ry=ry)
        for tri in ears:
            poly(P, [(x + ox, y + oy) for x, y in tri], c)
        for (x0, y0), (x1, y1) in zip(tail, tail[1:]):
            line(P, x0 + ox, y0 + oy, x1 + ox, y1 + oy, c, wid=tw)

    draw(rim, -0.7, -0.6)                                 # light on the sky side…
    draw(col, 0.0, 0.0)                                   # …then the cat itself



# rooftop per station (the sky and skyline come from CITY): bulbs, roof floor, parapet wall and cap, laundry, plants, water tower
ROOF = [
    dict(bulbs=[(255, 206, 130), (255, 160, 200), (200, 170, 255)], roof=[(56, 46, 68), (28, 22, 38)], wall=(98, 64, 74), cap=(142, 110, 122),
         cloth=[(255, 170, 200), (170, 210, 255), (255, 230, 160), (210, 240, 210)], plant=(70, 122, 92), tower=(40, 32, 56)),
    dict(bulbs=[(255, 230, 170), (255, 214, 140)], roof=[(32, 34, 50), (14, 14, 24)], wall=(56, 60, 82), cap=(86, 92, 118),
         cloth=[(200, 210, 240), (140, 160, 220), (240, 240, 250)], plant=(40, 82, 72), tower=(16, 18, 32)),
    dict(bulbs=[(255, 222, 150), (255, 182, 112)], roof=[(112, 82, 76), (70, 50, 48)], wall=(172, 102, 82), cap=(212, 152, 122),
         cloth=[(255, 255, 255), (255, 200, 120), (140, 200, 255), (255, 150, 140)], plant=(100, 142, 60), tower=(92, 52, 62)),
    dict(bulbs=[(255, 214, 150), (255, 240, 200), (190, 220, 255)], roof=[(150, 156, 180), (100, 106, 130)], wall=(102, 98, 122), cap=(222, 228, 242),
         cloth=[(204, 62, 62), (62, 102, 182), (240, 240, 250)], plant=(50, 92, 82), tower=(50, 56, 80)),
    dict(bulbs=[(60, 255, 230), (255, 60, 200), (250, 240, 90), (150, 90, 255)], roof=[(42, 20, 58), (16, 8, 28)], wall=(72, 30, 92), cap=(122, 60, 152),
         cloth=[(255, 60, 200), (60, 255, 230), (250, 240, 90)], plant=(40, 122, 112), tower=(30, 10, 46)),
    dict(bulbs=[(255, 236, 160), (222, 255, 172)], roof=[(34, 54, 50), (16, 28, 28)], wall=(62, 92, 82), cap=(102, 142, 122),
         cloth=[(255, 214, 232), (220, 255, 220), (255, 240, 200)], plant=(62, 132, 92), tower=(14, 34, 36)),
    dict(bulbs=[(255, 236, 190), (255, 214, 160)], roof=[(172, 162, 150), (132, 122, 112)], wall=(198, 122, 96), cap=(232, 192, 172),
         cloth=[(255, 255, 255), (255, 214, 90), (120, 200, 250), (255, 140, 140)], plant=(90, 162, 70), tower=(122, 96, 80)),
    dict(bulbs=[(240, 70, 56), (255, 236, 214), (250, 120, 70)], roof=[(60, 44, 64), (30, 22, 36)], wall=(110, 66, 84), cap=(150, 110, 130),   # tokyo
         cloth=[(60, 76, 140), (255, 190, 214), (240, 236, 226), (200, 60, 60)], plant=(70, 120, 90), tower=(44, 28, 54)),
    dict(bulbs=[(255, 190, 100), (255, 214, 140), (255, 160, 80)], roof=[(54, 36, 34), (26, 16, 18)], wall=(110, 50, 44), cap=(150, 96, 80),     # jazz
         cloth=[(170, 40, 50), (240, 230, 210), (40, 40, 50), (200, 150, 60)], plant=(70, 100, 60), tower=(40, 24, 26)),
    dict(bulbs=[(255, 214, 140), (255, 230, 170), (255, 196, 110)], roof=[(60, 50, 52), (30, 26, 30)], wall=(120, 70, 56), cap=(150, 140, 130),     # english
         cloth=[(40, 60, 110), (230, 226, 214), (150, 40, 50), (60, 90, 60)], plant=(70, 110, 64), tower=(36, 36, 44))
]


class RooftopScene:
    # ── rooftop ────────────────────────────────────────────────────────────
    def _roof_chimney(self, P, x, base, k, t):
        """A brick chimney stack standing on `base` (the parapet's top, so it reads as rising from the building
        behind), with clay pots, one of them smoking."""
        rect(P, x - 4 * k, base - 9 * k, x + 4 * k, base, (96, 52, 46))
        for yb in range(int(base - 9 * k) + 2, int(base), 3):
            P[yb, int(max(0, x - 4 * k)):int(min(P.shape[1], x + 4 * k))] *= 0.75
        rect(P, x - 4.6 * k, base - 10 * k, x + 4.6 * k, base - 9 * k, (150, 140, 130))
        for q in (-2.4, 0, 2.4):
            rect(P, x + (q - 0.9) * k, base - 13 * k, x + (q + 0.9) * k, base - 10 * k, (150, 84, 60))
        for q in range(3):
            u = (self.rclock * 0.2 + q / 3) % 1
            disc(P, x + 2.4 * k + u * 8 * k + math.sin(u * 6 + q), base - 13 * k - u * 10 * k, (1 + u * 2.5) * k,
                 (170, 170, 180), a=0.3 * (1 - u))

    def v_rooftop(self, cv, P, h, w):
        """A city rooftop: string lights chase the beat, laundry flaps in the wind (gusts on the
        bass), a cat watches from the parapet, the AC fan spins with the music, steam drifts from a
        vent, a radio plays along by the deck chair and a telescope points at the moon."""
        e = self.eng
        t = time.time()
        H = P.shape[0]
        V, C = self.var(ROOF), self.var(CITY)
        snow = e.st.get("weather") == "snow"
        day = self.noon()
        py = int(H * 0.58)
        wall = max(5, int(round(H * 0.075)))                # the parapet's height
        fl = py + wall
        yy, xx = grid(H, w)
        fh = H - fl
        k = max(1.0, fh / 14)                                # scale for things standing on the roof
        bh0, bw_ = 3, 18                                     # stairwell hut
        hut = w >= 90
        RL = self.layout("roof", h, w, lambda r: dict(
            ax=r.uniform(0.5, 0.6), sx=r.uniform(0.22, 0.3), moon=r.uniform(0.1, 0.4), tower=r.uniform(0.66, 0.9),
            tkind=r.randrange(3), cat=r.uniform(0.58, 0.7), catg=gen_cat(r)))
        ax = int(min(w * RL["ax"], w * 0.74 - 5 * k - 14))  # AC unit, kept clear of the deck chair
        sx_, sy_ = int(w * RL["sx"]), int(fl + fh * 0.35)    # skylight

        def build():
            B = np.zeros((H, w, 3), np.float32)
            fill_grad(B, 0, fl, list(zip((0, 0.5, 0.82, 1), C["sky"])))
            if C["sun"]:
                glow(B, w * RL["moon"], py * 0.7, H * 0.3, (255, 150, 90), 0.5)
                disc(B, w * RL["moon"], py * 0.7, max(3.0, H * 0.07), (255, 226, 160))
            else:
                self.sky_body(B, w * RL["moon"], py * 0.2, max(2.5, H * 0.05))
            rng = wrng("roofsky", h, w)
            hz = _c(C["haze"])
            tops = []
            for li, (lo, hi) in enumerate([(0.2, 0.46), (0.08, 0.32)]):
                top = np.full(w, float(fl))
                x = rng.randint(-5, 0)
                while x < w:
                    bw = rng.randint(5, 12)
                    top[max(0, x):min(w, x + bw)] = py - int(py * rng.uniform(lo, hi))
                    x += bw + rng.choice([0, 1, 2])
                msk = (yy >= top[None, :]) & (yy < fl)
                B[msk] = hz * 0.45 if li == 0 else hz * 0.2 + 4
                if day:
                    B[msk] = hz * (0.75 if li == 0 else 0.55)
                wm = msk & (yy > top[None, :] + 1) & (xx % 3 == 1) & (yy % 3 == 1) & (hash01(xx, yy, li + 3) < C["lit"] + 0.1)
                warm = _c((140, 170, 200) if day else (255, 204, 128)) * (0.45 if li == 0 else 0.85)
                B[wm] = warm
                if not day:
                    B += blur(wm.astype(np.float32), 1)[..., None] * warm * 0.3
                if li == 0:
                    fog_rows(B, py * 0.3, fl, C["haze"], 0.35)
                tops.append(top)
            skym = yy < tops[0][None, :]
            wtx = int(w * RL["tower"])                       # a water tower, a mast or a billboard next door
            tc = V["tower"]
            if wtx < w - 6 and RL["tkind"] == 1:
                base = int(tops[1][wtx])
                line(B, wtx, base - 16, wtx, base, tc, wid=1)
                for j in range(1, 4):
                    line(B, wtx - 3 + j * 0.6, base - j * 4, wtx + 3 - j * 0.6, base - j * 4, tc, wid=1)
            elif wtx < w - 10 and RL["tkind"] == 2:
                base = int(tops[1][wtx])
                for lx in (wtx - 3, wtx + 3):
                    line(B, lx, base - 4, lx, base, tc, wid=1)
                rect(B, wtx - 7, base - 11, wtx + 8, base - 4, tc)
                rect(B, wtx - 6, base - 10, wtx + 7, base - 5, mix(tc, C["haze"], 0.35))
            elif wtx < w - 6:
                base = int(tops[1][wtx])
                for lx in (wtx - 3, wtx + 3):
                    line(B, lx, base - 4, lx, base, tc, wid=1)
                rect(B, wtx - 4, base - 11, wtx + 5, base - 4, tc)
                poly(B, [(wtx - 5, base - 11), (wtx + 0.5, base - 15), (wtx + 6, base - 11)], tc)
                B[base - 9:base - 5, wtx - 4] += 18
            # parapet: a lit cap, brick courses with mortar, darker towards its foot
            fill_grad(B, py, fl, [(0, mix(V["wall"], V["cap"], 0.25)), (1, _c(V["wall"]) * 0.78)])
            Wy, Wx = yy[py + 2:fl] - py - 2, xx[py + 2:fl]
            course = Wy // 2
            mortar = ((Wy % 2) == 1) | (((Wx + course * 3) % 6) == 0)
            B[py + 2:fl] *= np.where(mortar, 0.72, 0.92 + 0.12 * hash01(Wx // 6 + course * 5, course, 11))[..., None]
            B[py] = mix(V["cap"], (255, 240, 220), 0.25)
            B[py + 1] = _c(V["cap"]) * 0.62                                          # shadow under the cap's lip
            fill_grad(B, fl, H, [(0, V["roof"][0]), (1, V["roof"][1])])                 # tiled roof floor
            F = B[fl:]
            ry_ = yy[fl:] - fl
            F[(ry_ % 4) == 0] *= 0.86
            F[((xx[fl:] + (ry_ // 4) * 5) % 12) == 0] *= 0.88
            F *= (0.95 + 0.05 * noise_tex(fh, w, 2, 9))[..., None]
            if snow:
                B[py] = (236, 240, 250)
                B[fl:] = B[fl:] * 0.45 + _c((206, 212, 230)) * 0.55
            # contact shadow where the floor meets the wall, so the two never run together
            cs = max(2, wall // 2)
            B[fl:fl + cs] *= (0.45 + 0.55 * (np.arange(cs) / cs) ** 0.7)[:, None, None]
            if hut:                                          # stairwell hut with a door
                top_ = fl - 10
                bot = int(fl + fh * 0.45)
                rect(B, bh0, top_, bh0 + bw_, bot, _c(V["wall"]) * 0.8)
                rect(B, bh0 - 1, top_ - 1, bh0 + bw_ + 1, top_, V["cap"])
                rect(B, bh0 + bw_ - 1, top_, bh0 + bw_, bot, _c(V["wall"]) * 0.6)
                rect(B, bh0 + 5, bot - 11, bh0 + 12, bot, (70, 50, 46))
                rect(B, bh0 + 6, bot - 10, bh0 + 11, bot - 6, (130, 160, 190) if day else (255, 206, 140))
                rect(B, bh0 + 10, bot - 5, bh0 + 11, bot - 4, (200, 170, 90))
                rect(B, bh0 + 14, top_ - 4, bh0 + 16, top_, (120, 116, 126))                # vent pipe on its roof
            sk = [(sx_, sy_), (sx_ + 16, sy_), (sx_ + 18, sy_ + 5), (sx_ - 2, sy_ + 5)]       # skylight
            if sx_ + 20 < ax:
                poly(B, [(x - 1, y - 1) for x, y in sk[:2]] + [(sk[2][0] + 1, sk[2][1] + 1), (sk[3][0] - 1, sk[3][1] + 1)], (150, 146, 156))
                poly(B, sk, (150, 180, 210) if day else (214, 172, 120))
                for j in range(1, 4):
                    line(B, sx_ + j * 4, sy_, sx_ - 2 + j * 5.5, sy_ + 5, (150, 146, 156), wid=1)
                line(B, sx_ - 1, sy_ + 2.5, sx_ + 17, sy_ + 2.5, (150, 146, 156), wid=1)
            rect(B, ax, fl - 1, ax + 12, fl + 7, (176, 180, 188))                          # AC unit
            rect(B, ax, fl - 1, ax + 12, fl, (210, 214, 222))
            rect(B, ax + 12, fl - 1, ax + 13, fl + 7, (130, 134, 142))
            disc(B, ax + 6, fl + 3, 3.3, (60, 64, 72))
            rect(B, ax + 15, fl - 6, ax + 17, fl + 6, (120, 116, 126))                     # steam vent
            rect(B, ax + 14, fl - 7, ax + 18, fl - 6, (150, 146, 156))
            return B, skym
        B, skym = self.tex("roof", (h, w, e.station_idx), build)
        P[:] = B
        s = getattr(self, "roofst", None)
        if s is None or s["size"] != (h, w):
            s = self.roofst = dict(size=(h, w), fan=0.0, puffs=[], last=t, star=None, drops=[], splash=[],
                                   stars=make_stars(wrng("roofstars", w), int(py * 0.36), w, w * h // 40))
        dt = min(0.1, max(0.0, t - s["last"]))
        s["last"] = t
        level = float(self.rms.mean())
        # sky: stars, a shooting star now and then, drifting clouds, a plane blinking across
        self.chord_wash(P, 0, py, 0.04)                           # the sky over the city takes each chord's colour
        self.stars(P, s["stars"], t, (0.3 if e.wet else 1.0) * C["stars"])
        if not day and not e.wet:
            if s["star"] is None and (random.random() < dt / 18 or ("change" in self.fired and random.random() < 0.12)):
                s["star"] = [random.uniform(w * 0.3, w), random.uniform(1, py * 0.3), 0.0]
            if s["star"] is not None:
                x0, y0, age = s["star"]
                L = age * 40
                line(P, x0 - L, y0 + L * 0.35, x0 - L + 6, y0 + L * 0.35 - 2, (255, 250, 230), wid=1, a=max(0.0, 1 - age * 1.4))
                s["star"][2] += dt
                if age > 0.7:
                    s["star"] = None
        clouds = self.tex("roofcl", (fl, w), lambda: noise_tex(fl, w, 7, 2))
        n = np.roll(clouds, int(t * 1.2) % w, axis=1)
        band = np.exp(-((np.arange(fl) / fl - 0.25) / 0.2) ** 2)[:, None]
        a = np.clip((n - (0.0 if e.wet else 0.5)) * 0.8, 0, 0.8) * band * skym[:fl]
        cc = vgrad(fl, [(0, C["cloud"][0]), (1, C["cloud"][1])])[:, None, :]
        P[:fl] += (cc - P[:fl]) * a[..., None] * (0.9 if day else 0.6)
        if e.wet:
            P[:py] = P[:py] * 0.75 + _c(C["cloud"][0]) * 0.2
        apx = (t * 7) % (w + 80) - 40
        apy = int(py * 0.12) + 2
        if 0 <= apx < w and not e.wet:
            on = int(t * 2) % 2
            P[apy, int(apx)] = (255, 90, 90) if on else (255, 255, 255)
            glow(P, apx, apy, 1.6, (255, 90, 90) if on else (255, 255, 255), 0.5)
        ev = self.rare(45.0)
        if ev:
            self._rare_roof(P, ev, py, w, day)
        bp = e.beat_pos
        # laundry line
        wind = (1.0 + 0.5 * self.energy) * math.sin(t * 0.9) + 0.8 * math.sin(t * 2.3 + 1) + 2.4 * self.bass + 1.2 * self.snare
        lx0, lx1 = (max(int(w * 0.17), bh0 + bw_ + 3), int(w * 0.46)) if hut else (int(w * 0.07), int(w * 0.42))
        if w >= 70:
            ly = max(int(H * 0.3), py - 14)
            for lx in (lx0, lx1):
                line(P, lx, ly - 1, lx, fl + 3, (70, 56, 50), wid=1)
            us = np.linspace(0, 1, lx1 - lx0 + 1)
            rope = ly + 3 * 4 * us * (1 - us)
            P[rope.astype(int), np.arange(lx0, lx1 + 1)] = (190, 180, 170)
            rng = wrng("laundry")
            u, i = 0.06, 0
            while u < 0.94:
                wd, ln = rng.choice([(6, 7), (4, 4), (7, 9), (2, 4)])
                x = lx0 + u * (lx1 - lx0)
                y = ly + 3 * 4 * u * (1 - u)
                col = V["cloth"][i % len(V["cloth"])]
                sw = wind * (0.6 + 0.4 * math.sin(t * 1.7 + i))
                poly(P, [(x - wd / 2, y), (x + wd / 2, y), (x + wd / 2 + sw, y + ln), (x - wd / 2 + sw, y + ln)], col)
                if wd >= 6:                                   # shirt sleeves
                    poly(P, [(x - wd / 2, y), (x - wd / 2 - 2 + sw * 0.3, y + 3), (x - wd / 2 + sw * 0.2, y + 3)], col)
                    poly(P, [(x + wd / 2, y), (x + wd / 2 + 2 + sw * 0.3, y + 3), (x + wd / 2 + sw * 0.2, y + 3)], col)
                line(P, x + sw * 0.5, y + 1, x + sw, y + ln - 1, _c(col) * 0.8, wid=1, a=0.6)
                rect(P, x - wd / 2 + 1, y - 1, x - wd / 2 + 2, y + 1, (210, 190, 150))
                u += (wd + 3) / (lx1 - lx0)
                i += 1
        # potted plants and a cat on the parapet
        rng = wrng("pots", w)
        cx = int(w * RL["cat"])
        if self.theme() == "english":                        # the building's chimney stack, rising behind the parapet
            chx = next((x_ for x_ in (w * 0.53, w * 0.9, w * 0.62) if abs(x_ - cx) > 10 * k), w * 0.53)
            self._roof_chimney(P, chx, py, k, t)
        for _ in range(max(2, w // 30)):
            px = rng.uniform(w * 0.5, w - 6)
            if abs(px - cx) < 7:
                continue
            ph = rng.random() * TAU
            poly(P, [(px - 3, py - 3), (px + 3, py - 3), (px + 2, py), (px - 2, py)], (172, 92, 62))
            for j in range(5):
                a_ = -math.pi / 2 + (j - 2) * 0.5 + math.sin(t * 1.2 + ph + j) * 0.1 * (1 + 2 * self.bass)
                line(P, px, py - 3, px + math.cos(a_) * 4, py - 3 + math.sin(a_) * 4, V["plant"], wid=1.5)
        rim = mix(C["haze"], (255, 226, 190), 0.5) if not day else (255, 250, 236)
        cat_sitting(P, cx, py, RL["catg"], (16, 14, 24), rim, t, sc=max(1.0, min(2.0, H / 60)), drop=wall)
        # string lights chasing the beat (dimmed by day)
        top = max(2, int(H * 0.12))
        nspan = max(1, round((w - 5) / 80))
        posts = np.linspace(2, w - 3, nspan + 1)
        for px in posts:
            line(P, px, top - 1, px, fl + 3, (52, 40, 40), wid=1.5)
        dimk = 0.35 if day else 1.0
        jp = self.theme() == "japan"
        kb = 0
        for i in range(nspan):
            a0, b0 = posts[i], posts[i + 1]
            for j, (dy, sag) in enumerate(((1, H * 0.13), (5, H * 0.08))):
                xs = np.arange(int(a0), int(b0) + 1)
                u = (xs - a0) / max(1, b0 - a0)
                ys = top + dy + sag * 4 * u * (1 - u)
                ok = (ys >= 0) & (ys < H)
                P[ys[ok].astype(int), xs[ok]] = (40, 30, 40)
                for xb in range(int(a0) + 3, int(b0) - 1, 6):
                    ub = (xb - a0) / max(1, b0 - a0)
                    yb = top + dy + sag * 4 * ub * (1 - ub) + 1
                    col = mix(V["bulbs"][(kb + j) % len(V["bulbs"])], tuple(self.tint), 0.45 * self.change)   # a chord change washes through
                    chase = 0.5 + 0.5 * math.cos(TAU * (kb / 8 - bp / 4))
                    br = (0.35 + 0.35 * chase + 0.3 * self.bass + 0.3 * self.kick) * dimk
                    if jp:                               # paper lanterns instead of bulbs
                        self.chochin(P, xb, yb + 1.5, 1.5, mix((60, 50, 60), col, min(1.0, br + 0.25)), lit=br, mark=False)
                    else:
                        disc(P, xb, yb, 1.0, mix((60, 50, 60), col, min(1.0, br + 0.25)))
                        glow(P, xb, yb, 2.6, col, 0.32 * br)
                    kb += 1
            glow(P, (a0 + b0) / 2, fl + 6, (b0 - a0) * 0.55, V["bulbs"][0], 0.07 * (1 + self.bass) * dimk, ry=6)
        # the stairwell door lamp and the skylight glow at night
        if hut and not day:
            bot = int(fl + fh * 0.45)
            glow(P, bh0 + 8.5, bot - 13, 3, (255, 214, 150), 0.7)
            glow(P, bh0 + 8.5, bot + 2, 10, (255, 200, 130), 0.18, ry=3)
        if sx_ + 20 < ax and not day:
            glow(P, sx_ + 8, sy_ + 2.5, 12, (255, 200, 130), 0.16 + 0.08 * self.bass, ry=5)
        # AC fan spinning with the music, steam puffing from the vent
        s["fan"] += dt * (3 + 40 * level)
        for j in range(3):
            a_ = s["fan"] + j * TAU / 3
            line(P, ax + 6, fl + 3, ax + 6 + math.cos(a_) * 2.8, fl + 3 + math.sin(a_) * 2.8, (150, 154, 162), wid=1.2)
        P[fl + 3, ax + 6] = (200, 204, 212)
        if random.random() < dt * 2.5:
            s["puffs"].append([ax + 16 + random.uniform(-0.5, 0.5), fl - 8.0, 1.2, 0.0])
        keep = []
        for pf in s["puffs"]:
            pf[3] += dt / 4
            pf[1] -= dt * 4
            pf[0] += dt * (1.5 + wind * 0.6)
            pf[2] += dt * 1.2
            if pf[3] < 1:
                keep.append(pf)
                disc(P, pf[0], pf[1], pf[2], (220, 222, 232), a=0.3 * (1 - pf[3]))
        s["puffs"] = keep[-40:]
        # deck chair, a crate with a radio and a mug, and a telescope
        chx = int(w * 0.74)
        cy = fl + fh * 0.78
        if chx + 9 * k < w and cy < H:
            wood = (128, 92, 62)
            F_, S_, T_ = (chx + 7 * k, cy - 3 * k), (chx, cy - 3.5 * k), (chx - 4 * k, cy - 10 * k)
            line(P, chx - 4.5 * k, cy, T_[0] + 0.5, T_[1], wood, wid=1.2)           # back legs
            line(P, chx + 8 * k, cy, F_[0], F_[1], wood, wid=1.2)                   # front legs
            line(P, chx - 1 * k, cy, chx + 5 * k, cy - 6 * k, wood, wid=1)
            stripes = [V["cloth"][0], (255, 255, 255)]
            for q in range(4):                               # striped canvas sling
                off_ = q * 0.6 * k
                col = stripes[q % 2]
                line(P, T_[0] + off_, T_[1] + off_ * 0.2, S_[0] + off_ * 0.3, S_[1], col, wid=1.2)
                line(P, S_[0] + off_ * 0.3, S_[1], F_[0], F_[1] + off_ * 0.1, col, wid=1.2)
            line(P, chx - 1 * k, cy - 6 * k, chx + 6 * k, cy - 6 * k, wood, wid=1)  # armrest
            crx = chx - 12 * k
            if crx > ax + 20:
                rect(P, crx, cy - 5 * k, crx + 8 * k, cy, (150, 110, 70))
                for q in range(1, 3):
                    rect(P, crx, cy - 5 * k + q * 1.7 * k, crx + 8 * k, cy - 5 * k + q * 1.7 * k + 0.6, (110, 78, 50))
                rx_, ry_ = crx + 0.5 * k, cy - 5 * k
                rect(P, rx_, ry_ - 4 * k, rx_ + 5 * k, ry_, (80, 70, 86))                # radio
                line(P, rx_ + 4.5 * k, ry_ - 4 * k, rx_ + 7 * k, ry_ - 9 * k, (170, 170, 180), wid=1)
                lv, _ = self.band_levels(5)
                self.note_sparks(P, "roof", lambda p, n: (rx_ + 7 * k + (n % 3 - 1), ry_ - 9 * k - p * 3), (255, 214, 170),
                                 rise=5.0, drift=1.0 + wind * 0.3, life=1.7, r=0.8)     # the tune floats off the aerial
                for b_ in range(5):
                    hgt = (0.5 + lv[b_] * 2.5) * k
                    rect(P, rx_ + 0.5 + b_ * k * 0.9, ry_ - 0.6 - hgt, rx_ + 0.5 + b_ * k * 0.9 + max(1, k * 0.6), ry_ - 0.6,
                         mix((120, 220, 200), V["bulbs"][0], b_ / 4))
                mx_ = crx + 6.5 * k                           # mug
                rect(P, mx_ - 1, ry_ - 2.5 * k, mx_ + 1.5 * k, ry_, (236, 230, 220))
                for q in range(2):
                    cyc = int(ry_ - 3 * k) // 2 - q
                    if cyc >= 0:
                        cv.put(cyc, int(mx_ + math.sin(t * 1.4 + q * 2)), "~" if (q + int(t * 2)) % 2 else "˜", (200, 196, 210))
            tsx = chx + 12 * k
            if tsx + 5 * k < w:
                for dxl in (-2.5 * k, 0, 2.5 * k):                                   # tripod
                    line(P, tsx, cy - 6 * k, tsx + dxl, cy, (90, 90, 100), wid=1)
                line(P, tsx + 2 * k, cy - 5 * k, tsx - 3 * k, cy - 9.5 * k, (200, 196, 210), wid=1.6 * min(k, 2))
                disc(P, tsx - 3 * k, cy - 9.5 * k, 0.9 * k, (230, 230, 240))
        th = self.theme()
        fx_, fb_ = w * 0.6, H - 2.0
        if th == "japan":                                    # a little rooftop shrine
            self.torii(P, fx_ - 7 * k, fb_, 9 * k)
            rect(P, fx_ - 2.5 * k, fb_ - 5 * k, fx_ + 2.5 * k, fb_, (150, 110, 80))
            rect(P, fx_ - 1 * k, fb_ - 4 * k, fx_ + 1 * k, fb_ - 1 * k, (60, 30, 30))
            poly(P, [(fx_ - 4 * k, fb_ - 5 * k), (fx_, fb_ - 8 * k), (fx_ + 4 * k, fb_ - 5 * k)], (70, 60, 70))
            glow(P, fx_, fb_ - 3 * k, 4 * k, (255, 180, 100), 0.3 + 0.2 * self.bass)
        elif th == "jazz":                                   # a gramophone and a stack of records out for the night
            fx_ = w * 0.5
            rect(P, fx_ - 3 * k, fb_ - 3 * k, fx_ + 3 * k, fb_, (110, 64, 40))
            line(P, fx_, fb_ - 3 * k, fx_ + 1.5 * k, fb_ - 6 * k, (200, 150, 60), wid=0.8)
            poly(P, [(fx_ + 1.5 * k, fb_ - 6 * k), (fx_ + 6 * k, fb_ - 12 * k), (fx_ + 8.5 * k, fb_ - 8 * k)], (226, 176, 72))
            disc(P, fx_ + 7.4 * k, fb_ - 10 * k, 2.3 * k, (250, 206, 110), ry=2.6 * k)
            disc(P, fx_ + 7.7 * k, fb_ - 10 * k, 1.3 * k, (110, 64, 30), ry=1.6 * k)
            for q in range(3):
                self.record(P, fx_ - 7 * k - q * 1.3, fb_ - 3 * k, 3 * k, ang=q, label=V["cloth"][q], sheen=0.6)
            self.notes(cv, int(fx_ + 8 * k), int(fb_ - 13 * k) // 2, V["bulbs"][1], n=3, rise=5, seed=12)
        if e.fire_on:                                        # fire bowl
            fbx, fby = w * 0.44, int(fl + fh * 0.62)
            glow(P, fbx, fby - 3, 14, (255, 140, 70), 0.35 + 0.2 * self.bass, ry=7)
            self.flames(P, fbx, fby - 1, 0.5 + self.bass, 3.5, "roof")
            disc(P, fbx, fby, 4.5, (40, 34, 38), ry=1.8)
        if e.wet:                                            # puddles holding the lights
            rng = wrng("puddles", w)
            for _ in range(max(2, w // 40)):
                px, pyy = rng.uniform(5, w - 5), rng.uniform(fl + 3, H - 2)
                disc(P, px, pyy, rng.uniform(4, 9), mix(C["sky"][1], (0, 0, 0), 0.3), a=0.6, ry=1.3)
                glow(P, px, pyy, 3, V["bulbs"][0], (0.2 + 0.2 * self.bass) * dimk, ry=1)
        self.sky = (0, w, 0, py // 3)
        self.weather(s, cv, 1, w - 1, 0, (H - 3) // 2)

    def _rare_roof(self, P, ev, py, w, day):
        """Rare event: a hot-air balloon drifts slowly across the sky, its burner flaring on heavy bass
        and lighting up the envelope from inside."""
        u, r = ev
        H = P.shape[0]
        rad = max(3.0, H * r.uniform(0.06, 0.09))
        f = r.choice((-1, 1))
        x = (-rad * 2 + u * (w + rad * 4)) if f > 0 else (w + rad * 2 - u * (w + rad * 4))
        y = py * r.uniform(0.18, 0.32) - u * py * 0.08 + math.sin(self.rclock * 0.4) * 0.8
        pal = r.choice([[(214, 70, 82), (250, 214, 120), (246, 240, 230)], [(70, 110, 190), (240, 240, 246), (226, 90, 80)],
                        [(110, 180, 120), (246, 210, 110), (240, 140, 80)], [(150, 96, 190), (250, 170, 200), (250, 240, 220)]])
        burn = max(0.0, (self.bass - 0.65) * 3) + (0.3 if (self.rclock % 7) < 0.6 else 0.0)
        k = 1.0 if day else 0.4 + 0.5 * min(1.0, burn)
        self.balloon(P, x, y, rad, [_c(c) * k for c in pal], burn=burn)
