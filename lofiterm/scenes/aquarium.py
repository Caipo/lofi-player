import math
import random
import time

import numpy as np

from ..paint import _c, disc, fill_grad, glow, grid, line, mix, noise_tex, poly, rect, ridge1d, vgrad
from ..world import wrng
from ..theory import TAU


# aquarium per station: water (top, bottom), sand (top, deep), plants (shade, lit), room wall, tank light, fish colours
AQUA = [
    dict(water=[(44, 112, 164), (12, 34, 84)], sand=[(200, 172, 132), (128, 106, 88)], plant=[(26, 92, 76), (96, 196, 136)],
         wall=[(22, 16, 38), (10, 8, 20)], lamp=(200, 232, 255), fish=[(255, 140, 80), (255, 204, 96), (255, 124, 176), (250, 250, 255)], tetra=((80, 200, 255), (255, 60, 80))),
    dict(water=[(30, 60, 132), (5, 14, 48)], sand=[(122, 122, 152), (70, 70, 100)], plant=[(18, 66, 80), (60, 152, 144)],
         wall=[(8, 10, 24), (4, 4, 12)], lamp=(160, 190, 255), fish=[(130, 200, 255), (240, 240, 255), (255, 120, 120)], tetra=((140, 220, 255), (200, 120, 255))),
    dict(water=[(84, 176, 172), (28, 92, 112)], sand=[(242, 212, 152), (192, 152, 102)], plant=[(60, 122, 50), (152, 214, 92)],
         wall=[(72, 48, 38), (40, 26, 22)], lamp=(255, 232, 184), fish=[(255, 150, 40), (255, 222, 82), (244, 84, 60)], tetra=((90, 220, 255), (255, 80, 60))),
    dict(water=[(112, 152, 194), (40, 60, 102)], sand=[(222, 228, 242), (160, 170, 198)], plant=[(48, 90, 100), (132, 182, 194)],
         wall=[(32, 36, 52), (16, 18, 30)], lamp=(222, 238, 255), fish=[(255, 255, 255), (255, 152, 122), (162, 202, 255)], tetra=((200, 230, 255), (255, 120, 120))),
    dict(water=[(64, 22, 144), (10, 4, 50)], sand=[(92, 62, 142), (50, 30, 90)], plant=[(20, 120, 120), (60, 255, 212)],
         wall=[(14, 4, 26), (6, 2, 14)], lamp=(255, 92, 222), fish=[(60, 255, 230), (255, 60, 200), (250, 240, 90)], tetra=((60, 255, 230), (255, 60, 200))),
    dict(water=[(52, 142, 132), (10, 50, 60)], sand=[(172, 172, 132), (110, 110, 90)], plant=[(30, 100, 60), (122, 212, 122)],
         wall=[(12, 26, 24), (6, 14, 14)], lamp=(212, 255, 232), fish=[(255, 202, 122), (255, 122, 92), (222, 255, 182)], tetra=((120, 255, 220), (255, 100, 90))),
    dict(water=[(92, 202, 232), (30, 122, 182)], sand=[(250, 236, 200), (210, 190, 150)], plant=[(50, 152, 80), (152, 232, 112)],
         wall=[(212, 216, 226), (172, 178, 192)], lamp=(255, 255, 240), fish=[(255, 150, 40), (255, 232, 90), (255, 90, 122), (255, 255, 255)], tetra=((80, 200, 255), (255, 60, 80))),
    dict(water=[(60, 90, 160), (18, 20, 70)], sand=[(230, 210, 200), (160, 130, 140)], plant=[(30, 90, 80), (120, 200, 150)],      # tokyo
         wall=[(40, 22, 40), (18, 10, 20)], lamp=(255, 210, 230), fish=[(255, 120, 60), (250, 250, 255), (220, 50, 50), (255, 200, 90)], tetra=((255, 150, 190), (255, 255, 255))),
    dict(water=[(90, 80, 70), (26, 20, 24)], sand=[(200, 170, 120), (120, 96, 70)], plant=[(50, 80, 40), (160, 170, 80)],          # jazz
         wall=[(30, 14, 16), (14, 6, 8)], lamp=(255, 200, 130), fish=[(255, 190, 70), (230, 230, 220), (200, 80, 60)], tetra=((255, 210, 120), (200, 60, 60))),
    dict(water=[(70, 96, 90), (20, 34, 38)], sand=[(190, 180, 150), (110, 104, 90)], plant=[(40, 80, 50), (120, 160, 90)],         # english
         wall=[(26, 30, 34), (12, 14, 18)], lamp=(255, 214, 150), fish=[(240, 150, 60), (230, 226, 210), (90, 110, 160)], tetra=((200, 220, 255), (220, 60, 60)))
]


class AquariumScene:
    # ── aquarium ───────────────────────────────────────────────────────────
    def _fish(self, P, f, t):
        x, y, d, L, col = f["x"], f["y"], f["d"], f["L"], f["col"]
        wig = math.sin(t * 7 + f["ph"]) * L * 0.08
        kind = f["kind"]
        light = mix(col, (255, 255, 255), 0.35)
        if kind == "tetra":
            disc(P, x, y, L / 2, col, ry=max(0.7, L / 5))
            P[int(y), int(np.clip(x - d * L * 0.15, 0, P.shape[1] - 1))] = f["col2"]
            poly(P, [(x - d * L * 0.4, y), (x - d * L * 0.8, y - 1 + wig), (x - d * L * 0.8, y + 1 + wig)], col, a=0.8)
            return
        ry = L * (0.42 if kind == "angel" else 0.3 if kind == "gold" else 0.22)
        tail = L * (1.1 if kind == "betta" else 0.6)
        tx = x - d * L * 0.42
        poly(P, [(tx + d, y), (tx - d * tail, y - tail * 0.6 + wig), (tx - d * tail * 0.8, y + wig * 0.5), (tx - d * tail, y + tail * 0.6 + wig)],
             mix(col, (255, 255, 255), 0.15), a=0.75 if kind == "betta" else 0.9)
        if kind == "angel":                                  # tall triangular fins
            poly(P, [(x - d * L * 0.2, y - ry), (x + d * L * 0.1, y - ry), (x - d * L * 0.55, y - L * 0.95 + wig * 0.5)], col, a=0.85)
            poly(P, [(x - d * L * 0.2, y + ry), (x + d * L * 0.1, y + ry), (x - d * L * 0.55, y + L * 0.95 + wig * 0.5)], col, a=0.85)
        if kind == "betta":
            poly(P, [(x - d * L * 0.3, y - ry), (x + d * L * 0.2, y - ry), (x - d * L * 0.6, y - L * 0.6 + wig)], col, a=0.7)
        dark = mix(col, (10, 14, 40), 0.4)
        if kind not in ("angel", "betta"):                   # dorsal and belly fins
            poly(P, [(x - d * L * 0.25, y - ry * 0.8), (x + d * L * 0.12, y - ry * 0.8), (x - d * L * 0.3, y - ry * 1.7 + wig * 0.4)],
                 mix(col, (255, 255, 255), 0.15), a=0.85)
            poly(P, [(x - d * L * 0.05, y + ry * 0.7), (x + d * L * 0.15, y + ry * 0.7), (x - d * L * 0.15, y + ry * 1.45)], dark, a=0.8)
        disc(P, x, y, L / 2, col, ry=ry)
        disc(P, x - d * L * 0.05, y - ry * 0.5, L * 0.36, dark, a=0.5, ry=ry * 0.45)          # darker back
        disc(P, x + d * L * 0.05, y + ry * 0.4, L * 0.32, light, a=0.6, ry=ry * 0.4)           # pale belly
        if L >= 9:                                           # gill line and a pectoral fin on the bigger fish
            line(P, x + d * L * 0.16, y - ry * 0.55, x + d * L * 0.14, y + ry * 0.55, dark, wid=0.8, a=0.6)
            poly(P, [(x + d * L * 0.08, y + ry * 0.1), (x - d * L * 0.16, y + ry * 0.2 + wig * 0.5), (x - d * L * 0.1, y + ry * 0.75)],
                 light, a=0.7)
        if kind == "angel":
            for k in (-0.1, 0.18):
                line(P, x + d * L * k, y - ry * 0.9, x + d * L * k, y + ry * 0.9, (30, 30, 40), wid=1, a=0.55)
        ex, ey = int(x + d * L * 0.3), int(y - ry * 0.2)
        if 0 <= ex < P.shape[1] and 0 <= ey < P.shape[0]:
            P[ey, ex] = (16, 16, 26)

    def v_aquarium(self, cv, P, h, w):
        """A fish tank glowing in a dark room: rays of light, caustics on the sand, swaying
        plants, a treasure chest that pops open on the bass, and fish that dart on the beat."""
        e = self.eng
        t = time.time()
        H = P.shape[0]
        V = self.var(AQUA)
        level = float(self.rms.mean())
        fy = H - max(3, int(H * 0.08))
        tw = min(w - 6, max(30, int(w * 0.86)))
        tx0 = (w - tw) // 2
        tx1 = tx0 + tw
        ty0 = max(5, int(H * 0.14))
        ty1 = fy - max(6, int(H * 0.16))
        ws = ty0 + 3
        th = ty1 - ws

        def build():
            B = np.zeros((H, w, 3), np.float32)
            fill_grad(B, 0, fy, [(0, V["wall"][0]), (1, V["wall"][1])])
            fill_grad(B, fy, H, [(0, (40, 30, 34)), (1, (14, 10, 14))])
            glow(B, (tx0 + tx1) / 2, (ty0 + ty1) / 2, tw * 0.62, V["lamp"], 0.16, ry=(ty1 - ty0) * 0.9)
            B[ty1 + 1:fy, tx0 + 2:tx1 - 2] = vgrad(fy - ty1 - 1, [(0, (72, 52, 46)), (1, (44, 30, 28))])[:, None, :]
            mid = (tx0 + tx1) // 2
            for a, b in ((tx0 + 4, mid - 1), (mid + 1, tx1 - 4)):
                if b - a > 4 and fy - ty1 > 5:
                    rect(B, a, ty1 + 3, b, fy - 1, (58, 40, 36))
                    rect(B, a + 1, ty1 + 4, b - 1, fy - 2, (66, 46, 40))
            for kx in (mid - 3, mid + 2):
                rect(B, kx, (ty1 + fy) // 2, kx + 1, (ty1 + fy) // 2 + 1, (200, 170, 110))
            B[ty0:ws, tx0 + 1:tx1 - 1] = _c(V["water"][0]) * 0.35 + _c(V["wall"][0]) * 0.4
            B[ws:ty1, tx0 + 1:tx1 - 1] = vgrad(th, [(0, V["water"][0]), (1, V["water"][1])])[:, None, :]
            prof = ridge1d(tw, 12, 7)
            sb = (ty1 - 2 - th * (0.07 + 0.1 * prof)).astype(int)
            rows = np.arange(H)[:, None].astype(np.float32)
            xs = np.arange(tw)[None, :].astype(np.float32)
            sm = (rows >= sb[None, :]) & (rows < ty1)
            k = np.clip((rows - sb[None, :]) / 6, 0, 1)
            sand = _c(V["sand"][0]) * (1 - k[..., None]) + _c(V["sand"][1]) * k[..., None]
            sand *= (0.9 + 0.1 * np.sin(xs * 0.8 + rows * 1.7))[..., None]
            S = B[:, tx0:tx1]
            S[sm] = sand[sm]
            rng = wrng("aquabuild", h, w)
            for _ in range(max(2, tw // 22)):                 # rocks
                rx = rng.uniform(4, tw - 4)
                r = rng.uniform(2.5, 5)
                g = rng.uniform(70, 120)
                disc(S, rx, sb[int(rx)] + 1, r, (g, g * 0.95, g * 1.05), ry=r * 0.7)
                disc(S, rx - r * 0.3, sb[int(rx)] - r * 0.15, r * 0.5, (g + 40, g + 38, g + 46), a=0.5, ry=r * 0.3)
            B[ty0 - 3:ty0, tx0 - 1:tx1 + 1] = (34, 32, 44)       # lid with its light
            B[ty0, tx0 + 1:tx1 - 1] = V["lamp"]
            B[ty0:ty1 + 1, tx0] = (58, 60, 72)
            B[ty0:ty1 + 1, tx1 - 1] = (58, 60, 72)
            B[ty1, tx0:tx1] = (58, 60, 72)
            return B, sb
        B, sb = self.tex("aqua", (h, w, e.station_idx), build)
        A = getattr(self, "aqua", None)
        if A is None or A["size"] != (h, w, e.station_idx):
            rng = wrng("aqua", h, w, e.station_idx)
            fish = []
            for _ in range(max(2, tw * th // 1000)):
                kind = rng.choice(["gold", "angel", "betta", "gold"])
                fish.append(dict(kind=kind, x=rng.uniform(tx0 + 8, tx1 - 8), y=rng.uniform(ws + 5, ty1 - 10), ty=ws + th / 2,
                                 d=rng.choice((-1, 1)), v=rng.uniform(5, 10), L=rng.uniform(5, 8), col=rng.choice(V["fish"]),
                                 ph=rng.random() * TAU, boost=0.0))
            school = dict(x=tx0 + tw * 0.3, y=ws + th * 0.4, ty=ws + th * 0.4, d=1, v=9.0, boost=0.0,
                          fish=[(rng.uniform(-8, 8), rng.uniform(-4, 4), rng.random() * TAU) for _ in range(max(5, tw // 10))])
            plants = [(rng.choice([rng.uniform(tx0 + 3, tx0 + tw * 0.3), rng.uniform(tx1 - tw * 0.3, tx1 - 3), rng.uniform(tx0 + 3, tx1 - 3)]),
                       rng.uniform(th * 0.3, th * 0.75), rng.random() * TAU, rng.random()) for _ in range(max(4, tw // 7))]
            A = self.aqua = dict(size=(h, w, e.station_idx), fish=fish, school=school, plants=plants, bubbles=[],
                                 chest=int(tx0 + tw * rng.uniform(0.55, 0.8)), lid=0.0, hit=0.0, last=t)
        dt = min(0.1, max(0.0, t - A["last"]))
        A["last"] = t
        P[:] = B
        yy, xx = grid(H, w)
        sub = (slice(ws, ty1), slice(tx0 + 1, tx1 - 1))
        Yr, Xr = yy[sub], xx[sub]
        ray = np.clip(np.sin((Xr * 0.55 + Yr * 0.3) / 3.2 + t * 0.35) * np.sin((Xr * 0.21 - Yr * 0.1) / 2.3 - t * 0.2), 0, 1) ** 3
        fade = np.clip(1 - (Yr - ws) / max(1, th), 0, 1)
        P[sub] += (ray * fade * (0.14 + 0.2 * self.bass + 0.2 * self.kick))[..., None] * _c(V["lamp"]) * 0.6
        P[sub] += self.tint * (0.035 * (0.5 + self.change))           # the water takes the chord's colour
        C = self.tex("aquac", (H, tw), lambda: noise_tex(H, tw, 3, 5))
        caus = np.clip(1.2 - np.abs(np.roll(C, int(t * 3) % tw, axis=1) + np.roll(C[::-1], -int(t * 2) % tw, axis=1)), 0, 1) ** 4
        rows = np.arange(H)[:, None]
        sm = (rows >= sb[None, :] - 1) & (rows < ty1) & (rows >= ws)
        T = P[:, tx0:tx1]
        T[sm] += (caus[sm] * (50 + 160 * level))[:, None] * _c(V["lamp"]) / 255
        if e.fire_on:
            glow(P, 0, fy - 6, w * 0.3, (255, 150, 80), 0.22 * (0.9 + 0.1 * math.sin(t * 9)))
        # plants
        lo, hi = _c(V["plant"][0]), _c(V["plant"][1])
        for px, hgt, ph, shade in A["plants"]:
            base = sb[int(np.clip(px - tx0, 0, tw - 1))]
            n = int(hgt)
            for k in range(n):
                u = k / max(1, n)
                x = px + (math.sin(t * 1.1 + ph + k * 0.18) * (1.2 + 3 * self.bass) + math.sin(t * 0.5 + ph) * 1.2) * u
                y = base - k
                xi = int(x)
                if ws < y < H and tx0 < xi < tx1 - 2:
                    c = lo + (hi - lo) * (0.3 + 0.7 * u) * (0.7 + 0.3 * shade)
                    P[y, xi] = c
                    P[y, xi + 1] = c * 0.8
                    if k % 4 == 2 and tx0 + 2 < xi - 1 and xi + 2 < tx1 - 1:
                        P[y, xi + (2 if k % 8 == 2 else -1)] = c * 0.9
        theme = self.theme()
        if theme:                                    # an ornament on the sand: a shrine gate, a castle, a lost sax
            ox = int(tx0 + tw * 0.3)
            ob = int(sb[int(np.clip(ox - tx0, 0, tw - 1))]) + 1
            if theme == "japan":
                self.torii(P, ox, ob, min(14.0, th * 0.4))
                self.pagoda(P, ox + 12, ob, min(16.0, th * 0.45), (150, 130, 140), (60, 40, 60), tiers=3)
            elif theme == "english":                 # the classic aquarium castle, a bubble stream from its keep
                cs = min(18.0, th * 0.5)
                self.castle(P, ox, ob, cs, (150, 146, 136), lit=(40, 50, 60), flag=(180, 50, 60), t=t)
                rect(P, ox - cs * 0.9, ob - 1, ox + cs * 0.9, ob + 1, V["sand"][0])
                if random.random() < 0.15:
                    A["bubbles"].append([ox + random.uniform(-1, 1), ob - cs * 0.8, random.uniform(0.4, 1.0),
                                         random.uniform(5, 9), random.random() * TAU])
            else:
                ss = min(24.0, th * 0.62)
                self.sax(P, ox - ss * 0.3, ob - ss * 0.95, ss, (226, 180, 80))
                rect(P, ox - 6, ob - 2, ox + 8, ob + 1, V["sand"][0])           # half buried
                self.notes(cv, ox + 3, (ob - 18) // 2, (255, 220, 160), n=3, rise=6, seed=11)
        # treasure chest: the lid pops on heavy bass and lets out a puff of bubbles
        cx = A["chest"]
        cb = int(sb[int(np.clip(cx - tx0, 0, tw - 1))]) + 1
        A["lid"] = max(float(np.clip((self.bass - 0.6) * 3, 0, 1)), 0.5 * self.kick * self.energy, A["lid"] * 0.9)
        lid = int(round(A["lid"] * 2))
        glow(P, cx, cb - 4, 4, (255, 210, 120), A["lid"] * 0.7)
        rect(P, cx - 4, cb - 4, cx + 5, cb, (110, 70, 40))
        rect(P, cx - 4, cb - 3, cx + 5, cb - 2, (220, 180, 80))
        rect(P, cx, cb - 3, cx + 1, cb - 1, (255, 230, 140))
        rect(P, cx - 4, cb - 6 - lid, cx + 5, cb - 4 - lid, (130, 84, 48))
        rect(P, cx - 4, cb - 6 - lid, cx + 5, cb - 5 - lid, (220, 180, 80))
        if A["lid"] > 0.5 and random.random() < 0.5:
            A["bubbles"].append([cx + random.uniform(-3, 3), cb - 5, random.uniform(0.4, 1.2), random.uniform(6, 12), random.random() * TAU])
        # fish
        pace = 0.7 + level * 5
        if (self.bass > 0.72 or "snare" in self.fired and random.random() < 0.3) and t - A["hit"] > 0.6 and A["fish"]:
            A["hit"] = t
            random.choice(A["fish"] + [A["school"]])["boost"] = 4.0
        lo_y, hi_y = ws + 4, ty1 - th * 0.22

        def swim(f, margin):
            f["boost"] = max(0.0, f["boost"] - dt * 3)
            f["x"] += f["d"] * f["v"] * pace * (1 + f["boost"]) * dt
            if f["x"] < tx0 + margin:
                f["d"] = 1
            elif f["x"] > tx1 - margin:
                f["d"] = -1
            elif random.random() < dt * 0.04:
                f["d"] = -f["d"]
            if abs(f["y"] - f["ty"]) < 1 or random.random() < dt * 0.1:
                f["ty"] = random.uniform(lo_y, hi_y)
            f["y"] += (f["ty"] - f["y"]) * min(1.0, dt * 0.6)
        sc = A["school"]
        if "change" in self.fired and random.random() < 0.4 and tx0 + 20 < sc["x"] < tx1 - 20:
            sc["d"] = -sc["d"]                                         # the school wheels round on a chord change
        swim(sc, 12)
        for dx, dy, ph in sc["fish"]:
            f = dict(kind="tetra", x=sc["x"] + dx + math.sin(t * 1.3 + ph) * 1.5, y=sc["y"] + dy + math.sin(t * 0.9 + ph) * 0.8,
                     d=sc["d"], L=3.2, col=V["tetra"][0], col2=V["tetra"][1], ph=ph)
            self._fish(P, f, t)
        for f in A["fish"]:
            swim(f, 8)
            self._fish(P, f, t)
        ev = self.rare(26.0)
        if ev:
            self._rare_aqua(P, ev, tx0, tx1, ws, ty1)
        # bubbles from the air stone
        stone = tx0 + 5
        rect(P, stone - 2, ty1 - 3, stone + 2, ty1 - 1, (150, 150, 166))
        nb = (5 + 30 * level) * dt
        for _ in range(int(nb) + (random.random() < nb % 1)):
            A["bubbles"].append([stone + random.uniform(-1, 1), ty1 - 4, random.uniform(0.3, 1.2), random.uniform(10, 18), random.random() * TAU])
        if "note" in self.fired:                                       # a bubble for each melody note: high notes are small
            A["bubbles"].append([stone + random.uniform(-1, 1), ty1 - 4, 1.3 - 1.1 * self.pitch, 14.0, random.random() * TAU])
        keep = []
        for b in A["bubbles"]:
            b[1] -= b[3] * dt
            b[0] += math.sin(t * 3 + b[4]) * 0.12
            if b[1] > ws + 1:
                keep.append(b)
                cv.put(int(b[1]) // 2, int(b[0]), "°" if b[2] < 0.6 else "o" if b[2] < 1.0 else "O", (206, 234, 255))
        A["bubbles"] = keep[-300:]
        # the surface shimmers (more with the waves toggle)
        amp = 0.5 + (1.5 if e.waves_on else 0) + level * 3 + self.hat
        xs = np.arange(tx0 + 1, tx1 - 1)
        sh = np.sin(xs * 0.4 + t * 2.4) * amp
        P[ws, xs] = _c(V["water"][0]) * 1.3 + 30 + sh[:, None] * 10
        P[ws + 1, xs] += np.clip(sh, 0, None)[:, None] * 12
        for k in range(2):
            gx = tx0 + tw * 0.08 + k * 4
            line(P, gx, ty0 + 2, gx + th * 0.35, ty1 - 2, (255, 255, 255), wid=1, a=0.05)

    def _rare_aqua(self, P, ev, tx0, tx1, ws, ty1):
        """Rare event: a pufferfish potters across the tank and blows itself up into a spiky ball
        on a heavy bass hit (and once anyway, halfway), then slowly deflates."""
        u, r = ev
        T = self.rclock
        m = self.rare_mem
        dt = min(0.1, max(0.0, T - m.get("t", T)))
        m["t"] = T
        puff = m.get("puff", 0.0)
        if self.bass > 0.78 or 0.48 < u < 0.5:
            puff = 1.0
        puff = max(0.0, puff - dt * 0.22)
        m["puff"] = puff
        f = r.choice((-1, 1))
        tw = tx1 - tx0
        x = (tx0 - 8 + u * (tw + 16)) if f > 0 else (tx1 + 8 - u * (tw + 16))
        y = ws + (ty1 - ws) * r.uniform(0.3, 0.55) + math.sin(T * 0.8) * 2
        rad = max(3.0, (ty1 - ws) * 0.085) * (1 + 0.7 * puff)
        V = P[ws + 1:ty1, tx0 + 1:tx1 - 1]                              # stays inside the glass
        x, y = x - tx0 - 1, y - ws - 1
        skin, belly = (222, 196, 120), (246, 240, 222)
        tail = x - f * rad
        poly(V, [(tail, y), (tail - f * rad * 0.7, y - rad * 0.5 + math.sin(T * 6) * 0.6),
                 (tail - f * rad * 0.7, y + rad * 0.5 + math.sin(T * 6) * 0.6)], mix(skin, (0, 0, 0), 0.2))
        if puff > 0.3:                                                  # spines
            for j in range(14):
                a = j * TAU / 14 + 0.2
                line(V, x + math.cos(a) * rad * 0.9, y + math.sin(a) * rad * 0.9, x + math.cos(a) * rad * (1 + 0.35 * puff),
                     y + math.sin(a) * rad * (1 + 0.35 * puff), (200, 176, 110), wid=0.6)
        disc(V, x, y, rad, skin, ry=rad * (0.8 + 0.2 * puff))
        disc(V, x, y + rad * 0.35, rad * 0.85, belly, ry=rad * 0.45)
        for j in range(5):                                              # spots on its back
            disc(V, x + (j - 2) * rad * 0.32, y - rad * (0.35 + 0.15 * (j % 2)), max(0.5, rad * 0.11), (120, 96, 60))
        ex, ey = x + f * rad * 0.5, y - rad * 0.2
        disc(V, ex, ey, max(0.8, rad * 0.22), (250, 250, 250))
        disc(V, ex + f * 0.3, ey, max(0.5, rad * 0.13), (20, 20, 30))
        fin = math.sin(T * 9) * rad * 0.3                               # little pectoral fin, buzzing
        line(V, x + f * rad * 0.1, y + rad * 0.1, x - f * rad * 0.25, y + rad * 0.1 + fin, mix(skin, (255, 255, 255), 0.3), wid=0.8)
