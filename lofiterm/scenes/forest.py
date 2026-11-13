import math
import random
import time

import numpy as np

from ..paint import _box, _c, disc, fill_grad, fog_rows, glow, grid, hash01, line, make_stars, mix, noise_tex, poly, rect, vgrad
from ..world import wrng
from ..term import C_BROWN, C_DARK, C_DIM, C_FIRE, C_YELLOW
from ..theory import TAU


FOREST = [  # misty night, midnight, autumn dusk, winter, aurora, firefly garden
    dict(sky=[(7, 9, 28), (22, 30, 66), (60, 74, 110)], far=[(96, 104, 150), (52, 62, 100)], mid=[(46, 58, 94), (32, 44, 72)],
         fog=(84, 96, 138), back=[(34, 50, 76), (20, 30, 52), (66, 86, 120)], front=[(40, 84, 68), (16, 40, 40), (88, 128, 126)],
         aurora=False, flies=1.0),
    dict(sky=[(2, 3, 14), (8, 12, 38), (26, 32, 66)], far=[(58, 64, 104), (32, 38, 70)], mid=[(28, 34, 62), (18, 24, 46)],
         fog=(44, 52, 90), back=[(20, 30, 52), (12, 18, 34), (48, 60, 96)], front=[(26, 58, 52), (10, 26, 28), (70, 100, 120)],
         aurora=False, flies=1.0),
    dict(sky=[(44, 32, 76), (150, 82, 96), (244, 150, 96)], far=[(160, 100, 116), (110, 70, 94)], mid=[(96, 62, 78), (66, 44, 60)],
         fog=(214, 136, 116), back=[(84, 62, 54), (54, 40, 40), (160, 110, 90)], front=[(110, 92, 40), (58, 46, 24), (230, 160, 90)],
         aurora=False, flies=0.5),
    dict(sky=[(14, 18, 40), (40, 50, 84), (96, 110, 146)], far=[(130, 140, 180), (80, 92, 130)], mid=[(70, 82, 118), (50, 62, 94)],
         fog=(130, 140, 176), back=[(46, 62, 88), (28, 40, 62), (96, 116, 150)], front=[(44, 80, 74), (18, 42, 44), (120, 150, 160)],
         aurora=False, flies=0.0),
    dict(sky=[(2, 6, 18), (6, 22, 38), (16, 48, 60)], far=[(40, 76, 96), (22, 44, 62)], mid=[(20, 44, 58), (12, 28, 40)],
         fog=(30, 80, 90), back=[(18, 40, 52), (10, 24, 34), (60, 140, 130)], front=[(22, 64, 58), (8, 26, 28), (80, 200, 170)],
         aurora=True, flies=1.0),
    dict(sky=[(3, 14, 22), (12, 40, 52), (40, 90, 96)], far=[(70, 120, 120), (40, 76, 82)], mid=[(34, 66, 70), (22, 46, 52)],
         fog=(60, 110, 110), back=[(22, 52, 54), (12, 32, 36), (70, 130, 120)], front=[(36, 90, 64), (14, 44, 36), (110, 190, 150)],
         aurora=False, flies=2.5),
    dict(sky=[(84, 156, 234), (148, 198, 246), (212, 236, 252)], far=[(150, 182, 204), (112, 152, 172)], mid=[(92, 142, 120), (70, 120, 100)],
         fog=(200, 226, 242), back=[(70, 122, 80), (46, 90, 60), (142, 192, 122)], front=[(56, 122, 78), (22, 62, 56), (178, 216, 120)],
         aurora=False, flies=0.0),
    dict(sky=[(18, 10, 40), (80, 40, 96), (210, 120, 150)], far=[(150, 96, 140), (100, 64, 110)], mid=[(80, 50, 92), (56, 36, 70)],
         fog=(200, 130, 170), back=[(54, 44, 76), (34, 28, 52), (150, 100, 150)], front=[(46, 70, 70), (20, 34, 40), (230, 150, 190)],
         aurora=False, flies=0.6),                                                   # tokyo: shrine path at dusk
    dict(sky=[(8, 6, 12), (30, 20, 26), (76, 50, 46)], far=[(84, 64, 64), (54, 42, 46)], mid=[(44, 36, 40), (30, 24, 30)],
         fog=(110, 84, 70), back=[(36, 34, 36), (22, 20, 24), (110, 84, 60)], front=[(40, 54, 44), (16, 24, 22), (200, 150, 90)],
         aurora=False, flies=2.0),                                                    # jazz: smoky amber night
    dict(sky=[(8, 10, 16), (26, 32, 42), (70, 78, 86)], far=[(70, 80, 84), (48, 56, 62)], mid=[(36, 46, 44), (24, 32, 32)],
         fog=(120, 126, 126), back=[(30, 40, 36), (18, 24, 22), (120, 120, 100)], front=[(38, 58, 40), (14, 24, 18), (210, 180, 120)],
         aurora=False, flies=0.8)                                                     # english: misty moor
]


# broadleaf trees among the pines, per station: (share of the trees, [(lit, shade, rim), …])
LEAFY = {
    2: (0.7, [((226, 128, 50), (120, 54, 34), (255, 200, 100)), ((204, 76, 44), (104, 36, 34), (255, 150, 90)),
              ((236, 176, 60), (132, 84, 30), (255, 226, 130))]),                     # autumn maples and birches
    5: (0.3, [((50, 112, 82), (18, 52, 44), (130, 204, 150))]),
    6: (0.45, [((118, 178, 76), (36, 84, 62), (210, 236, 140)), ((150, 186, 82), (54, 98, 58), (226, 238, 150)),
              ((92, 160, 86), (30, 74, 66), (190, 230, 150))]),        # oak, lime and beech greens in the sun
    7: (0.5, [((240, 160, 196), (150, 80, 130), (255, 216, 234)), ((250, 190, 214), (164, 96, 140), (255, 232, 242))]),
    9: (0.6, [((62, 84, 50), (22, 34, 24), (190, 172, 112)), ((78, 88, 52), (30, 38, 24), (204, 180, 116))]),
}


class ForestScene:
    def v_forest(self, cv, P, h, w):
        t = time.time()
        H = P.shape[0]
        gy = h - 3                                   # ground row (cells)
        gp = 2 * gy                                  # ground row (pixels)

        def camp(r):                                 # where the campfire burns and which side the tent is on
            cx_ = int(w * r.uniform(0.36, 0.64))
            ts = r.choice((-1, 1))
            if not (0 <= cx_ + ts * 26 < w):
                ts = -ts
            return cx_, ts
        cx, ts = self.layout("camp", h, w, camp)
        f = getattr(self, "forest", None)
        if f is None or f["size"] != (h, w):
            rng = wrng("forest", h, w)
            xs = np.arange(w)
            p = [rng.random() * TAU for _ in range(6)]
            q = [rng.uniform(0.6, 1.5) for _ in range(6)]           # how jagged each range is
            hf, hm = rng.uniform(0.32, 0.46), rng.uniform(0.18, 0.28)
            far = gp - (H * hf + H * 0.1 * q[0] * np.sin(xs * 0.045 * q[1] + p[0]) + H * 0.06 * np.sin(xs * 0.12 * q[2] + p[1])
                        + H * 0.025 * q[3] * np.sin(xs * 0.37 + p[2]))
            mid = gp - (H * hm + H * 0.06 * np.sin(xs * 0.07 * q[4] + p[3]) + H * 0.04 * np.sin(xs * 0.19 * q[5] + p[4])
                        + H * 0.015 * np.sin(xs * 0.6 + p[5]))
            bg0, bg1 = rng.randint(3, 5), rng.randint(6, 9)          # how dense the two rows of pines are
            fg0, fg1 = rng.randint(8, 13), rng.randint(14, 21)
            th0, th1 = rng.uniform(0.18, 0.26), rng.uniform(0.34, 0.46)
            back, x = [], rng.randint(0, 3)
            while x < w + 4:
                if abs(x - cx) > 5:
                    back.append((x, rng.randint(max(8, int(H * th0 * 0.8)), max(9, int(H * th1 * 0.85))), rng.random()))
                x += rng.randint(bg0, bg1)
            front, x = [], rng.randint(0, 6)
            while x < w + 6:
                if abs(x - cx) > 14:
                    # the clearing opens around the fire: small trees beside it, giants framing the edges
                    far_ = min(1.0, abs(x - cx) / (w * 0.42))
                    hg = rng.uniform(H * (th0 + 0.2), H * (th1 + 0.26)) * (0.62 + 0.75 * far_ ** 1.3)
                    front.append((x, max(12, int(hg)), rng.random() * TAU, rng.random()))
                x += rng.randint(fg0, fg1)
            bushes = [x for x in range(2, w - 2, 4) if abs(x - cx) > 13 and rng.random() < 0.4]
            owl = back[rng.randrange(len(back))] if back else (cx, 12)
            flies = [[rng.uniform(1, w - 1), rng.uniform(H * 0.42, gp - 2), rng.random() * TAU]
                     for _ in range(3 * max(6, w * h // 90))]
            old = f["depth"] if f is not None and len(f["depth"]) == w else np.zeros(w)
            f = self.forest = dict(size=(h, w), far=far, mid=mid, back=back, front=front, bushes=bushes, owl=owl,
                                   stars=make_stars(rng, int(gp * 0.55), w, w * h // 36), flies=flies,
                                   sparks=[], drops=[], splash=[], depth=old, mist=noise_tex(H, w, 6, 3),
                                   moon=(w * rng.choice((rng.uniform(0.06, 0.3), rng.uniform(0.7, 0.94))), H * rng.uniform(0.09, 0.2)))
        level = float(self.rms.mean())
        snowing = self.eng.st.get("weather") == "snow"
        depth = f["depth"]
        if snowing:
            depth += 0.00015                         # ~2 min to a full white blanket
            depth[:] = 0.96 * depth + 0.02 * (np.roll(depth, 1) + np.roll(depth, -1))
        else:
            depth -= 0.003
        warm = np.abs(np.arange(w) - cx) <= 6
        depth[warm] = np.minimum(depth[warm], 0.12)  # the fire melts it
        np.clip(depth, 0, 1.2, out=depth)
        cover = float(np.clip(depth.mean() * 1.15, 0, 1))
        heat = 0.35 + self.bass + (0.6 if self.eng.fire_on else 0.0)
        yy, xx = grid(H, w)
        Y = yy[:gp]
        V = self.var(FOREST)
        leafy = LEAFY.get(self.eng.station_idx, (0.0, []))

        # ── the static backdrop (sky, ranges, haze, back pines) is cached; only the sky animates ──
        key = (h, w, self.eng.station_idx, int(cover * 20))
        if f.get("bg_key") != key:
            sky = np.zeros_like(P)
            fill_grad(sky, 0, gp, list(zip((0, 0.5, 1), V["sky"])))
            fog_rows(sky, gp * 0.45, gp, V["fog"], 0.4)
            fog_rows(sky, gp * 0.55, gp, V["fog"], 0.3)
            B = np.zeros_like(P)
            fill_grad(B, 0, gp, list(zip((0, 0.5, 1), V["sky"])))
            m = Y >= f["far"][None, :]
            g = vgrad(gp, [(0, V["far"][0]), (1, V["far"][1])])
            B[:gp][m] = np.broadcast_to(g[:, None, :], (gp, w, 3))[m]
            tmin = f["far"].min()
            hf = np.clip((tmin + H * 0.13 - f["far"]) / (H * 0.13), 0, 1)
            thick = 1 + hf * min(7.0, H * 0.06) + cover * 4
            depth_in = Y - f["far"][None, :]
            cap = m & (depth_in < thick[None, :]) & ((hf[None, :] > 0) | (cover > 0.05)) \
                & (hash01(xx[:gp], Y) < 0.45 + 0.55 * np.clip(1 - depth_in / thick[None, :], 0, 1) + cover * 0.3)
            litside = (np.gradient(f["far"]) < 0)[None, :]
            B[:gp][cap & litside] = (218, 226, 246)
            B[:gp][cap & ~litside] = (160, 170, 206)
            fog_rows(B, gp * 0.45, gp, V["fog"], 0.4)
            m = Y >= f["mid"][None, :]
            g = vgrad(gp, [(0, V["mid"][0]), (1, V["mid"][1])])
            B[:gp][m] = np.broadcast_to(g[:, None, :], (gp, w, 3))[m]
            fog_rows(B, gp * 0.55, gp, V["fog"], 0.3)
            lift = max(2, int(H * 0.045))                # the far row stands up the slope, behind the mist
            for i, (x, hg, u_) in enumerate(f["back"]):
                if u_ < leafy[0]:
                    lt, sh, rm = leafy[1][i % len(leafy[1])]
                    self.broadleaf(B, x, gp - lift, int(hg * 0.8), mix(lt, V["fog"], 0.55), mix(sh, V["fog"], 0.45),
                                   snow=cover * 0.8, rim=mix(rm, V["fog"], 0.5), trunk=mix((70, 52, 44), V["fog"], 0.5), seed=i)
                else:
                    self.pine(B, x, gp - lift, hg, V["back"][0], V["back"][1], snow=cover * 0.8, rim=V["back"][2],
                              trunk=(26, 28, 44), seed=i, fat=0.8 + 0.5 * u_)
            B[gp - lift:gp] = B[gp - lift:gp] * 0.55 + _c(V["back"][1]) * 0.45
            f["bg"], f["sky"], f["bg_key"] = B, sky, key
            f["land"] = np.abs(B[:gp] - sky[:gp]).sum(-1) > 0.5
        S = f["sky"][:gp].copy()
        self.stars(S, f["stars"], t, 0.4 if V["sky"][2][0] > 200 else 1.0)
        self.sky_body(S, f["moon"][0], f["moon"][1], max(3.0, H * 0.065))
        if V["aurora"]:                                  # northern lights: curtains of green and violet
            X = xx[:gp]
            drive = 0.55 + 0.35 * self.energy + 0.4 * self.bass + 0.5 * self.kick
            for j, (yc, sp) in enumerate(((0.3, 0.22), (0.44, -0.16))):
                cen = gp * yc + np.sin(X * 0.035 + t * sp + j * 2) * H * 0.09 + np.sin(X * 0.11 - t * 0.3 + j) * H * 0.035
                dy = Y - cen
                band = np.where(dy < 0, np.exp(dy / (H * 0.2)), np.exp(-(dy / (H * 0.02)) ** 2))   # long rays, a crisp hem
                env = np.clip(0.5 + 0.65 * np.sin(X * 0.023 + j * 1.7 + t * 0.07) + 0.25 * np.sin(X * 0.07 + j), 0, 1)
                rays = 0.45 + 0.55 * np.clip(np.sin(X * 0.55 + np.sin(X * 0.13 + t * 0.6) * 2.5 + j), 0, 1)
                k = np.clip(-dy / (H * 0.22), 0, 1)[..., None]
                S += (band * env * rays * drive * (1.5 - 0.5 * j))[..., None] * (_c((60, 255, 170)) * (1 - k) + _c((190, 80, 255)) * k)
        self.chord_wash(S, 0, int(gp * 0.6), 0.05)        # the night sky takes on each chord's colour
        P[:] = f["bg"]
        P[:gp] = np.where(f["land"][..., None], P[:gp], S)
        ox, oh = f["owl"][:2]
        oy = (gp - oh + oh // 2) // 2
        blink = (t % 5.0) < 0.18 or self.change > 0.85     # the owl blinks at every chord change
        cv.text(oy, ox - 1, "-.-" if blink else "°v°", C_DARK if blink else C_YELLOW)
        if self.theme() == "english":                # a thatched cottage up the clearing, behind the camp
            cs_ = min(gp * 0.34, 17.0)
            self.cottage(P, cx - ts * 4, gp - max(2.0, gp * 0.05), cs_, wall=(170, 162, 146), thatch=(110, 88, 58),
                         glow_k=0.8 + self.bass * 0.4, smoke=1.0, t=self.rclock)
        # ground mist drifting between the rows
        n = np.roll(f["mist"], int(t * 2) % w, axis=1)[:gp]
        a = np.clip(n * 0.35 + 0.15, 0, 0.55) * np.exp(-((Y - (gp - 5)) / 5) ** 2)
        P[:gp] += (_c(V["fog"]) * 1.15 - P[:gp]) * a[..., None]
        # ── front pines sway with the volume ──
        amp = 0.5 + 2.5 * level
        for i, (x, hg, ph, u_) in enumerate(f["front"]):
            if u_ < leafy[0]:
                lt, sh, rm = leafy[1][i % len(leafy[1])]
                self.broadleaf(P, x, gp, int(hg * 0.78), lt, sh, amp * math.sin(t * 0.7 + ph), snow=cover, rim=rm, seed=100 + i,
                               trunk=(96, 70, 52) if self.noon() else (58, 40, 36))
            else:
                self.pine(P, x, gp, hg, V["front"][0], V["front"][1], amp * math.sin(t * 0.7 + ph), snow=cover,
                          rim=V["front"][2], seed=100 + i, fat=0.78 + 0.5 * u_)
        for x in f["bushes"]:
            col = (220, 226, 240) if depth[x] > 0.7 else (36, 74, 58)
            disc(P, x, gp - 1.2, 2.4, col, ry=1.8)
            disc(P, x - 0.8, gp - 1.8, 1.2, (60, 100, 80) if depth[x] <= 0.7 else (240, 244, 252), a=0.6)

        th = self.theme()
        if th == "japan":                            # a shrine gate up the path, a stone lantern beside it
            gx = cx + ts * 30
            self.torii(P, gx, gp, min(gp * 0.42, 22.0))
            lx = gx + ts * 10
            rect(P, lx - 1, gp - 6, lx + 2, gp, (120, 118, 130))
            rect(P, lx - 2.5, gp - 9, lx + 3.5, gp - 6, (150, 146, 156))
            rect(P, lx - 1, gp - 8.5, lx + 2, gp - 6.5, (255, 200, 120))
            glow(P, lx + 0.5, gp - 7.5, 3, (255, 170, 90), 0.45 + self.bass * 0.2 + self.pluck * 0.3)
            poly(P, [(lx - 3.5, gp - 9), (lx + 0.5, gp - 12), (lx + 4.5, gp - 9)], (120, 116, 128))

        ev = self.rare(34.0)
        if ev:
            self._rare_forest(P, ev, gp, w, H, cx, ts, V)

        # ── campsite: tent, bench ──
        m_ = -ts                                     # +1: the fire is to the tent's right
        tx = cx + ts * 12
        roof_snow = 0.75 if cover > 0.5 else 0.0
        poly(P, [(tx, gp - 12), (tx - 6 * m_, gp - 12), (tx - 14 * m_, gp - 0.5), (tx - 8 * m_, gp)], (104, 62, 54))
        tent = poly(P, [(tx - 8, gp), (tx, gp - 12.5), (tx + 8, gp)], (176, 110, 82))
        b = _box(P, tx - 8, gp - 13, tx + 9, gp + 1)
        if b is not None and tent is not None:
            v, _, bx = b
            k_ = np.clip((m_ * (bx - tx) + 8) / 16, 0, 1)                 # brighter toward the fire
            v[tent] *= (0.7 + 0.55 * k_[tent])[:, None]
            if roof_snow:
                v[tent] += (_c((230, 234, 246)) - v[tent]) * roof_snow * 0.8
        poly(P, [(tx - 2.5 * m_, gp), (tx + 0.5 * m_, gp - 7), (tx + 3.5 * m_, gp)], (38, 22, 24))
        glow(P, tx + 0.5 * m_, gp - 2, 2.5, (255, 170, 90), 0.35)
        b0, b1 = sorted((cx + 7 * m_, cx + 14 * m_))
        b0, b1 = max(0, b0), min(w, b1)
        if b0 < b1:
            P[gp - 3:gp - 1, b0:b1] = (100, 62, 44)
            P[gp - 3, b0:b1] = (146, 96, 66)
        disc(P, cx + 14 * m_, gp - 2, 1.2, (176, 124, 82))
        if th == "jazz":                             # a gramophone on the bench, playing to the fire
            gx = cx + 11 * m_
            rect(P, gx - 2, gp - 6, gx + 2, gp - 3, (110, 64, 40))
            rect(P, gx - 2, gp - 6, gx + 2, gp - 5, (150, 96, 60))
            line(P, gx, gp - 6, gx + 1.5 * m_, gp - 9, (200, 150, 60), wid=0.8)
            poly(P, [(gx + 1.5 * m_, gp - 9), (gx + 6 * m_, gp - 15), (gx + 8 * m_, gp - 11)], (226, 176, 72))
            disc(P, gx + 7 * m_, gp - 13, 2.3, (250, 206, 110), ry=2.8)
            disc(P, gx + 7.3 * m_, gp - 13, 1.4, (120, 70, 30), ry=1.9)
            self.notes(cv, gx + 8 * m_, (gp - 15) // 2, (255, 214, 140), n=3, rise=6, seed=3)

        # ── ground: spectrum grass or snow drifts ──
        gh = H - gp
        P[gp:] = vgrad(gh, [(0, (34, 58, 44)), (1, (14, 26, 22))])[:, None, :]
        P[gp:] += (f["mist"][gp:] * 5)[..., None] * _c((1, 1.3, 1))
        sn = np.clip((depth - 0.25) / 0.45, 0, 1)
        P[gp:] += (vgrad(gh, [(0, (228, 234, 248)), (1, (168, 178, 210))])[:, None, :] - P[gp:]) * sn[None, :, None]
        dirt = np.clip(1 - np.abs(np.arange(w) - cx) / 7, 0, 1) * (1 - sn)
        P[gp:] += (_c((80, 52, 38)) - P[gp:]) * (dirt * 0.8)[None, :, None]
        lv, _ = self.band_levels(w)
        hb = lv * 7
        grow = (depth < 0.3) | (lv * 1.2 > depth + 0.45)
        for k in range(1, 8):
            mk = (hb >= k) & grow & (np.arange(w) % 2 == 0) & (dirt < 0.3)
            P[gp - k][mk] = _c((86, 146, 92)) * (0.55 + 0.45 * k / 7)

        # ── fire, logs and its light ──
        P[gp - 1, cx - 4:cx + 5] = (90, 56, 40)
        P[gp, cx - 3:cx + 4] = (120, 70, 46)
        emb = np.random.random(9) < 0.35
        P[gp - 1, cx - 4:cx + 5][emb] = (255, 96, 46)
        self.flames(P, cx + 0.5, gp - 1, heat, 4.5, "camp")
        self._fl = fl = 0.8 * getattr(self, "_fl", 1.0) + 0.2 * (0.85 + 0.15 * random.random())
        glow(P, cx, gp - 3, 9 + heat * 6, (255, 128, 50), (0.2 + 0.18 * heat + 0.12 * self.kick) * fl)
        glow(P, cx, gp + 1, 18 + heat * 6, (255, 110, 40), 0.12 * fl, ry=5)

        # ── sparks stream up, drift on the wind, cool and fade (cell coords) ──
        fh = 2 + int(heat * 2.2)
        n_new = 0.6 + heat * 2.0 + (4.0 * self.snare if "snare" in self.fired else 0.0)   # a log cracks on the snare
        for _ in range(int(n_new) + (random.random() < n_new % 1)):
            f["sparks"].append([cx + random.uniform(-fh * 0.5, fh * 0.5), gy - 1 - random.uniform(0, fh),
                                random.uniform(-0.08, 0.08), random.uniform(0.25, 0.6), 0, random.random() * TAU])
        wind = 0.06 + 0.1 * math.sin(t * 0.23)
        alive = []
        for s in f["sparks"]:
            s[4] += 1
            s[1] -= s[3]
            s[3] *= 0.985
            s[0] += s[2] + wind + 0.18 * math.sin(t * 3 + s[5])
            age = s[4]
            if s[1] < 0 or age > 70 or not (0 <= s[0] < w) or random.random() < 0.012:
                continue
            alive.append(s)
            if age < 8:
                ch, col = random.choice("*•"), C_YELLOW
            elif age < 22:
                ch, col = random.choice("*··"), C_FIRE
            elif age < 40:
                ch, col = "·", C_BROWN
            else:
                ch, col = "˙", C_DIM
            cv.put(int(s[1]), int(s[0]), ch, col)
        f["sparks"] = alive[-250:]

        def land(x):
            if snowing:
                depth[x] += 0.04
        self.sky = (0, w, 0, gp // 6)
        self.weather(f, cv, 1, w - 1, 0, gy, on_land=land)
        # ── fireflies blink brighter on the beat (they hide in the snow) ──
        if not snowing:
            flies = f["flies"][: int(len(f["flies"]) / 3 * V["flies"])]
            fx = f.get("fx")
            if fx is None or len(fx) != len(flies):
                fx = f["fx"] = np.zeros(len(flies), np.float32)
            fx *= 0.93
            if "note" in self.fired and len(flies):           # each melody note: a firefly near its pitch flares
                ty = gp - 2 - self.pitch * (gp - 2 - H * 0.35)
                near = sorted(range(len(flies)), key=lambda j: abs(flies[j][1] - ty) + (j * 7 + self.frame) % 5)
                fx[near[0]] = 1.0
            for j, p in enumerate(flies):
                p[2] += 0.08
                p[0] = min(w - 2, max(1, p[0] + math.cos(p[2] * 0.7) * 0.25 + random.uniform(-0.1, 0.1)))
                p[1] = min(gp - 2, max(H * 0.35, p[1] + math.sin(p[2] * 0.5) * 0.24))
                b_ = 0.5 + 0.5 * math.sin(t * 2 + p[2] * 3) + self.bass * 0.4 + self.hat * 0.3 + fx[j] * 1.5
                if b_ > 0.6:
                    k_ = min(1.0, b_ - 0.4)
                    P[int(p[1]), int(p[0])] = _c((236, 255, 150)) * k_
                    glow(P, int(p[0]) + 0.5, int(p[1]) + 0.5, 1.6 + 2 * fx[j], (170, 255, 110), 0.5 * k_ + 0.4 * fx[j])

    def _rare_forest(self, P, ev, gp, w, H, cx, ts, V):
        """Rare event: a deer steps out between the trees, stops to graze (glancing at the camp, eyes catching
        the firelight), then wanders off the other way."""
        u, r = ev
        f = r.choice((-1, 1))
        stop = cx - ts * max(20.0, r.uniform(0.18, 0.3) * w)            # the far side of the fire from the tent
        stop = min(w - 8.0, max(8.0, stop))
        sz = 6.5                                                  # shoulder height: a little under the tent (12 px)
        x0, x2 = (-sz * 2, w + sz * 2) if f > 0 else (w + sz * 2, -sz * 2)
        if u < 0.32:
            k = u / 0.32
            x, moving = x0 + (stop - x0) * (1 - (1 - k) ** 1.6), True
        elif u < 0.68:
            x, moving = stop, False
        else:
            k = (u - 0.68) / 0.32
            x, moving = stop + (x2 - stop) * k ** 1.4, True
        step = x * 0.9 if moving else 0.0
        graze = 0.0 if moving else min(1.0, (u - 0.32) / 0.03, (0.68 - u) / 0.03)
        look = (not moving) and 0.45 < u < 0.53                     # head up, looking at the camp
        if look:
            graze = 0.0
        day = self.noon()
        col = (132, 92, 62) if day else tuple(int(c) for c in _c(mix((120, 84, 60), V["fog"], 0.35)) * 0.62)
        warm = max(0.0, 1 - abs(x - cx) / (w * 0.4))                  # the side facing the fire catches its light
        rim = mix(col, (255, 150, 80), 0.25 + 0.4 * warm) if not day else mix(col, (255, 230, 190), 0.3)
        eye = (255, 210, 140) if look and not day else None
        side = 1 if cx > x else -1
        self.quadruped(P, x + 0.7 * side, gp, sz, f, step, rim, head_dn=graze, kind="deer")    # firelit edge
        self.quadruped(P, x, gp, sz, f, step, col, head_dn=graze, kind="deer", eye=eye)
