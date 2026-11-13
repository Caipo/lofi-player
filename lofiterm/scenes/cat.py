import math
import time

import numpy as np

from ..paint import _box, _c, disc, fill_grad, glow, grid, hash01, line, make_stars, mix, moon, poly, rect, vgrad
from ..world import wrng
from ..theory import SR, TAU
from .shared import BOKEH, COATS, SWEATERS


# sleepy cat's room per station: wall, floor, window sky, rooftops outside, their lit windows, rug, bed (outer, inner,
# rim), curtains, the light falling in, the bedside lamp; dusk = a sunset outside, day = a bright noon
CATROOM = [
    dict(wall=[(22, 20, 46), (40, 34, 66)], floor=[(66, 46, 58), (34, 24, 34)], sky=[(10, 10, 34), (52, 44, 96)], roofs=(18, 16, 36),
         lit=(220, 170, 100), rug=[(120, 80, 120), (168, 124, 156)], bed=[(126, 100, 170), (86, 66, 120), (140, 112, 186)],
         curtain=(150, 104, 150), beam=(170, 190, 255), lamp=(255, 180, 110)),
    dict(wall=[(12, 16, 36), (26, 30, 56)], floor=[(44, 40, 56), (20, 18, 30)], sky=[(4, 6, 22), (20, 26, 64)], roofs=(10, 10, 24),
         lit=(230, 190, 120), rug=[(60, 80, 130), (96, 120, 170)], bed=[(70, 90, 150), (44, 56, 100), (90, 110, 170)],
         curtain=(70, 86, 140), beam=(150, 170, 255), lamp=(255, 200, 130)),
    dict(wall=[(96, 60, 52), (132, 86, 64)], floor=[(122, 80, 54), (72, 46, 34)], sky=[(90, 70, 140), (255, 172, 112)], roofs=(70, 40, 60),
         lit=(255, 210, 130), rug=[(190, 110, 70), (230, 160, 100)], bed=[(200, 120, 90), (150, 84, 64), (222, 142, 102)],
         curtain=(206, 130, 90), beam=(255, 190, 120), lamp=(255, 190, 120), dusk=True),
    dict(wall=[(40, 46, 70), (64, 70, 96)], floor=[(80, 70, 80), (44, 40, 50)], sky=[(30, 36, 62), (110, 120, 156)], roofs=(26, 28, 46),
         lit=(255, 214, 150), rug=[(110, 130, 170), (160, 180, 210)], bed=[(120, 140, 190), (84, 100, 140), (150, 168, 210)],
         curtain=(130, 150, 196), beam=(200, 214, 255), lamp=(255, 206, 150)),
    dict(wall=[(30, 12, 52), (56, 22, 80)], floor=[(52, 24, 64), (24, 10, 34)], sky=[(16, 4, 40), (150, 30, 120)], roofs=(20, 6, 36),
         lit=(80, 255, 230), rug=[(180, 40, 150), (60, 200, 196)], bed=[(150, 60, 200), (96, 36, 140), (180, 80, 230)],
         curtain=(120, 50, 170), beam=(255, 120, 220), lamp=(255, 90, 200)),
    dict(wall=[(14, 34, 36), (28, 56, 54)], floor=[(56, 52, 44), (28, 26, 24)], sky=[(4, 18, 28), (30, 90, 96)], roofs=(8, 24, 28),
         lit=(255, 226, 140), rug=[(70, 130, 110), (120, 180, 150)], bed=[(90, 150, 130), (56, 104, 90), (110, 176, 150)],
         curtain=(80, 140, 120), beam=(180, 255, 230), lamp=(255, 220, 140)),
    dict(wall=[(236, 222, 196), (214, 196, 168)], floor=[(196, 150, 104), (150, 108, 72)], sky=[(90, 160, 240), (192, 228, 255)],
         roofs=(122, 140, 172), lit=(150, 176, 210), rug=[(110, 180, 200), (250, 210, 120)], bed=[(240, 140, 120), (200, 100, 90), (255, 170, 150)],
         curtain=(250, 200, 110), beam=(255, 240, 190), lamp=(255, 240, 200), day=True),
    dict(wall=[(54, 30, 46), (84, 50, 64)], floor=[(150, 130, 84), (96, 82, 54)], sky=[(30, 16, 60), (230, 130, 150)], roofs=(36, 20, 50),
         lit=(255, 200, 150), rug=[(60, 80, 130), (200, 70, 80)], bed=[(200, 70, 80), (150, 46, 60), (230, 100, 110)],
         curtain=(60, 76, 130), beam=(255, 190, 210), lamp=(255, 150, 110)),                       # tokyo: tatami, indigo
    dict(wall=[(36, 18, 22), (58, 30, 32)], floor=[(96, 58, 36), (52, 30, 22)], sky=[(8, 6, 18), (60, 40, 60)], roofs=(14, 10, 18),
         lit=(255, 200, 120), rug=[(110, 30, 40), (200, 150, 70)], bed=[(130, 40, 50), (90, 26, 36), (160, 60, 64)],
         curtain=(120, 40, 46), beam=(255, 200, 130), lamp=(255, 180, 90)),                         # jazz: oxblood, brass
    dict(wall=[(40, 48, 44), (66, 78, 70)], floor=[(104, 72, 46), (58, 38, 26)], sky=[(10, 12, 22), (70, 76, 96)], roofs=(16, 16, 22),
         lit=(255, 206, 130), rug=[(110, 36, 40), (40, 70, 60)], bed=[(60, 90, 70), (40, 62, 50), (190, 170, 120)],
         curtain=(110, 40, 44), beam=(255, 214, 150), lamp=(255, 190, 110))                         # english: library green, claret
]


class CatScene:
    # ── sleepy cat ─────────────────────────────────────────────────────────
    @staticmethod
    def _fur(P, cx, cy, rx, ry, base, stripe, striped, phase=0.0, light=(-0.6, -0.8)):
        """A furry ellipse lit from the window side, with optional tabby stripes and a soft
        darker outline so overlapping parts (head on body) read clearly."""
        b = _box(P, cx - rx - 1, cy - ry - 1, cx + rx + 2, cy + ry + 2)
        if b is None:
            return None
        v, yy, xx = b
        nx, ny = (xx + 0.5 - cx) / rx, (yy + 0.5 - cy) / ry
        d = np.sqrt(nx * nx + ny * ny)
        fuzz = hash01(xx, yy, 5) * 0.12                      # ragged fur edge
        m = d < 1.0 - fuzz * (d > 0.85)
        shade = np.clip(1.0 - 0.28 * (nx * light[0] + ny * light[1]) - 0.25 * d ** 3, 0.5, 1.15)
        col = _c(base) * shade[..., None]
        if striped:
            st = (np.sin((xx - cx) * 0.75 + phase + np.sin((yy - cy) * 0.5) * 1.2) > 0.55) & (ny < 0.55)
            col[st] = _c(stripe) * shade[st][:, None]
        inner = m.copy()
        inner[1:] &= m[:-1]
        inner[:-1] &= m[1:]
        inner[:, 1:] &= m[:, :-1]
        inner[:, :-1] &= m[:, 1:]
        edge = m & ~inner & (ny > -0.2)                       # outline the shadowed underside
        col[edge] *= 0.62
        v[m] = col[m]
        return m

    def _dream(self, P, kind, cx, cy, R, t):
        """What the cat is dreaming about: a fish, a mouse, a butterfly or a ball of yarn."""
        s = math.sin(t * 0.9)
        x = cx + s * R * 0.45
        d = 1 if math.cos(t * 0.9) > 0 else -1
        y = cy + math.sin(t * 2.1) * 0.8
        if kind == 0:                                        # fish
            disc(P, x, y, 2.2, (120, 180, 255), ry=1.3)
            poly(P, [(x - d * 2, y), (x - d * 4, y - 1.6), (x - d * 4, y + 1.6)], (100, 150, 240))
            P[int(y), int(x + d * 1.2)] = (20, 20, 40)
        elif kind == 1:                                      # mouse
            disc(P, x, y + 0.5, 2.2, (150, 146, 160), ry=1.4)
            disc(P, x + d * 1.8, y - 0.6, 0.9, (236, 170, 186))
            line(P, x - d * 2, y + 1, x - d * 4.5, y - 0.5 + math.sin(t * 6), (236, 170, 186), 0.6)
            P[int(y), int(x + d * 2.3)] = (20, 20, 40)
        elif kind == 2:                                      # butterfly
            f = abs(math.sin(t * 7)) * 0.8 + 0.2
            for side in (-1, 1):
                poly(P, [(x, y), (x + side * 2.6 * f, y - 2.4), (x + side * 2.2 * f, y + 0.4)], (255, 170, 90))
                poly(P, [(x, y), (x + side * 1.8 * f, y + 1.8), (x + side * 0.6 * f, y + 0.6)], (255, 120, 150))
            P[int(y) - 1:int(y) + 2, int(x)] = (60, 40, 50)
        else:                                                # yarn ball rolling
            disc(P, x, y, 2.2, (236, 110, 150))
            a = t * 4 * d
            line(P, x + math.cos(a) * 2, y + math.sin(a) * 2, x - math.cos(a) * 2, y - math.sin(a) * 2, (255, 170, 196), 0.6)
            line(P, x - d * 2, y + 1.5, x - d * 5, y + 1.8, (236, 110, 150), 0.5)

    def v_cat(self, cv, P, h, w):
        e = self.eng
        t = time.time()
        H = P.shape[0]
        base, stripe, belly, striped, points = COATS[e.station_idx % len(COATS)]
        R = self.var(CATROOM)
        day, dusk = R.get("day", False), R.get("dusk", False)
        snow = e.st.get("weather") == "snow"
        yy, xx = grid(H, w)
        fy = int(H * 0.7)                                    # floor line
        beats = e.played / SR * e.st["bpm"] / 60
        # ── wall, skirting, floor planks, rug ──
        fill_grad(P, 0, fy, [(0, R["wall"][0]), (1, R["wall"][1])])
        P[:fy][((xx[:fy] + (yy[:fy] // 6) * 3) % 12) == 0] *= 0.92          # faint wallpaper
        P[fy - 3:fy] = _c(R["floor"][0]) * 0.85
        P[fy - 3] = _c(R["floor"][0]) * 1.1
        P[fy:] = vgrad(H - fy, [(0, R["floor"][0]), (1, R["floor"][1])])[:, None, :]
        P[fy:][((yy[fy:] - fy) % 4) == 3] *= 0.78
        P[fy:][(((xx[fy:] + ((yy[fy:] - fy) // 4) * 11) % 23) == 0)] *= 0.82
        CL = self.layout("catroom", h, w, lambda r: dict(
            bx=r.uniform(0.55, 0.64), wx=r.uniform(0.05, 0.13), ww=r.uniform(0.25, 0.33), lx=r.uniform(0.85, 0.9),
            pic=r.randrange(3), pal=r.choice(BOKEH), plant=[(r.uniform(-2, 2), r.uniform(-6.5, -2), r.random() < 0.5)
                                                          for _ in range(r.randint(4, 7))]))
        bx, by = w * CL["bx"], fy + (H - fy) * 0.2           # where the cat sleeps
        rx = max(8.0, min(w * 0.12, H * 0.2))
        ry = rx * 0.55
        r0, r1 = R["rug"]
        for k, col in enumerate([r0, r1, r0, mix(r1, (255, 255, 255), 0.15)]):
            disc(P, bx - rx * 0.1, by + ry * 1.0, rx * (2.0 - 0.35 * k), col, ry=ry * (0.75 - 0.13 * k))
        # ── window: sky by station (stars + moon, a sunset, or a bright noon) ──
        wx0 = int(w * CL["wx"])
        wx1 = wx0 + max(14, int(w * CL["ww"]))
        wy0, wy1 = max(4, int(H * 0.09)), int(H * 0.5)
        W_ = P[wy0 + 1:wy1, wx0 + 1:wx1]
        vh, vw = W_.shape[:2]
        if vh > 4 and vw > 6:
            W_[:] = vgrad(vh, [(0, R["sky"][0]), (1, R["sky"][1])])[:, None, :]
            if day:
                for k in range(3):                           # clouds drifting past
                    cxk = (k * vw * 0.45 + t * (1.0 + k * 0.4)) % (vw + 24) - 12
                    for ox, oy, rr in ((0, 0, 3.2), (3, -1.2, 2.6), (-3, 0.4, 2.4), (5.5, 0.6, 2.0)):
                        disc(W_, cxk + ox, vh * (0.2 + 0.14 * k) + oy, rr, (255, 255, 255), a=0.85, ry=rr * 0.7)
                self.sky_body(W_, vw * 0.78, vh * 0.2, max(2.5, vh * 0.12))
                bxp = (t * 5) % (vw + 20) - 10                # a bird flaps across
                if 0 < bxp < vw - 3:
                    fl_ = int(t * 6) % 2
                    by_ = int(vh * 0.4 + math.sin(t) * 2)
                    W_[by_ - fl_, int(bxp)] = W_[by_ - fl_, int(bxp) + 2] = (60, 60, 80)
                    W_[by_, int(bxp) + 1] = (60, 60, 80)
            else:
                stars = self.tex("catstars", (vh, vw), lambda: make_stars(wrng("catstars"), vh, vw, vw * vh // 12))
                self.stars(W_, stars, t, 0.0 if dusk else (0.35 if e.wet else 1.0))
                if dusk:
                    glow(W_, vw * 0.35, vh * 0.72, vh * 0.8, (255, 150, 90), 0.5)
                    disc(W_, vw * 0.35, vh * 0.72, max(2.5, vh * 0.13), (255, 222, 150))
                else:
                    moon(W_, vw * 0.7, vh * 0.3, max(2.5, vh * 0.13), dim=0.6 if e.wet else 1.0)
            def gen_roofs():                                 # the houses across the street
                r = wrng("catroofs", vh, vw)
                out, x = np.zeros(vw), 0
                while x < vw:
                    bw_, top_ = r.randint(3, 9), vh * r.uniform(0.62, 0.84)
                    pitch = r.uniform(0.4, 0.9) if r.random() < 0.45 else 0.0
                    for j in range(x, min(vw, x + bw_)):
                        out[j] = top_ - (bw_ / 2 - abs(j + 0.5 - x - bw_ / 2)) * pitch
                    if r.random() < 0.3 and x + 1 < vw:      # a chimney
                        out[x + 1] -= 2
                    x += bw_
                return out
            roofs = self.tex("catroofs", (vh, vw), gen_roofs)
            if self.theme() == "japan":                  # a pagoda over the rooftops, Fuji beyond
                poly(W_, [(vw * 0.35, vh), (vw * 0.62, vh * 0.42), (vw * 0.72, vh * 0.42), (vw * 1.05, vh)], mix(R["sky"][1], R["roofs"], 0.6))
                poly(W_, [(vw * 0.62, vh * 0.42), (vw * 0.72, vh * 0.42), (vw * 0.77, vh * 0.52), (vw * 0.57, vh * 0.52)], (240, 226, 236))
                self.pagoda(W_, vw * 0.25, vh * 0.8, vh * 0.62, R["roofs"], mix(R["roofs"], (0, 0, 0), 0.3), tiers=4)
            elif self.theme() == "english":              # a church spire over the rooftops, a castle on the hill
                poly(W_, [(vw * 0.4, vh), (vw * 0.62, vh * 0.6), (vw * 0.9, vh * 0.58), (vw * 1.1, vh)], mix(R["sky"][1], R["roofs"], 0.55))
                self.castle(W_, vw * 0.76, vh * 0.6, vh * 0.26, mix(R["sky"][1], R["roofs"], 0.7), lit=R["lit"])
                self.spire(W_, vw * 0.25, vh * 0.85, vh * 0.75, R["roofs"], lit=R["lit"])
            Yw, _ = grid(vh, vw)
            W_[Yw >= roofs[None, :]] = R["roofs"]
            if snow:
                W_[(Yw >= roofs[None, :]) & (Yw < roofs[None, :] + 1.2)] = (226, 232, 246)
            lit = self.tex("catlit", (vh, vw), lambda: (lambda r: [(int(roofs[x] + r.randint(2, 4)), x) for x in range(1, vw)
                                                                   if r.random() < 0.2])(wrng("catlit", vh, vw)))
            for ly_, lx_ in lit:
                if ly_ < vh and (lx_ * 7 + int(t * 0.2)) % 5:
                    W_[ly_, lx_] = R["lit"]
            if e.wet:                                        # drops running down the glass
                for k in range(vw // 4):
                    gx = (k * 7.3) % vw
                    gy = (t * (3 + k % 3) + k * 13) % (vh + 6) - 3
                    if 0 <= gy < vh - 1:
                        W_[int(gy), int(gx)] += 40
                        if gy >= 1:
                            W_[int(gy) - 1, int(gx)] += 16
        wood = _c(mix(R["floor"][0], (90, 60, 60), 0.4))
        P[wy0, wx0:wx1 + 1] = wood * 1.3
        P[wy1 - 1:wy1 + 1, wx0:wx1 + 1] = wood
        P[wy0:wy1, wx0] = wood
        P[wy0:wy1, wx1] = wood
        P[wy0:wy1, (wx0 + wx1) // 2] = wood
        P[(wy0 + wy1) // 2, wx0:wx1] = wood
        P[wy1:wy1 + 2, max(0, wx0 - 2):wx1 + 3] = wood * 1.35
        P[wy1 + 2, max(0, wx0 - 2):wx1 + 3] *= 0.6
        P[wy1 - 3:wy1, wx1 - 5:wx1 - 2] = (176, 96, 72)      # little plant on the sill
        for dx, dy, alt in CL["plant"]:
            disc(P, wx1 - 3.5 + dx, wy1 - 3 + dy, 1.2, (78, 140, 96) if alt else (60, 116, 82))
        cur = R["curtain"]
        for side in (0, 1):                                  # curtains with soft folds
            cx0 = wx0 - 5 if side == 0 else wx1 - 1
            ys = slice(max(0, wy0 - 3), wy1 + 4)
            xs_ = np.arange(max(0, cx0), min(w, cx0 + 6))
            fold = 0.78 + 0.22 * np.sin(xs_ * 1.6 + side)
            P[ys, xs_] = _c(cur)[None, None, :] * fold[None, :, None]
        rod = max(0, wy0 - 4)
        P[rod, max(0, wx0 - 6):wx1 + 6] = _c(cur) * 0.7     # curtain rod with fairy lights
        for i, fx in enumerate(range(max(0, wx0 - 5), min(w, wx1 + 6), 3)):
            col = mix(BOKEH[i % 4], tuple(self.tint), 0.5 * self.change)     # they blush toward each new chord
            k = 0.3 + 0.55 * (0.5 + 0.5 * math.cos(TAU * (i / 6 - beats / 4))) + 0.35 * self.kick
            k *= 0.4 if day else 1.0
            P[min(H - 1, rod + 1 + (i % 2)), fx] = _c(col) * (0.5 + 0.5 * k)
            glow(P, fx, rod + 1.5, 1.8, col, 0.3 * k)
        # light falling in from the window, with dust drifting in it
        beam = R["beam"]
        L0, L1 = (wx0 + 1, wx1), (wx0 + w * 0.12, wx1 + w * 0.22)
        poly(P, [(L0[0], wy1 + 2), (L0[1], wy1 + 2), (L1[1], H), (L1[0], H)], beam, 0.2 if day or dusk else 0.09)
        for i in range(max(6, w // 8)):
            sp = (i * 0.618 + t * 0.012 * (1 + i % 3)) % 1.0
            rp = (i * 0.37) % 1.0 + 0.06 * math.sin(t * 0.3 + i)
            my = wy1 + 2 + sp * (H - wy1 - 2)
            mx_ = L0[0] + (L1[0] - L0[0]) * sp + ((L0[1] + (L1[1] - L0[1]) * sp) - (L0[0] + (L1[0] - L0[0]) * sp)) * rp
            if 0 <= int(my) < H and 0 <= int(mx_) < w:
                P[int(my), int(mx_)] += _c(beam) * (0.35 if day else 0.25) * (0.5 + 0.5 * math.sin(t * 2 + i))
        self.sky = (wx0 + 1, wx1 - 1, wy0 // 2 + 1, wy1 // 2 - 1)
        self.weather(self.tex("catwx", (h, w), lambda: dict(drops=[], splash=[])), cv, wx0 + 1, wx1,
                     wy0 // 2 + 1, wy1 // 2, splash=False)
        # ── bedside table + lamp (its glow breathes with the bass; off by day) ──
        lx = int(w * CL["lx"])
        if lx + 5 < w:
            P[fy - 12:fy, lx - 5:lx + 6] = mix(R["floor"][0], (60, 40, 50), 0.4)
            P[fy - 12, lx - 6:lx + 7] = mix(R["floor"][0], (255, 255, 255), 0.1)
            th = self.theme()
            if th == "japan":                            # a paper lantern lamp
                P[fy - 14:fy - 12, lx - 1:lx + 2] = (60, 40, 40)
                self.chochin(P, lx + 0.5, fy - 18, 3.2, (236, 226, 210), lit=0.6 + 0.4 * self.bass)
            elif th == "english":                        # a candlestick and a pile of old books
                rect(P, lx - 5, fy - 14, lx + 4, fy - 12, (110, 36, 40))
                rect(P, lx - 4, fy - 16, lx + 3, fy - 14, (40, 70, 60))
                P[fy - 15, lx - 4:lx + 3] = (190, 160, 90)
                self.candles(P, lx + 0.5, fy - 16, 8, beats * 0.3, n=1)
            elif th == "jazz":                           # a record player instead of a lamp, spinning
                rect(P, lx - 5, fy - 15, lx + 6, fy - 12, (70, 40, 30))
                disc(P, lx + 0.5, fy - 15, 4.6, (20, 16, 22), ry=1.4)
                disc(P, lx + 0.5, fy - 15, 1.4, (200, 70, 60), ry=0.5)
                arm = 0.3 * math.sin(beats * 0.1)
                line(P, lx + 5, fy - 17, lx + 2 + arm, fy - 15, (200, 200, 210), wid=0.6)
                self.notes(cv, lx, (fy - 18) // 2, R["lamp"], n=3, rise=6, seed=6, t=beats * 0.5)
            else:
                P[fy - 15:fy - 12, lx] = (140, 120, 110)
                poly(P, [(lx - 4, fy - 15), (lx - 2, fy - 20), (lx + 3, fy - 20), (lx + 5, fy - 15)],
                     mix(R["lamp"], (255, 255, 255), 0.3))
            if not day:
                glow(P, lx + 0.5, fy - 16, 9, R["lamp"], 0.2 + 0.12 * self.bass + 0.08 * self.kick)
                glow(P, lx + 0.5, fy - 2, 14, R["lamp"], 0.1, ry=4)
        # ── wall clock with the real time; its pendulum swings on the beat ──
        ccx = lx - 13 if lx + 5 < w else w * 0.8
        ccy = max(6.0, H * 0.2)
        cr = max(3.5, min(H * 0.075, 6.0))
        if ccx - cr > wx1 + 6 and ccx + cr < w - 2 and ccx - cr > bx + 8:
            line(P, ccx, ccy + cr, ccx, ccy + cr * 2.6, (110, 76, 60), wid=cr * 1.3)
            sw = 0.35 * math.sin(math.pi * beats)
            line(P, ccx, ccy + cr, ccx + math.sin(sw) * cr * 1.4, ccy + cr + math.cos(sw) * cr * 1.4, (210, 170, 90), 0.8)
            disc(P, ccx + math.sin(sw) * cr * 1.5, ccy + cr + math.cos(sw) * cr * 1.5, 0.9, (240, 200, 110))
            disc(P, ccx, ccy, cr + 1, (110, 76, 60))
            disc(P, ccx, ccy, cr, (246, 238, 220))
            lt = time.localtime()
            for ang, ln, wd in ((TAU * ((lt.tm_hour % 12) + lt.tm_min / 60) / 12, cr * 0.5, 1.0),
                                (TAU * (lt.tm_min + lt.tm_sec / 60) / 60, cr * 0.82, 0.7)):
                line(P, ccx, ccy, ccx + math.sin(ang) * ln, ccy - math.cos(ang) * ln, (50, 40, 50), wd)
            for k in range(12):
                a = TAU * k / 12
                P[int(ccy - math.cos(a) * (cr - 0.8)), int(ccx + math.sin(a) * (cr - 0.8))] = (150, 130, 120)
        # ── a framed fish picture above the bed ──
        fx0, fy0 = bx - 6, max(3.0, H * 0.24)
        if fy0 + 8 < by - ry * 2.5 and fx0 > wx1 + 8:
            rect(P, fx0 - 1, fy0 - 1, fx0 + 12, fy0 + 9, (120, 86, 64))
            rect(P, fx0, fy0, fx0 + 11, fy0 + 8, mix(R["sky"][1], (255, 255, 255), 0.3))
            if CL["pic"] == 0:                               # a fish
                disc(P, fx0 + 5, fy0 + 4, 2.5, (240, 140, 90), ry=1.5)
                poly(P, [(fx0 + 7, fy0 + 4), (fx0 + 9.5, fy0 + 2.3), (fx0 + 9.5, fy0 + 5.7)], (240, 140, 90))
            elif CL["pic"] == 1:                             # mountains under a sun
                disc(P, fx0 + 8, fy0 + 2.5, 1.5, CL["pal"])
                poly(P, [(fx0, fy0 + 8), (fx0 + 4, fy0 + 3), (fx0 + 7, fy0 + 6), (fx0 + 9, fy0 + 4.5), (fx0 + 11, fy0 + 8)], (90, 110, 140))
            else:                                            # something abstract
                rect(P, fx0 + 1, fy0 + 1, fx0 + 5, fy0 + 7, CL["pal"])
                disc(P, fx0 + 7.5, fy0 + 4.5, 2.2, mix(CL["pal"], (40, 40, 60), 0.5))
        gap0, gap1 = bx + rx * 1.6, lx - 9
        if gap1 - gap0 >= 22:                                # wide room: a bookshelf full of books
            sw_ = int(min(28, gap1 - gap0 - 4))
            sx0 = int((gap0 + gap1) / 2 - sw_ / 2)
            st_top = max(2, int(fy - min(H * 0.55, fy * 0.8)))
            P[st_top:fy, sx0:sx0 + sw_] = (70, 48, 58)
            P[st_top:fy, sx0] = (96, 68, 78)
            P[st_top, sx0:sx0 + sw_] = (104, 74, 84)
            rng_ = wrng("books", sw_, st_top)
            nshelf = max(2, (fy - st_top) // 9)
            sh_h = (fy - st_top) / nshelf
            for k in range(nshelf):
                yb_ = int(st_top + (k + 1) * sh_h)
                P[min(H - 1, yb_ - 1), sx0:sx0 + sw_] = (110, 78, 88)
                x = sx0 + 2
                while x < sx0 + sw_ - 2:
                    bw_ = rng_.choice((1, 1, 2, 2, 3))
                    bh_ = rng_.randint(int(sh_h * 0.5), max(int(sh_h * 0.5) + 1, int(sh_h) - 2))
                    col = _c(rng_.choice(SWEATERS + BOKEH[:3] + [(220, 210, 190)])) * 0.7
                    if rng_.random() < 0.12:
                        x += 2                               # a gap, or a book leaning over
                        continue
                    P[yb_ - 1 - bh_:yb_ - 1, x:min(sx0 + sw_ - 2, x + bw_)] = col
                    P[yb_ - 1 - bh_, x:min(sx0 + sw_ - 2, x + bw_)] = col * 1.3
                    x += bw_
            disc(P, sx0 + sw_ * 0.7, st_top - 2.5, 2.5, (70, 130, 90))
            P[st_top - 2:st_top, int(sx0 + sw_ * 0.7) - 1:int(sx0 + sw_ * 0.7) + 2] = (176, 96, 72)
        if e.fire_on:                                        # a fireplace just off-screen
            fl = 0.8 + 0.2 * math.sin(t * 9) * math.sin(t * 5.3)
            glow(P, w + 2, fy, H * 0.5, (255, 120, 50), 0.22 * fl, ry=H * 0.45)
        ev = self.rare(18.0)
        alert = self._rare_cat(P, ev, fy, w, bx - rx * 1.9) if ev else False
        # ── food and water bowls ──
        bwx, bwy = bx + rx * 1.75, fy + (H - fy) * 0.62
        if bwx + 12 < min(w - 2, lx - 6):
            for k, fill in enumerate(((200, 130, 80), (150, 200, 240))):
                cxb = bwx + k * 7
                disc(P, cxb, bwy, 3.2, (220, 90, 100) if k == 0 else (90, 140, 210), ry=1.6)
                disc(P, cxb, bwy - 0.6, 2.4, fill, ry=0.9)
                if k == 1:
                    P[int(bwy - 1), int(cxb - 1 + (int(t * 1.5) % 3))] = (230, 246, 255)   # a glint on the water
        # ── yarn ball ──
        yx, yy_ = max(w * 0.25, bx - rx * 1.4 - 30), fy + (H - fy) * 0.55
        disc(P, yx, yy_, 3.2, (226, 110, 140))
        for k in range(3):
            line(P, yx - 2.5, yy_ - 1.5 + k * 1.4, yx + 2.5, yy_ - 2.5 + k * 1.6, (255, 160, 186), 0.8)
        line(P, yx + 3, yy_ + 1, bx - rx * 1.4, by + ry * 1.3, (226, 110, 140), 0.7)
        # ── the cat bed (back half), the cat, then the bed's front rim ──
        bo, bi, brim = R["bed"]
        bed_c = (bx - rx * 0.1, by + ry * 0.55)
        disc(P, bed_c[0], bed_c[1], rx * 1.4, bo, ry=ry * 0.95)
        disc(P, bed_c[0], bed_c[1] - 0.5, rx * 1.15, bi, ry=ry * 0.65)
        breath = 0.5 - 0.5 * math.cos(TAU * beats / 4)       # one slow breath every bar
        bry = ry * (0.94 + 0.08 * breath)
        cyb = by - ry * 0.04 * breath
        self._fur(P, bx, cyb, rx, bry, base, stripe, striped, phase=1.0)
        hx, hy = bx - rx * 0.62, cyb + bry * 0.28            # head resting on its paws, front-left
        hr = rx * 0.42
        for ex, tilt in ((-0.55, -0.2), (0.45, 0.25)):       # ears (one twitches now and then)
            flick = 1.5 if ex > 0 and ((t % 9.0) < 0.25 or (alert and (t % 1.3) < 0.3)
                                       or (self.snare > 0.6 and int(beats) % 8 == 5)) else 0.0   # an ear for the backbeat
            ear = [(hx + hr * (ex - 0.34), hy - hr * 0.45), (hx + hr * (ex + tilt * 0.4) + flick * 0.5, hy - hr * 1.2 - flick),
                   (hx + hr * (ex + 0.34), hy - hr * 0.5)]
            poly(P, ear, _c(stripe if points else base) * 0.8)
            poly(P, [(x_ + (hx + hr * ex - x_) * 0.18, y_ + 0.6) for x_, y_ in ear], stripe if points else base)
            inner = [(hx + hr * (ex - 0.16), hy - hr * 0.52), (hx + hr * (ex + tilt * 0.35) + flick * 0.5, hy - hr * 0.98 - flick),
                     (hx + hr * (ex + 0.16), hy - hr * 0.54)]
            poly(P, inner, (236, 150, 170))
        self._fur(P, hx, hy, hr, hr * 0.84, base, stripe, striped, phase=0.3)
        if points:
            disc(P, hx, hy + hr * 0.2, hr * 0.55, stripe, ry=hr * 0.45, a=0.6)
        disc(P, hx, hy + hr * 0.42, hr * 0.42, belly, ry=hr * 0.3)          # muzzle
        dark = (34, 26, 34) if sum(base) > 200 else (170, 160, 190)
        for ex in (-0.38, 0.38):                                             # closed eyes ‿
            exi, eyi = hx + hr * ex, hy + hr * 0.05
            line(P, exi - 1.6, eyi - 0.4, exi, eyi + 0.6, dark, 0.9)
            line(P, exi, eyi + 0.6, exi + 1.6, eyi - 0.4, dark, 0.9)
        disc(P, hx, hy + hr * 0.3, 0.8, (236, 130, 150))                     # nose
        for side in (-1, 1):                                                 # whiskers
            for dyw in (0.0, 0.18):
                line(P, hx + side * hr * 0.4, hy + hr * (0.45 + dyw), hx + side * hr * 1.3, hy + hr * (0.3 + dyw * 1.6),
                     (226, 226, 236), 0.5, 0.55)
        for px_ in (hx - hr * 0.55, hx + hr * 0.35):                         # front paws
            disc(P, px_, hy + hr * 0.95, hr * 0.32, belly, ry=hr * 0.2)
        b = _box(P, bed_c[0] - rx * 1.45, bed_c[1] - ry, bed_c[0] + rx * 1.45, bed_c[1] + ry * 1.05)
        if b is not None:                                    # front rim of the bed over the cat
            v, Yb, Xb = b
            d_out = ((Xb + 0.5 - bed_c[0]) / (rx * 1.4)) ** 2 + ((Yb + 0.5 - bed_c[1]) / (ry * 0.95)) ** 2
            d_in = ((Xb + 0.5 - bed_c[0]) / (rx * 1.15)) ** 2 + ((Yb + 0.5 - bed_c[1] + 0.5) / (ry * 0.65)) ** 2
            rim = (d_out < 1) & (d_in >= 1) & (Yb + 0.5 > bed_c[1] - ry * 0.1)
            shade = np.clip(1.15 - 0.4 * (Yb - bed_c[1]) / ry, 0.7, 1.2)
            v[rim] = (_c(brim) * shade[..., None])[rim]
        tail_col = stripe if points else base                # tail curls round, the tip twitches on the beat
        twitch = 0.25 * math.sin(TAU * beats) * (1 if int(beats) % 4 == 3 else 0.2)
        n = 14
        prev = None
        for i in range(n + 1):
            a = 0.15 + 2.2 * i / n + (twitch * (i / n) ** 3)
            px = bx + (rx * 1.02) * math.cos(a)
            py = cyb + (bry * 0.95) * math.sin(a)
            if prev is not None:
                band = striped and i % 3 == 0
                col = stripe if (band or (points and i > n * 0.6)) else tail_col
                line(P, prev[0], prev[1], px, py, col, (3.2 - 1.4 * i / n, 3.2 - 1.4 * (i + 1) / n))
            prev = (px, py)
        glow(P, bx, cyb, rx * 1.6, R["beam"], 0.05 + 0.04 * breath)          # light on the fur
        # ── dream bubble: a crisp little cloud; the dream changes every four bars ──
        dx0, dy0 = hx + hr * 1.15, hy - hr * 1.3
        bub, out = (236, 232, 252), (140, 130, 190)
        for k, rr in enumerate((0.9, 1.3, 1.8)):
            disc(P, dx0 + k * 2.6, dy0 - k * 2.4, rr + 0.6, out)
            disc(P, dx0 + k * 2.6, dy0 - k * 2.4, rr, bub)
        cxb, cyb_ = dx0 + 12, dy0 - 11
        Rb = 6.0 + 0.5 * self.bass + 0.5 * self.kick
        bb = _box(P, cxb - Rb * 1.9, cyb_ - Rb * 1.4, cxb + Rb * 1.9, cyb_ + Rb * 1.3)
        if bb is not None and cyb_ - Rb * 1.2 > 0:
            v, Yb, Xb = bb
            m = np.zeros(v.shape[:2], bool)
            for ox, oy, rr in ((-0.85, 0.15, 0.72), (0.85, 0.15, 0.72), (0, -0.45, 0.8), (0, 0.38, 0.72), (-0.4, -0.2, 0.7), (0.45, -0.25, 0.7)):
                m |= ((Xb + 0.5 - (cxb + ox * Rb)) / (Rb * rr)) ** 2 + ((Yb + 0.5 - (cyb_ + oy * Rb)) / (Rb * rr * 0.72)) ** 2 < 1
            inner = m.copy()
            inner[1:] &= m[:-1]
            inner[:-1] &= m[1:]
            inner[:, 1:] &= m[:, :-1]
            inner[:, :-1] &= m[:, 1:]
            v[m] = v[m] * 0.12 + _c(bub) * 0.88
            v[m & ~inner] = out
            self._dream(P, int(beats // 16) % 4, cxb, cyb_, Rb, t)
            self.note_sparks(P, "catdream", lambda p, n: (cxb + (p - 0.5) * Rb * 2.4, cyb_ + Rb * 0.5),
                             (255, 214, 120), rise=4.0, life=1.0, r=0.55)     # it dreams the melody too
        # ── floating Zzz ──
        zs = self.tex("catz", (h, w), lambda: [])
        if int(beats * 2) % 8 == 0 and getattr(self, "_zlast", -1) != int(beats * 2):
            self._zlast = int(beats * 2)
            zs.append([hx - hr * 0.2, (hy - hr) / 2, 0])
        keep = []
        for z in zs:
            z[2] += 1
            z[1] -= 0.06
            z[0] -= 0.05 + 0.04 * math.sin(z[2] * 0.08)
            if z[2] < 120 and z[1] > 0:
                keep.append(z)
                k = z[2] / 120
                col = (int(230 - 110 * k), int(220 - 110 * k), int(255 - 90 * k)) if not day else (int(120 + 60 * k), int(110 + 60 * k), int(170 + 40 * k))
                cv.put(int(z[1]), int(z[0]), "z" if z[2] < 40 else "Z", col)
        zs[:] = keep[-8:]

    def _rare_cat(self, P, ev, fy, w, near):
        """Rare event: a mouse scurries out along the skirting board, stops to sniff the air (the cat's
        ear twitches in its sleep), then dashes off. Returns True while the mouse is stopped."""
        u, r = ev
        H = P.shape[0]
        f = r.choice((-1, 1))
        s = max(1.4, H / 50)
        stop = max(6.0, near - r.uniform(2, 12))                      # pauses just short of the cat bed
        x0, x2 = -6 * s, w + 6 * s
        if f < 0:
            x0, x2 = x2, x0
        T = self.rclock
        if u < 0.3:
            x, still = x0 + (stop - x0) * (u / 0.3), False
        elif u < 0.62:
            x, still = stop + math.sin(T * 1.3) * 0.6 * (u > 0.45), True
        else:
            x, still = stop + (x2 - stop) * ((u - 0.62) / 0.38) ** 1.3, False
        y = fy + 2.5 * s
        col = (150, 140, 156)
        hop = 0.0 if still else abs(math.sin(T * 22)) * 0.6
        disc(P, x, y - 1.2 * s - hop, 2.6 * s, col, ry=1.4 * s)                    # body
        hx = x + f * 2.6 * s
        sniff = math.sin(T * 25) * 0.3 if still else 0.0
        disc(P, hx + sniff, y - 1.4 * s - hop, 1.3 * s, col, ry=1.0 * s)             # head
        for px, py, c in ((hx - f * 0.6 * s, y - 2.6 * s, (236, 170, 186)), (hx + f * 1.4 * s + sniff, y - 1.6 * s, (60, 40, 50))):
            if 0 <= px < w and 0 <= py < H:                                            # ear, nose
                P[int(py), int(px)] = c
        line(P, x - f * 2.4 * s, y - 0.6 * s, x - f * 6 * s, y - 0.2 * s + math.sin(T * 6) * 0.6, (190, 150, 160), wid=0.6)
        return still
