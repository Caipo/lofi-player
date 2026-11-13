import math
import time

import numpy as np

from ..paint import _c, blur, disc, fill_grad, glow, grid, line, make_stars, mix, noise_tex, poly, vgrad
from ..world import wrng
from ..stations import STATIONS
from ..term import C_DIM


# ── per-station looks for every scene (index = station) ──
SPEC_GRAD = [
    [(255, 224, 178), (255, 160, 206), (196, 156, 255), (118, 206, 226), (104, 148, 240)],
    [(220, 236, 255), (150, 190, 255), (100, 130, 240), (70, 90, 200), (50, 60, 160)],
    [(255, 240, 170), (255, 200, 110), (255, 140, 80), (230, 90, 80), (170, 60, 90)],
    [(255, 255, 255), (220, 236, 255), (170, 200, 240), (130, 160, 220), (100, 120, 190)],
    [(255, 255, 140), (60, 255, 230), (255, 60, 200), (170, 60, 255), (90, 40, 220)],
    [(240, 255, 220), (170, 240, 180), (100, 210, 170), (60, 160, 150), (40, 110, 120)],
    [(255, 250, 190), (255, 214, 90), (140, 230, 120), (80, 200, 230), (60, 150, 240)],
    [(255, 240, 246), (255, 184, 214), (246, 110, 150), (200, 60, 96), (96, 60, 150)],      # sakura → indigo
    [(255, 236, 190), (246, 196, 104), (214, 140, 60), (170, 70, 60), (96, 34, 60)],        # brass & bourbon
    [(250, 240, 214), (220, 196, 140), (150, 160, 120), (90, 110, 120), (60, 66, 96)],       # parchment → moor fog
]


class SpectrumScene:
    # ════════════════════════════════════════════════════════════════════════
    #  Visualizers. Each paints pixels into P (2·h × w) and text into cv (h × w)
    # ════════════════════════════════════════════════════════════════════════
    def v_spectrum(self, cv, P, h, w):
        H = P.shape[0]
        t = time.time()
        floor = max(6, int(H * 0.74))
        fill_grad(P, 0, floor, [(0, (11, 9, 27)), (1, (40, 24, 64))])
        fill_grad(P, floor, H, [(0, (26, 16, 44)), (1, (9, 7, 20))])
        neb = np.roll(self.tex("specneb", (floor, w), lambda: noise_tex(floor, w, 8, 41)), int(t * 0.6) % w, axis=1)
        P[:floor] += np.clip(neb - 0.4, 0, None)[..., None] * _c(self.var(SPEC_GRAD)[3]) * 0.07
        P[:floor] += np.clip(neb - 0.3, 0, None)[..., None] * self.tint * (0.05 + 0.12 * self.change)   # the chord's colour
        self.stars(P, self.layout("specstars", h, w, lambda r: make_stars(r, int(floor * 0.7), w, w * h // 60)), t, 0.7)
        # a far ridge on the horizon, in the station's colours, so the stage has somewhere to stand
        ridge = self.tex("specridge", (floor, w), lambda: self._spec_ridge(floor, w))
        P[:floor] += (_c(mix(self.var(SPEC_GRAD)[4], (14, 10, 34), 0.72)) - P[:floor]) * ridge[..., None]
        th = self.theme()
        if th == "japan":                                   # a striped rising sun behind a dark Fuji
            self._spec_japan(P, floor, w, t)
        elif th == "english":                               # a castle on a hill under the moon, fog below
            self._spec_english(P, floor, w, t)
        elif th == "jazz":                                  # two spotlights from the rafters
            for sx, ph in ((0.2, 0.0), (0.8, 2.0)):
                yy_, xx_ = grid(floor, w)
                cx = w * sx + math.sin(t * 0.25 + ph) * w * 0.06
                spread = 0.12 + yy_ / floor * 0.32
                beam = np.clip(1 - np.abs(xx_ - cx - (yy_ / floor) * (w * 0.5 - cx) * 0.4) / (spread * w * 0.35 + 1), 0, 1)
                P[:floor] += (beam ** 1.5 * (0.25 + 0.75 * yy_ / floor))[..., None] * _c((255, 214, 150)) * (0.13 + 0.12 * self.snare)
        bw = 2 if w >= 90 else 1
        nb = max(4, (w - 2) // (bw + 1))
        lv, pk = self.band_levels(nb)
        x0 = (w - nb * (bw + 1)) // 2 + 1
        maxh = floor - 5
        hcol = np.zeros(w, np.float32)
        pcol = np.zeros(w, np.float32)
        shade = np.ones(w, np.float32)
        for i in range(nb):
            x = x0 + i * (bw + 1)
            hcol[x:x + bw] = lv[i] * maxh
            pcol[x:x + bw] = pk[i] * maxh
            if bw == 2:
                shade[x + 1:x + 2] = 0.78
        rows = vgrad(floor, list(zip((0, 0.3, 0.55, 0.8, 1), self.var(SPEC_GRAD))))
        yy, _ = grid(H, w)
        dist = floor - 1 - yy[:floor]
        cov = np.clip(hcol[None, :] - dist, 0, 1)
        B = np.zeros_like(P)
        B[:floor] = rows[:, None, :] * (cov * shade[None, :])[..., None]
        edge = (cov > 0) & (hcol[None, :] - dist < 1.5)            # bright lip on each bar
        B[:floor][edge] = np.minimum(255, B[:floor][edge] * 1.3 + 30)
        cap = np.nonzero((pcol > hcol + 1.5) & (pcol > 1))[0]
        B[np.clip(floor - 1 - pcol[cap], 0, floor - 1).astype(int), cap] = (255, 242, 228)
        G = blur(blur(B, 3), 3)
        P += B + G * (0.55 + 0.5 * self.kick)
        nw = nb * (bw + 1)
        self.note_sparks(P, "spec", lambda p, n: (x0 + 0.5 + p * (nw - 2), floor - 1 - hcol[min(w - 1, int(x0 + p * (nw - 2)))] - 2),
                         (255, 236, 220), rise=9.0, drift=0.0, life=1.4, r=0.9)
        ev = self.rare(30.0)
        if ev:
            self._rare_spectrum(P, ev, floor, hcol)
        rh = H - floor
        k = np.arange(rh)
        src = np.clip(floor - 1 - k, 0, floor - 1)
        level = float(self.rms.mean())
        shift = np.round(np.sin(k * 0.35 + t * 2.0) * (0.4 + level * 2.5 + 1.5 * self.snare)).astype(int)
        idx = (np.arange(w)[None, :] - shift[:, None]) % w
        fade = (0.4 * np.exp(-k / max(1, rh * 0.5)))[:, None, None]
        P[floor:] += (B[src[:, None], idx] + G[src[:, None], idx] * 0.5) * fade
        # the floor: a grid running to the horizon, rolling toward you in time with the beat
        gcol = _c(self.var(SPEC_GRAD)[3])
        ph = (self.eng.beat_pos / 2) % 1 if not self.eng.paused else 0.0
        dep = (k + 0.5) / rh                                      # 0 at the horizon → 1 at the bottom edge
        rows_on = ((1.2 / dep + ph) % 1) < 1.2 / (dep * dep * rh)                # one row thick, however near
        xs = (np.arange(w)[None, :] - w / 2) / (dep[:, None] * w * 0.9 + 1e-3)
        cols_on = np.abs((xs * 12.0 + 0.5) % 1 - 0.5) < 0.55 / (dep[:, None] * w * 0.075 + 0.5)
        gridm = (rows_on[:, None] | cols_on) * (np.clip(dep * 4, 0, 1) * (0.1 + 0.22 * dep))[:, None] * (1 + 0.8 * self.kick)
        P[floor:] += gridm[..., None] * gcol
        P[floor] += _c((64, 48, 100)) * (1 + 1.6 * self.kick)
        if th == "jazz":                                    # the stage edge is a keyboard; keys light under the bars
            self._spec_keys(P, floor, w, hcol, maxh)
        cv.text(0, 1, STATIONS[self.eng.station_idx]["mood"][: w - 2], C_DIM)

    @staticmethod
    def _spec_ridge(floor, w):
        r = wrng("specridge", w)
        xs = np.arange(w)
        top = floor - floor * (0.1 + 0.07 * np.sin(xs * 0.05 + r.uniform(0, 6)) + 0.04 * np.sin(xs * 0.13 + r.uniform(0, 6))
                               + 0.02 * np.sin(xs * 0.41 + r.uniform(0, 6)))
        yy, _ = grid(floor, w)
        return np.clip(yy - top[None, :] + 0.5, 0, 1).astype(np.float32) * 0.85

    def _spec_japan(self, P, floor, w, t):
        R = min(floor * 0.36, w * 0.2)
        cx, cy = w * 0.5, floor * 0.44
        yy, xx = grid(floor, w)
        d = np.hypot(xx + 0.5 - cx, yy + 0.5 - cy)
        q = (yy + 0.5 - cy) / R                                            # -1 top … 1 bottom
        cut = (q > 0.0) & (((q * 5 - (t * 0.15) % 1) % 1) < 0.12 + q * 0.45)   # slats widening toward the horizon
        m = np.clip(R - d + 0.5, 0, 1) * ~cut
        col = _c((255, 210, 130)) + (_c((236, 60, 98)) - _c((255, 210, 130))) * np.clip(q * 0.5 + 0.5, 0, 1)[..., None]
        P[:floor] += (col * 0.5 - P[:floor]) * m[..., None]
        glow(P, cx, cy, R * 1.6, (200, 50, 90), 0.14 + 0.14 * self.kick)
        my = cy + R * 0.8
        poly(P, [(cx - R * 2.6, floor), (cx - R * 0.42, my), (cx + R * 0.42, my), (cx + R * 2.6, floor)], (48, 28, 70))
        poly(P, [(cx - R * 0.42, my), (cx + R * 0.42, my), (cx + R * 0.7, my + R * 0.22), (cx + R * 0.3, my + R * 0.14),
                 (cx, my + R * 0.3), (cx - R * 0.3, my + R * 0.14), (cx - R * 0.7, my + R * 0.22)], (196, 186, 222))

    def _spec_english(self, P, floor, w, t):
        s = min(floor * 0.36, w * 0.14)
        cx = w * 0.5
        glow(P, cx + s * 1.8, floor * 0.22, s * 0.9, (240, 236, 210), 0.2 + 0.15 * self.note)
        disc(P, cx + s * 1.8, floor * 0.22, max(2.0, s * 0.22), (236, 232, 214))
        hill = (62, 58, 88)
        poly(P, [(cx - s * 3.4, floor), (cx - s * 1.1, floor - s * 0.52), (cx + s * 1.1, floor - s * 0.56), (cx + s * 3.4, floor)], hill)
        self.castle(P, cx, floor - s * 0.5, s, hill, lit=(255, 206, 130), t=t)
        fog = np.clip(np.sin(np.arange(w) * 0.07 + t * 0.2) * 0.5 + 0.5, 0, 1)[None, :, None]
        yb = int(floor - s * 0.3)
        P[yb:floor] += (_c((150, 150, 170)) - P[yb:floor]) * 0.25 * fog

    def _spec_keys(self, P, floor, w, hcol, maxh):
        H = P.shape[0]
        kh = max(3, int((H - floor) * 0.55))
        y0 = H - kh
        kw = 3 if w < 90 else 4
        lit = np.clip(hcol / max(1.0, maxh), 0, 1)
        for k in range(w // kw + 1):
            x0, x1 = k * kw, min(w, k * kw + kw - 1)
            if x0 >= w:
                break
            glowk = float(lit[x0:x1 + 1].max()) if x1 >= x0 else 0.0
            P[y0:, x0:x1] = mix((222, 214, 196), (255, 214, 140), glowk * 0.8)
            P[y0:, x1:x1 + 1] = (60, 48, 50)
            if k % 7 not in (2, 6):                                       # black keys between most whites
                bx = x1
                P[y0:y0 + max(2, kh * 3 // 5), max(0, bx - 1):bx + 1 + (kw > 3)] = (24, 18, 26)
        P[y0 - 1] = (120, 70, 50)

    def _rare_spectrum(self, P, ev, floor, hcol):
        """Rare event: a little cat strolls along the tops of the bars, hopping up as they rise
        and dropping down as they fall."""
        u, r = ev
        H, w = P.shape[:2]
        f = r.choice((-1, 1))
        sc = max(1.0, H / 60)
        x = (-8 + u * (w + 16)) if f > 0 else (w + 8 - u * (w + 16))
        xa, xb = int(max(0, x - 4 * sc)), int(min(w, x + 4 * sc + 1))
        top = floor - 1 - (hcol[xa:xb].max() if xb > xa else 0.0)
        m = self.rare_mem
        cy = m.get("cy", top)
        cy = top if top < cy else min(top, cy + 2.2)          # hop up at once, fall a little slower
        m["cy"] = cy
        col = (238, 230, 255)
        T = self.rclock
        by = cy - 2.6 * sc
        disc(P, x, by, 3.6 * sc, col, ry=1.7 * sc)                                   # body
        for k, ox in enumerate((-2.6, -1.6, 1.8, 2.8)):                              # legs
            sw = math.sin(T * 9 + k * 1.7) * 0.9 * sc
            line(P, x + f * ox * sc, by + 1, x + f * ox * sc + sw, cy, col, wid=0.9)
        hx, hy = x + f * 4.0 * sc, by - 1.8 * sc
        disc(P, hx, hy, 1.7 * sc, col)                                               # head and ears
        for e_ in (-0.9, 0.7):
            poly(P, [(hx + (e_ - 0.6) * sc, hy - 0.8 * sc), (hx + e_ * sc, hy - 2.8 * sc), (hx + (e_ + 0.6) * sc, hy - 0.8 * sc)], col)
        if 0 <= hy < H and 0 <= hx + f * 0.8 * sc < w:
            P[int(hy), int(hx + f * 0.8 * sc)] = (40, 30, 70)                         # eye
        wag = math.sin(T * 2.5) * 1.2 * sc                                           # tail up, swaying
        line(P, x - f * 3.4 * sc, by - 0.5, x - f * 4.8 * sc + wag, by - 4.5 * sc, col, wid=(1.2, 0.8))
