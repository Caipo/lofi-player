import math
import random
import time

import numpy as np

from ..paint import _c, blur, disc, fill_grad, glow, grid, mix, rect, vgrad
from ..world import wrng
from ..theory import TAU


CHROME = [(0, (64, 64, 84)),(0.28, (214, 214, 230)), (0.42, (255, 255, 255)), (0.68, (130, 130, 150)), (1, (44, 44, 60))]


# lava lamp colours per station: liquid top, liquid bottom, wax rim, wax core
LAVA = [
    ((36, 26, 104), (108, 48, 150), (255, 176, 206), (255, 92, 140)),
    ((12, 54, 66), (28, 108, 110), (255, 206, 124), (255, 108, 40)),
    ((140, 80, 18), (226, 166, 58), (255, 128, 104), (204, 36, 48)),
    ((56, 84, 136), (128, 168, 212), (255, 255, 255), (206, 196, 255)),
    ((66, 10, 88), (156, 28, 140), (184, 255, 232), (36, 226, 188)),
    ((22, 18, 68), (58, 38, 118), (255, 238, 152), (238, 166, 48)),
    ((120, 200, 240), (250, 240, 170), (255, 160, 60), (255, 90, 40)),
    ((40, 20, 70), (120, 40, 100), (255, 220, 236), (255, 110, 160)),      # tokyo: sakura wax in plum
    ((50, 20, 14), (130, 60, 30), (255, 226, 150), (240, 160, 50)),        # jazz: honey wax in bourbon
    ((30, 40, 34), (90, 40, 40), (255, 236, 200), (200, 140, 60)),        # english: claret and candle wax
]


class LavaScene:
    # ── lava lamp ──────────────────────────────────────────────────────────
    @staticmethod
    def _trap(P, cx, y0, y1, hw0, hw1, stops):
        """Chrome-shaded trapezoid between rows y0 and y1 (half-widths hw0 → hw1)."""
        H, W = P.shape[:2]
        y0c, y1c = max(0, int(y0)), min(H, int(math.ceil(y1)))
        if y0c >= y1c:
            return
        yy, xx = grid(H, W)
        Y, X = yy[y0c:y1c], xx[y0c:y1c]
        hw = hw0 + (hw1 - hw0) * np.clip((Y + 0.5 - y0) / max(1e-3, y1 - y0), 0, 1)
        m = np.abs(X + 0.5 - cx) <= hw
        k = np.clip((X + 0.5 - (cx - hw)) / (2 * hw), 0, 1)
        col = np.stack([np.interp(k, [p for p, _ in stops], [c[i] for _, c in stops]) for i in range(3)], -1)
        v = P[y0c:y1c]
        v[m] = col[m]

    def v_lava(self, cv, P, h, w):
        e = self.eng
        t = time.time()
        H = P.shape[0]
        table = H - max(3, int(H * 0.08))
        base_h = max(6, int(H * 0.19))
        cap_h = max(4, int(H * 0.11))
        yt0 = max(cap_h + 1, int(H * 0.05) + cap_h)
        gl0 = max(8, table - base_h - yt0)
        rb0 = max(5.0, min(w * 0.17, gl0 * 0.27))
        # one lamp in a narrow window; a row of smaller ones either side when it's wide
        lamps = [(w / 2, 1.0, 0)]
        x, k = rb0 * 3.4, 1
        while k <= 6:
            sc = self.layout("lava", 0, 0, lambda r: [r.uniform(0.5, 0.78) for _ in range(6)])[k - 1]
            if w / 2 + x + rb0 * sc * 1.5 > w:
                break
            lamps += [(w / 2 - x, sc, k), (w / 2 + x, sc * 0.94, k + 6)]
            x += rb0 * (sc + 0.72) * 2.4
            k += 1
        # ── the room, lit by the lamps ──
        fill_grad(P, 0, table, [(0, (9, 7, 16)), (1, (20, 15, 28))])
        fill_grad(P, table, H, [(0, (44, 30, 36)), (1, (16, 11, 16))])
        P[table] = (70, 52, 58)
        th = self.theme()
        # clear stretches of wall between the lamps, widest first, for pictures and records to hang in
        occ = sorted((cx - rb0 * sc * 1.35, cx + rb0 * sc * 1.35) for cx, sc, _ in lamps)
        gaps, x_ = [], 0.0
        for a, b in occ:
            if a > x_:
                gaps.append((x_, a))
            x_ = max(x_, b)
        if x_ < w:
            gaps.append((x_, float(w)))
        gaps.sort(key=lambda g: g[0] - g[1])
        # places to hang things: (x, y, room across, room down) in the gaps, and over the shorter lamps' caps
        spots = [((g0 + g1) / 2, table * 0.34, g1 - g0 - 2, table * 0.6) for g0, g1 in gaps]
        for cx, sc, k in lamps:
            top = table - (table - yt0 + cap_h) * sc - 3
            if k and top > 8:
                spots.append((cx, top / 2, rb0 * sc * 2.4, top - 2))
        spots.sort(key=lambda q: -min(q[2] / 2, q[3]))
        if th == "japan":                                    # shoji: paper panes in a wooden lattice
            yy_, xx_ = grid(table, w)
            cw = max(6, int(table / 4))
            P[:table] = vgrad(table, [(0, (30, 22, 32)), (1, (56, 42, 50))])[:, None, :]
            frame = ((xx_ % cw) < 1) | ((yy_ % max(5, cw * 4 // 3)) < 1) | ((xx_ % (cw * 3)) < 2)
            P[:table][frame] = (16, 11, 14)
        elif th == "english":                                # damask wallpaper in green, a dado rail, a gilt portrait
            yy_, xx_ = grid(table, w)
            P[:table] = vgrad(table, [(0, (14, 22, 18)), (1, (30, 44, 36))])[:, None, :]
            u, v = (xx_ % 10) - 4.5, (yy_ + (xx_ // 10 % 2) * 7) % 14 - 7
            motif = (u * u / 9 + v * v / 30 < 1) & ~(u * u / 3 + v * v / 12 < 1) | ((np.abs(u) < 0.6) & (np.abs(v) < 6))
            P[:table][motif] *= 1.45
            dy = int(table * 0.72)
            P[dy:table] *= 0.7
            P[dy] = (120, 90, 50)
            best = max(spots, key=lambda q: min(q[2] / 2.2, q[3] / 2.8), default=None)
            fw_ = min(table * 0.16, w * 0.06, best[2] / 2.2, best[3] / 2.8) if best else 0
            if fw_ >= 3:
                fx_, fy_ = best[0], best[1]
                rect(P, fx_ - fw_ - 1, fy_ - fw_ * 1.3 - 1, fx_ + fw_ + 1, fy_ + fw_ * 1.3 + 1, (176, 140, 60))
                rect(P, fx_ - fw_, fy_ - fw_ * 1.3, fx_ + fw_, fy_ + fw_ * 1.3, (40, 30, 30))
                disc(P, fx_, fy_ - fw_ * 0.3, fw_ * 0.45, (190, 160, 140))                     # a sitter in a ruff
                disc(P, fx_, fy_ + fw_ * 0.9, fw_ * 0.85, (30, 26, 40), ry=fw_ * 0.5)
                disc(P, fx_, fy_ + fw_ * 0.25, fw_ * 0.55, (236, 230, 220), ry=fw_ * 0.18)
        elif th == "jazz":                                   # records and their sleeves pinned to the wall
            sleeves = ((150, 40, 40), (40, 80, 140), (200, 150, 60))
            for k, ((sx_, sy_, rw, rh), sl) in enumerate(zip(spots, sleeves)):
                r_ = min(table * 0.15, w * 0.07, rw / 3.9, rh / 2.2)     # sleeve plus the record beside it
                if r_ < 3:
                    continue
                x_, y_ = sx_ - r_ * 0.35, sy_
                rect(P, x_ - r_, y_ - r_, x_ + r_, y_ + r_, mix(sl, (10, 8, 14), 0.45))
                rect(P, x_ - r_ * 0.7, y_ - r_ * 0.2, x_ + r_ * 0.1, y_ + r_ * 0.6, mix(sl, (230, 200, 160), 0.2))
                self.record(P, x_ + r_ * 0.75, y_, r_ * 0.95, ang=k * 1.3, label=mix(sl, (240, 220, 190), 0.3), sheen=0.5)
        pulse = 0.8 + 0.25 * self.bass + 0.25 * self.kick
        L = getattr(self, "lava", None)
        if L is None or L["size"] != (h, w):
            L = self.lava = dict(size=(h, w), lamps={}, last=t)
        dt = min(0.1, max(0.0, t - L["last"]))
        L["last"] = t
        geo = []
        for cx, sc, k in lamps:
            pal = tuple(_c(c) for c in LAVA[(e.station_idx + k) % len(LAVA)])
            g = dict(cx=cx, k=k, pal=pal, base_h=base_h * sc, cap_h=max(3, int(cap_h * sc)))
            g["yb"] = int(table - g["base_h"])
            g["yt"] = int(table - (table - yt0 + cap_h) * sc + g["cap_h"])
            g["gl"] = max(6, g["yb"] - g["yt"])
            g["rb"] = max(3.5, rb0 * sc)
            g["rt"] = g["rb"] * self.layout("lavashape", 0, 0, lambda r: r.uniform(0.36, 0.6), k)
            g["col"] = pal[3] * 0.6 + pal[1] * 0.4
            glow(P, cx, g["yt"] + g["gl"] * 0.55, g["rb"] * 2.6, g["col"], 0.32 * pulse * sc, ry=g["gl"] * 0.75)
            halo = g["col"] * 0.7 + self.tint * (0.15 + 0.25 * self.change)         # the wide halo takes the chord's colour
            glow(P, cx, g["yt"] + g["gl"] * 0.55, g["rb"] * 6, halo, 0.1 * pulse * sc, ry=g["gl"] * 1.2)
            glow(P, cx, table + 1.5, g["rb"] * 3.2, g["col"], 0.35 * pulse * sc, ry=2.5)
            geo.append(g)
        for g in geo:
            self._lava_lamp(P, g, table, L, dt, t)
        ev = self.rare(22.0)
        if ev:
            self._rare_lava(P, ev, geo[0], w)
        if th == "jazz":
            self.notes(cv, int(w / 2), max(2, geo[0]["yt"] // 2 - 2), (255, 214, 150), n=3, rise=5, seed=4)
        elif th == "english":                                # a candelabra at the end of the table
            cs_ = min(14.0, table * 0.3)
            spot = [g for g in gaps if g[1] - g[0] >= cs_ * 0.8 + 2]
            if spot:                                         # a candelabra on the table, clear of the lamps
                g0, g1 = max(spot, key=lambda g: g[1])
                self.candles(P, (g0 + g1) / 2 if g1 < w else max(g0 + cs_ * 0.4 + 1, w - cs_ * 0.6), table, cs_, self.rclock)
        # reflection of the lamps on the table
        if table + 1 < H:
            src = P[max(0, table - 1 - (H - table - 1) * 2):table][::-2][: H - table - 1]
            n = len(src)
            if n:
                P[table + 1:table + 1 + n] = P[table + 1:table + 1 + n] * 0.75 + src * 0.25

    def _lava_lamp(self, P, g, table, L, dt, t):
        e = self.eng
        H, w = P.shape[:2]
        cx, yb, yt, gl, rb, rt = g["cx"], g["yb"], g["yt"], g["gl"], g["rb"], g["rt"]
        lt, lb, wr, wc = g["pal"]
        st = L["lamps"].get(g["k"])
        if st is None:                       # blobs of wax for this lamp
            rng = wrng("lavablobs", g["k"])
            st = L["lamps"][g["k"]] = dict(bubbles=[], blobs=[
                [0.9, rng.uniform(-0.6, 0.6), rng.uniform(0.24, 0.52), 0.0, rng.uniform(0, TAU), rng.random() * TAU]
                for _ in range(rng.randint(5, 9) if g["k"] == 0 else rng.randint(3, 6))])
        heat = 0.8 + 1.5 * self.bass + 0.6 * self.energy + (0.6 if e.fire_on else 0.0)
        for b in st["blobs"]:
            u, s, r, vy, T, ph = b                             # T is this blob's cycle phase
            period = 25 + 40 * ((ph * 7.3) % 1)                # each blob rises and sinks at its own pace
            T += dt * TAU / period * heat
            ru = r * rb / gl
            lo, hi = ru * 0.6, 1 - ru * 0.4
            nu = lo + (hi - lo) * (0.5 + 0.5 * math.tanh(2.2 * math.cos(T)) / math.tanh(2.2))   # lingers at the ends
            vy = (nu - u) / dt if dt > 0 else 0.0
            u = nu
            s += math.sin(t * 0.25 + ph) * 0.004
            s = float(np.clip(s, -0.7, 0.7))
            b[:] = [u, s, r, float(np.clip(vy, -0.075, 0.075)), T, ph]
        # ── glass: liquid, wax (metaballs), highlights ──
        y0, y1 = max(0, yt), min(H, yb)
        x0, x1 = max(0, int(cx - rb - 2)), min(w, int(cx + rb + 3))
        if y0 < y1 and x0 < x1:
            yy, xx = grid(H, w)
            Y, X = yy[y0:y1, x0:x1] + 0.5, xx[y0:y1, x0:x1] + 0.5
            uu = (Y - yt) / gl
            hw = rt + (rb - rt) * uu
            inside = np.abs(X - cx) <= hw
            F = np.zeros(Y.shape, np.float32)
            wob = 1 + 0.04 * self.bass + 0.06 * self.kick              # the wax throbs with the kick
            for u, s, r, vy, T, ph in st["blobs"]:
                by_ = yt + u * gl
                hwb = rt + (rb - rt) * u
                rp = r * rb * wob * (1 + 0.04 * math.sin(t * 1.3 + ph))
                bx_ = cx + s * max(0.0, hwb - rp * 0.7)
                sy = 1 + 0.25 * abs(vy) / 0.075                       # stretch while moving
                F += np.clip(1 - (((X - bx_) * sy) ** 2 + ((Y - by_) / sy) ** 2) / (1.6 * rp) ** 2, 0, 1) ** 2
            pr = rb * 0.6                                             # molten pool on the bottom
            F += np.clip(1 - (((X - cx) / 1.9) ** 2 + (Y - (yb + pr * 0.4)) ** 2) / (1.6 * pr) ** 2, 0, 1) ** 2
            tr_ = rt * 0.45                                           # a little wax stuck to the top
            F += np.clip(1 - (((X - cx) / 1.6) ** 2 + (Y - (yt - tr_ * 0.3)) ** 2) / (1.6 * tr_) ** 2, 0, 1) ** 2
            liquid = lt[None, None, :] * (1 - uu[..., None]) + lb[None, None, :] * uu[..., None]
            wax = F > 0.35
            depth = np.clip((F - 0.35) / 0.6, 0, 1)[..., None]
            lit = (0.75 + 0.45 * np.clip(uu, 0, 1))[..., None]        # the bulb is below
            waxcol = (wr * (1 - depth) + wc * depth) * lit
            Gw = blur(blur(wax.astype(np.float32), 2), 2)             # wax glows into the liquid
            col = np.where(wax[..., None], waxcol, liquid + Gw[..., None] * wc * 0.45)
            edge = inside & (np.abs(X - cx) > hw - 1.0)
            spec = inside & (np.abs(X - (cx - hw * 0.55)) < 0.75) & (uu > 0.08) & (uu < 0.92)
            v = P[y0:y1, x0:x1]
            v[inside] = col[inside]
            v[edge] = v[edge] * 0.55 + _c((210, 210, 232)) * 0.25
            v[spec] = v[spec] * 0.6 + _c((255, 255, 255)) * 0.4
            if random.random() < 0.1 + 0.3 * self.bass + 0.3 * self.hat:   # tiny bubbles drifting up
                st["bubbles"].append([random.uniform(-0.7, 0.7), 0.95])
            if "note" in self.fired:                                  # one for each melody note, across by pitch
                st["bubbles"].append([self.pitch * 1.4 - 0.7, 0.95, 1.0])
            keep = []
            for bb in st["bubbles"]:
                bb[1] -= dt * 0.12
                if bb[1] > 0.04:
                    keep.append(bb)
                    py = yt + bb[1] * gl
                    px = cx + bb[0] * (rt + (rb - rt) * bb[1]) * 0.8
                    if 0 <= int(py) < H and 0 <= int(px) < w:
                        P[int(py), int(px)] += _c((60, 60, 70)) * (2.2 if len(bb) > 2 else 1.0)
            st["bubbles"] = keep[-40:]
        # ── chrome cap and base, bulb light leaking at the seam ──
        cap_h, base_h = g["cap_h"], g["base_h"]
        self._trap(P, cx, yt - cap_h, yt, rt * 0.55, rt + 0.6, CHROME)
        P[max(0, yt - cap_h), max(0, int(cx - rt * 0.55)):max(0, int(cx + rt * 0.55) + 1)] = (90, 90, 110)
        self._trap(P, cx, yb, yb + base_h * 0.45, rb + 0.8, rb * 0.72, CHROME)
        self._trap(P, cx, yb + base_h * 0.45, table, rb * 0.72, rb * 1.3, CHROME)
        beat = 0.5 + 0.5 * (int(e.beat_pos) % 2 == 0)
        if 0 <= yb < H:
            P[yb, max(0, int(cx - rb)):max(0, min(w, int(cx + rb) + 1))] = np.minimum(255, g["col"] * 0.5 + 150)
        glow(P, cx, yb + 0.5, rb * 0.9, (255, 220, 160), 0.2 + 0.15 * beat + 0.3 * self.kick, ry=1.5)
        vy_ = int(yb + base_h * 0.7)
        for k in range(-2, 3):                                        # vent slots
            vx = int(cx + k * rb * 0.3)
            if 0 <= vy_ < H and 0 <= vx < w:
                P[vy_, vx] = (30, 26, 38)

    def _rare_lava(self, P, ev, g, w):
        """Rare event: a moth flutters in out of the dark, loops round and round the lamp (its shadow
        swooping huge across the wall behind) and finally drifts off."""
        u, r = ev
        H = P.shape[0]
        cx, cy = g["cx"], g["yt"] + g["gl"] * 0.45
        R = g["rb"] + 4
        T = self.rclock
        sx_, ex_ = (-4, w + 4) if r.random() < 0.5 else (w + 4, -4)
        sy_, ey_ = r.uniform(0.05, 0.4) * H, r.uniform(0.05, 0.4) * H
        orbit = lambda a: (cx + math.cos(a) * R * 1.3 + math.sin(a * 2.7) * 2, cy + math.sin(a) * g["gl"] * 0.45
                           + math.sin(a * 3.3 + 1) * 2)
        a = T * 1.9 + math.sin(T * 0.7) * 1.5
        ox, oy = orbit(a)
        if u < 0.18:
            k = u / 0.18
            x, y = sx_ + (ox - sx_) * k, sy_ + (oy - sy_) * k
        elif u > 0.82:
            k = (u - 0.82) / 0.18
            x, y = ox + (ex_ - ox) * k, oy + (ey_ - oy) * k
        else:
            x, y = ox, oy
        x += math.sin(T * 13) * 0.6                                   # the flutter
        y += math.cos(T * 17) * 0.6
        lamp = g["col"]
        d = math.hypot(x - cx, y - cy) / (R * 3)
        lit = max(0.25, 1.2 - d)
        sx, sy = cx + (x - cx) * 2.6, cy + (y - cy) * 2.6              # its shadow, cast big on the wall
        if 0 <= sx < w and 0 <= sy < g["yt"] + g["gl"] * 1.6:
            disc(P, sx, sy, 3.2, (6, 4, 10), a=0.45, ry=1.8)
        up = math.sin(T * 40) > 0
        col = _c((150, 128, 108)) * 0.5 + _c(lamp) * 0.5 * lit
        xi, yi = int(x), int(y)
        for dx_, dy_ in ((0, 0), (-1, -1 if up else 0), (1, -1 if up else 0)):
            if 0 <= xi + dx_ < w and 0 <= yi + dy_ < H:
                P[yi + dy_, xi + dx_] = col * (1.0 if dx_ == 0 else 0.85)
        glow(P, x, y, 1.6, lamp, 0.25 * lit)
