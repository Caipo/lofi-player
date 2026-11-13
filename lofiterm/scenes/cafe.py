import math
import random
import time

import numpy as np

from ..paint import (
    _box, _c, disc, fill_grad, glow, grad_at, grain_tex, grid, line, mix, noise_tex, poly, rect, rrect, vgrad)
from ..world import wseed
from ..term import C_DIM
from ..theory import TAU
from .shared import BOKEH, COATS, SWEATERS, gen_cat


def cat_loaf(P, x, base, g, coat, flick):
    """A cat curled up asleep on a sill (pixel row `base`), head on the left, tail flicking."""
    fur, dark, belly, striped = coat[0], coat[1], coat[2], coat[3]
    hr, L, ry = g["head"], g["long"], g["flat"]
    H, W = P.shape[:2]
    bx = x + hr * 1.6 + L
    disc(P, bx, base - ry, L, fur, ry=ry)                         # body
    if striped:
        for k in range(g["stripes"]):
            sx = bx - L * 0.5 + k * L / max(1, g["stripes"] - 1) * 0.9
            line(P, sx, base - ry * 1.9, sx + 0.6, base - ry * 1.1, dark, wid=0.9)
    hx, hy = x + hr * 1.2, base - hr * 0.95
    for sd in (-1, 1):
        ex = hx + sd * hr * g["ear_sp"]
        poly(P, [(ex - hr * 0.4, hy - hr * 0.4), (ex + sd * g["ear_tilt"] * 0.6, hy - hr * 0.4 - g["ear"] * 0.8),
                 (ex + hr * 0.4, hy - hr * 0.4)], fur)
    disc(P, hx, hy, hr, fur, ry=hr * 0.85)
    disc(P, hx + 0.2, hy + hr * 0.45, hr * 0.55, belly, ry=hr * 0.35)    # muzzle
    for sd in (-1, 1):                                            # closed eyes
        ex, ey = int(hx + sd * hr * 0.45), int(hy)
        if 0 <= ey < H and 0 <= ex < W:
            P[ey, ex] = dark
    disc(P, hx + hr * 0.9, base - 0.8, 1.1, belly, ry=0.7)       # paws
    disc(P, hx + hr * 1.9, base - 0.8, 1.1, belly, ry=0.7)
    prev = None
    for j in range(6):                                            # tail
        u = j / 5
        a = (0.25 + 1.1 * u * g["curl"]) * (1.25 if flick else 1.0)
        px_ = bx + L - 0.5 + math.cos(-a) * u * g["tail"]
        py_ = base - 1 + math.sin(-a) * u * g["tail"] * 0.8
        if prev is not None:
            line(P, prev[0], prev[1], px_, py_, dark if striped and j % 2 else fur, wid=1.1)
        prev = (px_, py_)


HAIR = [(42, 30, 32), (86, 54, 38), (30, 30, 42), (128, 84, 52)]


MENU = [("latte", 4), ("mocha", 5), ("chai", 4), ("matcha", 5), ("cocoa", 4), ("cortado", 4), ("flat white", 5),
        ("hojicha", 4), ("americano", 3), ("london fog", 5), ("affogato", 6), ("oolong", 4), ("yuzu tea", 5), ("toast", 3)]


CAFE = [
    dict(wall=[(40, 26, 38), (76, 48, 56), (86, 56, 58)], wain=[(98, 64, 48), (70, 44, 36)], sign="☾ open late", neon=(255, 150, 200)),
    dict(wall=[(18, 32, 30), (34, 58, 50), (40, 66, 56)], wain=[(70, 46, 36), (46, 30, 24)], sign="study hall", neon=(120, 220, 255)),
    dict(wall=[(90, 54, 40), (150, 96, 66), (166, 110, 76)], wain=[(120, 80, 54), (86, 56, 40)], sign="fresh brew", neon=(255, 170, 90)),
    dict(wall=[(34, 40, 58), (64, 74, 96), (74, 84, 106)], wain=[(88, 80, 90), (60, 54, 64)], sign="warm drinks", neon=(255, 230, 170)),
    dict(wall=[(30, 10, 46), (64, 24, 86), (74, 30, 96)], wain=[(40, 24, 60), (26, 14, 40)], sign="open 24/7", neon=(80, 255, 230)),
    dict(wall=[(32, 44, 36), (62, 84, 64), (72, 94, 72)], wain=[(90, 70, 50), (64, 48, 36)], sign="night tea", neon=(255, 214, 120)),
    dict(wall=[(122, 92, 62), (204, 164, 112), (216, 178, 126)], wain=[(142, 102, 66), (100, 70, 46)], sign="lunch special", neon=(255, 200, 60)),
    dict(wall=[(60, 34, 40), (122, 70, 70), (136, 82, 80)], wain=[(84, 52, 40), (56, 34, 28)], sign="kissa · open", neon=(255, 120, 150)),
    dict(wall=[(26, 12, 18), (58, 26, 34), (68, 32, 40)], wain=[(64, 36, 26), (40, 22, 18)], sign="live jazz", neon=(255, 190, 90)),
    dict(wall=[(28, 34, 30), (52, 66, 58), (64, 80, 70)], wain=[(70, 46, 30), (44, 28, 20)], sign="tea room", neon=(255, 214, 150))
]


class CafeScene:
    # ── coffee shop ────────────────────────────────────────────────────────
    @staticmethod
    def _cafe_view(vh, vw, seed):
        """The night street seen through one café window."""
        rng = random.Random(seed)
        bokeh = [(rng.uniform(1, vh * 0.75), rng.uniform(1, vw - 1), rng.choice(BOKEH), rng.random() * 6,
                  rng.uniform(1.0, 2.4)) for _ in range(vw * vh // 30)]
        sk = np.array([vh * 0.55 + vh * 0.12 * math.sin(x * 0.3 + seed) + (rng.random() * 3 if x % 5 == 0 else 0)
                       for x in range(vw)])
        for x in range(0, vw, rng.randint(4, 7)):
            sk[x:x + rng.randint(3, 6)] -= rng.randint(0, int(vh * 0.2) + 1)
        fwy, fwx = [], []
        for x in range(vw):
            for y in range(int(sk[x]) + 2, vh, 3):
                if rng.random() < 0.18 and x % 2 == 0:
                    fwy.append(y)
                    fwx.append(x)
        layers = np.zeros((3, vh, vw, 3), np.float32)        # blurry lights in three twinkle groups
        for i, (y, x, col, ph, r) in enumerate(bokeh):
            disc(layers[i % 3], x, y, r, col, a=0.3)
            glow(layers[i % 3], x, y, r * 1.3, col, 0.25)
        return dict(bokeh=layers, skyline=sk, fwin=(np.array(fwy, int), np.array(fwx, int)),
                    walkers=[], glass=[], drops=[], splash=[])

    def _maneki(self, P, x, cp, t):
        """A beckoning cat on the counter, its paw going with the beat."""
        white, red, gold = (246, 242, 236), (220, 60, 56), (246, 196, 80)
        disc(P, x, cp - 3, 3.2, white, ry=3)                         # body
        disc(P, x, cp - 8, 2.8, white, ry=2.4)                       # head
        for sx in (-1, 1):
            poly(P, [(x + sx * 2.8, cp - 9), (x + sx * 2.2, cp - 12), (x + sx * 0.8, cp - 10)], white)
        P[int(cp - 8), int(x - 1)] = P[int(cp - 8), int(x + 1)] = (40, 30, 40)
        rect(P, x - 2.5, cp - 6, x + 2.5, cp - 5, red)               # collar and bell
        disc(P, x, cp - 4.6, 0.9, gold)
        up = 0 if self.eng.paused else (1.0 if (self.eng.beat_pos % 1) < 0.5 else 0.0)
        line(P, x + 2.6, cp - 5, x + 3.6, cp - 9 - up * 1.5, white, wid=1.6)
        disc(P, x + 3.6, cp - 9.4 - up * 1.5, 1.0, white)
        rect(P, x - 1.6, cp - 2.5, x + 1.6, cp - 0.5, gold, 0.8)        # the koban coin

    def _upright(self, P, x, base):
        """A double bass resting on its side against the wall; only what clears the counter shows."""
        wood, dark = (176, 92, 46), (96, 46, 24)
        cx = x + 5
        disc(P, cx, base - 6, 5.0, wood, ry=4.6)
        disc(P, cx, base - 14, 3.8, wood, ry=3.8)
        rect(P, cx - 3, base - 12, cx + 3, base - 8, wood)
        for sx in (-1.6, 1.6):                                       # f-holes
            line(P, cx + sx, base - 11, cx + sx, base - 7, dark, wid=0.7)
        line(P, cx, base - 17, cx + 2, base - 32, (40, 24, 20), wid=1.6)   # neck and scroll
        disc(P, cx + 2.2, base - 33, 1.2, dark)
        for k in (-0.6, 0.6):
            line(P, cx + k, base - 4, cx + 2 + k * 0.4, base - 31, (230, 220, 200), wid=0.4, a=0.6)

    def _cafe_window(self, cv, P, t, wx0, wx1, wy0, wy1, k=0):
        wp0, wp1 = 2 * wy0, 2 * wy1
        V = P[wp0 + 1:wp1, wx0 + 1:wx1]
        vh, vw = V.shape[:2]
        c = self.tex(f"cafewin{k}", (vh, vw), lambda: self._cafe_view(vh, vw, wseed("cafeview", k) % 100000))
        day = self.noon()
        V[:] = vgrad(vh, [(0, (84, 150, 232)), (0.6, (150, 200, 246)), (1, (206, 228, 250))] if day else
                     [(0, (14, 16, 42)), (0.6, (44, 32, 74)), (1, (96, 52, 84))])[:, None, :]
        Yv, Xv = grid(vh, vw)
        sk = c["skyline"]
        V[Yv >= sk[None, :]] = (150, 160, 182) if day else (28, 22, 48)
        fy, fx = c["fwin"]
        V[fy, fx] = (110, 140, 176) if day else (190, 140, 96)
        for i in range(3):
            V += c["bokeh"][i] * (0.55 + 0.45 * math.sin(t * 0.8 + i * 2.1)) * (0.12 if day else 1.0)
        # people passing on the street outside (umbrellas up when it's wet)
        walkers = c["walkers"]
        if random.random() < (0.025 if self.eng.cafe_on else 0.004):
            d = random.choice((-1, 1))
            walkers.append([1.0 if d > 0 else vw - 3.0, d * random.uniform(0.12, 0.25), self.eng.wet,
                            random.choice(BOKEH)])
        keep = []
        for wk in walkers:
            wk[0] += wk[1]
            if 0 < wk[0] < vw - 2:
                keep.append(wk)
                x = int(wk[0])
                body = (22, 18, 36)
                V[vh - 8:vh - 2, x:x + 2] = body
                disc(V, x + 1, vh - 9, 1.3, body)
                step = int(wk[0] * 2) % 2
                V[vh - 2:vh, x + step] = body
                if wk[2]:
                    poly(V, [(x - 3, vh - 10), (x + 1, vh - 13.5), (x + 5, vh - 10)], wk[3])
                    V[vh - 10:vh - 8, x + 1] = (60, 56, 70)
        c["walkers"] = keep
        if self._bus:
            self._rare_cafe(V, self._bus, wx0 + 1, vh, vw)
        V[((Xv + Yv * 0.9) % 29) < 2] += 14                       # reflections on the glass
        if self.eng.wet:                                           # droplets sliding down the pane
            g = c["glass"]
            if random.random() < 0.4:
                g.append([random.uniform(0, vw - 1), random.uniform(0, vh * 0.5), random.uniform(0.05, 0.4)])
            alive = []
            for d in g:
                d[1] += d[2] * (1 if random.random() < 0.3 else 0.1)
                if d[1] < vh - 1:
                    alive.append(d)
                    V[int(d[1]), int(d[0])] += (70, 80, 110)
                    if d[1] >= 1:
                        V[int(d[1]) - 1, int(d[0])] += (20, 24, 34)
            c["glass"] = alive[-120:]
        if k == 0:
            self.sky = (wx0 + 1, wx1 - 1, wy0 + 1, wy1 - 1)
        self.weather(c, cv, wx0 + 1, wx1, wy0 + 1, wy1, splash=False)
        # frame + sill
        wood, hi = _c((128, 84, 58)), _c((172, 118, 82))
        mx, my = (wx0 + wx1) // 2, (wp0 + wp1) // 2
        P[wp0, wx0:wx1 + 1] = hi
        P[wp0 + 1, wx0:wx1 + 1] = wood
        P[my, wx0:wx1 + 1] = wood
        for x in (wx0, wx1, mx):
            P[wp0:wp1, x] = wood
        P[wp0:wp1, wx0] = hi
        P[wp1:wp1 + 2, max(0, wx0 - 1):wx1 + 2] = (152, 104, 72)
        P[wp1 + 1, max(0, wx0 - 1):wx1 + 2] = (98, 64, 46)
        P[wp1 + 2, max(0, wx0 - 1):wx1 + 2] *= 0.6
        if k != 0 or wx1 - wx0 < 18:
            return
        # sleepy cat on the sill, tail flicks on the beat
        g = self.layout("cafecat", 0, 0, gen_cat)
        kx = int(wx1 - 3 - g["head"] * 1.6 - 2 * g["long"] - g["tail"] * 0.6)
        cat_loaf(P, kx, wp1, g, COATS[self.eng.station_idx % len(COATS)], int(self.eng.beat_pos) % 2 == 1)
        if int(t * 0.7) % 3 == 0:
            cv.put(wy1 - 4 - int(t * 1.4) % 2, kx + 1 + int(t * 1.4) % 2, "z", C_DIM)

    def v_cafe(self, cv, P, h, w):
        e = self.eng
        t = time.time()
        H = P.shape[0]
        level = float(self.rms.mean())
        cy = h - max(4, int(h * 0.12))               # counter top row (cells)
        cp = 2 * cy
        ww = max(14, min(int(w * 0.48) - 2, 62))     # windows never get absurdly wide…
        wx0, wx1 = 2, 2 + ww
        bx0 = int(w * 0.56) if w <= 150 else wx1 + 10
        bx1 = w - 3 if w <= 150 else min(w - 3, bx0 + 46)
        wins = [(wx0, wx1)]
        if w > 150:                                  # …a wide café just has more of them
            x, ww2 = bx1 + 12, min(ww, 50)
            while x + ww2 < w - 4:
                wins.append((x, x + ww2))
                x += ww2 + 14
        wy0, wy1 = 4, min(max(9, int(h * 0.6)), cy - 5)
        has_window = wy1 - wy0 >= 4
        wp0, wp1 = 2 * wy0, 2 * wy1
        c = getattr(self, "cafe", None)
        if c is None or c["size"] != (h, w):
            c = self.cafe = dict(size=(h, w), steam=[], wall=noise_tex(H, w, 3, 11),
                                 grain=grain_tex(H, w, 4))
        yy, xx = grid(H, w)
        # ── wall: wallpaper above, wood panelling below ──
        V = self.var(CAFE)
        fill_grad(P, 0, cp, list(zip((0, 0.6, 1), V["wall"])))
        P[:cp] *= (1 + 0.03 * c["wall"][:cp])[..., None]
        P[:cp][(xx[:cp] % 7) == 0] *= 0.9
        wt = min(cp - 2, (wp1 + 3) if has_window else int(H * 0.55))
        fill_grad(P, wt, cp, [(0, V["wain"][0]), (1, V["wain"][1])])
        P[wt:cp] *= (1 + 0.06 * c["grain"][wt:cp])[..., None]
        P[wt:cp][(xx[wt:cp] % 9) == 4] *= 0.68
        P[wt] = (146, 98, 70)
        P[wt + 1] = (60, 38, 30)
        ev = self.rare(16.0)                                   # (one seed, so every window sees the same bus)
        self._bus, self._bus_w, self._bus_wins = (ev[0], ev[1].random()) if ev else None, w, wins
        if has_window:
            for k, (ax, bx) in enumerate(wins):
                self._cafe_window(cv, P, t, ax, bx, wy0, wy1, k)
        # ── chalkboard menu, shelf of jars, neon sign ──
        by0 = 4
        by1 = min(by0 + 7, cy - 2)
        neon = None
        if bx1 - bx0 >= 16 and by1 - by0 >= 5:
            iw = bx1 - bx0 - 1
            P[2 * by0:2 * by1 + 2, bx0:bx1 + 1] = (120, 80, 56)
            P[2 * by0, bx0:bx1 + 1] = (160, 110, 76)
            B = P[2 * by0 + 2:2 * by1, bx0 + 1:bx1]
            B[:] = (32, 46, 42)
            B += (c["wall"][2 * by0 + 2:2 * by1, bx0 + 1:bx1] * 5)[..., None]
            chalk = (232, 234, 222)
            cv.text(by0 + 1, bx0 + 1, "~ menu ~".center(iw), chalk)
            for i, (item, price) in enumerate(self.layout("menu", 0, 0, lambda r: r.sample(MENU, 3))):
                if by0 + 2 + i < by1 - 1:
                    cv.text(by0 + 2 + i, bx0 + 2, f"{item} {'.' * (iw - len(item) - 5)} {price}", (170, 184, 172))
            chord = e.meta["chord"] if e.meta else ""
            cv.text(by1 - 1, bx0 + 2, f"♪ {chord}"[: iw - 2], mix((255, 176, 210), (255, 255, 255), self.change))   # chalk flash on a change
            sy = by1 + 2
            if sy < cy - 5:
                P[2 * sy, bx0:bx1 + 1] = (150, 100, 70)
                P[2 * sy + 1, bx0:bx1 + 1] = (84, 54, 40)
                def shelf(r):                            # jars, tins and bottles of all sizes
                    jars = [(110, 196, 190), (255, 196, 150), (196, 170, 240), (130, 170, 110), (230, 120, 110), (240, 220, 140)]
                    out, x = [], bx0 + 1 + r.randint(0, 1)
                    while x < bx1 - 2:
                        wd = r.choice((1, 2, 2, 2, 3))
                        out.append((x, wd, r.randint(2, 5), mix(r.choice(jars), (255, 255, 255), r.uniform(0, 0.3)), r.random() < 0.6))
                        x += wd + r.choice((1, 1, 2))
                    return out
                for x, wd, ht, col, lid in self.layout("shelf", bx0, bx1, shelf):
                    col, x1_ = _c(col), min(bx1 - 1, x + wd)
                    P[2 * sy - ht:2 * sy, x:x1_] = col
                    if lid:
                        P[2 * sy - ht, x:x1_] = (70, 60, 70)
                    P[2 * sy - ht + 1, x] = np.minimum(255, col + 60)
            ny = sy + 3
            if ny < cy - 5:
                sign = V["sign"]
                on = not (random.random() < 0.04 or (int(t * 3) % 23 == 0))
                nx = (bx0 + bx1) // 2 - len(sign) // 2
                col = V["neon"] if on else (90, 70, 90)
                cv.text(ny - 1, nx - 2, "╭" + "─" * (len(sign) + 2) + "╮", col)
                cv.text(ny, nx - 2, "│ " + sign + " │", col)
                cv.text(ny + 1, nx - 2, "╰" + "─" * (len(sign) + 2) + "╯", col)
                if on:
                    neon = (nx - 2, 2 * (ny - 1), len(sign) + 4, 6)
        # ── fireplace on the back wall ──
        fire = None
        if e.fire_on:
            fx0 = max(int(w * 0.72) + 9, int(w * 0.8))
            if fx0 + 12 < w:
                fire = fx0
                Fb = P[cp - 12:cp, fx0:fx0 + 12]
                Fb[:] = (126, 60, 48)
                Yf, Xf = grid(12, 12)
                Fb[(Yf % 3 == 0) | (((Xf + (Yf // 3) % 2 * 2) % 4) == 0)] = (88, 42, 36)
                P[cp - 13:cp - 11, fx0 - 1:fx0 + 13] = (116, 74, 52)
                P[cp - 13, fx0 - 1:fx0 + 13] = (160, 110, 76)
                P[cp - 8:cp, fx0 + 2:fx0 + 10] = (22, 12, 12)
        # ── regulars at the counter ──
        busy = e.cafe_on
        jit = self.layout("seats", h, w, lambda r: [r.uniform(-0.02, 0.02) for _ in range(6)])
        if w <= 150:
            seats = [int(w * (f + j)) for f, j in zip([0.2, 0.27, 0.53, 0.6, 0.66, 0.92] if busy else [0.6], jit)]
        elif busy:                                   # a long counter fills up all the way along
            seats = [x for j, x in enumerate(range(18, w - 6, 13)) if (j * 7) % 5 != 0]
        else:
            seats = list(range(int(w * 0.3), w - 6, 90))
        for i, x in enumerate(seats):
            if not (3 <= x < w - 4):
                continue
            lean = 1 if busy and (int(e.beat_pos) + i) % 4 == 0 else 0
            nod = (self.kick if i % 2 == 0 else self.snare) * (0.9 if busy else 0.6)    # heads nod along, half on each drum
            si, hi_, dh, bun, broad = self.layout("regulars", 0, 0, lambda r: [
                (r.randrange(len(SWEATERS)), r.randrange(len(HAIR)), r.uniform(-1.0, 1.5), r.random() < 0.3, r.uniform(-0.5, 0.8))
                for _ in range(48)])[i % 48]
            sw, hair = _c(SWEATERS[si]), HAIR[hi_]
            k = max(1.5, min(2.1, (cp - 6) / 15))          # people are far bigger than the mugs on the counter
            dh, nod = dh * k, nod * k
            sh_y, top = cp - 7.5 * k - dh, cp - 9 * k - dh    # shoulders, neck
            poly(P, [(x - (4.6 + broad) * k, cp), (x - (4.2 + broad) * k, cp - 4.5 * k), (x - 2.2 * k, sh_y), (x + 2.2 * k, sh_y),
                     (x + (4.2 + broad) * k, cp - 4.5 * k), (x + (4.6 + broad) * k, cp)], sw)
            poly(P, [(x + 1.2 * k, sh_y), (x + 2.2 * k, sh_y), (x + (4.2 + broad) * k, cp - 4.5 * k), (x + (4.6 + broad) * k, cp),
                     (x + 2.4 * k, cp)], sw * 0.78)                                   # the side away from the lamps
            line(P, x - 1.2 * k, sh_y + 0.5, x + 1.2 * k, sh_y + 0.5, np.minimum(255, sw * 1.25), wid=1.0)   # collar
            rect(P, x - 0.9 * k + lean, top - 0.5 + nod, x + 0.9 * k + lean + 1, sh_y + 1, (196, 150, 124))   # neck
            hx_, hy_ = x + lean * k * 0.6 + 0.5, top - 1.7 * k + nod
            disc(P, hx_, hy_, 2.3 * k, hair, ry=2.5 * k)                              # seen from behind: all hair
            disc(P, hx_ - 0.7 * k, hy_ - 0.8 * k, 1.2 * k, np.minimum(255, _c(hair) + 30), a=0.55)
            if bun:
                disc(P, hx_ - 0.3, hy_ - 2.9 * k, 1.2 * k, hair)
            else:
                rect(P, hx_ - 2.2 * k, hy_, hx_ + 2.2 * k + 1, hy_ + 2.2 * k, hair)   # hair down to the collar
            if (int(t * 0.3) + i) % 7 == 0:                                           # lifting a cup
                rect(P, x + 3.4 * k, cp - 6 * k, x + 3.4 * k + 3, cp - 6 * k + 3, (240, 236, 228))
            if busy and (int(t * 0.6) + i * 2) % 5 == 0:
                cv.text(cy - 12 - (i % 2), x + 4, random.choice(("…", "ha", "♪", "~", "…!")), C_DIM)
        # ── counter ──
        P[cp:] = vgrad(H - cp, [(0, (108, 70, 50)), (1, (64, 40, 32))])[:, None, :]
        P[cp:] *= (1 + 0.07 * c["grain"][cp:])[..., None]
        P[cp:][(xx[cp:] % 12) == 0] *= 0.7
        P[cp] = (184, 128, 90)
        P[cp + 1] = (140, 92, 64)
        # ── espresso machine ──
        ex = 3
        chrome = vgrad(9, [(0, (110, 112, 128)), (0.35, (236, 238, 248)), (0.6, (180, 182, 198)), (1, (96, 98, 114))])
        P[cp - 11:cp, ex:ex + 9] = chrome[None, :, :]
        P[cp - 11, ex:ex + 9] = (60, 60, 74)
        P[cp - 9:cp - 6, ex + 1:ex + 8] = (40, 40, 54)
        disc(P, ex + 4.5, cp - 7.5, 1.2, (236, 236, 242))
        P[cp - 8, ex + 4] = (40, 40, 54)
        P[cp - 7, ex + 7] = (120, 230, 140)
        P[cp - 5:cp - 3, ex + 3:ex + 6] = (70, 70, 84)
        line(P, ex + 5, cp - 4.2, ex + 11, cp - 3.6, (36, 30, 34), 1.4)
        line(P, ex + 8.5, cp - 9, ex + 10.5, cp - 3, (160, 160, 176), 0.9)
        P[cp - 13:cp - 11, ex + 1:ex + 3] = (236, 232, 226)
        P[cp - 13:cp - 11, ex + 5:ex + 7] = (236, 232, 226)
        P[cp - 2:cp, ex + 3:ex + 6] = (240, 236, 228)
        P[cp - 1, ex:ex + 9] = (50, 50, 62)
        # ── little radio with a live equalizer ──
        rx = int(w * 0.36)
        line(P, rx + 11.5, cp - 7, rx + 15, cp - 14, (150, 150, 168), 1)
        body = rrect(H, w, rx, cp - 8, 13, 8, 2)
        P[body] = grad_at(H, cp - 8, cp, [(0, (204, 158, 210)), (1, (140, 100, 160))])[yy[body].astype(int)]
        gr = body & (xx < rx + 4) & (xx > rx) & (yy > cp - 7) & ((xx + yy) % 2 == 0)
        P[gr] *= 0.55 + 0.35 * self.kick                  # the speaker thumps
        P[cp - 7:cp - 1, rx + 5:rx + 12] = (24, 20, 38)
        lv, _ = self.band_levels(7)
        for i, v in enumerate(lv):
            hh = int(round(v * 6))
            if hh:
                P[cp - 1 - hh:cp - 1, rx + 5 + i] = (255, 160, 206) if i % 2 else (128, 222, 214)
        self.note_sparks(P, "cafe", lambda p, n: (rx + 15 + (n % 3 - 1) * 1.5, cp - 14 - p * 3), (255, 196, 226),
                         rise=6.0, drift=1.2, life=1.8, r=0.8)       # the melody drifts off the radio's aerial
        # ── mug of coffee ──
        mugs = [int(w * 0.72)] + ([int(w * f) for f in (0.16, 0.5, 0.9)] if w > 150 else [])
        for ux in mugs:
            if ux + 9 >= w:
                continue
            mug = grad_at(w, ux, ux + 6, [(0, (250, 246, 240)), (0.6, (226, 220, 214)), (1, (170, 164, 160))])
            P[cp - 8:cp, ux:ux + 6] = mug[ux:ux + 6][None, :, :]
            P[cp - 8, ux:ux + 6] = (92, 56, 40)
            P[cp - 7, ux + 1:ux + 5] = (120, 76, 52)
            P[cp - 6, ux + 6:ux + 8] = (210, 204, 200)
            P[cp - 5:cp - 3, ux + 8] = (200, 194, 190)
            P[cp - 3, ux + 6:ux + 8] = (190, 184, 180)
        th = self.theme()
        if th == "japan":
            if has_window:                           # noren split curtains over the windows
                for ax, bx in wins:
                    nh = max(3, (wp1 - wp0) * 2 // 5)
                    pw = max(4, (bx - ax - 1) // 4)
                    for k in range(ax + 1, bx, pw):
                        rect(P, k, wp0 + 1, min(bx, k + pw - 1), wp0 + 1 + nh, (36, 44, 96))
                        P[wp0 + 1, k:min(bx, k + pw - 1)] = (22, 26, 60)
                        P[wp0 + nh, k:min(bx, k + pw - 1)] = (150, 156, 196)          # the hem
                        P[wp0 + 1:wp0 + 1 + nh, min(bx, k + pw - 1) - 1:min(bx, k + pw - 1)] *= 0.6
                    disc(P, (ax + bx) / 2, wp0 + 1 + nh * 0.55, max(1.5, nh * 0.28), (236, 230, 220))
                    disc(P, (ax + bx) / 2, wp0 + 1 + nh * 0.55, max(0.8, nh * 0.14), (44, 58, 118))
            self._maneki(P, int(w * 0.36) + 19, cp, t)
        elif th == "jazz":
            if has_window:                           # blinds half down
                for ax, bx in wins:
                    for y in range(wp0 + 1, wp0 + 1 + (wp1 - wp0) // 2, 2):
                        P[y, ax + 1:bx] = mix((190, 150, 110), (60, 40, 30), 0.3)
            ux = int(w * 0.47)
            if ux + 12 < w:                          # an upright bass leaning behind the counter
                self._upright(P[:cp], ux, cp)
            self.notes(cv, int(w * 0.36) + 8, (cp - 16) // 2, (255, 200, 130), n=3, rise=5, seed=1)
        elif th == "english":
            if has_window:                           # leaded glass: diamond panes in lead cames
                for ax, bx in wins:
                    sub = P[wp0 + 1:wp1, ax + 1:bx]
                    yy_, xx_ = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
                    came = ((xx_ + yy_) % 5 == 0) | ((xx_ - yy_) % 5 == 0)
                    sub[came] = sub[came] * 0.35 + 8
            self.candles(P, int(w * 0.36) + 20, cp, 11, self.rclock)
            px_ = int(w * 0.36) + 27                 # a teapot beside them
            if px_ + 6 < w:
                disc(P, px_ + 2.5, cp - 2.5, 2.6, (236, 232, 222), ry=2.2)
                line(P, px_ + 5, cp - 3, px_ + 7, cp - 5, (236, 232, 222), wid=0.8)
                disc(P, px_ + 2.5, cp - 4.8, 0.9, (40, 70, 120))
                rect(P, px_ + 0.5, cp - 2.6, px_ + 4.5, cp - 2, (40, 70, 120))
        # ── light: lamps, fireplace, neon (light map multiplies the room) ──
        n_l = max(3, round(w / 36))
        lj = self.layout("lamps", h, w, lambda r: [r.uniform(-0.035, 0.035) for _ in range(3)])
        lamps = [int(w * (f + j)) for f, j in zip((0.25, 0.62, 0.86), lj)] if w <= 150 else [int((i + 0.5) * w / n_l) for i in range(n_l)]
        bright = 0.5 + 0.5 * self.bass + 0.1 * self.energy + 0.1 * self.kick
        ly = 6

        def lightmap():
            L0 = np.zeros((H, w), np.float32)
            for lx in lamps:
                dy = yy - ly
                spread = 1.5 + np.maximum(dy, 0) * 0.55
                L0 += (dy > 0) * np.clip(1 - np.abs(xx - lx) / spread, 0, 1) ** 0.7 * np.exp(-dy / (H * 0.9)) * 0.8
                L0 += np.exp(-((xx - lx) ** 2 + (yy - ly) ** 2) / 196) * 0.5
            return L0
        Lm = self.tex("cafelight", (H, w, tuple(lamps)), lightmap) * bright
        if fire is not None:
            Lm += np.exp(-((xx - fire - 6) ** 2 + (yy - cp + 4) ** 2) / 300) * (0.6 + 0.4 * random.random())
        mult = 0.55 + np.minimum(Lm, 1.5)[..., None] * _c((1.05, 0.86, 0.62))
        if has_window:
            for ax, bx in wins:
                mult[wp0 + 2:wp1, ax + 1:bx] = 1.0
        P *= mult
        for lx in lamps:
            P[0:ly - 2, lx] = (90, 84, 100)
            poly(P, [(lx - 3.5, ly + 0.5), (lx - 1, ly - 3), (lx + 2, ly - 3), (lx + 4.5, ly + 0.5)], (214, 154, 84))
            P[ly - 2, lx - 1:lx + 2] = (246, 196, 120)
            P[ly, lx - 1:lx + 2] = (255, 242, 204)
            glow(P, lx + 0.5, ly + 1, 3, (255, 210, 140), 0.8 * bright)
        if fire is not None:
            self.flames(P, fire + 6, cp - 1, 0.6 + self.bass, 3.6, "cafe")
            P[cp - 1, fire + 3:fire + 9] = (110, 66, 44)
            glow(P, fire + 6, cp - 4, 5, (255, 130, 50), 0.4)
        if neon is not None:
            nx, ny0, nw, nh = neon
            b_ = _box(P, nx - 8, ny0 - 8, nx + nw + 8, ny0 + nh + 8)     # glow only around the sign
            if b_ is not None:
                v_, by_, bx_ = b_
                dx_ = np.maximum(np.maximum(nx - bx_, bx_ - (nx + nw)), 0)
                dy_ = np.maximum(np.maximum(ny0 - by_, by_ - (ny0 + nh)), 0)
                v_ += (np.exp(-(dx_ ** 2 + dy_ ** 2) / 12) * (0.38 + 0.3 * self.snare))[..., None] * _c(V["neon"])
        # mug steam thickens as the music swells
        for ux in mugs:
            if random.random() < 0.2 + 2 * level + 0.6 * self.kick:
                c["steam"].append([ux + random.uniform(1.5, 3.5), cy - 5.0, random.random() * TAU])
        alive = []
        for s in c["steam"]:
            s[1] -= 0.22
            if s[1] > cy - 13:
                alive.append(s)
                dx = math.sin(t * 2 + s[2] + s[1] * 0.6)
                ch = ")" if dx > 0.35 else "(" if dx < -0.35 else "~"
                cv.put(int(s[1]), int(s[0] + dx), ch, (200, 190, 196) if s[1] > cy - 9 else (130, 116, 128))
        c["steam"] = alive[-80 * len(mugs):]

    BUS = [(196, 64, 60), (60, 140, 140), (232, 176, 60), (220, 210, 186), (70, 100, 170)]

    def _rare_cafe(self, V, ev, ox, vh, vw):
        """Rare event: a city bus pulls up outside, idles at the stop with its windows lit and a few
        passengers inside, then pulls away. It passes every window in turn (`ox` is this window's x)."""
        u, r = ev[0], random.Random(ev[1])
        w = self._bus_w
        f = r.choice((-1, 1))
        L = max(26.0, min(70.0, vh * 3.2))
        bh = max(6, min(vh - 2, int(vh * 0.55)))
        ax, bx = r.choice(self._bus_wins)                                       # it stops in front of a window
        stop = (ax + bx) / 2
        start, end = (-L / 2 - 10, w + L / 2 + 10) if f > 0 else (w + L / 2 + 10, -L / 2 - 10)
        if u < 0.32:
            k = u / 0.32
            mid = start + (stop - start) * (1 - (1 - k) ** 2)                  # brakes into the stop
        elif u < 0.62:
            mid = stop
        else:
            k = (u - 0.62) / 0.38
            mid = stop + (end - stop) * k ** 2                                 # pulls away
        x0 = mid - L / 2                                                        # body, in screen columns
        a, b = int(round(x0 - ox)), int(round(x0 + L - ox))
        lo, hi = max(0, a), min(vw, b)
        if lo >= hi:
            return
        day = self.noon()
        body = _c(r.choice(self.BUS)) * (1.0 if day else 0.55)
        y0, y1 = vh - bh, vh - 1
        V[y0:y1, lo:hi] = body
        V[y0, lo:hi] = body * 1.25
        V[y1 - 2:y1, lo:hi] = body * 0.6                                         # skirt
        wy0, wy1 = y0 + 2, y0 + max(3, int(bh * 0.42))
        winc = _c((70, 88, 112)) if day else _c((255, 214, 150))
        for xw in range(a + 3, b - 3, 6):                                      # windows, a passenger or two
            ws, we = max(lo, xw), min(hi, xw + 4)
            if ws < we:
                V[wy0:wy1, ws:we] = winc
                if (xw * 7 + int(L)) % 3 == 0 and we - ws >= 2:
                    V[wy1 - 2:wy1, ws + 1:ws + 2] = (40, 32, 50)
                    V[wy1 - 3, ws + 1] = (40, 32, 50)
        for k in (0.18, 0.8):                                                   # wheels
            wx = a + L * k
            if lo - 2 <= wx < hi + 2:
                disc(V, wx, y1, 2.2, (24, 22, 28))
                disc(V, wx, y1, 0.9, (120, 120, 130))
        nose = b - 1 if f > 0 else a
        if lo <= nose < hi and not day:
            glow(V, nose, y1 - 2, 6, (255, 240, 200), 0.6)
            V[y1 - 3:y1 - 1, nose] = (255, 250, 230)
        door = int(nose - f * 4)                                                # the door opens at the stop
        if 0.36 < u < 0.58 and lo <= door < hi - 1:
            V[wy0:y1 - 1, door:door + 2] = winc * 0.6
