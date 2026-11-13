import math
import random
import time

import numpy as np

from ..paint import _c, blur, disc, draw_stars, fill_grad, fog_rows, glow, grid, line, make_stars, mix, noise_tex, poly, rect, vgrad
from ..world import wrng
from ..theory import TAU
from .shared import BOKEH, CITY


class CityScene:
    def v_city(self, cv, P, h, w):
        e = self.eng
        t = time.time()
        H = P.shape[0]
        level = float(self.rms.mean())
        gp = min(H - 3, 2 * max(4, int(h * 0.8)))
        c = self.city
        V = self.var(CITY)
        if c is None or c["size"] != (h, w, e.station_idx):
            rng = wrng("city", h, w)
            tall = rng.uniform(0.8, 1.25)                    # this city's character: how tall, how wide, how fancy
            fancy = rng.uniform(0.25, 0.7)
            layers = []
            for li, (lo, hi, bmin, bmax) in enumerate([(0.32, 0.64, 5, 10), (0.22, 0.52, 4, 9), (0.12, 0.42, 5, 12)]):
                lo, hi = lo * tall * rng.uniform(0.85, 1.15), min(0.92, hi * tall * rng.uniform(0.85, 1.15))
                bmin = max(3, bmin + rng.randint(-1, 2))
                bmax = max(bmin + 1, bmax + rng.randint(-2, 4))
                top = np.full(w, float(gp))
                wy, wx, ants, signs = [], [], [], []
                x = rng.randint(-4, 0)
                while x < w:
                    bw = rng.randint(bmin, bmax)
                    yt = gp - int(gp * rng.uniform(lo, hi))
                    xa, xb = max(0, x), min(w, x + bw)
                    if xa < xb:
                        top[xa:xb] = np.minimum(top[xa:xb], yt)
                        roof = rng.random() / fancy
                        if bw >= 6 and roof < 0.35:                      # stepped roof
                            ys_, ins = yt, 0
                            for _ in range(rng.randint(1, 3)):
                                ins += rng.randint(1, max(1, bw // 4))
                                m0, m1 = max(0, x + ins), min(w, x + bw - ins)
                                ys_ -= rng.randint(1, 4)
                                top[m0:m1] = np.minimum(top[m0:m1], ys_)
                                if m1 - m0 < 4:
                                    break
                        elif bw >= 5 and roof < 0.5:                     # pitched roof or a spire
                            sl = rng.choice((0.5, 0.8, 1.6, 2.5))
                            cxb = x + bw / 2
                            for xx_ in range(xa, xb):
                                top[xx_] = min(top[xx_], yt - int((bw / 2 - abs(xx_ + 0.5 - cxb)) * sl))
                        elif bw >= 7 and roof < 0.6:                     # a dome
                            rad = bw / 2 - 0.5
                            for xx_ in range(xa, xb):
                                dx_ = xx_ + 0.5 - (x + bw / 2)
                                top[xx_] = min(top[xx_], yt - int(math.sqrt(max(0.0, rad * rad - dx_ * dx_)) * 0.8))
                        if li == 2 and rng.random() < 0.3:
                            ants.append((min(w - 1, max(0, x + bw // 2)), int(top[min(w - 1, max(0, x + bw // 2))]), rng.random() * TAU))
                        if li == 2 and rng.random() < V["signs"] and bw >= 6:
                            signs.append((min(w - 2, max(0, x + 1)), yt + 4, rng.randint(6, 10), rng.choice(BOKEH[:3]), rng.random() * 9))
                        py_, px_ = rng.choice((2, 3, 3, 4)), rng.choice((2, 2, 3))   # this building's window grid
                        band = rng.random() < 0.2                        # or long ribbon windows
                        for yy_ in range(yt + 3, gp - 2, py_):
                            for xx_ in range(x + 1, x + bw - 1, 1 if band else px_):
                                if 0 <= xx_ < w:
                                    wy.append(yy_)
                                    wx.append(xx_)
                    x += bw + rng.choice([0, 0, 1, 2])
                n = len(wy)
                layers.append(dict(top=top, wy=np.array(wy, int), wx=np.array(wx, int),
                                   lit=np.array([rng.random() < V["lit"] for _ in range(n)], bool),
                                   tone=np.array([rng.random() for _ in range(n)]),
                                   ph=np.array([rng.random() for _ in range(n)]), ants=ants, signs=signs))
            c = self.city = dict(size=(h, w, e.station_idx), layers=layers, drops=[], splash=[],
                                 stars=make_stars(rng, int(gp * 0.55), w, w * h // 30),
                                 clouds=noise_tex(gp, w, 7, 2), lamps=list(range(rng.randint(2, 14), w - 3, rng.randint(17, 32))),
                                 moon=(rng.uniform(0.12, 0.9) * w, H * rng.uniform(0.1, 0.24)))
        yy, xx = grid(H, w)
        Y = yy[:gp]
        # sky, moon, clouds lit by the city
        fill_grad(P, 0, gp, list(zip((0, 0.5, 0.82, 1), V["sky"])))
        self.chord_wash(P, gp // 3, gp, 0.05)                 # the glow over the city takes the chord's colour
        draw_stars(P, c["stars"], t, (0.35 if e.wet else 1.0) * V["stars"])
        mx_, my_ = c["moon"]
        if V["sun"]:                                     # golden hour: a low sun behind the towers
            glow(P, mx_, gp * 0.78, H * 0.3, (255, 150, 90), 0.5)
            disc(P, mx_, gp * 0.78, max(4.0, H * 0.09), (255, 226, 160))
        else:
            self.sky_body(P, mx_, my_, max(3.0, H * 0.065), dim=0.7 if e.wet else 1.0)
        n = np.roll(c["clouds"], int(t * 1.5) % w, axis=1)
        band = np.exp(-((np.arange(gp) / gp - 0.3) / 0.2) ** 2)[:, None]
        a = np.clip((n - (0.0 if e.wet else 0.45)) * 0.8, 0, 0.85) * band
        P[:gp] += (vgrad(gp, [(0, V["cloud"][0]), (1, V["cloud"][1])])[:, None, :] - P[:gp]) * a[..., None]
        ev = self.rare(24.0)
        if ev:
            self._rare_city(P, ev, gp, w, np.min([L["top"] for L in c["layers"]], axis=0))
        # three layers of buildings, hazier with distance
        hz = _c(V["haze"])
        noon = self.noon()
        cols = ([(hz * 0.96, hz * 0.9), (_c((206, 190, 176)), _c((176, 160, 152))),       # by day: sandstone and brick in the sun
                 (_c((204, 150, 112)), _c((150, 104, 86)))] if noon else
                [(hz * 0.62, hz * 0.5), (hz * 0.31, hz * 0.24), (hz * 0.14 + 5, hz * 0.08 + 4)])
        wk = [0.4, 0.7, 1.0]
        th = self.theme()
        for li, L in enumerate(c["layers"] if th != "english" else ()):     # (Candlelit Manor has its own old town)
            if li == 2 and th == "japan":                # a lit lattice tower rising behind the front row
                self._city_tower(P, w, gp, H, t)
            m = Y >= L["top"][None, :]
            g = vgrad(gp, [(0, cols[li][0]), (1, cols[li][1])])
            P[:gp][m] = np.broadcast_to(g[:, None, :], (gp, w, 3))[m]
            rim = m & (Y < L["top"][None, :] + 1)
            P[:gp][rim] += np.array([28, 20, 40], np.float32) * (1 - li * 0.3)
            step = np.diff(L["top"], prepend=L["top"][0])        # each block's light-facing wall catches the sky; the far wall is in shade
            toward = 1 if mx_ < w / 2 else -1
            P[:gp][m & (step * toward < -1)[None, :]] *= 1.22 if noon else 1.12
            P[:gp][m & (np.roll(step, -1) * toward > 1)[None, :]] *= 0.8
            lit = L["lit"]
            lit ^= np.random.random(len(lit)) < (0.03 if "change" in self.fired else 0.0015)   # lights go on and off with the chords
            wy, wx = L["wy"], L["wx"]
            ok = wy >= L["top"][wx] + 1
            on, off = lit & ok, ~lit & ok
            warm = np.where((L["tone"][on] < 0.78)[:, None], (255, 204, 128), (168, 200, 255)).astype(np.float32)
            if noon:                                    # windows catch the sky by day
                warm[:] = (150, 180, 214)
            hot = (self.kick > 0.5) & (L["ph"][on] < 0.3)
            warm[hot] = (255, 168, 158)
            warm *= wk[li]
            if li == 2:
                P[np.clip(wy[on] + 1, 0, gp - 1), wx[on]] = warm * 0.8
                P[wy[off], wx[off]] = (74, 96, 130) if noon else (36, 32, 56)
            P[wy[on], wx[on]] = warm
            if li == 2:                                  # the melody: a window flares for every note, higher notes higher up
                fl = c.get("flare")
                if fl is None or len(fl) != len(wy):
                    fl = c["flare"] = np.zeros(len(wy), np.float32)
                fl *= 0.9
                if "note" in self.fired and len(fl):
                    cand = np.nonzero(ok & (np.abs(wy - (gp - 3 - self.pitch * gp * 0.45)) < 3))[0]
                    if len(cand):
                        fl[cand[(self.frame * 7) % len(cand)]] = 1.0
                for j in np.nonzero(fl > 0.05)[0]:
                    P[wy[j], wx[j]] = mix((255, 236, 190), tuple(self.tint), 0.3)
                    glow(P, wx[j], wy[j], 2.2, tuple(self.tint), 0.5 * fl[j])
            Gm = np.zeros((H, w), np.float32)
            Gm[wy[on], wx[on]] = 1
            if not noon:
                P += blur(Gm, 1)[..., None] * np.array([255, 180, 110], np.float32) * 0.5 * wk[li]
            for x, yt, ph in L["ants"]:
                P[max(0, yt - 4):max(0, yt), x] = cols[li][0] * 0.8
                if ((t * 1.3 + ph) % 2 < 0.5 or (self.kick > 0.6 and ph < 3)) and yt >= 5:
                    P[yt - 5, x] = (255, 80, 90)
                    glow(P, x, yt - 5, 1.8, (255, 60, 70), 0.5)
            for x, y0, sh, col, ph in L["signs"]:
                on_ = not (random.random() < 0.03 or int(t * 3 + ph) % 29 == 0)
                k_ = 1.0 if on_ else 0.2
                P[y0:y0 + sh, x:x + 2] = _c(col) * k_
                if on_:
                    glow(P, x + 1, y0 + sh / 2, 3, col, 0.28 + 0.35 * self.snare, ry=sh * 0.7)
            if li < 2:
                fog_rows(P, gp * 0.4, gp, V["haze"], 0.3)
        # street lamps
        for lx in c["lamps"] if th != "english" else ():
            P[gp - 12:gp, lx] = (42, 36, 58)
            P[gp - 12, lx:lx + 3] = (42, 36, 58)
            P[gp - 11, min(w - 1, lx + 2)] = (214, 216, 222) if noon else (255, 236, 186)
            if not noon:
                glow(P, lx + 2, gp - 11, 3.5, (255, 200, 130), 0.55 + 0.2 * self.kick)
        if th == "japan":                                # strings of red paper lanterns over the street
            self._city_lanterns(P, w, gp, t)
        elif th == "jazz":                               # a club's marquee, bulbs chasing, music leaking out
            self._city_marquee(P, cv, w, gp, t)
        elif th == "english":                            # a medieval town: cathedral, timber-framed houses
            self._city_old(P, gp, w, h, t, V, noon)
        # wet street mirrors everything above it, rippling with the music
        sh = H - gp
        P[gp:] = vgrad(sh, [(0, (150, 146, 150)), (1, (96, 94, 104))] if noon else [(0, (34, 26, 50)), (1, (12, 10, 22))])[:, None, :]
        k = np.arange(sh)
        src = np.clip(gp - 1 - (k * 2.2).astype(int), 0, gp - 1)
        amp = 0.6 + level * 8 + 1.5 * self.snare
        shift = np.round(np.sin(k * 1.1 + t * 2.2) * amp * (0.4 + k / max(1, sh))).astype(int)
        idx = (np.arange(w)[None, :] - shift[:, None]) % w
        R = P[src[:, None], idx]
        lv, _ = self.band_levels(w)
        a = (0.25 + 0.5 * lv[None, :] + (0.1 if e.wet else 0)) * (1 - k / max(1, sh) * 0.55)[:, None]
        P[gp:] += (R - P[gp:]) * a[..., None]
        P[gp] = (84, 70, 104)
        P[gp + 1] *= 0.7
        for lx in c["lamps"] if not noon and th != "english" else ():
            glow(P, lx + 2, gp + 2.5, 6, (255, 190, 120), 0.22, ry=2.5)
        if th == "english":                              # cobbles under the puddles, lantern light pooling on them
            cob = self.tex("cobbles", (sh, w), lambda: np.where(
                (np.arange(sh)[:, None] % 2 == 1) | (((np.arange(w)[None, :] + (np.arange(sh)[:, None] // 2 % 2) * 2) % 4) == 0),
                0.72, 1.0).astype(np.float32))
            P[gp + 1:] *= cob[1:, :, None]
            for lx, ly in self.city_old["lanterns"] if not noon else ():
                glow(P, lx, gp + 2.5, 5, (255, 180, 100), 0.25 + 0.1 * self.kick, ry=2.5)
        self.sky = (0, w, 0, gp // 4)
        self.weather(c, cv, 1, w - 1, 0, gp // 2)

    def _city_old_build(self, h, w, gp, V, noon):
        """The old town, built once per size and world: a gothic cathedral and
        a row of distant gables (mid layer), and a street of timber-framed houses with jettied upper floors,
        steep gables, chimneys and the odd stone tower (front layer). Each layer is pixels plus a mask;
        windows, lanterns and signs are listed so they can flicker and react every frame."""
        r = wrng("oldtown", h, w)
        hz = _c(V["haze"])
        dim = 1.0 if noon else 0.5
        Bs = [np.zeros((gp, w, 3), np.float32) for _ in range(2)]
        Ms = [np.zeros((gp, w, 3), np.float32) for _ in range(2)]
        wins, lanterns, signs = [], [], []

        def R(li, x0, y0, x1, y1, col):
            rect(Bs[li], x0, y0, x1, y1, col)
            rect(Ms[li], x0, y0, x1, y1, (1, 1, 1))

        def Pg(li, pts, col):
            poly(Bs[li], pts, col)
            poly(Ms[li], pts, (1, 1, 1))

        def win(li, x, y, ww, wh, p_lit):
            wins.append((int(y), int(x), ww, wh, li, r.random() < p_lit, r.random()))

        # ── mid layer: the cathedral and a jumble of distant roofs ──
        mid = hz * 0.36
        s_ = gp * r.uniform(0.6, 0.7)
        cx = w * r.uniform(0.25, 0.75)
        nl, nr = cx - s_ * 0.5, cx + s_ * 0.45
        R(0, nl, gp - s_ * 0.34, nr, gp, mid)                                     # nave
        Pg(0, [(nl, gp - s_ * 0.34), (nl + s_ * 0.12, gp - s_ * 0.46), (nr - s_ * 0.06, gp - s_ * 0.46), (nr, gp - s_ * 0.34)], mid * 0.85)
        for k in range(int((nr - nl) / max(2.0, s_ * 0.09))):                      # buttresses and lancet windows
            bx = nl + s_ * 0.2 + k * max(2.0, s_ * 0.09)
            if bx < nr - 1:
                R(0, bx, gp - s_ * 0.3, bx + 1, gp, mid * 0.75)
                win(0, bx + max(1.0, s_ * 0.03), gp - s_ * 0.26, 1, max(2, int(s_ * 0.1)), 0.8)
        tw = max(2.0, s_ * 0.08)
        for tx in (nl, nl + s_ * 0.2 - tw):                                      # the west towers, pinnacled
            R(0, tx, gp - s_ * 0.66, tx + tw, gp, mid * 1.05)
            Pg(0, [(tx - 0.5, gp - s_ * 0.66), (tx + tw / 2, gp - s_ * 0.8), (tx + tw + 0.5, gp - s_ * 0.66)], mid * 0.9)
            win(0, tx + tw / 2 - 0.5, gp - s_ * 0.56, 1, max(2, int(s_ * 0.08)), 0.6)
        ccx = nl + s_ * 0.6                                                        # the crossing tower and its spire
        R(0, ccx - tw * 0.7, gp - s_ * 0.62, ccx + tw * 0.7, gp - s_ * 0.3, mid * 1.05)
        Pg(0, [(ccx - tw * 0.7, gp - s_ * 0.62), (ccx, gp - s_), (ccx + tw * 0.7, gp - s_ * 0.62)], mid * 0.9)
        rw = max(1.0, s_ * 0.045)                                                  # the rose window over the west door
        wins.append((int(gp - s_ * 0.5), int(nl + s_ * 0.1 - rw / 2), int(rw * 2) or 1, int(rw * 2) or 1, 0, True, 0.5))
        x = r.randint(-3, 0)
        while x < w:                                                               # distant gables
            ww = r.randint(5, 9)
            hh = gp * r.uniform(0.14, 0.26)
            if not (nl - 2 < x + ww / 2 < nr + 2) or r.random() < 0.4:
                R(0, x, gp - hh, x + ww, gp, mid * r.uniform(0.85, 1.0))
                Pg(0, [(x - 0.5, gp - hh), (x + ww / 2, gp - hh - ww * 0.75), (x + ww + 0.5, gp - hh)], mid * 0.8)
                if r.random() < 0.55:
                    win(0, x + ww // 2, gp - hh + 2, 1, 1, 0.6)
            x += ww + r.choice((0, 0, 1))
        # ── front layer: timber-framed houses ──
        PLASTER = [(214, 200, 170), (206, 186, 150), (196, 190, 176), (220, 208, 186), (200, 170, 140)]
        ROOFS = [(70, 72, 86), (64, 60, 66), (136, 76, 56), (124, 100, 64)]                   # slate, slate, tile, thatch
        timber = _c((44, 32, 30)) * (1.4 if noon else 1.0)
        x = r.randint(-6, -1)
        while x < w + 4:
            wid = r.randint(9, 14)
            n = r.choice((2, 2, 3))
            sh = max(4, int(gp * r.uniform(0.075, 0.095)))
            top = gp - n * sh
            if r.random() < 0.1 and wid >= 10:                                    # a stone tower or town gate
                stone = _c(mix((150, 144, 132), V["haze"], 0.3)) * dim
                th_ = top - sh
                R(1, x, th_, x + wid, gp, stone)
                m = max(1.0, wid / 7)
                k = x
                while k + m <= x + wid + 0.01:
                    R(1, k, th_ - m, k + m, th_, stone)
                    k += m * 2
                for yy_ in range(int(th_) + 2, gp - 1, 2):                         # coursed stonework
                    rect(Bs[1], x, yy_, x + wid, yy_ + 0.5, stone * 0.85)
                disc(Bs[1], x + wid / 2, gp - sh * 0.9, wid * 0.28, (24, 18, 20))
                rect(Bs[1], x + wid / 2 - wid * 0.28, gp - sh * 0.9, x + wid / 2 + wid * 0.28, gp, (24, 18, 20))
                win(1, x + wid / 2, th_ + sh * 0.6, 1, 2, 0.7)
                lanterns.append((x + wid / 2, gp - sh * 1.4))
                x += wid + r.choice((0, 1))
                continue
            plaster = _c(mix(r.choice(PLASTER), V["haze"], 0.25)) * dim
            for k in range(n):                                                     # storeys, each jettied out a pixel
                x0, x1 = x - k, x + wid + k
                y0, y1 = gp - (k + 1) * sh, gp - k * sh
                R(1, x0, y0, x1, y1, plaster)
                rect(Bs[1], x0, y1 - 1, x1, y1, timber)                           # the beam along each floor
                posts = list(range(x0, x1, 4)) + [x1 - 1]
                for px in posts:
                    rect(Bs[1], px, y0, px + 1, y1, timber)
                if k:
                    for a_, b_ in zip(posts, posts[1:]):
                        if r.random() < 0.6 and b_ - a_ >= 3:                    # braces between the posts
                            if r.random() < 0.5:
                                line(Bs[1], a_ + 0.5, y1 - 1, b_ + 0.5, y0 + 0.5, timber, wid=0.8)
                            else:
                                line(Bs[1], a_ + 0.5, y0 + 0.5, b_ + 0.5, y1 - 1, timber, wid=0.8)
                        elif b_ - a_ >= 3:
                            win(1, a_ + 1, y0 + max(1, sh // 3), 2, 2, 0.55)
                else:                                                              # ground floor: door, shop window
                    dx = x + r.randint(1, max(1, wid - 4))
                    disc(Bs[1], dx + 1.5, y1 - sh + 2.5, 1.5, (30, 22, 22))
                    rect(Bs[1], dx, y1 - sh + 2.5, dx + 3, y1, (30, 22, 22))
                    sx = x + 1 if dx > x + wid / 2 else x + wid - 5
                    win(1, sx, y1 - sh + 2, 3, 2, 0.6)
            ex0, ex1, ey = x - n, x + wid + n, top                              # eaves overhang the top storey
            roof = _c(r.choice(ROOFS)) * dim * 0.9
            ah = (ex1 - ex0) * r.uniform(0.55, 0.8)
            if r.random() < 0.7:                                                   # gable end to the street
                Pg(1, [(ex0 - 0.5, ey), ((ex0 + ex1) / 2, ey - ah), (ex1 + 0.5, ey)], roof)
                poly(Bs[1], [(ex0 + 1.5, ey), ((ex0 + ex1) / 2, ey - ah + 2.2), (ex1 - 1.5, ey)], plaster)
                rect(Bs[1], (ex0 + ex1) / 2 - 0.5, ey - ah + 2, (ex0 + ex1) / 2 + 0.5, ey, timber)
                rect(Bs[1], ex0 + 1.5, ey - 1, ex1 - 1.5, ey, timber)
                if ah > 7:
                    win(1, (ex0 + ex1) / 2 - 2, ey - ah * 0.45, 1, 2, 0.5)
                    win(1, (ex0 + ex1) / 2 + 1, ey - ah * 0.45, 1, 2, 0.5)
            else:                                                                  # ridge along the street
                Pg(1, [(ex0 - 0.5, ey), (ex0 + ah * 0.45, ey - ah * 0.6), (ex1 - ah * 0.45, ey - ah * 0.6), (ex1 + 0.5, ey)], roof)
                for yy_ in range(int(ey - ah * 0.6) + 1, int(ey), 2):
                    rect(Bs[1], ex0 + 1, yy_, ex1 - 1, yy_ + 0.5, roof * 0.75)
                if r.random() < 0.6:                                               # a dormer
                    dxm = (ex0 + ex1) / 2 - 1.5
                    R(1, dxm, ey - ah * 0.45, dxm + 3, ey - 1, plaster)
                    Pg(1, [(dxm - 0.5, ey - ah * 0.45), (dxm + 1.5, ey - ah * 0.45 - 2), (dxm + 3.5, ey - ah * 0.45)], roof)
                    win(1, dxm + 1, ey - ah * 0.45 + 1, 1, 1, 0.6)
            if r.random() < 0.55:                                                  # a chimney stack
                cx_ = x + wid * r.uniform(0.15, 0.8)
                R(1, cx_, ey - ah * 0.75, cx_ + 2, ey - ah * 0.2, _c((96, 84, 78)) * dim)
                R(1, cx_ - 0.5, ey - ah * 0.75 - 1, cx_ + 2.5, ey - ah * 0.75, _c((120, 104, 96)) * dim)
            if r.random() < 0.4:                                                   # a wall lantern on a bracket
                lx = x + wid + n - 2 if r.random() < 0.5 else x - n + 2
                ly = gp - sh - 2
                lanterns.append((lx, ly))
            if r.random() < 0.18 and n >= 2:                                       # a tavern sign hanging over the street
                signs.append((x + wid, gp - sh - 3, r.choice([(150, 60, 50), (60, 90, 70), (170, 130, 60)])))
            x += wid + n - 1 + r.choice((0, 0, 1, 2))
        M = [m[..., 0] > 0.5 for m in Ms]
        both = M[0] | M[1]
        sky_top = np.where(both.any(0), np.argmax(both, axis=0), gp).astype(np.float32)
        return dict(B=Bs, M=M, wins=wins, lanterns=lanterns, signs=signs,
                    top=sky_top, lit=np.array([wv[5] for wv in wins], bool),
                    flare=np.zeros(len(wins), np.float32))

    def _city_old(self, P, gp, w, h, t, V, noon):
        o = self.city_old
        if o is None or o["key"] != (h, w, gp, noon):
            o = self.city_old = self._city_old_build(h, w, gp, V, noon)
            o["key"] = (h, w, gp, noon)
        # mid, then the street in front
        lit = o["lit"]
        lit ^= np.random.random(len(lit)) < (0.04 if "change" in self.fired else 0.001)   # candles lit and snuffed with the chords
        fl = o["flare"]
        fl *= 0.9
        if "note" in self.fired and len(fl):                                      # the melody: a window brightens, higher notes higher up
            ys = np.array([wv[0] for wv in o["wins"]])
            cand = np.nonzero(np.abs(ys - (gp - 3 - self.pitch * gp * 0.5)) < 3)[0]
            if len(cand):
                fl[cand[(self.frame * 7) % len(cand)]] = 1.0
        for li in (0, 1):
            m = o["M"][li]
            P[:gp][m] = o["B"][li][m]
            Gm = np.zeros((gp, w), np.float32)
            for j, (y, x, ww, wh, wl, _, ph) in enumerate(o["wins"]):
                if wl != li or not (0 <= x < w):
                    continue
                if lit[j] or fl[j] > 0.05:
                    k = (0.75 + 0.25 * math.sin(t * (5 + ph * 4) + ph * 40)) * (0.55 if li == 0 else 1.0)   # candlelight flickers
                    col = mix((255, 196, 110), (255, 236, 190), fl[j]) if fl[j] > 0.05 else (255, 186, 100)
                    rect(P, x, y, x + ww, y + wh, _c(col) * (k + 0.6 * fl[j]))
                    Gm[max(0, y):y + wh, max(0, x):x + ww] = k * (0.5 if li == 0 else 1.0) + fl[j]
                else:
                    rect(P, x, y, x + ww, y + wh, (22, 20, 28))
            if not noon:
                P[:gp] += blur(Gm, 1)[..., None] * np.array([255, 170, 90], np.float32) * 0.35
            if li == 0:
                fog_rows(P, gp * 0.5, gp, V["haze"], 0.22)
        for sx, sy, col in o["signs"]:                                             # tavern signs swing on their brackets
            line(P, sx - 1, sy, sx + 4, sy, (40, 30, 28), wid=0.8)
            sw_ = math.sin(t * 1.3 + sx) * 0.6
            line(P, sx + 1, sy, sx + 1 + sw_, sy + 1.5, (40, 30, 28), wid=0.6)
            line(P, sx + 3, sy, sx + 3 + sw_, sy + 1.5, (40, 30, 28), wid=0.6)
            rect(P, sx + 0.5 + sw_, sy + 1.5, sx + 3.5 + sw_, sy + 4, _c(col) * (1.0 if noon else 0.7))
        for lx, ly in o["lanterns"]:                                               # wall lanterns, pulsing on the kick
            line(P, lx - 1.5, ly - 1, lx + 0.5, ly - 1, (40, 30, 28), wid=0.7)
            rect(P, lx - 0.5, ly - 0.5, lx + 1.5, ly + 1.5, (60, 48, 40))
            if not noon:
                k = (0.85 + 0.15 * math.sin(t * 11 + lx)) * (0.8 + 0.4 * self.kick + 0.2 * self.snare)
                P[int(ly), int(min(w - 1, max(0, lx)))] = _c((255, 220, 150)) * k
                glow(P, lx + 0.5, ly + 0.5, 4.5, (255, 170, 90), 0.45 * k)

    def _city_tower(self, P, w, gp, H, t):
        r = wrng("citytower", w)
        x, s = w * r.uniform(0.25, 0.75), gp * 0.85
        base = gp
        y0 = int(max(0, base - s))
        for y in range(y0, base):
            v = min(1.0, (base - y) / s)
            hw = s * 0.2 * (1 - v) ** 2.3 + 0.6
            band = int(v * 9) % 2
            col = _c((255, 128, 60) if band == 0 else (255, 214, 170)) * (0.55 + 0.45 * v)
            xa, xb = int(max(0, x - hw)), int(min(w, x + hw + 1))
            if xb <= xa:
                continue
            row = P[y, xa:xb]
            xs = np.arange(xa, xb) + 0.5 - x
            lat = (np.abs(((xs + y) % 4) - 2) < 0.8) | (np.abs(((xs - y) % 4) - 2) < 0.8) | (np.abs(np.abs(xs) - hw) < 1)
            if v < 0.2:                                  # the arch between the four legs
                lat &= np.abs(xs) > hw * 0.75 * (1 - v / 0.2) ** 0.5
            row[lat] = col
            if abs(v - 0.36) < 0.025 or abs(v - 0.62) < 0.015:     # observation decks
                P[y, int(max(0, x - hw - 1)):int(min(w, x + hw + 2))] = (255, 236, 200)
        glow(P, x, base - s * 0.5, s * 0.35, (255, 120, 60), 0.12, ry=s * 0.5)
        if (t * 1.2) % 2 < 1 or self.kick > 0.5:
            glow(P, x, y0, 2, (255, 60, 60), 0.9)

    def _city_lanterns(self, P, w, gp, t):
        r = wrng("citylant", w)
        y0, sag, gap = gp - 15, r.uniform(2, 4), r.randint(5, 7)
        cols = [(232, 64, 50), (240, 110, 60), (236, 220, 196)]
        for x in range(0, w):
            y = y0 + sag * math.sin(math.pi * (x % 30) / 30)
            P[int(y), x] = (50, 36, 40)
            if x % gap == 2:
                sw = math.sin(t * 1.5 + x) * (0.3 + 0.5 * self.kick) if not self.eng.paused else 0.0
                self.chochin(P, x + sw, y + 3, 1.7, cols[(x // gap) % 3], lit=0.8 + 0.3 * self.pluck, mark=False)

    def _city_marquee(self, P, cv, w, gp, t):
        r = wrng("citymarq", w)
        x0 = int(w * r.uniform(0.15, 0.6))
        y0 = (gp - 14) // 2 * 2                          # on even pixel rows, so the text sits on cells
        mw = 12
        rect(P, x0, y0, x0 + mw, y0 + 6, (40, 14, 24))
        rect(P, x0 + mw // 2 - 3, y0 + 6, x0 + mw // 2 + 3, gp, (60, 30, 30))            # the door, warm inside
        rect(P, x0 + mw // 2 - 2, y0 + 7, x0 + mw // 2 + 2, gp, (255, 170, 90), 0.75)
        T = int(self.eng.beat_pos * 2)                  # the bulbs chase in eighth notes
        for k in range(mw + 1):                          # chasing bulbs round the board
            for yy_ in (y0, y0 + 5):
                on = (k + T + (yy_ == y0)) % 3 == 0
                P[yy_, x0 + k if x0 + k < w else w - 1] = (255, 226, 140) if on else (120, 80, 50)
        glow(P, x0 + mw / 2, y0 + 3, mw * 0.6, (255, 150, 60), 0.35 + self.bass * 0.4, ry=4)
        cv.text(y0 // 2 + 1, x0 + 2, "J A Z Z"[: max(0, w - x0 - 2)], (255, 214, 120))
        self.notes(cv, x0 + mw // 2, y0 // 2 - 1, (255, 200, 140), n=3, rise=5, seed=x0)

    FIREWORK = [(255, 120, 150), (255, 210, 120), (140, 220, 255), (190, 150, 255), (150, 255, 170), (255, 255, 230)]

    def _rare_city(self, P, ev, gp, w, skyline):
        """Rare event: fireworks over the skyline (behind the towers). At noon, hot-air balloons drift over instead."""
        u, r = ev
        T = u * 24.0
        if self.noon():
            for i in range(r.randint(2, 3)):
                f, sp, y0 = r.choice((-1, 1)), r.uniform(0.8, 1.2), gp * r.uniform(0.15, 0.4)
                rad = max(3.0, gp * r.uniform(0.06, 0.09))
                x = (u * sp * (w + 60) - 30 - i * 25) if f > 0 else (w + 30 + i * 25 - u * sp * (w + 60))
                cols = [r.choice(self.FIREWORK), r.choice(self.FIREWORK), (250, 246, 236)]
                self.balloon(P, x, y0 - u * gp * 0.1, rad, cols, burn=max(0.0, self.bass - 0.6) * 2)
            return
        H = P.shape[0]
        shells = r.randint(10, 16)
        for i in range(shells):
            t0 = 0.6 + i * 18.5 / shells + r.uniform(-0.5, 0.5)      # a steady show, never a long gap
            x0 = r.uniform(0.1, 0.9) * w
            x1 = x0 + r.uniform(-6, 6)
            yp = gp * r.uniform(0.1, 0.42)
            col = r.choice(self.FIREWORK)
            n, R = r.randint(30, 48), r.uniform(7, 13) * max(1.0, H / 70)
            jseed = r.randrange(1 << 30)                         # (draw everything before any `continue`)
            xa, xb = int(max(0, x1 - R)), int(min(w, x1 + R + 1))
            if xb > xa:                                          # burst clear of the towers in front
                yp = max(R * 0.8 + 1, min(yp, float(skyline[xa:xb].min()) - R * 0.9))
            age = T - t0
            if age < 0 or age > 3.6:
                continue
            if age < 0.9:                                        # the shell climbing, trailing sparks
                k = age / 0.9
                y = gp - (gp - yp) * (1 - (1 - k) ** 2)
                x = x0 + (x1 - x0) * k
                for j in range(4):
                    yj = int(y + j * 1.5)
                    if 0 <= yj < gp:
                        P[yj, int(x) % w] += (_c((255, 220, 170)) - P[yj, int(x) % w]) * (0.8 - j * 0.18)
                continue
            a_ = age - 0.9                                       # the burst: a ring of embers, falling and fading
            ang = np.arange(n) * TAU / n + i
            rad = R * (1 - math.exp(-a_ * 3.2)) * np.random.default_rng(jseed).uniform(0.6, 1.0, n)
            fade = max(0.0, 1 - a_ / 2.7)
            glow(P, x1, yp + 1.5 * a_ * a_, R * 1.5, col, (0.6 if a_ < 0.2 else 0.22) * fade)   # lights up the smoke
            hot = mix(col, (255, 255, 240), max(0.0, 0.7 - a_ * 0.6))                # white-hot, cooling to colour
            for tr in range(4):                                  # each ember streaks out from the centre
                rr = rad * (1 - tr * 0.13)
                xs = (x1 + np.cos(ang) * rr).astype(int)
                ys = (yp + np.sin(ang) * rr * 0.9 + 3.0 * a_ * a_ - tr * 0.4).astype(int)
                ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < gp)
                tw_ = np.where(np.random.random(n) < 0.2 * a_, 0.25, 1.0)[ok]    # crackle as they die
                P[ys[ok], xs[ok]] += (_c(hot if tr == 0 else col) * 1.15 - P[ys[ok], xs[ok]]) * (min(1.0, fade * 1.4) * (1 - tr * 0.22) * tw_)[:, None]
