import math
import random
import time

import numpy as np

from ..paint import _box, _c, blur, disc, draw_stars, flame_colours, glow, hash01, line, mix, moon, poly, rect
from ..stations import STATIONS
from ..world import wrng
from ..term import C_RAIN, C_WHITE


def gen_cat(rng):
    """Proportions for one generated cat: head, ears, body, tail."""
    return dict(head=rng.uniform(1.8, 2.5), ear=rng.uniform(1.3, 2.4), ear_sp=rng.uniform(0.4, 0.7),
                ear_tilt=rng.uniform(-0.3, 0.4), body=rng.uniform(2.4, 3.6), hunch=rng.uniform(4.0, 5.4),
                long=rng.uniform(4.2, 6.2), flat=rng.uniform(1.7, 2.4), tail=rng.uniform(3.5, 6.5),
                curl=rng.uniform(0.6, 1.5), stripes=rng.randint(2, 4))


SWEATERS = [(126, 92, 146), (88, 112, 156), (156, 92, 96), (92, 134, 112), (168, 124, 82), (112, 100, 156)]


BOKEH = [(255, 150, 190), (255, 200, 140), (120, 220, 214), (255, 230, 150), (200, 170, 255), (255, 120, 110)]


# sleepy cat coats per station: fur, stripes / points, belly, striped?, dark points?
COATS = [
    ((150, 150, 172), (92, 92, 114), (230, 230, 242), True, False),     # grey tabby
    ((46, 42, 58), (24, 22, 32), (86, 82, 104), False, False),          # black cat
    ((236, 152, 82), (186, 98, 50), (252, 226, 192), True, False),      # orange tabby
    ((240, 238, 248), (196, 196, 214), (255, 255, 255), False, False),  # white
    ((58, 56, 72), (34, 32, 44), (238, 238, 246), False, False),        # tuxedo
    ((236, 220, 196), (96, 68, 56), (250, 242, 228), False, True),      # siamese
    ((246, 214, 162), (214, 168, 108), (255, 248, 236), True, False),     # cream tabby
    ((244, 240, 236), (60, 50, 52), (255, 255, 255), True, False),      # calico-ish white with dark patches
    ((112, 108, 120), (70, 66, 80), (200, 196, 210), False, True),      # russian blue with smoky points
    ((128, 104, 84), (70, 54, 44), (226, 214, 196), True, False),       # british brown tabby
]


CITY = [   # dusk, midnight, golden sunset, snowy evening, neon, teal night, noon, tokyo plum, jazz amber
    dict(sky=[(9, 8, 28), (36, 24, 70), (110, 56, 106), (212, 116, 120)], cloud=[(52, 40, 86), (176, 98, 126)],
         haze=(150, 82, 124), lit=0.4, signs=0.18, stars=1.0, sun=False),
    dict(sky=[(3, 4, 16), (12, 16, 44), (32, 34, 78), (70, 62, 112)], cloud=[(26, 30, 60), (70, 66, 110)],
         haze=(66, 64, 118), lit=0.28, signs=0.12, stars=1.2, sun=False),
    dict(sky=[(56, 56, 116), (166, 104, 146), (248, 148, 110), (255, 206, 140)], cloud=[(190, 116, 140), (255, 188, 150)],
         haze=(236, 146, 120), lit=0.22, signs=0.08, stars=0.0, sun=True),
    dict(sky=[(28, 34, 58), (66, 76, 108), (126, 136, 166), (186, 192, 216)], cloud=[(104, 110, 136), (176, 182, 204)],
         haze=(166, 172, 198), lit=0.5, signs=0.1, stars=0.3, sun=False),
    dict(sky=[(10, 4, 30), (48, 10, 70), (128, 20, 110), (255, 60, 140)], cloud=[(60, 20, 90), (200, 60, 150)],
         haze=(196, 40, 138), lit=0.45, signs=0.55, stars=0.6, sun=False),
    dict(sky=[(4, 14, 24), (10, 40, 56), (30, 80, 90), (90, 150, 140)], cloud=[(20, 50, 64), (80, 140, 140)],
         haze=(70, 130, 130), lit=0.3, signs=0.15, stars=1.0, sun=False),
    dict(sky=[(64, 136, 228), (104, 166, 238), (156, 204, 248), (206, 230, 255)], cloud=[(236, 242, 255), (255, 255, 255)],
         haze=(172, 200, 228), lit=0.04, signs=0.04, stars=0.0, sun=False),
    dict(sky=[(12, 8, 34), (52, 26, 84), (150, 60, 120), (246, 130, 150)], cloud=[(70, 34, 96), (220, 110, 150)],
         haze=(170, 80, 140), lit=0.42, signs=0.4, stars=0.8, sun=False),                 # tokyo: plum dusk
    dict(sky=[(10, 6, 14), (34, 18, 30), (80, 40, 50), (150, 80, 70)], cloud=[(46, 26, 40), (130, 70, 66)],
         haze=(130, 76, 66), lit=0.5, signs=0.3, stars=0.5, sun=False),                    # jazz: smoky amber
    dict(sky=[(10, 12, 20), (34, 40, 56), (84, 90, 104), (150, 146, 140)], cloud=[(54, 58, 72), (130, 128, 130)],
         haze=(120, 120, 124), lit=0.45, signs=0.0, stars=0.3, sun=False),                # english: foggy gaslight
]


RARE_GAP = (120.0, 300.0)      # seconds of watching between rare events


class SharedScene:
    # ── rare events: something small and unexpected, every few minutes ─────
    def rare(self, dur):
        """The current scene's rare event. While one is on, returns (u, r): u runs 0→1 over `dur` seconds
        of scene time (`self.rclock`, which stops on pause) and r is a `random.Random` that is the same
        every frame of this event, for its direction, size and colours. Otherwise None.
        `rare_mem` is a dict that lives as long as the event. `rare_force = u` pins every scene's
        event at u (for --preview)."""
        if self.rare_force is not None:
            return self.rare_force, random.Random(7)
        ev = self.rare_ev
        if ev is not None and ev[0] != self.viz:           # the scene changed mid-event: it's gone
            ev = self.rare_ev = None
            self.rare_next = self.rclock + random.uniform(*RARE_GAP)
        if ev is None:
            if self.rclock < self.rare_next:
                return None
            ev = self.rare_ev = [self.viz, self.rclock, random.randrange(1 << 30)]
            self.rare_mem = {}                                 # scratch state for this one event
        u = (self.rclock - ev[1]) / dur
        if u >= 1:
            self.rare_ev = None
            self.rare_next = self.rclock + random.uniform(*RARE_GAP)
            return None
        return u, random.Random(ev[2])

    # ── the melody made visible ────────────────────────────────────────────
    def note_sparks(self, P, name, spawn, col, rise=5.0, drift=0.0, life=1.6, r=1.0, kinds=("note",), k=1.0):
        """A glint for every lead note (or riff pluck, with kinds=("pluck",)) as it plays: `spawn(pitch, n)`
        gives where it appears (pitch 0 low … 1 high, n counts sparks so far), then it rises `rise` px/s,
        drifts sideways and fades over `life` seconds of scene time (so it freezes on pause)."""
        if self.sparks is None:
            self.sparks = {}
        S = self.sparks.setdefault(name, [[], self.rclock, 0])
        dt = self.rclock - S[1]
        S[1] = self.rclock
        dt = dt if 0 <= dt < 0.5 else 0.0
        for kind in kinds:
            if kind in self.fired:
                p = self.pitch if kind == "note" else self.ppitch
                x, y = spawn(p, S[2])
                S[2] += 1
                S[0].append([x, y, 0.0])
        keep = []
        for sp in S[0]:
            sp[2] += dt
            sp[1] -= rise * dt
            sp[0] += drift * dt
            a = 1 - sp[2] / life
            if a <= 0:
                continue
            keep.append(sp)
            glow(P, sp[0], sp[1], r * 2.6, col, 0.45 * a * k)
            disc(P, sp[0], sp[1], r * (0.4 + 0.6 * a), mix(col, (255, 255, 255), 0.55), a=min(1.0, 1.4 * a) * k)
        S[0] = keep[-32:]

    def chord_wash(self, P, y0, y1, k=0.06):
        """Wash a band of rows with the current chord's colour (additive light), brighter just after a change."""
        if y1 > y0:
            P[y0:y1] += self.tint * (k * (0.5 + 0.9 * self.change) * min(1.0, 0.3 + self.energy))

    def summon_rare(self):
        """`e`: bring the scene's rare event on now."""
        if self.rare_ev is None or self.rare_ev[0] != self.viz:
            self.rare_ev, self.rare_next = None, self.rclock

    # ── station themes: props the Japanese and jazz stations add to every scene ──
    def theme(self):
        """"japan", "jazz", "english" or None for the station on air (follows `var`, i.e. the station index)."""
        return STATIONS[self.eng.station_idx % len(STATIONS)].get("theme")

    def torii(self, P, x, base, s, col=(212, 58, 46), dark=(40, 22, 30), a=1.0):
        """A shrine gate `s` pixels tall standing on `base`, centred on x."""
        pw, half = max(1.0, s * 0.12), s * 0.42
        for sx in (-1, 1):
            rect(P, x + sx * half * 0.72 - pw / 2, base - s * 0.86, x + sx * half * 0.72 + pw / 2, base, col, a)
            rect(P, x + sx * half * 0.72 - pw * 0.7, base - max(1, s * 0.07), x + sx * half * 0.72 + pw * 0.7, base, dark, a)
        rect(P, x - half * 0.95, base - s * 0.68, x + half * 0.95, base - s * 0.68 + max(1, s * 0.07), col, a)   # nuki
        top, th = base - s * 0.92, max(1.0, s * 0.1)                                                          # kasagi
        poly(P, [(x - half * 1.25, top - th * 0.9), (x, top), (x + half * 1.25, top - th * 0.9),
                 (x + half * 1.1, top + th), (x - half * 1.1, top + th)], col, a)
        rect(P, x - half * 1.15, top - th * 0.9 - max(0.8, th * 0.5), x + half * 1.15, top - th * 0.9 + 0.3, dark, a)
        rect(P, x - pw * 0.4, top + th, x + pw * 0.4, base - s * 0.68, col, a)                               # gakuzuka

    def chochin(self, P, x, y, r, col=(230, 60, 46), lit=1.0, mark=True):
        """A round paper lantern centred at (x, y): ribbed, capped, glowing from inside."""
        glow(P, x, y, r * 2.2, col, 0.35 * lit)
        disc(P, x, y, r, mix(col, (255, 220, 160), 0.25 * lit), ry=r * 1.2)
        b = _box(P, x - r, y - r * 1.2, x + r + 1, y + r * 1.2 + 1)
        if b is not None:
            v, yy, xx = b
            rib = (np.abs(((yy - y) / max(1.0, r * 0.45)) % 1 - 0.5) > 0.38) & ((xx - x) ** 2 / r ** 2 + (yy - y) ** 2 / (r * 1.2) ** 2 < 1)
            v[rib] *= 0.78
        if mark and r >= 2.5:
            rect(P, x - r * 0.18, y - r * 0.55, x + r * 0.22, y + r * 0.55, (60, 20, 24), 0.7)
        for sy in (-1, 1):
            rect(P, x - r * 0.55, y + sy * r * 1.2 - (1 if sy < 0 else 0), x + r * 0.55, y + sy * r * 1.2 + (0 if sy < 0 else 1), (40, 28, 30))

    def pagoda(self, P, x, base, s, col, roof, tiers=4):
        """A pagoda silhouette `s` tall: stacked tiers with deep, flared roofs and a spire."""
        th = s * 0.78 / tiers
        rim = mix(roof, (255, 220, 200), 0.25)
        for k in range(tiers):
            wd = s * (0.3 - 0.035 * k)
            y1 = base - k * th
            rect(P, x - wd * 0.42, y1 - th * 0.7, x + wd * 0.42, y1, col)
            yr = y1 - th * 0.62
            poly(P, [(x - wd * 1.3, yr - th * 0.05), (x - wd * 0.9, yr - th * 0.28), (x - wd * 0.35, yr - th * 0.4),
                     (x + wd * 0.35, yr - th * 0.4), (x + wd * 0.9, yr - th * 0.28), (x + wd * 1.3, yr - th * 0.05),
                     (x + wd * 0.9, yr + th * 0.08), (x - wd * 0.9, yr + th * 0.08)], roof)
            line(P, x - wd * 1.3, yr - th * 0.05, x + wd * 1.3, yr - th * 0.05, rim, wid=0.7, a=0.6)
        line(P, x, base - tiers * th, x, base - s, roof, wid=max(0.8, s * 0.03))

    def record(self, P, x, y, r, ang=0.0, label=(196, 64, 60), sheen=1.0):
        """A vinyl record seen face on, turning with `ang`: grooves, a light sheen and a paper label."""
        b = _box(P, x - r - 1, y - r - 1, x + r + 2, y + r + 2)
        if b is None:
            return
        v, yy, xx = b
        d = np.hypot(xx + 0.5 - x, (yy + 0.5 - y)) / r
        m = np.clip((1 - d) * r + 0.5, 0, 1)
        a = np.arctan2(yy + 0.5 - y, xx + 0.5 - x)
        groove = 14 + 8 * (np.sin(d * r * 3.1) > 0.3) + 26 * sheen * np.clip(np.cos(2 * (a - ang)), 0, 1) ** 6 * (d > 0.38)
        col = np.where((d < 0.36)[..., None], _c(label) * (0.9 + 0.1 * np.cos(a - ang))[..., None], groove[..., None] + _c((4, 2, 8)))
        col = np.where((d < 0.06)[..., None], _c((10, 8, 14)), col)
        v += (col - v) * m[..., None]

    def notes(self, cv, x, y, col, n=3, rise=6, seed=0, t=None):
        """Music notes (♪ ♫) drifting up and away from cell (x, y), swaying; they freeze on pause."""
        T = self.rclock if t is None else t
        for k in range(n):
            u = ((T * 0.22 + k / n + seed * 0.37) % 1.0)
            cx = x + math.sin(u * 5 + k * 2.1 + seed) * 1.6 + u * 2.5 * (1 if k % 2 else -1)
            cy = y - u * rise
            if u < 0.92:
                cv.put(int(round(cy)), int(round(cx)), "♪♫♩"[(k + seed) % 3], mix(col, (40, 30, 60), u * 0.7))

    def sax(self, P, x, y, s, col=(226, 176, 72)):
        """A brass saxophone `s` pixels tall, bell at the bottom right of (x, y) (the neck's top)."""
        line(P, x, y, x + s * 0.12, y + s * 0.08, (40, 34, 40), wid=max(0.8, s * 0.05))           # mouthpiece
        line(P, x + s * 0.12, y + s * 0.08, x + s * 0.2, y + s * 0.18, col, wid=max(0.8, s * 0.07))
        line(P, x + s * 0.2, y + s * 0.18, x + s * 0.22, y + s * 0.78, col, wid=(max(1, s * 0.1), max(1.4, s * 0.17)))
        disc(P, x + s * 0.34, y + s * 0.84, max(1.0, s * 0.15), col, ry=max(1.0, s * 0.11))     # the U-bend
        line(P, x + s * 0.42, y + s * 0.82, x + s * 0.52, y + s * 0.5, col, wid=(max(1.2, s * 0.15), max(1.6, s * 0.2)))
        disc(P, x + s * 0.54, y + s * 0.46, max(1.0, s * 0.12), mix(col, (60, 40, 20), 0.35), ry=max(0.8, s * 0.06))
        for k in range(3):                                                                       # keys
            disc(P, x + s * 0.21, y + s * (0.32 + 0.14 * k), max(0.6, s * 0.035), (255, 240, 200))

    # ── old English props ─────────────────────────────────────────────────
    def castle(self, P, x, base, s, col, lit=(255, 200, 120), flag=(170, 40, 50), t=0.0, seed=0):
        """A castle `s` pixels tall standing on `base`, centred on x: a curtain wall with a gate, round
        corner towers, a tall keep, crenellations everywhere, lit arrow slits and a pennant on the keep."""
        top = mix(col, (255, 240, 220), 0.12)

        def merlons(x0, x1, y):
            m = max(1.0, s * 0.05)
            k = x0
            while k + m <= x1 + 0.01:
                rect(P, k, y - m, k + m, y, col)
                k += m * 2
        ww, wh = s * 0.72, s * 0.42                                          # curtain wall
        rect(P, x - ww, base - wh, x + ww, base, col)
        merlons(x - ww, x + ww, base - wh)
        gw = max(1.0, s * 0.09)
        disc(P, x, base - wh * 0.45, gw, (20, 14, 18), ry=gw * 1.1)
        rect(P, x - gw, base - wh * 0.45, x + gw, base, (20, 14, 18))
        for sx in (-1, 1):                                                   # corner towers
            tx, tw, tht = x + sx * ww, s * 0.15, s * 0.62
            rect(P, tx - tw, base - tht, tx + tw, base, col)
            merlons(tx - tw - s * 0.02, tx + tw + s * 0.02, base - tht)
            rect(P, tx - tw - s * 0.02, base - tht, tx + tw + s * 0.02, base - tht + max(1, s * 0.03), top)
        kw, kh = s * 0.24, s * 0.86                                          # the keep
        rect(P, x - kw, base - kh, x + kw, base - wh * 0.5, col)
        merlons(x - kw, x + kw, base - kh)
        line(P, x + kw * 0.6, base - kh, x + kw * 0.6, base - s * 1.08, col, wid=0.7)
        wave = math.sin(t * 3 + seed) * s * 0.02
        poly(P, [(x + kw * 0.6, base - s * 1.08), (x + kw * 0.6 + s * 0.16, base - s * 1.04 + wave),
                 (x + kw * 0.6, base - s * 1.0)], flag)
        for k, (ox, oy) in enumerate(((-0.1, 0.62), (0.1, 0.62), (0, 0.78), (-ww / s, 0.45), (ww / s, 0.45))):
            if hash01(np.float32(k), np.float32(seed), 7) < 0.75:            # arrow slits, most of them lit
                fx, fy = x + ox * s, base - oy * s
                glow(P, fx, fy, max(1.5, s * 0.06), lit, 0.25)
                rect(P, fx - 0.5, fy - max(1, s * 0.06), fx + 0.5, fy + max(1, s * 0.03), lit)

    def spire(self, P, x, base, s, col, lit=(255, 206, 130), clock=True):
        """A parish church's tower and needle spire, `s` tall on `base`: belfry louvres, a lit clock face."""
        tw, th = max(1.5, s * 0.13), s * 0.5
        rect(P, x - tw, base - th, x + tw, base, col)
        rect(P, x - tw * 1.2, base - th - max(1, s * 0.03), x + tw * 1.2, base - th, mix(col, (255, 240, 220), 0.1))
        for sx in (-1, 1):                                                   # corner pinnacles
            line(P, x + sx * tw, base - th, x + sx * tw, base - th - s * 0.08, col, wid=0.8)
        poly(P, [(x - tw * 0.9, base - th), (x, base - s), (x + tw * 0.9, base - th)], col)
        line(P, x, base - s, x, base - s * 1.06, col, wid=0.6)               # the weathervane rod
        if clock and s >= 14:
            cy = base - th * 0.72
            disc(P, x, cy, tw * 0.62, lit)
            glow(P, x, cy, tw * 1.6, lit, 0.2)
            line(P, x, cy, x, cy - tw * 0.45, (40, 30, 30), wid=0.5)
            line(P, x, cy, x + tw * 0.32, cy, (40, 30, 30), wid=0.5)
        rect(P, x - tw * 0.3, base - th * 0.38, x + tw * 0.3, base - th * 0.15, mix(col, lit, 0.4))   # belfry

    def cottage(self, P, x, base, s, wall=(214, 200, 176), beam=(58, 40, 34), thatch=(150, 116, 70),
                lit=(255, 196, 110), glow_k=1.0, smoke=0.0, t=0.0):
        """A half-timbered cottage `s` wide (centred on x, standing on `base`) under a deep thatched roof,
        a lit leaded window and a chimney, with wisps of smoke when `smoke` > 0."""
        w2, wh = s * 0.5, s * 0.36
        rect(P, x - w2, base - wh, x + w2, base, wall)
        for k in range(5):                                                   # timber frame
            bx = x - w2 + k * s / 4
            rect(P, bx - 0.4, base - wh, bx + 0.6, base, beam)
        rect(P, x - w2, base - wh * 0.5 - 0.4, x + w2, base - wh * 0.5 + 0.5, beam)
        line(P, x - w2, base, x - w2 + s / 4, base - wh * 0.5, beam, wid=0.6)
        line(P, x + w2, base, x + w2 - s / 4, base - wh * 0.5, beam, wid=0.6)
        rh = s * 0.42                                                        # a thatched roof, rounded eaves
        poly(P, [(x - w2 * 1.18, base - wh + 1), (x - w2 * 0.5, base - wh - rh), (x + w2 * 0.5, base - wh - rh),
                 (x + w2 * 1.18, base - wh + 1)], thatch)
        line(P, x - w2 * 0.5, base - wh - rh, x + w2 * 0.5, base - wh - rh, mix(thatch, (255, 230, 180), 0.3), wid=0.8)
        cx = x + w2 * 0.55                                                   # chimney
        rect(P, cx - s * 0.05, base - wh - rh * 1.15, cx + s * 0.05, base - wh - rh * 0.5, (110, 70, 60))
        for k in range(3 if smoke else 0):
            u = (t * 0.25 + k / 3) % 1
            disc(P, cx + u * s * 0.3 + math.sin(u * 6 + k) * 1.2, base - wh - rh * 1.2 - u * s * 0.6,
                 1 + u * s * 0.06, (190, 186, 196), a=0.35 * smoke * (1 - u))
        wx, wy, ws = x - w2 * 0.45, base - wh * 0.62, max(1.0, s * 0.09)     # the leaded window
        glow(P, wx, wy, ws * 3, lit, 0.3 * glow_k)
        rect(P, wx - ws, wy - ws, wx + ws, wy + ws, mix(beam, lit, glow_k))
        line(P, wx, wy - ws, wx, wy + ws, beam, wid=0.5)
        rect(P, x + w2 * 0.05, base - wh * 0.6, x + w2 * 0.3, base, beam)    # the door

    def gaslamp(self, P, x, base, s, lit=(255, 214, 140), col=(30, 28, 34), k=1.0):
        """A Victorian gas lamp post `s` tall on `base`, its lantern glowing gold."""
        line(P, x, base, x, base - s * 0.8, col, wid=max(0.8, s * 0.06))
        rect(P, x - s * 0.08, base - s * 0.08, x + s * 0.08, base, col)
        line(P, x - s * 0.12, base - s * 0.72, x + s * 0.12, base - s * 0.72, col, wid=0.6)
        ly = base - s * 0.88
        glow(P, x, ly, s * 0.9, lit, 0.35 * k)
        poly(P, [(x - s * 0.1, ly - s * 0.08), (x + s * 0.1, ly - s * 0.08), (x + s * 0.07, ly + s * 0.08),
                 (x - s * 0.07, ly + s * 0.08)], mix(col, lit, k))
        poly(P, [(x - s * 0.13, ly - s * 0.08), (x, ly - s * 0.17), (x + s * 0.13, ly - s * 0.08)], col)

    def candles(self, P, x, base, s, t, n=3, wax=(236, 226, 200), holder=(196, 160, 80)):
        """A brass candelabra `s` tall on `base` holding n candles, flames flickering (they still on pause,
        since `t` is scene time)."""
        line(P, x, base, x, base - s * 0.45, holder, wid=max(0.8, s * 0.06))
        rect(P, x - s * 0.15, base - max(1, s * 0.05), x + s * 0.15, base, holder)
        span = s * 0.32 if n > 1 else 0
        if n > 1:
            line(P, x - span, base - s * 0.45, x + span, base - s * 0.45, holder, wid=0.7)
        for k in range(n):
            cx = x + (k / (n - 1) * 2 - 1) * span if n > 1 else x
            ch = s * (0.32 if n > 1 and k == n // 2 else 0.26)
            top = base - s * 0.45 - ch - (s * 0.06 if n > 1 and k == n // 2 else 0)
            rect(P, cx - max(0.5, s * 0.04), top, cx + max(0.5, s * 0.04), base - s * 0.45, wax)
            fl = 1 + 0.15 * math.sin(t * 11 + k * 2.3) + 0.1 * math.sin(t * 17 + k)
            glow(P, cx, top - s * 0.06, s * 0.5 * fl, (255, 170, 80), 0.3)
            disc(P, cx + math.sin(t * 7 + k) * 0.25, top - s * 0.06 * fl, max(0.6, s * 0.045), (255, 236, 170),
                 ry=max(0.9, s * 0.09 * fl))

    def tune_fx(self, P):
        """Turning the dial: the picture tears and snows for a moment, like an old set between channels."""
        k = 1 - (time.monotonic() - self.tuned_at) / 0.75
        if k <= 0:
            return
        H, W = P.shape[:2]
        for y in np.nonzero(np.random.random(H) < 0.3 * k)[0]:
            P[y] = np.roll(P[y], random.randint(-8, 8), axis=0)
        P += (P.mean(-1, keepdims=True) - P) * (0.7 * k)
        snow = np.random.random((H, W)).astype(np.float32)[..., None] * 230
        P += (snow - P) * (0.55 * k * k)
        yb = int((1 - k) * H * 1.6) % H                       # a bright band rolling down
        P[yb:yb + 3] += 50 * k

    # ── scene transitions ──────────────────────────────────────────────────
    TRANS_KINDS = ("wipe", "iris", "blinds", "dissolve")

    def transition(self, P, cv):
        """Changing scene: the new picture is revealed over the last frame of the old one through a
        shape that changes each time (a diagonal wipe, an iris, venetian blinds, a dithered dissolve),
        with a soft band of light at the moving edge. The old frame's text overlay goes with it."""
        tr = self.trans
        if tr is None:
            return
        old = tr["P"]
        k = (time.monotonic() - tr["t0"]) / tr["dur"]
        if k >= 1 or old.shape != P.shape:
            self.trans = None
            return
        u = k * k * (3 - 2 * k)                                            # ease in and out
        H, W = P.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        kind = tr["kind"]
        if kind == "wipe":                                                 # diagonal, a ragged edge
            f = (xx / W * 0.75 + yy / H * 0.25) + 0.06 * np.sin(yy * 0.9 + xx * 0.13)
            f = f * 0.85 + 0.075
        elif kind == "iris":                                               # a circle opening from the middle
            f = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2) * 0.6) / 1.18
        elif kind == "blinds":                                             # slats turning, top to bottom
            f = (yy % 8) / 8 * 0.55 + yy / H * 0.45
        else:                                                              # an ordered-dither dissolve
            bay = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16
            f = bay[yy.astype(int) % 4, xx.astype(int) % 4] * 0.8 + hash01(xx // 4, yy // 4, 3) * 0.2
        edge = u * 1.12 - 0.06 - f                                         # > 0: the new scene shows
        new = edge > 0
        P[:] = np.where(new[..., None], P, old)
        band = np.clip(1 - np.abs(edge) / 0.05, 0, 1) * (kind != "dissolve")
        P += band[..., None] * _c(tr["col"]) * 0.8
        oc, oo = tr["ch"], tr["co"]                                        # old text in the old part
        cell_new = new[0::2] | new[1::2]
        for y in range(min(cv.h, len(oc))):
            row_new = cell_new[y]
            for x in range(min(cv.w, len(oc[y]))):
                if not row_new[x]:
                    cv.ch[y][x], cv.co[y][x] = oc[y][x], oo[y][x]

    def balloon(self, P, x, y, r, cols, burn=0.0, seed=0):
        """A hot-air balloon: an envelope of coloured gores centred at (x, y) with radius r, a basket
        below and, with `burn` > 0, the burner flaring inside."""
        b = _box(P, x - r - 1, y - r - 1, x + r + 2, y + r * 1.25 + 1)
        if b is None:
            return
        v, Y, X = b
        dx, dy = (X + 0.5 - x) / r, (Y + 0.5 - y) / r
        half = 1 - np.clip(dy, 0, None) ** 1.4 * 0.75                # tapers toward the mouth
        m = (dy > -1) & (dy < 1.15) & (np.abs(dx) < np.sqrt(np.clip(1 - np.clip(dy, -1, 0) ** 2, 0, 1)) * half)
        gore = ((np.arcsin(np.clip(dx / np.maximum(half, 0.2), -1, 1)) / np.pi + 0.5) * 6).astype(int) % len(cols)
        col = np.asarray(cols, np.float32)[gore]
        col *= (1.1 - 0.45 * np.clip(dx + 0.3, 0, 1) - 0.15 * np.clip(dy, 0, 1))[..., None]   # round, lit from the left
        v[m] = col[m]
        by = y + r * 1.15
        line(P, x - r * 0.3, by, x - r * 0.18, by + r * 0.42, (60, 46, 40), wid=0.6)
        line(P, x + r * 0.3, by, x + r * 0.18, by + r * 0.42, (60, 46, 40), wid=0.6)
        rect(P, x - max(1.0, r * 0.2), by + r * 0.4, x + max(1.0, r * 0.2), by + r * 0.4 + max(1.5, r * 0.22), (120, 82, 50))
        if burn > 0.05:
            glow(P, x, by - r * 0.15, r * 0.9, (255, 170, 80), 0.55 * burn)
            disc(P, x, by, max(0.6, r * 0.12), (255, 220, 140), a=min(1.0, burn))

    def quadruped(self, P, x, base, s, facing, step, col, head_dn=0.0, kind="deer", eye=None):
        """A side-on silhouette of a deer or fox standing on pixel row `base`. `s` is the shoulder height,
        `step` the walk-cycle phase (radians), `head_dn` 0…1 lowers the head to graze."""
        f = facing
        body_y = base - s * 0.95
        bl = s * (0.9 if kind == "deer" else 1.05)                  # body length
        bh = s * (0.42 if kind == "deer" else 0.36)
        disc(P, x, body_y, bl * 0.55, col, ry=bh * 0.55)
        legs = ((-0.38, 0.0), (-0.3, math.pi), (0.32, math.pi * 0.5), (0.4, math.pi * 1.5))
        lw = max(0.7, s * 0.08)
        for ox, ph in legs:
            sw = math.sin(step + ph) * s * 0.18
            lx = x + f * ox * bl
            line(P, lx, body_y + bh * 0.2, lx + f * sw, base, col, wid=lw)
        nx, ny = x + f * bl * 0.5, body_y - bh * 0.2                # neck to head
        if kind == "deer":
            hx = nx + f * s * (0.25 + 0.2 * head_dn)
            hy = body_y - s * 0.65 * (1 - head_dn) + s * 0.75 * head_dn
        else:
            hx = nx + f * s * (0.25 + 0.1 * head_dn)
            hy = body_y - s * 0.2 * (1 - head_dn) + s * 0.6 * head_dn
        line(P, nx, ny, hx, hy, col, wid=max(1.0, s * 0.17))
        disc(P, hx + f * s * 0.08, hy + s * 0.02, s * 0.14, col, ry=s * 0.1)
        disc(P, hx + f * s * 0.22, hy + s * 0.06, s * 0.08, col, ry=s * 0.06)          # muzzle
        if kind == "deer":
            line(P, hx - f * s * 0.02, hy - s * 0.08, hx - f * s * 0.12, hy - s * 0.25, col, wid=max(0.6, s * 0.06))
            disc(P, x - f * bl * 0.55, body_y - bh * 0.25, s * 0.07, mix(col, (255, 255, 255), 0.6))   # white tail
        else:
            for k in (-0.06, 0.08):
                poly(P, [(hx + f * (k - 0.06) * s, hy - s * 0.08), (hx + f * k * s, hy - s * 0.3), (hx + f * (k + 0.07) * s, hy - s * 0.08)], col)
            tx, ty = x - f * bl * 0.55, body_y
            line(P, tx, ty, tx - f * s * 0.55, ty + s * 0.25 + math.sin(step * 0.5) * s * 0.05, col, wid=max(1.0, s * 0.2))
            disc(P, tx - f * s * 0.58, ty + s * 0.28, max(0.6, s * 0.1), (236, 230, 220))
        if eye is not None and 0 <= hy < P.shape[0] and 0 <= hx + f * s * 0.1 < P.shape[1]:
            P[int(hy), int(hx + f * s * 0.1)] = eye
            glow(P, hx + f * s * 0.1, hy, 1.4, eye, 0.5)
    def sky_body(self, P, cx, cy, r, dim=1.0, haze=1.0, tint=(252, 240, 214)):
        """The moon, or on the noon station a bright midday sun."""
        if self.noon():
            glow(P, cx, cy, r * 7, (255, 244, 200), 0.3 * dim)
            glow(P, cx, cy, r * 2.4, (255, 252, 230), 0.55 * dim)
            disc(P, cx, cy, r * 1.15, (255, 254, 240))
        else:
            moon(P, cx, cy, r, dim=dim, haze=haze, tint=tint)

    def stars(self, P, stars, t, dim=1.0):
        if not self.noon():
            draw_stars(P, stars, t, dim)

    # ── shared rain / snow particles, reacting to loudness ────────────────
    def weather(self, s, cv, x0, x1, ytop, ground, splash=True, on_land=None):
        e = self.eng
        snow = e.st.get("weather") == "snow"
        intensity = (1.0 if e.rain_on else 0.0) + (1.3 if e.storm_on else 0.0) + (0.6 if snow else 0.0)
        level = float(self.rms.mean())
        span = max(1, x1 - x0)
        spawn = intensity * span / 30 * (0.4 + 4 * level) * (0.4 if snow else 1)
        for _ in range(int(spawn) + (random.random() < spawn % 1)):
            s["drops"].append([float(ytop), random.randrange(x0, max(x0 + 1, x1)),
                               random.uniform(0.25, 0.5) if snow else random.uniform(0.9, 1.7)])
        alive = []
        for d in s["drops"]:
            d[0] += d[2]
            if snow:
                d[1] = min(x1 - 1, max(x0, d[1] + random.choice([-1, 0, 0, 0, 1])))
            gr = ground if isinstance(ground, (int, float)) else ground[d[1]]
            if d[0] >= gr:
                if on_land is not None:
                    on_land(d[1])
                if splash and not snow:
                    s["splash"].append([d[1], 3, int(gr)])
                continue
            alive.append(d)
            cv.put(int(d[0]), d[1], ("*" if d[2] > 0.38 else "·") if snow else ("│" if d[2] > 1.3 else "╎"),
                   C_WHITE if snow else C_RAIN)
        s["drops"] = alive[-600:]
        keep = []
        for sp in s["splash"]:
            cv.put(sp[2], sp[0], "˙.,"[3 - sp[1]], C_RAIN)
            sp[1] -= 1
            if sp[1] > 0:
                keep.append(sp)
        s["splash"] = keep

    # ── forest ─────────────────────────────────────────────────────────────
    def broadleaf(self, P, x, base, hgt, lit, shade, sway=0.0, snow=0.0, rim=None, trunk=(58, 40, 36), seed=0):
        """A round-crowned tree standing on pixel row `base`: a flared trunk forking into a crown of leaf
        clumps. Each clump is lit from the upper left and shaded below, in a few dithered tones so it reads
        as pixel-art foliage, with a ragged leafy edge and sky gaps between clumps."""
        H, W = P.shape[:2]
        r = wrng("leaf", seed)
        cr = hgt * r.uniform(0.26, 0.32)
        cy = base - hgt + cr * 1.05
        tw = max(1.0, hgt / 18)
        tcol, tdark = _c(trunk), _c(trunk) * 0.6
        poly(P, [(x - tw * 1.6, base), (x - tw * 0.6, base - tw * 1.5), (x - tw * 0.5, cy + cr * 0.4),
                 (x + tw * 0.5 + 1, cy + cr * 0.4), (x + tw * 0.6 + 1, base - tw * 1.5), (x + tw * 1.6 + 1, base)], tcol)
        rect(P, x + 0.5, cy + cr * 0.5, x + tw * 0.6 + 1, base, tdark)          # the shaded side of the trunk
        for sd in (-1, 1):
            line(P, x + 0.5, cy + cr * 0.75, x + 0.5 + sd * cr * 0.5, cy + cr * 0.05, trunk, wid=max(0.7, tw * 0.55))
        sx = x + 0.5 + sway * 0.6
        blobs = [(0.0, -0.15, 0.62)] + [(r.uniform(-0.75, 0.75), r.uniform(-0.45, 0.4), r.uniform(0.38, 0.56)) for _ in range(7)]
        y0, y1 = max(0, int(cy - cr * 1.05)), min(H, int(cy + cr * 0.95) + 1)
        x0, x1 = max(0, int(sx - cr * 1.4)), min(W, int(sx + cr * 1.4) + 1)
        if y0 >= y1 or x0 >= x1:
            return
        Y = np.arange(y0, y1, dtype=np.float32)[:, None] + 0.5
        X = np.arange(x0, x1, dtype=np.float32)[None, :] + 0.5
        best = np.full((y1 - y0, x1 - x0), -9.0, np.float32)
        light = np.zeros_like(best)
        for bx, by, br in blobs:
            rx, ry = br * cr, br * cr * 0.82
            dx, dy = (X - sx - bx * cr) / rx, (Y - cy - by * cr) / ry
            f = 1 - (dx * dx + dy * dy)
            upd = f > best
            best = np.where(upd, f, best)
            light = np.where(upd, 0.62 - 0.42 * dx - 0.55 * dy, light)       # each clump is lit on its upper left
        hx = hash01(X.astype(np.int64) + 0 * Y.astype(np.int64), Y.astype(np.int64) + 0 * X.astype(np.int64), seed + 3)
        inside = best + 0.28 * (hx - 0.5) > 0                                # a ragged, leafy edge
        light += 0.18 * (cy - Y) / cr + 0.3 * (hx - 0.5)                    # the top of the crown is in the sun; dither
        tone = np.clip((light * 3).astype(int), 0, 3)
        pal = np.array([_c(shade) * 0.8, _c(shade), _c(mix(shade, lit, 0.55)), _c(lit)], np.float32)
        col = pal[tone]
        edge = inside & ~np.roll(inside, 1, axis=0) & ~np.roll(inside, 1, axis=1) & (light > 0.6)
        if rim is not None:
            col[edge] = rim
        if snow > 0:
            col[inside & (light > 0.75 - 0.2 * snow)] = (232, 238, 250)
        v = P[y0:y1, x0:x1]
        v[inside] = col[inside]

    def pine(self, P, x, base, hgt, lit, shade, sway=0.0, snow=0.0, rim=None, trunk=(58, 40, 36), seed=0, fat=1.0):
        """A tiered pine standing on pixel row `base`: lit left side, shaded right,
        a rim of moonlight, optional snow on each tier."""
        H, W = P.shape[:2]
        tr = max(2, hgt // 7)
        crown = hgt - tr
        top = base - hgt
        maxw = max(2.0, hgt * 0.27 * fat)
        T = crown / max(2, crown // 6)
        y0, y1 = max(0, int(top)), min(H, int(base - tr) + 1)
        xs0, xs1 = max(0, int(x - maxw - abs(sway) - 2)), min(W, int(x + maxw + abs(sway) + 3))
        P[max(0, int(base - tr) - 1):min(H, int(base)), max(0, int(x)):min(W, int(x) + (2 if hgt > 30 else 1))] = trunk
        if y0 >= y1 or xs0 >= xs1:
            return
        rows = np.arange(y0, y1, dtype=np.float32)[:, None]
        cols = np.arange(xs0, xs1, dtype=np.float32)[None, :]
        r = rows - top
        u = np.clip(r / crown, 0, 1)
        tp = (r % T) / T
        hw = maxw * (0.1 + 0.9 * u) * (0.42 + 0.58 * tp) + 0.4
        dx = cols - (x + 0.5 + sway * (1 - u))
        jag = hash01(cols + 0 * rows, rows + 0 * cols, seed) * 0.9
        inside = np.abs(dx) <= hw - jag * tp
        k = np.clip(0.5 + dx / (hw * 1.3), 0, 1)
        col = _c(lit) * (1 - k[..., None]) + _c(shade) * k[..., None]
        col *= (1.12 - 0.32 * tp)[..., None]
        v = P[y0:y1, xs0:xs1]
        v[inside] = col[inside]
        if rim is not None:
            left = inside & ~np.roll(inside, 1, axis=1) & (dx < 0)
            v[left] = rim
        if snow > 0:
            sn = inside & (tp < 0.28 + 0.2 * snow) & (hash01(cols + 0 * rows, rows + 0 * cols, seed + 7) < snow)
            v[sn] = (226, 232, 246)

    def flames(self, P, cx, base, heat, width, key):
        H, W = P.shape[:2]
        fh = 4 + heat * 7
        x0, x1 = max(0, int(cx - width)), min(W, int(cx + width) + 1)
        y0, y1 = max(0, int(base - fh * 1.7)), min(H, int(base))
        if x0 >= x1 or y0 >= y1:
            return
        st = self._flick.get(key)
        if st is None or len(st) != x1 - x0:
            st = np.random.random(x1 - x0)
        st = self._flick[key] = st * 0.55 + np.random.random(x1 - x0) * 0.45
        dx = (np.arange(x0, x1) + 0.5 - cx) / width
        colh = fh * np.clip(1 - dx * dx, 0, 1) ** 0.7 * (0.55 + 0.75 * st)
        hgt = (base - np.arange(y0, y1, dtype=np.float32))[:, None]
        u = hgt / (colh[None, :] + 1e-3)
        a = np.clip((1.15 - u) * 2.2, 0, 1)
        v = P[y0:y1, x0:x1]
        v += (flame_colours(np.clip(u, 0, 1)) - v) * a[..., None]

    def lightning(self, P):
        """Thunderstorm: flash the scene and fork a glowing bolt across its sky,
        triggered by the audio callback exactly when a strike is heard."""
        e = self.eng
        if e.amb_g["storm"] < 0.05:
            return
        dt = time.monotonic() - e.flash_at
        if dt > 0.5:
            return
        power = e.flash_power
        H, W = P.shape[:2]
        if getattr(self, "_bolt_for", None) != e.flash_at:          # new strike → new bolt
            self._bolt_for, self._bolt = e.flash_at, None
            if self.sky and power > 0.6:
                x0, x1, y0, y1 = self.sky
                Y0, Y1 = 2 * y0, min(H, 2 * y1)
                if x1 - x0 > 6 and Y1 - Y0 > 6:
                    M = np.zeros((H, W), np.float32)
                    x = random.randint(x0 + 3, x1 - 4)
                    fork = random.randint(Y0 + 2, Y1 - 2)
                    fx = x
                    for y in range(Y0, Y1):
                        x += random.choice((-1, 0, 0, 1)) + (random.choice((-2, 2)) if random.random() < 0.12 else 0)
                        x = max(x0, min(x1 - 1, x))
                        M[y, x] = 1
                        if y == fork:
                            fx = x
                    for k in range(random.randint(4, 9)):
                        yb, xb = fork + k, fx + int(k * 1.3) + 1
                        if yb < H and xb < W:
                            M[yb, xb] = 0.8
                    self._bolt = (M, blur(blur(M, 2), 2))
        on = dt < 0.06 or 0.11 < dt < 0.2 or (power > 0.6 and 0.28 < dt < 0.33)
        if not on:
            return
        P += (_c((210, 220, 255)) - P) * (0.42 * power)
        if self._bolt is not None:
            M, G = self._bolt
            P += G[..., None] * _c((150, 160, 255)) * 2.5
            P[M > 0] = (255, 255, 255)

    def analog_fx(self, ch, fg, bg, a):
        """Vinyl on → worn film / VHS: faded warm colour, vignette, grain, dust,
        wandering scratches, a rolling tracking bar and the odd frame jump."""
        h, w = ch.shape
        if getattr(self, "_vig_key", None) != (h, w):
            yy, xx = np.mgrid[0:h, 0:w]
            d = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
            self._vig = np.clip(1 - 0.42 * np.clip(d - 0.25, 0, None), 0.3, 1).astype(np.float32)[..., None]
            self._vig_key = (h, w)
            self._scratches = []
        t = time.time()
        vig = 1 - (1 - self._vig) * a
        tint = _c((1.07, 0.98, 0.84))
        for arr in (fg, bg):
            lum = arr.mean(-1, keepdims=True)
            arr *= 1 - 0.3 * a
            arr += lum * tint * 0.3 * a
            arr *= vig
            arr += 9 * a
        if getattr(self, "_grain", None) is None or self._grain.shape[:2] != (h, w) or random.random() < 0.05:
            self._grain = np.random.normal(0, 5, (h, w, 1)).astype(np.float32)
        bg += self._grain * a
        def mark(ys, xs, chars, col):
            bg[ys, xs] = (fg[ys, xs] + bg[ys, xs]) * 0.5
            ch[ys, xs] = chars
            fg[ys, xs] = bg[ys, xs] * 0.35 + _c(col) * 0.65
        n = int(w * h / 260 * a)                        # dust on the lens
        ys, xs = np.random.randint(0, h, n), np.random.randint(0, w, n)
        free = np.isin(ch[ys, xs], (" ", "▀"))
        if free.any():
            mark(ys[free], xs[free], np.random.choice(list("˙·'`,."), int(free.sum())), (200, 190, 170))
        if random.random() < 0.05 * a:                 # film scratches that wander
            y0 = random.randrange(h)
            self._scratches.append([random.randrange(w), y0, min(h, y0 + random.randint(3, h)), random.randint(2, 6)])
        keep = []
        for s in self._scratches:
            s[0] = min(w - 1, max(0, s[0] + random.choice((-1, 0, 0, 1))))
            ys = np.arange(s[1], s[2])
            ys = ys[np.random.random(len(ys)) < 0.8]
            if len(ys):
                mark(ys, np.full(len(ys), s[0]), np.where(np.random.random(len(ys)) < 0.7, "│", "╎"), (214, 204, 186))
            s[3] -= 1
            if s[3] > 0:
                keep.append(s)
        self._scratches = keep
        if (t % 14.0) < 4.0:                            # VHS tracking bar rolling down
            yb = int((t % 14.0) / 4.0 * (h + 4)) - 2
            for y in (yb, yb + 1):
                if 0 <= y < h:
                    sh = random.randint(1, 2)
                    ch[y] = np.roll(ch[y], sh)
                    fg[y] = np.roll(fg[y], sh, axis=0) + 14
                    bg[y] = np.roll(bg[y], sh, axis=0) + 14
                    xs = np.random.randint(0, w, w // 8)
                    mark(np.full(len(xs), y), xs, "░", (130, 120, 140))
        if random.random() < 0.01 * a:                 # the odd frame jump
            ch[:] = np.roll(ch, 1, axis=1)
            fg[:] = np.roll(fg, 1, axis=1)
            bg[:] = np.roll(bg, 1, axis=1)
