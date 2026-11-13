import math
import random
import time

import numpy as np

from ..paint import _c, disc, draw_stars, fbm3, glow, grid, line, make_stars, moon, noise_tex, ramp
from ..world import wrng, wseed
from ..theory import SR, TAU


# the planet per station: what kind of world it is and how it's coloured and lit. The terrain, moons, tilt
# and (sometimes) a ring come from the world seed; the station decides oceans, ice, clouds, lights, glow
PLANET = [
    dict(kind="ocean world", style="terra", sea=0.45, water=[(0, (70, 120, 200)), (0.35, (36, 62, 150)), (1, (16, 22, 80))],
         land=[(0, (196, 182, 150)), (0.12, (86, 140, 104)), (0.6, (70, 96, 92)), (1, (150, 140, 170))], caps=0.3,
         clouds=0.55, cloud=(232, 226, 248), atm=(160, 130, 255), dusk=(255, 130, 170), sun=(-0.75, -0.35, 0.55),
         sunc=(255, 214, 236), amb=0.05, gloss=40, lights=(255, 200, 120), lights_k=0.5,
         space=(8, 6, 22), n1=(60, 22, 80), n2=(16, 52, 80), ringc=[(0, (170, 150, 200)), (1, (230, 210, 240))]),
    dict(kind="city planet", style="terra", sea=0.1, water=[(0, (40, 60, 120)), (1, (8, 12, 40))],
         land=[(0, (96, 100, 120)), (0.5, (70, 74, 96)), (1, (130, 134, 156))], caps=0.25,
         clouds=0.3, cloud=(190, 200, 230), atm=(110, 150, 255), dusk=(140, 120, 255), sun=(-0.9, -0.2, -0.15),
         sunc=(200, 214, 255), amb=0.03, gloss=50, lights=(255, 206, 120), lights_k=1.4,
         space=(3, 4, 14), n1=(20, 30, 90), n2=(10, 50, 70), ringc=[(0, (120, 130, 170)), (1, (200, 206, 236))]),
    dict(kind="desert world", style="desert", sea=-9, water=[(0, (0, 0, 0)), (1, (0, 0, 0))],
         land=[(0, (150, 70, 46)), (0.35, (206, 120, 70)), (0.7, (236, 176, 112)), (1, (255, 226, 176))], caps=0.12,
         clouds=0.12, cloud=(246, 210, 170), atm=(255, 160, 90), dusk=(255, 90, 60), sun=(-0.7, -0.45, 0.55),
         sunc=(255, 200, 140), amb=0.06, gloss=0, ring=True,
         space=(14, 6, 10), n1=(100, 40, 30), n2=(80, 60, 20), ringc=[(0, (170, 120, 90)), (0.5, (236, 196, 150)), (1, (200, 150, 110))]),
    dict(kind="ice world", style="terra", sea=0.2, water=[(0, (200, 226, 246)), (1, (140, 176, 220))],
         land=[(0, (226, 232, 246)), (0.5, (184, 196, 222)), (1, (250, 252, 255))], caps=0.75,
         clouds=0.45, cloud=(244, 248, 255), atm=(200, 220, 255), dusk=(255, 190, 210), sun=(-0.7, -0.4, 0.58),
         sunc=(230, 240, 255), amb=0.07, gloss=12, aurora=(120, 255, 190),
         space=(6, 9, 20), n1=(40, 60, 100), n2=(70, 80, 120), ringc=[(0, (180, 200, 230)), (1, (240, 246, 255))]),
    dict(kind="gas giant", style="gas", sea=-9, bands=[(0, (40, 10, 70)), (0.25, (150, 30, 140)), (0.5, (255, 90, 200)),
         (0.7, (90, 40, 180)), (0.85, (40, 220, 220)), (1, (230, 255, 250))], nb=9, spot=(60, 255, 230),
         clouds=0.0, cloud=(255, 255, 255), atm=(255, 60, 200), dusk=(120, 40, 255), sun=(-0.75, -0.3, 0.6),
         sunc=(255, 140, 230), amb=0.06, gloss=0, aurora=(60, 255, 230), ring=True,
         space=(8, 2, 20), n1=(120, 0, 100), n2=(0, 90, 110), ringc=[(0, (60, 255, 230)), (0.5, (150, 90, 255)), (1, (255, 60, 200))]),
    dict(kind="jungle world", style="terra", sea=0.2, water=[(0, (40, 140, 140)), (1, (6, 40, 56))],
         land=[(0, (60, 120, 90)), (0.3, (26, 86, 60)), (0.8, (20, 60, 50)), (1, (90, 120, 110))], caps=0.2,
         clouds=0.35, cloud=(200, 240, 226), atm=(120, 255, 200), dusk=(255, 210, 140), sun=(-0.8, -0.3, 0.35),
         sunc=(220, 255, 236), amb=0.04, gloss=30, bio=(90, 255, 200),
         space=(3, 12, 14), n1=(20, 80, 60), n2=(40, 60, 100), ringc=[(0, (120, 180, 160)), (1, (210, 250, 230))]),
    dict(kind="garden world", style="terra", sea=0.25, water=[(0, (70, 190, 230)), (0.3, (30, 110, 200)), (1, (14, 50, 130))],
         land=[(0, (238, 222, 170)), (0.08, (110, 186, 80)), (0.45, (70, 140, 60)), (0.8, (150, 130, 90)), (1, (250, 250, 250))],
         caps=0.32, clouds=0.42, cloud=(255, 255, 255), atm=(130, 200, 255), dusk=(255, 180, 120), sun=(-0.45, -0.35, 0.82),
         sunc=(255, 248, 220), amb=0.08, gloss=36,
         space=(10, 14, 34), n1=(50, 90, 160), n2=(160, 140, 90), ringc=[(0, (200, 210, 230)), (1, (255, 250, 236))]),
    dict(kind="sakura world", style="terra", sea=0.3, water=[(0, (90, 120, 200)), (0.4, (44, 60, 150)), (1, (20, 22, 80))],
         land=[(0, (236, 214, 200)), (0.1, (250, 170, 200)), (0.45, (220, 110, 160)), (0.75, (120, 80, 120)), (1, (250, 246, 250))],
         caps=0.28, clouds=0.4, cloud=(255, 236, 244), atm=(255, 150, 200), dusk=(255, 110, 120), sun=(-0.7, -0.35, 0.6),
         sunc=(255, 214, 230), amb=0.06, gloss=34, lights=(255, 110, 80), lights_k=0.9,
         space=(12, 6, 26), n1=(110, 30, 90), n2=(30, 40, 110), ringc=[(0, (230, 150, 190)), (1, (255, 230, 240))]),
    dict(kind="vinyl giant", style="gas", sea=-9, bands=[(0, (40, 18, 14)), (0.3, (120, 60, 30)), (0.5, (220, 150, 70)),
         (0.7, (150, 80, 40)), (0.85, (250, 214, 150)), (1, (90, 40, 30))], nb=11, spot=(250, 200, 110),
         clouds=0.0, cloud=(255, 255, 255), atm=(255, 170, 90), dusk=(255, 110, 60), sun=(-0.7, -0.35, 0.6),
         sunc=(255, 200, 140), amb=0.05, gloss=0, ring=True, vinyl=True,
         space=(10, 6, 8), n1=(90, 40, 20), n2=(60, 20, 40), ringc=[(0, (34, 28, 32)), (0.5, (58, 50, 54)), (1, (30, 24, 28))]),
    dict(kind="green isle", style="terra", sea=0.5, water=[(0, (90, 120, 130)), (0.4, (44, 70, 90)), (1, (18, 30, 50))],
         land=[(0, (200, 196, 170)), (0.1, (100, 140, 80)), (0.45, (70, 100, 64)), (0.75, (120, 100, 110)), (1, (230, 230, 226))],
         caps=0.2, clouds=0.7, cloud=(226, 228, 232), atm=(170, 190, 220), dusk=(255, 170, 120), sun=(-0.7, -0.35, 0.6),
         sunc=(255, 236, 210), amb=0.06, gloss=30, lights=(255, 206, 120), lights_k=0.8,
         space=(8, 8, 16), n1=(40, 50, 80), n2=(60, 40, 50), ringc=[(0, (170, 170, 180)), (1, (230, 230, 236))])
]


class PlanetScene:
    # ── planet ─────────────────────────────────────────────────────────────
    PLANET_TH, PLANET_TW = 128, 256

    def _planet_maps(self):
        """This world's planet surface, wrapped on a sphere as lat/lon maps: terrain height, canyons, clouds,
        gas-giant turbulence, city lights and glowing jungle. The station only decides how they're coloured."""
        TH, TW = self.PLANET_TH, self.PLANET_TW
        lat = (0.5 - (np.arange(TH) + 0.5) / TH) * np.pi
        lon = (np.arange(TW) + 0.5) / TW * TAU
        LA, LO = np.meshgrid(lat, lon, indexing="ij")
        x, y, z = np.cos(LA) * np.sin(LO), np.sin(LA), np.cos(LA) * np.cos(LO)
        r = wrng("planetmaps")
        k = r.uniform(0.25, 0.5)
        wx, wy, wz = (fbm3(x, y, z, ("warp", i), 3, 1.4) for i in range(3))
        hgt = fbm3(x + k * wx, y + k * wy, z + k * wz, "height", 5, r.uniform(1.0, 1.5))
        canyon = np.exp(-(fbm3(x + 0.3 * wx, y + 0.3 * wy, z + 0.3 * wz, "canyon", 3, 1.8) / 0.2) ** 2)
        cloud = fbm3(x + 0.6 * wx, y * 1.8 + 0.6 * wy, z + 0.6 * wz, "cloud", 4, 1.5)
        turb = fbm3(x * 0.5, y * 3.0, z * 0.5, "bands", 4, 1.6)
        city = fbm3(x, y, z, "city", 3, 3.0)
        dots = np.random.default_rng(wseed("lights")).random((TH, TW)).astype(np.float32)
        lights = np.clip(city - 0.1, 0, None) * (dots > 0.82) * dots + (dots > 0.985) * 0.4
        bio = np.clip(fbm3(x, y, z, "bio", 4, 5.0) - 0.9, 0, None)
        return dict(hgt=hgt, canyon=canyon, cloud=cloud, turb=turb, lights=lights.astype(np.float32), bio=bio)

    def v_planet(self, cv, P, h, w):
        """A procedurally generated planet that turns once every 8 bars, so it spins faster on faster
        stations and stops when the music is paused. Every station repaints it as another kind of world."""
        e = self.eng
        H = P.shape[0]
        t = time.time()
        V = self.var(PLANET)
        TH, TW = self.PLANET_TH, self.PLANET_TW
        Mp = self.tex("planetmaps", 0, self._planet_maps)
        G = self.layout("planet", 0, 0, lambda r: dict(
            tilt=r.uniform(-0.5, 0.5), tip=r.uniform(0.2, 0.5), ring=r.random() < 0.3,
            rin=r.uniform(1.3, 1.55), rout=r.uniform(1.9, 2.3),
            gaps=[(r.uniform(0.1, 0.9), r.uniform(0.02, 0.07)) for _ in range(r.randint(1, 4))],
            moons=[(r.uniform(1.6, 2.6), r.uniform(0, TAU), r.uniform(0.25, 0.7) * r.choice((1, -1)), r.uniform(0.07, 0.15),
                    r.uniform(-0.15, 0.15), (r.randint(190, 255), r.randint(180, 240), r.randint(180, 255)))
                   for _ in range(r.choice((0, 1, 1, 2, 2, 3)))],
            spot=(r.uniform(0, TAU), r.uniform(-0.5, 0.5) or 0.2, r.uniform(0.18, 0.3)),
            name=r.choice(("Ka", "Ve", "Lo", "Mi", "Su", "Or", "Ne", "Ta", "Yu", "Ri", "Ae", "Zo"))
                 + r.choice(("ra", "lune", "mi", "sol", "the", "nia", "vin", "do", "ka", "ris"))
                 + f"-{r.randint(2, 99)}{r.choice('bcdef')}"))
        s = getattr(self, "pl", None)
        if s is None:
            s = self.pl = dict(rot=0.0, spd=0.0, last=t)
        # spin: one turn per 8 bars (32 beats) of the current tempo; it eases between tempos and to a stop on pause
        dt = min(0.25, max(0.0, t - s["last"]))
        s["last"] = t
        target = 0.0 if e.paused else TAU * e.st["bpm"] / 60 / 32
        s["spd"] += (target - s["spd"]) * min(1.0, dt * 1.5)
        s["rot"] += s["spd"] * dt
        rot = s["rot"]
        beats = e.played / SR * e.st["bpm"] / 60
        pulse = 0.0 if e.paused else max(0.5 * math.exp(-(beats % 1) * 5), self.kick)   # the kick drum, or the beat in quiet parts
        bass = self.bass
        ring = bool(V.get("ring") or G["ring"])
        R = max(4.0, min(H * 0.36, w * (0.2 if ring else 0.3)))
        cx, cy = w / 2 - 0.5, H / 2 - 0.5
        yy, xx = grid(H, w)
        X, Y = (xx - cx) / R, (yy - cy) / R
        D = np.sqrt(X * X + Y * Y)
        # orientation: tilt the axis sideways (tilt) and toward us (tip); body = Mr @ view
        ct, st_ = math.cos(G["tilt"]), math.sin(G["tilt"])
        cb, sb = math.cos(G["tip"]), math.sin(G["tip"])
        Mr = np.array([[1, 0, 0], [0, cb, -sb], [0, sb, cb]]) @ np.array([[ct, -st_, 0], [st_, ct, 0], [0, 0, 1]])
        axis = Mr.T @ np.array([0.0, -1.0, 0.0])
        Lv = np.array(V["sun"], float)
        Lv /= np.linalg.norm(Lv)
        atm = _c(V["atm"])

        # ── space: nebula, stars, the far sun ──
        n1 = self.tex("pneb1", (H, w), lambda: noise_tex(H, w, 9, 15))
        n2 = self.tex("pneb2", (H, w), lambda: noise_tex(H, w, 6, 19))
        P[:] = _c(V["space"])
        P += np.clip(n1 - 0.3, 0, None)[..., None] * _c(V["n1"]) * 0.45
        P += np.clip(n2 - 0.6, 0, None)[..., None] * _c(V["n2"]) * 0.45
        P += np.clip(n1 - 0.4, 0, None)[..., None] * self.tint * (0.08 + 0.2 * self.change)    # the nebula glows in the chord's colour
        draw_stars(P, self.tex("pstars", (H, w), lambda: make_stars(wrng("pstars"), H, w, w * h // 16)), t, 0.8)
        self.note_sparks(P, "planet", lambda p, n: (w * ((n * 0.618 + 0.1) % 1), H * (0.92 - 0.84 * p)), (240, 236, 255),
                         rise=0.0, life=1.3, r=0.8)              # a star flares for each note, high notes high
        sx, sy = cx + Lv[0] * R * 3.4, cy + Lv[1] * R * 3.4
        glow(P, sx, sy, R * 1.4, V["sunc"], 0.22 * max(0.2, Lv[2] + 0.6))
        disc(P, sx, sy, max(1.2, R * 0.08), (255, 252, 236), 0.9 if Lv[2] > -0.2 else 0.0)

        ev = self.rare(16.0)
        if ev:
            self._rare_planet(P, ev, sx, sy, w)
        u_ax0, v_ax = Mr.T @ np.array([1.0, 0, 0]), Mr.T @ np.array([0, 0, 1.0])
        # ── ring geometry (equatorial plane, normal = axis) and the moons ──
        if ring:
            rz = -(axis[0] * X + axis[1] * Y) / axis[2]
            rr = np.sqrt(X * X + Y * Y + rz * rz)
            u = (rr - G["rin"]) / (G["rout"] - G["rin"])
            on = (u > 0) & (u < 1)
            if V.get("vinyl"):                       # the ring is a record: dense, finely grooved, a sheen that turns
                dens = 0.95 + 0.05 * np.sin(u * 260.0)
            else:
                dens = 0.55 + 0.35 * np.sin(u * 47.0) * np.sin(u * 13.0 + 1.0)
            for gc, gw in G["gaps"]:
                dens = dens * (1 - 0.85 * np.exp(-((u - gc) / gw) ** 2))
            ra = np.where(on, np.clip(dens, 0, 1) * np.clip(np.minimum(u, 1 - u) * 12, 0, 1), 0.0)
            pl = X * Lv[0] + Y * Lv[1] + rz * Lv[2]                                       # the planet's shadow
            perp = (X - pl * Lv[0]) ** 2 + (Y - pl * Lv[1]) ** 2 + (rz - pl * Lv[2]) ** 2
            rlit = (0.35 + 0.65 * abs(float(axis @ Lv))) * np.where((pl < 0) & (perp < 1), 0.18, 1.0)
            rcol = ramp(np.clip(u, 0, 1), V["ringc"]) * rlit[..., None] * (0.9 + 0.25 * bass)
            if V.get("vinyl"):
                ang = np.arctan2(X * v_ax[0] + Y * v_ax[1] + rz * v_ax[2], X * u_ax0[0] + Y * u_ax0[1] + rz * u_ax0[2])
                rcol = rcol + (np.clip(np.cos(2 * (ang - rot * 4)), 0, 1) ** 8 * 70 * rlit)[..., None] * _c((1, 0.9, 0.8))

            def draw_ring(sel):
                a = (ra * sel)[..., None] * 0.85
                P[:] = P * (1 - a) + rcol * a
        u_ax, v_ax = Mr.T @ np.array([1.0, 0, 0]), Mr.T @ np.array([0, 0, 1.0])
        moons = []
        for orb, a0, spd, mr, inc, tint in G["moons"]:
            a = a0 + rot * spd
            p = orb * (math.cos(a) * u_ax + math.sin(a) * v_ax) + orb * inc * math.sin(a) * axis
            moons.append((p, max(1.3, mr * R), tint))

        def draw_moons(front):
            for p, mr, tint in moons:
                if (p[2] > 0) == front:
                    if not front and math.hypot(p[0], p[1]) < 1.0:
                        continue                                            # behind the planet
                    moon(P, cx + p[0] * R, cy + p[1] * R, mr, dim=0.9 if front else 0.75, haze=0.35, tint=tint)

        draw_moons(False)
        if ring:
            draw_ring(rz < 0)

        # ── the planet ──
        m = D < 1 + 1.0 / R
        xs, ys = X[m], Y[m]
        zs = np.sqrt(np.clip(1 - xs * xs - ys * ys, 0, 1))
        N = np.stack([xs, ys, zs])
        B = Mr @ N
        lat = np.arcsin(np.clip(-B[1], -1, 1))
        lon = np.arctan2(B[0], B[2]) - rot
        ti = np.clip(((0.5 - lat / np.pi) * TH).astype(int), 0, TH - 1)
        tj = ((lon / TAU % 1) * TW).astype(int) % TW
        tc = (((lon - rot * 0.12 - t * 0.004) / TAU % 1) * TW).astype(int) % TW
        hv = Mp["hgt"][ti, tj]
        alat = np.abs(lat)
        n = len(xs)
        water = np.zeros(n, np.float32)
        emit = np.zeros((n, 3), np.float32)
        if V["style"] == "gas":
            bv = np.sin(lat * V["nb"] + 0.55 * Mp["turb"][ti, tj] + 0.15 * np.sin(lon * 3 + lat * 5))
            col = ramp(bv * 0.5 + 0.5, V["bands"])
            slon, slat, ssz = G["spot"]
            dl = (lon - slon + np.pi) % TAU - np.pi
            sd = np.sqrt((dl * np.cos(lat) / 1.7) ** 2 + (lat - slat) ** 2) / ssz
            sw = np.clip(1 - sd, 0, 1)
            col += (_c(V["spot"]) - col) * (sw ** 0.6 * (0.6 + 0.3 * np.sin(sd * 9 - t * 1.2)))[:, None]
            land = np.zeros(n, bool)
        else:
            sea = V["sea"]
            land = hv > sea
            lo = max(sea, -2.2)
            col = np.where(land[:, None], ramp(np.clip((hv - lo) / (2.6 - lo), 0, 1), V["land"]),
                           ramp(np.clip((sea - hv) / 1.6, 0, 1), V["water"]))
            water = (~land).astype(np.float32)
            if V["style"] == "desert":
                col *= (1 - 0.45 * Mp["canyon"][ti, tj])[:, None]
            cap = np.clip((alat - (np.pi / 2 - V["caps"]) + 0.12 * hv) / 0.06, 0, 1)
            if e.st.get("weather") == "snow":
                cap = np.maximum(cap, np.clip((hv - sea - 1.0) / 0.3, 0, 1))
            col += (_c((240, 246, 255)) - col) * cap[:, None]
            water *= 1 - cap
            if V.get("lights"):
                emit += _c(V["lights"]) * (Mp["lights"][ti, tj] * land * (1 - cap) * V["lights_k"]
                                           * (0.75 + 0.3 * np.sin(t * 3 + tj) + 0.3 * self.hat))[:, None]
            if V.get("bio"):
                emit += _c(V["bio"]) * (Mp["bio"][ti, tj] * land * (0.5 + 0.9 * bass + 0.5 * pulse))[:, None]
        # clouds (thicker when it rains)
        cover = V["clouds"] + (0.1 if e.wet and V["clouds"] > 0 else 0.0)
        cmask = np.zeros(n, np.float32)
        if cover > 0:
            thr = (0.5 - cover) * 2.4 + 0.3
            cmask = np.clip((Mp["cloud"][ti, tc] - thr) / 0.9, 0, 1) ** 1.3 * 0.88
            col += (_c(V["cloud"]) - col) * cmask[:, None]
            emit *= (1 - cmask * 0.85)[:, None]
            water *= 1 - cmask
        # sunlight, the glow along the terminator, sun glint on the water, the atmosphere at the rim
        diff = Lv[0] * xs + Lv[1] * ys + Lv[2] * zs
        lit = np.clip(diff * 1.1 + 0.05, 0, 1) ** 0.8
        if ring:                                                           # the ring's shadow on the planet
            dn = float(axis @ Lv)
            if abs(dn) > 1e-3:
                ad = axis[0] * xs + axis[1] * ys + axis[2] * zs
                sdist = -ad / dn
                hx, hy, hz = xs + sdist * Lv[0], ys + sdist * Lv[1], zs + sdist * Lv[2]
                hu = (np.sqrt(hx * hx + hy * hy + hz * hz) - G["rin"]) / (G["rout"] - G["rin"])
                lit = lit * np.where((sdist > 0) & (hu > 0) & (hu < 1), 0.45, 1.0)
        night = np.clip(-diff * 4 + 0.15, 0, 1)
        col = col * (V["amb"] + lit)[:, None]
        col += _c(V["dusk"]) * (np.exp(-(diff / 0.13) ** 2) * 0.3)[:, None]
        if V["gloss"]:
            Hh = Lv + np.array([0, 0, 1.0])
            Hh /= np.linalg.norm(Hh)
            sp = np.clip(Hh[0] * xs + Hh[1] * ys + Hh[2] * zs, 0, 1) ** V["gloss"]
            col += _c(V["sunc"]) * (sp * water * 0.9 * (diff > 0))[:, None]
        col += emit * night[:, None]
        fres = (1 - zs) ** 2.6
        col += atm * (fres * (0.2 + 0.9 * np.clip(diff + 0.35, 0, 1)) * (0.85 + 0.5 * bass))[:, None]
        if V.get("aurora"):                                                # polar lights that flare on the beat
            au = np.exp(-((alat - 1.12) / 0.07) ** 2) * (0.55 + 0.45 * np.sin(lon * 7 + t * 1.3))
            col += _c(V["aurora"]) * (au * (0.25 + 0.75 * pulse) * (0.3 + 0.7 * night))[:, None] * 0.9
        # thunderstorm: lightning lights up the cloud tops
        if e.amb_g["storm"] > 0.05 and time.monotonic() - e.flash_at < 0.25 and cover > 0:
            fr = random.Random(e.flash_at)
            fx, fy = fr.uniform(-0.6, 0.6), fr.uniform(-0.6, 0.6)
            col += _c((220, 226, 255)) * (cmask * np.exp(-((xs - fx) ** 2 + (ys - fy) ** 2) / 0.03))[:, None] * e.flash_power
        a = np.clip((1 - D[m]) * R + 0.5, 0, 1)[:, None]
        P[m] = P[m] * (1 - a) + col * a
        # the atmosphere's haze just outside the disc, brightest on the sunny side
        out = D > 0.97
        hl = np.clip(0.3 + 0.9 * (X * Lv[0] + Y * Lv[1]) / np.maximum(D, 1e-3), 0.08, 1.2)
        halo = np.exp(-np.clip(D - 1, 0, None) * R / (1.0 + R * 0.06)) * hl * (0.5 + 0.35 * bass + 0.25 * self.kick) * out
        P += halo[..., None] * atm * 0.6

        if ring:
            draw_ring(rz >= 0)
        draw_moons(True)

        bpm = e.st["bpm"]
        lab = f"{G['name']} · {V['kind']}"
        info = f"1 turn / 8 bars · {bpm} bpm"
        if w >= 30:
            cv.text(h - 2, 2, lab[: w - 4], (150, 140, 180))
            cv.text(h - 1, 2, info[: w - 4], (96, 90, 124))

    def _rare_planet(self, P, ev, sx, sy, w):
        """Rare event: a comet sails across the background (behind the planet), its dust tail and
        fainter blue ion tail both streaming away from the sun."""
        u, r = ev
        H = P.shape[0]
        a = r.uniform(0, TAU)
        d = math.hypot(w, H) * 0.6
        cx, cy = w / 2 + r.uniform(-0.25, 0.25) * w, H / 2 + r.uniform(-0.25, 0.25) * H
        x0, y0 = cx - math.cos(a) * d, cy - math.sin(a) * d
        x1, y1 = cx + math.cos(a) * d, cy + math.sin(a) * d
        x, y = x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
        ux, uy = x - sx, y - sy
        n = math.hypot(ux, uy) or 1.0
        ux, uy = ux / n, uy / n
        L = max(10.0, min(w, H * 2) * 0.22)
        line(P, x, y, x + ux * L, y + uy * L, (255, 236, 200), wid=(2.2, 0.6), a=0.35)            # dust tail
        line(P, x, y, x + ux * L * 0.6, y + uy * L * 0.6, (255, 246, 226), wid=(1.4, 0.4), a=0.4)
        bx, by = ux * 0.94 - uy * 0.34, uy * 0.94 + ux * 0.34                                      # ion tail, off-angle
        line(P, x, y, x + bx * L * 1.2, y + by * L * 1.2, (120, 190, 255), wid=(1.0, 0.4), a=0.3)
        glow(P, x, y, 4, (220, 236, 255), 0.6)
        disc(P, x, y, 0.9, (255, 255, 255))
