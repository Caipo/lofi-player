import math
import os
import random
import time

import numpy as np

from .paint import Canvas, compose
from .world import WORLD, wrng
from .scenes import MIXINS, SCENES, SCENE_DESC, VIZ_NAMES
from .scenes.spectrum import SPEC_GRAD
from .settings import (
    CYCLE_CHOICES, IDLE_CHOICES, SAVER_DEFAULT, TIMER_DEFAULT, TIMER_RANGE, _num, save_settings, settings_path)
from .composer import station_bpm
from .stations import STATIONS
from .term import C_DIM, C_LAV, C_PEACH, C_PINK, C_TEAL, C_WHITE, LINE, PANEL, Screen
from .theory import SR


HANN = np.hanning(2048)
AMB_KEYS = [("r", "rain", "rain"), ("c", "vinyl", "vinyl"), ("f", "fire", "fire"),
            ("w", "waves", "waves"), ("m", "café", "cafe"), ("t", "storm", "storm")]
LEVEL_CH = " ▁▂▃▄▅▆▇█"
PANELS = [("scenes", "scenes", "g"), ("mix", "mixer", "a"), ("timer", "focus timer", "o"), ("saver", "settings", ",")]


def mmss(s):
    s = max(0, int(math.ceil(s)))
    return f"{s // 60:02d}:{s % 60:02d}"


def chord_colour(root, minor):
    """A soft colour for a chord: its root's place on the circle of fifths picks the hue (so related chords
    sit close), minor chords are cooler and dimmer."""
    hue = ((root * 7) % 12) / 12 + (0.06 if minor else 0.0)
    r, g, b = (0.5 + 0.5 * math.cos(math.tau * (hue - o)) for o in (0.0, 1 / 3, 2 / 3))
    k = 0.8 if minor else 1.0
    return tuple(255 * k * (0.45 + 0.55 * c) for c in (r, g, b))


def key_name(st):
    return f"{st['key'][0]} {'major' if st['key'][1] == 'maj' else 'minor'}"


class App(*MIXINS):
    def __init__(self, term, eng, settings=None):
        self.term, self.eng = term, eng
        self.sel = eng.station_idx
        self.viz = 0
        self.show_side = True
        self.levels = {}
        self.bass = 0.0
        self.init_groove()
        self.city = self.city_old = None
        self.vu_peak = [0.0, 0.0]
        self.wave_gain = 3.0
        self.frame = 0
        self.edges = {}
        self.texs = {}
        self._flick = {}
        self.sky = None
        self.static = True                       # tuning static between stations
        self.transitions = True                  # a wipe / iris / blinds / dissolve between scenes
        self.trans, self.trans_n, self._last = None, 0, None
        self.tuned_at = -1e9
        self.rclock, self._rlast = 0.0, time.monotonic()      # scene time for rare events (stops on pause)
        self.rare_ev, self.rare_next, self.rare_force = None, random.uniform(50, 120), None
        self.rare_mem = {}
        self.new_world(int(os.environ["LOFI_SEED"]) if os.environ.get("LOFI_SEED", "").isdigit() else None)
        # ui state
        self.panel, self.prow, self.edit = None, 0, None
        self.digit = None                         # first digit of a two-digit scene number
        self.card = False                        # now-playing card
        self.toast = None                        # (title, line, until)
        self.toasts = True
        self.mouse = True
        self.saver, self.saver_switch, self.saver_at = False, 0.0, 0.0
        self.saver_opt = dict(SAVER_DEFAULT)
        self.last_input = time.monotonic()
        self.hits, self.drag = [], None
        self.timer = dict(TIMER_DEFAULT)
        self.pomo = dict(phase="focus", left=self.timer["focus"] * 60.0, running=False, round=0, last=time.monotonic())
        if settings:
            self.apply_settings(settings)
        self._seen_station = -1                  # → a "now playing" pop-up on the first frame
        self._seen_song = None
        self._saved = self.settings_dict()

    def var(self, table):
        return table[self.eng.station_idx % len(table)]

    # every scene keeps its generated layout in one of these; a new world throws them all away
    SCENE_STATE = ("city", "forest", "cafe", "ocean", "blossom", "lava", "koi", "train", "aqua", "roofst", "trail", "pl", "city_old",
                   "sparks")

    def new_world(self, seed=None):
        """Re-roll everything generated: skylines, trees, coasts, rooms, creatures, textures."""
        WORLD["seed"] = random.randrange(1 << 30) if seed is None else seed
        for name in self.SCENE_STATE:
            setattr(self, name, None)
        self.texs = {}
        self._flick = {}
        self.eng.recompose()                      # the music is generated from the world too

    def layout(self, name, h, w, fn, *extra):
        """Generated parameters for one scene at one size, stable until the world changes."""
        return self.tex("layout:" + name, (h, w) + extra, lambda: fn(wrng(name, h, w, *extra)))

    def noon(self):
        return bool(self.eng.st.get("noon"))

    def tex(self, name, key, fn):
        k = (name, key)
        if k not in self.texs:
            self.texs[k] = fn()
        return self.texs[k]

    # ── remembered settings ────────────────────────────────────────────────
    def settings_dict(self):
        e = self.eng
        return dict(version=1, station=e.station_idx, visual=VIZ_NAMES[self.viz], volume=round(e.volume, 2),
                    sidebar=self.show_side, now_playing=self.card, toasts=self.toasts, mouse=self.mouse,
                    static=self.static, transitions=self.transitions,
                    ambience={n: dict(on=bool(e.amb_on[n]), level=round(e.amb_lvl[n], 2)) for n in e.amb_names},
                    timer=dict(self.timer), screensaver=dict(self.saver_opt))

    def apply_settings(self, d):
        e = self.eng
        st = d.get("station")
        if isinstance(st, int) and not isinstance(st, bool) and 0 <= st < len(STATIONS):
            e._reset(st)                         # before the engine starts: no fade needed
            self.sel = st
        if d.get("visual") in VIZ_NAMES:
            self.viz = VIZ_NAMES.index(d["visual"])
        e.volume = _num(d.get("volume"), 0.0, 1.0, e.volume)
        self.show_side = bool(d.get("sidebar", True))
        self.card = bool(d.get("now_playing", False))
        self.toasts = bool(d.get("toasts", True))
        self.mouse = bool(d.get("mouse", True))
        self.static = bool(d.get("static", True))
        self.transitions = bool(d.get("transitions", True))
        amb = d.get("ambience")
        if isinstance(amb, dict):
            for n in e.amb_names:
                a = amb.get(n)
                if isinstance(a, dict):
                    e.amb_on[n] = bool(a.get("on", e.amb_on[n]))
                    e.amb_lvl[n] = _num(a.get("level"), 0.0, 1.0, 1.0)
        tm = d.get("timer")
        if isinstance(tm, dict):
            for k, (lo, hi, _) in TIMER_RANGE.items():
                self.timer[k] = int(_num(tm.get(k), lo, hi, self.timer[k]))
            for k in ("auto", "chime"):
                self.timer[k] = bool(tm.get(k, self.timer[k]))
        sv = d.get("screensaver")
        if isinstance(sv, dict):
            if sv.get("idle") in IDLE_CHOICES:
                self.saver_opt["idle"] = sv["idle"]
            if sv.get("cycle") in CYCLE_CHOICES:
                self.saver_opt["cycle"] = sv["cycle"]
            self.saver_opt["clock"] = bool(sv.get("clock", True))
        self.pomo["left"] = self.timer["focus"] * 60.0

    def autosave(self, force=False):
        d = self.settings_dict()
        if force or d != self._saved:
            save_settings(d)
            self._saved = d

    # ── analysis ───────────────────────────────────────────────────────────
    def analyze(self):
        s = self.eng.snapshot(4096)
        self.raw = s
        seg = s[-2048:]
        self.mono = s.mean(axis=1)
        self.smooth = np.convolve(self.mono, np.ones(12) / 12, "same")   # display-only, hides hiss
        self.rms = np.sqrt((seg.astype(np.float64) ** 2).mean(axis=0))
        self.spec = np.abs(np.fft.rfft(seg.mean(axis=1) * HANN)) / 512.0
        b = self.spec[1:8].max()
        bn = float(np.clip((20 * math.log10(b + 1e-9) + 34) / 30, 0, 1))
        self.bass = max(bn, self.bass * 0.86)
        self.groove()

    # ── what the band is playing right now ─────────────────────────────────
    # The engine lists every bar's drum hits, lead and riff notes and chord changes (meta["evs"]). These turn
    # the ones that just played into signals the scenes can move with, exactly in time with what you hear:
    #   kick / snare / hat / note / pluck   0…1, jump on a hit and decay (taiko counts as kick, brushes as snare)
    #   pitch / ppitch   the last lead / riff note, 0 (low) … 1 (high)
    #   change           1 on a chord change, fading over a second; root, minor: that chord
    #   tint             RGB the current chord's colour eases toward (circle of fifths round the hue wheel)
    #   energy           how busy the song section is (intro, turnaround ≈ .2 … melody 1), eased over seconds
    #   fired            the kinds that hit this frame ("kick", "snare", "hat", "note", "pluck", "change")
    GROOVE_KIND = dict(k=("kick", 0.85), T=("kick", 0.5), s=("snare", 0.5), b=("snare", 0.3), h=("hat", 0.16),
                       r=("hat", 0.2), w=("hat", 0.11), n=("note", 1.0), p=("pluck", 0.2), c=("change", 1.0))
    GROOVE_TAU = dict(kick=0.15, snare=0.17, hat=0.07, note=0.4, pluck=0.16, change=1.0)
    SECTION_ENERGY = dict(intro=0.25, turn=0.2, breakdown=0.45, groove=0.75, melody=1.0)

    def init_groove(self):
        for k in self.GROOVE_TAU:
            setattr(self, k, 0.0)
        self.pitch = self.ppitch = 0.5
        self.root, self.minor, self.energy = 0, False, 0.0
        self.tint = np.array(chord_colour(0, False), np.float32)
        self.fired = set()
        self._gmeta, self._gpos, self._gt = None, -1, time.monotonic()
        self.groove_force = None

    def groove(self):
        e, now = self.eng, time.monotonic()
        dt = min(0.2, now - self._gt)
        self._gt = now
        for k, tau in self.GROOVE_TAU.items():
            setattr(self, k, getattr(self, k) * math.exp(-dt / tau))
        self.fired = set()
        cur, pos = e.cur, e.pos
        target = 0.0
        if cur is not None and not e.paused and not e.muted:
            meta = cur[2]
            if meta is not self._gmeta:                  # a new bar: finish the old one, start this one
                if self._gmeta is not None:
                    self._fire(self._gmeta.get("evs", ()), self._gpos, 1 << 60)
                self._gmeta, self._gpos = meta, -1
            if pos > self._gpos:
                self._fire(meta.get("evs", ()), self._gpos, pos)
                self._gpos = pos
            target = self.SECTION_ENERGY.get(meta.get("section"), 0.6)
        self.energy += (target - self.energy) * (1 - math.exp(-dt / 2.5))
        self.tint += (np.array(chord_colour(self.root, self.minor), np.float32) - self.tint) * (1 - math.exp(-dt / 0.6))
        if self.groove_force is not None:            # --preview: everything mid-hit, and hitting every few frames
            for k in self.GROOVE_TAU:
                setattr(self, k, self.groove_force)
            self.energy = 1.0
            self._gpos += 1
            if self._gpos % 5 == 0:
                self.fired = set(self.GROOVE_TAU)
                self.pitch = self.ppitch = (self._gpos * 0.37) % 1

    def _fire(self, evs, a, b):
        for off, kind, v in evs:
            if not a < off <= b:
                continue
            ch, ref = self.GROOVE_KIND.get(kind, (None, 1))
            if ch is None:
                continue
            self.fired.add(ch)
            if ch == "note":
                self.note, self.pitch = 1.0, float(np.clip((v - 67) / 21, 0, 1))
            elif ch == "pluck":
                self.pluck, self.ppitch = 1.0, float(np.clip((v - 48) / 40, 0, 1))
            elif ch == "change":
                self.change, self.root, self.minor = 1.0, int(v) % 12, v % 1 > 0
            else:
                setattr(self, ch, max(getattr(self, ch), min(1.0, v / ref)))

    def band_levels(self, nb):
        if nb not in self.edges:
            f = np.geomspace(40, 9000, nb + 1)
            idx = np.clip(np.round(f / (SR / 2048)).astype(int), 1, 1024)
            lo = idx[:-1]
            hi = np.maximum(idx[1:], lo + 1)
            cen = np.sqrt(f[:-1] * f[1:])
            self.edges[nb] = (lo, hi, np.clip(4.5 * np.log2(cen / 250), 0, None))
        lo, hi, tilt = self.edges[nb]
        vals = np.array([self.spec[a:b].max() for a, b in zip(lo, hi)])
        norm = np.clip((20 * np.log10(vals + 1e-9) + tilt + 50) / 44, 0, 1)
        lv, pk = self.levels.get(nb, (np.zeros(nb), np.zeros(nb)))
        lv = np.maximum(norm, lv - 0.04)
        pk = np.maximum(lv, pk - 0.012)
        self.levels[nb] = (lv, pk)
        return lv, pk

    # ── main loop ──────────────────────────────────────────────────────────
    def run(self):
        self.term.set_mouse(self.mouse)
        while True:
            t0 = time.time()
            for k in self.term.keys():
                if self.handle(k) is False:
                    return
            self.pomo_tick()
            self.saver_tick()
            self.analyze()
            self.draw()
            self.frame += 1
            if self.frame % 30 == 0:
                self.autosave()
            time.sleep(max(0, 1 / 30 - (time.time() - t0)))

    def handle(self, k):
        """One input event: a key name or a ("MOUSE", button, x, y, pressed) tuple."""
        mouse = isinstance(k, tuple)
        if mouse and (k[1] & 32 or not k[4]):              # drag motion / release
            if k[1] & 32 and self.drag is not None:
                self.drag(k[2])
            if not k[4]:
                self.drag = None
            return None
        self.last_input = time.monotonic()
        if self.saver:                                   # anything wakes the screensaver
            self.saver = False
            return None
        return self.click(*k[1:4]) if mouse else self.key(k)

    def tune(self, i):
        self.sel = i
        self.eng.paused = False
        if self.static and i != self.eng.station_idx:
            self.eng.static()
            self.tuned_at = time.monotonic()
        self.eng.tune(i)

    def set_viz(self, i):
        i %= len(VIZ_NAMES)
        if i != self.viz and self.transitions and self._last is not None:
            P, cv = self._last
            self.trans = dict(t0=time.monotonic(), dur=0.9, kind=self.TRANS_KINDS[self.trans_n % len(self.TRANS_KINDS)],
                              P=P, ch=cv.ch, co=cv.co, col=self.var(SPEC_GRAD)[1])
            self.trans_n += 1
        self.viz = i

    def digit_key(self, d):
        """Scene by number, like a TV remote: 1–9 (0 = 10) switch at once, and a second
        digit typed within a second makes a two-digit number (1 then 3 → scene 13)."""
        now = time.monotonic()
        p = self.digit
        if p is not None and now - p[1] < 1.0 and p[0] * 10 + d <= len(VIZ_NAMES):
            self.set_viz(p[0] * 10 + d - 1)
            self.digit = None
            return
        n = 10 if d == 0 else d
        if n <= len(VIZ_NAMES):
            self.set_viz(n - 1)
        self.digit = (d, now) if 0 < d and d * 10 <= len(VIZ_NAMES) else None

    def key(self, k):
        e, n = self.eng, len(STATIONS)
        if self.panel and self.panel_key(k):
            return None
        if k in ("q", "Q"):
            return False
        if k == "ESC":
            if self.card:
                self.card = False
                return None
            return False
        if k in ("UP", "k"):
            self.sel = (self.sel - 1) % n
        elif k in ("DOWN", "j"):
            self.sel = (self.sel + 1) % n
        elif k == "ENTER":
            self.tune(self.sel)
        elif k == "n":
            self.tune((e.station_idx + 1) % n)
        elif k == "p":
            self.tune((e.station_idx - 1) % n)
        elif k == " ":
            e.paused = not e.paused
        elif k in ("RIGHT", "v", "l"):
            self.set_viz(self.viz + 1)
        elif k in ("LEFT", "V", "h"):
            self.set_viz(self.viz - 1)
        elif len(k) == 1 and k.isdigit():
            self.digit_key(int(k))
        elif k in ("+", "="):
            e.volume = min(1.0, round(e.volume + 0.05, 2))
        elif k in ("-", "_"):
            e.volume = max(0.0, round(e.volume - 0.05, 2))
        elif k in ("TAB", "s"):
            self.show_side = not self.show_side
        elif k in [a[0] for a in AMB_KEYS]:
            name = next(a[2] for a in AMB_KEYS if a[0] == k)
            e.amb_on[name] = not e.amb_on[name]
        elif k in [p[2] for p in PANELS]:
            self.open_panel(next(p[0] for p in PANELS if p[2] == k))
        elif k == "O":
            self.pomo_toggle()
        elif k == "i":
            self.card = not self.card
        elif k == "z":
            self.start_saver()
        elif k == "e":
            self.summon_rare()
        elif k == "x":
            self.new_world()
            self.show_toast(f"✦ a new world · #{WORLD['seed'] % 100000:05d}",
                            "every scene has been regenerated", secs=3.0, force=True)
        return None

    # ── mouse ──────────────────────────────────────────────────────────────
    def hit(self, y, x0, x1, click=None, drag=False, wheel=None):
        """Register a clickable strip for this frame. click(mx) / wheel(+1 | -1)."""
        self.hits.append((y, x0, x1, click, drag, wheel))

    def click(self, b, mx, my):
        wheel = {64: -1, 65: 1}.get(b)
        if b not in (0, 64, 65):
            return None
        for y, x0, x1, fn, drag, wfn in reversed(self.hits):
            if y == my and x0 <= mx < x1:
                if wheel is not None and wfn is not None:
                    wfn(wheel)
                    return None
                if wheel is None and fn is not None:
                    fn(mx)
                    if drag:
                        self.drag = fn
                    return None
                if wheel is None:
                    return None
        if wheel is not None:                            # the wheel anywhere else: volume
            self.eng.volume = float(np.clip(round(self.eng.volume - wheel * 0.05, 2), 0, 1))
        elif self.panel:                                 # a click outside the panel closes it
            self.panel = None
        return None

    # ── focus (pomodoro) timer ─────────────────────────────────────────────
    def pomo_len(self, phase):
        return self.timer[phase] * 60.0

    def pomo_tick(self):
        p, now = self.pomo, time.monotonic()
        if p["running"]:
            p["left"] -= now - p["last"]
        p["last"] = now
        if p["running"] and p["left"] <= 0:
            self.pomo_next(done=True)

    def pomo_next(self, done=False):
        """Move on to the next phase. done=True when the clock ran out (chime + pop-up)."""
        p = self.pomo
        if p["phase"] == "focus":
            p["round"] += 1
            every = self.timer["every"]
            p["phase"] = "long" if every and p["round"] % every == 0 else "short"
        else:
            p["phase"] = "focus"
        p["left"] = self.pomo_len(p["phase"])
        p["last"] = time.monotonic()
        if done:
            p["running"] = self.timer["auto"]
            mins = self.timer[p["phase"]]
            if p["phase"] == "focus":
                msg = ("◷ back to focus", f"{mins} minutes. you've got this.")
                notes = [72, 76, 79, 84]
            elif p["phase"] == "long":
                msg = ("◷ long break", f"{mins} minutes. you earned it — go for a walk.")
                notes = [84, 79, 76, 72, 67]
            else:
                msg = ("◷ break time", f"{mins} minutes. stretch, sip some water.")
                notes = [84, 79, 76]
            if not p["running"]:
                msg = (msg[0], msg[1] + "  (o → start)")
            self.show_toast(*msg, secs=8, force=True)
            if self.timer["chime"]:
                self.eng.chime(notes)

    def pomo_toggle(self):
        p = self.pomo
        p["running"] = not p["running"]
        p["last"] = time.monotonic()

    def pomo_reset(self):
        self.pomo.update(phase="focus", left=self.pomo_len("focus"), running=False, round=0, last=time.monotonic())

    def pomo_active(self):
        p = self.pomo
        return p["running"] or p["round"] > 0 or p["phase"] != "focus" or p["left"] < self.pomo_len("focus") - 0.5

    def pomo_label(self):
        p = self.pomo
        name = {"focus": "focus", "short": "break", "long": "long break"}[p["phase"]]
        return f"◷ {name} {mmss(p['left'])}{'' if p['running'] else ' ❚❚'}"

    def set_timer(self, k, v):
        lo, hi, _ = TIMER_RANGE[k]
        old = self.timer[k]
        self.timer[k] = int(min(hi, max(lo, v)))
        p = self.pomo
        if k == p["phase"] and not p["running"] and abs(p["left"] - old * 60) < 1:
            p["left"] = self.pomo_len(k)        # not started yet: follow the new length

    # ── screensaver ────────────────────────────────────────────────────────
    def start_saver(self):
        self.saver, self.panel = True, None
        self.saver_at = self.saver_switch = time.monotonic()

    def saver_tick(self):
        now = time.monotonic()
        idle = self.saver_opt["idle"]
        if not self.saver and idle and now - self.last_input > idle * 60:
            self.start_saver()
        cyc = self.saver_opt["cycle"]
        if self.saver and cyc and now - self.saver_switch > cyc * 60:
            self.saver_switch = now
            self.set_viz(self.viz + 1)

    def show_toast(self, title, line, secs=5.0, force=False):
        if force or self.toasts:
            self.toast = (title, line, time.monotonic() + secs)

    # ── panels: mixer, focus timer, settings ───────────────────────────────
    def open_panel(self, name):
        if self.panel == name:
            self.panel = None
        else:
            self.panel, self.prow, self.edit = name, (self.viz if name == "scenes" else 0), None
            rows = self.panel_rows()
            while self.prow < len(rows) - 1 and rows[self.prow]["kind"] == "info":
                self.prow += 1

    def panel_rows(self):
        e = self.eng
        if self.panel == "scenes":
            return [dict(kind="scene", idx=i, label=name) for i, name in enumerate(VIZ_NAMES)]
        if self.panel == "mix":
            def setvol(v):
                e.volume = v
            rows = [dict(kind="slider", label="master", key="", get=lambda: e.volume, set=setvol, on=lambda: True)]
            for key, label, name in AMB_KEYS:
                def setl(v, n=name):
                    e.amb_lvl[n] = v
                    if v > 0:
                        e.amb_on[n] = True

                def flip(n=name):
                    e.amb_on[n] = not e.amb_on[n]
                rows.append(dict(kind="slider", label=label, key=key, get=lambda n=name: e.amb_lvl[n], set=setl,
                                 on=lambda n=name: e.amb_on[n], toggle=flip))
            return rows
        if self.panel == "timer":
            p = self.pomo

            def every_fmt(v):
                return "never" if not v else f"after {v} focus" if v == 1 else f"every {v} focus"
            return [
                dict(kind="info", label=lambda: f"{ {'focus': 'FOCUS', 'short': 'BREAK', 'long': 'LONG BREAK'}[p['phase']]}"
                                              f"  {mmss(p['left'])}   round {p['round'] + (p['phase'] == 'focus')}"),
                dict(kind="action", label=lambda: "❚❚ pause" if p["running"] else "▶ start", act=self.pomo_toggle),
                dict(kind="action", label=lambda: "⏭ skip to " + ("break" if p["phase"] == "focus" else "focus"),
                     act=self.pomo_next),
                dict(kind="action", label=lambda: "↺ reset", act=self.pomo_reset),
                dict(kind="num", label="focus", key="focus", fmt=lambda v: f"{v} min"),
                dict(kind="num", label="short break", key="short", fmt=lambda v: f"{v} min"),
                dict(kind="num", label="long break", key="long", fmt=lambda v: f"{v} min"),
                dict(kind="num", label="long break", key="every", fmt=every_fmt),
                dict(kind="toggle", label="auto-start next", get=lambda: self.timer["auto"],
                     set=lambda v: self.timer.__setitem__("auto", v)),
                dict(kind="toggle", label="chime", get=lambda: self.timer["chime"],
                     set=lambda v: self.timer.__setitem__("chime", v)),
            ]
        so = self.saver_opt

        def set_mouse(v):
            self.mouse = v
            self.term.set_mouse(v)
        return [
            dict(kind="action", label=lambda: "▶ start screensaver now  (z)", act=self.start_saver),
            dict(kind="choice", label="screensaver when idle", values=IDLE_CHOICES, get=lambda: so["idle"],
                 set=lambda v: so.__setitem__("idle", v), fmt=lambda v: f"{v} min" if v else "off"),
            dict(kind="choice", label="screensaver: next scene", values=CYCLE_CHOICES, get=lambda: so["cycle"],
                 set=lambda v: so.__setitem__("cycle", v), fmt=lambda v: f"every {v} min" if v else "off"),
            dict(kind="toggle", label="screensaver: clock", get=lambda: so["clock"],
                 set=lambda v: so.__setitem__("clock", v)),
            dict(kind="toggle", label="now-playing pop-ups", get=lambda: self.toasts,
                 set=lambda v: setattr(self, "toasts", v)),
            dict(kind="toggle", label="now-playing card  (i)", get=lambda: self.card,
                 set=lambda v: setattr(self, "card", v)),
            dict(kind="toggle", label="tuning static", get=lambda: self.static,
                 set=lambda v: setattr(self, "static", v)),
            dict(kind="toggle", label="scene transitions", get=lambda: self.transitions,
                 set=lambda v: setattr(self, "transitions", v)),
            dict(kind="toggle", label="mouse", get=lambda: self.mouse, set=set_mouse),
        ]

    def row_adjust(self, r, d):
        """←/→ on a row (d = −1 or +1)."""
        k = r["kind"]
        if k == "slider":
            r["set"](float(np.clip(round(r["get"]() + 0.05 * d, 2), 0, 1)))
        elif k == "num":
            lo, hi, stp = TIMER_RANGE[r["key"]]
            v = self.timer[r["key"]]
            v = (v // stp + 1) * stp if d > 0 else (v - 1) // stp * stp      # snap to the step
            self.set_timer(r["key"], max(lo, v))
        elif k == "choice":
            vals = r["values"]
            i = vals.index(r["get"]()) if r["get"]() in vals else 0
            r["set"](vals[max(0, min(len(vals) - 1, i + d))])
        elif k == "toggle":
            r["set"](not r["get"]())

    def row_activate(self, r):
        k = r["kind"]
        if k == "scene":
            self.set_viz(r["idx"])
            self.panel = None
        elif k == "slider":
            (r.get("toggle") or (lambda: None))()
        elif k == "action":
            r["act"]()
        elif k == "toggle":
            r["set"](not r["get"]())
        elif k == "choice":
            vals = r["values"]
            i = vals.index(r["get"]()) if r["get"]() in vals else 0
            r["set"](vals[(i + 1) % len(vals)])
        elif k == "num":
            self.edit = ""

    def panel_key(self, k):
        """Keys while a panel is open. Returns True when the key was used."""
        rows = self.panel_rows()
        self.prow = min(self.prow, len(rows) - 1)
        r = rows[self.prow]
        if self.edit is not None:                        # typing a number of minutes
            if len(k) == 1 and k.isdigit():
                self.edit = (self.edit + k)[:3]
            elif k in ("\x7f", "\x08"):
                self.edit = self.edit[:-1]
            elif k == "ENTER":
                if self.edit and r["kind"] == "num":
                    self.set_timer(r["key"], int(self.edit))
                self.edit = None
            elif k == "ESC":
                self.edit = None
            else:
                return False
            return True
        if k == "ESC" or k == next(p[2] for p in PANELS if p[0] == self.panel):
            self.panel = None
        elif k in ("UP", "k", "DOWN", "j"):
            d = -1 if k in ("UP", "k") else 1
            i = self.prow
            for _ in range(len(rows)):
                i = (i + d) % len(rows)
                if rows[i]["kind"] != "info":
                    break
            self.prow = i
            if self.panel == "scenes":
                self.set_viz(i)
        elif k in ("LEFT", "h", "RIGHT", "l") and self.panel == "scenes":
            return False
        elif k in ("LEFT", "h", "RIGHT", "l"):
            self.row_adjust(r, -1 if k in ("LEFT", "h") else 1)
        elif k in ("ENTER", " "):
            self.row_activate(r)
        elif k in ("TAB", "BTAB"):
            names = [p[0] for p in PANELS]
            self.open_panel(names[(names.index(self.panel) + (1 if k == "TAB" else -1)) % len(names)])
        elif len(k) == 1 and k.isdigit() and r["kind"] == "num":
            self.edit = k
        elif len(k) == 1 and k.isdigit() and self.panel == "scenes":
            self.digit_key(int(k))
            self.prow = self.viz
        else:
            return False
        return True

    def draw_panel(self, scr, y0, x0, vh, vw):
        rows = self.panel_rows()
        if self.panel == "scenes":
            self.prow = self.viz                          # ←/→ and clicks elsewhere move it too
        self.prow = min(self.prow, len(rows) - 1)
        pw = min(52, vw - 2)
        fit =min(len(rows), vh - 6 - (self.panel == "saver"))
        if pw < 36 or fit < 3:
            self.panel = None
            return
        top = min(max(0, self.prow - fit // 2), len(rows) - fit)
        ph = fit + 6
        x = x0 + (vw - pw) // 2
        y = y0 + max(0, (vh - ph) // 2)
        bgc, sel_bg = (27, 23, 44), (206, 182, 255)
        scr.box(y, x, ph, pw, bgc, (120, 104, 170))
        self.hit(y, x, x + pw)                            # swallow clicks on the frame
        # tabs
        tx = x + 2
        for name, label, hk in PANELS:
            s = f" {label} "
            on = name == self.panel
            scr.text(y + 1, tx, s, (34, 26, 50) if on else C_DIM, sel_bg if on else None, at=1 if on else 0)
            self.hit(y + 1, tx, tx + len(s), click=lambda mx, n=name: self.open_panel(n) if self.panel != n else None)
            tx += len(s) + 1
        scr.text(y + 1, x + pw - 4, " ✕ ", C_PINK)
        self.hit(y + 1, x + pw - 4, x + pw - 1, click=lambda mx: setattr(self, "panel", None))
        for yy in range(y + 2, y + ph - 1):
            self.hit(yy, x, x + pw)
        scr.text(y + 2, x + 1, "─" * (pw - 2), (60, 54, 92))
        inner = pw - 4
        for i in range(top, top + fit):
            r = rows[i]
            yy, xx = y + 3 + i - top, x + 2
            sel = i == self.prow
            if sel:
                scr.text(yy, x + 1, " " * (pw - 2), C_WHITE, (44, 38, 70))
            self.draw_row(scr, r, i, yy, xx, inner, sel)
        hint = {"scenes": "↑↓ preview · ⏎ pick · or type a number",
                "mix": "←→ level · ⏎ on/off · drag bars",
                "timer": "←→ adjust · type minutes ⏎",
                "saver": "←→ change · ⏎ toggle"}[self.panel]
        foot = f"{hint} · ⇥ tab · esc"
        scr.text(y + ph - 2, x + 2, foot[:inner], C_DIM)
        if top > 0:
            scr.text(y + 2, x + pw - 6, " ▴ ", C_DIM)
        if top + fit < len(rows):
            scr.text(y + ph - 3, x + pw - 6, " ▾ ", C_DIM)
        if self.panel == "saver":
            p = settings_path().replace(os.path.expanduser("~"), "~")
            scr.text(y + ph - 3, x + 2, ("saved to " + p)[:inner], (90, 84, 120), at=2)

    def draw_row(self, scr, r, i, y, x, w, sel):
        k = r["kind"]
        lab = r["label"]() if callable(r["label"]) else r["label"]
        fgc = C_WHITE if sel else (206, 200, 224)

        def pick(mx):
            self.prow = i
        if k == "info":
            scr.text(y, x, lab.center(w)[:w], C_PEACH, at=1)
            return
        if k == "scene":
            on = r["idx"] == self.viz
            scr.text(y, x, f"{'◉' if on else '○'} {r['idx'] + 1:>2}  {lab}", C_TEAL if on else fgc, at=1 if on else 0)
            scr.text(y, x + 22, SCENE_DESC[r["idx"]][: w - 22], (150, 142, 180) if sel else C_DIM)
            self.hit(y, x - 1, x + w + 1, click=lambda mx, r=r: (pick(mx), self.row_activate(r)),
                     wheel=lambda d: (setattr(self, "prow", (self.prow + d) % len(VIZ_NAMES)), self.set_viz(self.prow)))
            return
        if k == "slider":
            on = r["on"]()
            dot = "●" if on else "○"
            scr.text(y, x, dot, C_TEAL if on else C_DIM)
            if "toggle" in r:
                self.hit(y, x, x + 2, click=lambda mx, r=r: (pick(mx), r["toggle"]()))
            scr.text(y, x + 2, f"{r['key']:1} {lab:<7}", fgc if on else C_DIM)
            bx, bw = x + 13, max(6, w - 19)
            v = r["get"]()
            fill = int(round(v * bw))
            col = (C_LAV if r["key"] == "" else C_TEAL) if on else C_DIM
            scr.text(y, bx, "━" * fill, col)
            scr.text(y, bx + fill, "─" * (bw - fill), (70, 64, 100))
            scr.text(y, bx + min(bw - 1, max(0, fill - 1)), "●", C_WHITE if sel else col)
            scr.text(y, bx + bw + 1, f"{int(round(v * 100)):>3}%", fgc if on else C_DIM)

            def setx(mx, r=r, bx=bx, bw=bw):
                self.prow = i
                r["set"](float(np.clip((mx - bx + 1) / bw, 0, 1)))
            self.hit(y, bx - 1, bx + bw + 1, click=setx, drag=True,
                     wheel=lambda d, r=r: self.row_adjust(r, -d))
            self.hit(y, x + 2, bx - 1, click=pick, wheel=lambda d, r=r: self.row_adjust(r, -d))
            return
        if k == "action":
            scr.text(y, x, lab[:w], C_PEACH if sel else fgc, at=1 if sel else 0)
            self.hit(y, x, x + w, click=lambda mx, r=r: (pick(mx), r["act"]()))
            return
        if k == "toggle":
            on = r["get"]()
            scr.text(y, x, f"{lab:<{w - 8}}"[: w - 8], fgc)
            scr.text(y, x + w - 6, "● on " if on else "○ off", C_TEAL if on else C_DIM, at=1 if on else 0)
            self.hit(y, x, x + w, click=lambda mx, r=r: (pick(mx), r["set"](not r["get"]())))
            return
        # num / choice: label  ◂ value ▸
        if k == "num":
            v = self.timer[r["key"]]
            val = (self.edit + "▏ min") if (sel and self.edit is not None) else r["fmt"](v)
        else:
            val = r["fmt"](r["get"]())
        vw_ = 16
        scr.text(y, x, f"{lab:<{w - vw_ - 4}}"[: w - vw_ - 4], fgc)
        vx = x + w - vw_ - 3
        scr.text(y, vx, "◂", C_LAV if sel else C_DIM)
        scr.text(y, vx + 2, val.center(vw_)[:vw_], C_PEACH if (sel and self.edit is not None) else fgc, at=1 if sel else 0)
        scr.text(y, vx + vw_ + 2, "▸", C_LAV if sel else C_DIM)
        self.hit(y, x, vx, click=pick, wheel=lambda d, r=r: self.row_adjust(r, -d))
        self.hit(y, vx - 1, vx + 2, click=lambda mx, r=r: (pick(mx), self.row_adjust(r, -1)))
        self.hit(y, vx + 2, vx + vw_ + 2, click=lambda mx, r=r: (pick(mx), self.row_activate(r) if r["kind"] == "num" else self.row_adjust(r, 1)),
                 wheel=lambda d, r=r: self.row_adjust(r, -d))
        self.hit(y, vx + vw_ + 2, vx + vw_ + 4, click=lambda mx, r=r: (pick(mx), self.row_adjust(r, 1)))

    # ── now playing ────────────────────────────────────────────────────────
    def now_playing(self):
        e = self.eng
        st, meta = e.st, e.meta or {}
        return st, meta.get("chord", "…"), meta.get("section", "intro"), meta.get("bar", 0), meta.get("step", 0)

    def draw_card(self, scr, y0, x0, vh, vw):
        e = self.eng
        st, chord, section, bar, step = self.now_playing()
        cw = min(40, vw - 4)
        ch_ = 10
        if cw < 30 or ch_ + 2 > vh:
            return
        y, x = y0 + 1, x0 + 2
        scr.box(y, x, ch_, cw, (24, 20, 40), (96, 84, 140))
        self.hit(y, x, x + cw, click=lambda mx: setattr(self, "card", False))
        scr.text(y, x + 2, " now playing ", C_PINK, at=1)
        scr.text(y, x + cw - 4, " ✕ ", C_DIM)
        mark = "❚❚" if e.paused else ("♪" if self.frame // 8 % 2 else "♫")
        scr.text(y + 1, x + 2, f"{mark} {st['name']}"[: cw - 4], C_WHITE, at=1)
        scr.text(y + 2, x + 2, f"“{st.get('title', '')}” · song {st.get('song', 0) + 1}"[: cw - 4], C_PEACH, at=2)
        scr.text(y + 3, x + 2, st["mood"][: cw - 4], C_DIM, at=2)
        scr.text(y + 4, x + 2, f"{key_name(st)} · {st['bpm']} bpm · {st['keys']} + {st['lead']}{' + ' + st['riff_inst'] + ' riff' if st.get('riff_inst') else ''}{' + ' + st['arp_inst'] + ' arpeggios' if st.get('arp_inst') else ''}"[: cw - 4], C_LAV)
        prog = (e.meta or {}).get("prog") or st["prog"]
        xx = x + 2                                         # the progression, current chord lit
        for j, c in enumerate(prog):
            s = c if j == len(prog) - 1 else c + " → "
            if xx + len(s) > x + cw - 2:
                break
            scr.text(y + 5, xx, c, C_PEACH if j == step else (150, 140, 180), at=1 if j == step else 0)
            scr.text(y + 5, xx + len(c), s[len(c):], (90, 84, 120))
            xx += len(s)
        bp = int(e.beat_pos)
        beats = " ".join("●" if i == bp and not e.paused else "○" for i in range(4))
        secs = e.played // SR
        part = (e.meta or {}).get("part", "A")
        scr.text(y + 6, x + 2, f"bar {bar + 1:<4} {section} {part}"[:15], C_TEAL)
        scr.text(y + 6, x + 2 + 16, beats, C_PINK)
        scr.text(y + 6, x + cw - 8, mmss(secs), C_DIM)
        amb = [label for k, label, n in AMB_KEYS if e.amb_on[n]]
        scr.text(y + 7, x + 2, ("with " + ", ".join(amb) if amb else "no ambience")[: cw - 4], (120, 112, 150), at=2)
        scr.text(y + 8, x + 2, f"visual · {VIZ_NAMES[self.viz]}"[: cw - 4], (120, 112, 150))

    def draw_toast(self, scr, y0, x0, vh, vw):
        if not self.toast:
            return
        title, line, until = self.toast
        left = until - time.monotonic()
        if left <= 0:
            self.toast = None
            return
        tw = min(vw - 4, max(len(title), len(line)) + 6)
        if tw < 20 or vh < 6:
            return
        x = x0 + (vw - tw) // 2
        y = y0 + vh - 5
        scr.box(y, x, 4, tw, (30, 25, 50), (150, 128, 210))
        scr.text(y + 1, x + 3, title[: tw - 6], C_PEACH, at=1)
        scr.text(y + 2, x + 3, line[: tw - 6], C_WHITE)
        self.hit(y, x, x + tw, click=lambda mx: setattr(self, "toast", None))

    # ── layout ─────────────────────────────────────────────────────────────
    def render_scene(self, vh, vw):
        P = np.zeros((vh * 2, vw, 3), np.float32)
        cv = Canvas(vh, vw)
        self.sky = None
        now = time.monotonic()
        if not self.eng.paused:
            self.rclock += min(0.1, now - self._rlast)
        self._rlast = now
        getattr(self, SCENES[self.viz][1])(cv, P, vh, vw)
        self.lightning(P)
        self.transition(P, cv)
        self.tune_fx(P)
        self._last = (P, cv)                     # the frame a scene change dissolves away from
        ch, fg, bg = compose(P, cv)
        if self.eng.cg > 0.05:
            self.analog_fx(ch, fg, bg, self.eng.cg)
        return ch, fg, bg

    def draw(self):
        e = self.eng
        H, W = self.term.size()
        scr = Screen(H, W)
        self.hits = []
        if e.station_idx != self._seen_station and not e.muted:
            self._seen_station, self._seen_song = e.station_idx, e.st.get("song")
            self.show_toast(f"♫ now playing · {e.st['name']}", f"“{e.st.get('title', '')}” · {e.st['mood']}")
        elif e.st.get("song") != self._seen_song and e.station_idx == self._seen_station and not e.muted:
            self._seen_song = e.st.get("song")
            self.show_toast(f"♫ next up · {e.st.get('title', '')}", f"{key_name(e.st)} · {e.st['bpm']} bpm · {e.st['name']}")
        self.term.set_title(f"{'❚❚' if e.paused else '♫'} {e.st['name']} · lofi.term")
        if H < 16 or W < 50:
            scr.text(0, 0, "lofi.term needs at least 50×16 — make me bigger ☾"[: W - 1], C_PINK)
            self.term.present(scr)
            return
        if self.saver:
            self.draw_saver(scr, H, W)
            self.term.present(scr)
            return
        side = 28 if (W >= 84 and self.show_side) else 0
        if side:
            scr.fill(1, H - 4, 1, side, PANEL)
        scr.text(0, 0, "╭" + "─" * (W - 2) + "╮", LINE)
        for y in range(1, H - 1):
            scr.text(y, 0, "│", LINE)
            scr.text(y, W - 1, "│", LINE)
        scr.text(H - 4, 0, "├" + "─" * (W - 2) + "┤", LINE)
        scr.text(H - 1, 0, "╰" + "─" * (W - 2) + "╯", LINE)
        if side:
            for y in range(1, H - 4):
                scr.text(y, side, "│", LINE)
            scr.text(0, side, "┬", LINE)
            scr.text(H - 4, side, "┴", LINE)
        scr.text(0, 2, " ☾ lofi.term ", C_PINK, at=1)
        vname = f" {self.viz + 1}·{VIZ_NAMES[self.viz]} ▾ "
        if self.digit is not None and time.monotonic() - self.digit[1] < 1.0:
            vname = f" {self.digit[0]}_ " + vname
        scr.text(0, W - len(vname) - 2, vname, C_LAV)
        self.hit(0, W - len(vname) - 2, W - 2, click=lambda mx: self.open_panel("scenes"),
                 wheel=lambda d: self.set_viz(self.viz + d))
        if side:
            self.draw_side(scr, 1, 1, H - 5, side - 1)
        vx = side + 1 if side else 1
        vh, vw = H - 5, W - vx - 1
        ch, fg, bg = self.render_scene(vh, vw)
        scr.blit(1, vx, ch, fg, bg)
        if self.card:
            self.draw_card(scr, 1, vx, vh, vw)
        self.draw_toast(scr, 1, vx, vh, vw)
        if self.panel:
            self.draw_panel(scr, 1, vx, vh, vw)
        self.draw_status(scr, H - 3, 2, W - 4)
        self.term.present(scr)

    def draw_saver(self, scr, H, W):
        """Screensaver: the scene fills the whole terminal, with a quiet clock."""
        e = self.eng
        ch, fg, bg = self.render_scene(H, W)
        scr.blit(0, 0, ch, fg, bg)

        def label(y, x, s, col, at=0):
            if 0 <= y < H and x >= 0:
                n = min(len(s), W - x)
                scr.bg[y, x:x + n] = (scr.bg[y, x:x + n] * 0.35).astype(np.uint8)
                scr.text(y, x, s, col, at=at)
        if self.saver_opt["clock"] and H >= 8:
            now = time.strftime("%H:%M")
            st, chord, *_ = self.now_playing()
            np_ = f"{'❚❚' if e.paused else '♫'} {st['name']} · {chord}"
            label(H - 3, W - len(now) - 3, now, C_WHITE, at=1)
            label(H - 2, W - len(np_) - 3, np_, (190, 184, 214))
            if self.pomo_active():
                s = self.pomo_label()
                label(H - 3, 2, s, C_PEACH, at=1)
        if time.monotonic() - self.saver_at < 3:
            hint = " any key or click to wake "
            label(1, (W - len(hint)) // 2, hint, C_DIM)
        self.draw_toast(scr, 0, 0, H, W)

    def draw_side(self, scr, y, x, h, w):
        e = self.eng
        bottom = y + h
        scr.text(y, x + 1, "STATIONS", C_PINK, at=1)
        for i, st in enumerate(STATIONS):
            yy = y + 1 + i
            playing = i == e.station_idx
            mark = ("♪" if self.frame // 8 % 2 else "♫") if playing and not e.paused else ("▶" if playing else " ")
            label = f"{mark} {st['name']:<{w - 10}}{station_bpm(i):>3}bpm"[: w - 1]
            if i == self.sel:
                scr.text(yy, x, " " + label.ljust(w - 2), (34, 26, 50), C_LAV, at=1)
            else:
                scr.text(yy, x + 1, label, C_PEACH if playing else C_WHITE)
            self.hit(yy, x, x + w, click=lambda mx, i=i: self.tune(i),
                     wheel=lambda d: setattr(self, "sel", (self.sel + d) % len(STATIONS)))
        # bottom-up: mood text, focus timer, ambience; the visuals list gets what's left
        mood = STATIONS[self.sel]["mood"]
        lines, ln = [], ""
        for word in mood.split():
            if len(ln) + len(word) + 1 > w - 3:
                lines.append(ln)
                ln = word
            else:
                ln = (ln + " " + word).strip()
        lines.append(ln)
        y2 = y + len(STATIONS) + 2
        if y2 + 2 > bottom:                              # a short window: just the stations
            return
        sections = dict(amb=5, timer=4, mood=len(lines) + 1)
        n = len(VIZ_NAMES)
        for drop in ("mood", "timer", "amb"):
            if bottom - y2 - 1 - sum(sections.values()) >= min(n, 4):
                break
            sections.pop(drop)
        show = max(1, min(n, bottom - y2 - 1 - sum(sections.values())))
        start = min(max(0, self.viz - show // 2), n - show)
        more = (" ▴" if start > 0 else "") + (" ▾" if start + show < n else "")
        scr.text(y2, x + 1, "VISUALS", C_PINK, at=1)
        scr.text(y2, x + 9, more, C_DIM)
        scr.text(y2, x + w - 9, "g scenes", C_DIM)
        self.hit(y2, x, x + w, click=lambda mx: self.open_panel("scenes"))
        for j in range(show):
            i = start + j
            on = i == self.viz
            scr.text(y2 + 1 + j, x + 1, f"{'◉' if on else '○'} {i + 1:>2} {VIZ_NAMES[i]}"[: w - 2], C_TEAL if on else C_DIM)
            self.hit(y2 + 1 + j, x, x + w, click=lambda mx, i=i: self.set_viz(i), wheel=lambda d: self.set_viz(self.viz + d))
        yy = y2 + 1 + show + 1
        if "amb" in sections:
            scr.text(yy, x + 1, "AMBIENCE", C_PINK, at=1)
            scr.text(yy, x + w - 9, "a mixer", C_DIM)
            self.hit(yy, x, x + w, click=lambda mx: self.open_panel("mix"))
            for j, (k, label, name) in enumerate(AMB_KEYS):
                on = e.amb_on[name]
                lv = LEVEL_CH[int(round(e.amb_lvl[name] * 8))] if on else " "
                cx = x + 1 + (j % 2) * 13
                scr.text(yy + 1 + j // 2, cx, f"{'●' if on else '○'} {k} {label:<5}", C_TEAL if on else C_DIM)
                scr.text(yy + 1 + j // 2, cx + 10, lv, C_LAV)

                def adj(d, n=name):
                    e.amb_lvl[n] = float(np.clip(round(e.amb_lvl[n] - d * 0.1, 2), 0, 1))
                    e.amb_on[n] = e.amb_lvl[n] > 0 or e.amb_on[n]
                self.hit(yy + 1 + j // 2, cx, cx + 12, click=lambda mx, n=name: e.amb_on.__setitem__(n, not e.amb_on[n]),
                         wheel=adj)
            yy += 5
        if "timer" in sections:
            p = self.pomo
            scr.text(yy, x + 1, "FOCUS TIMER", C_PINK, at=1)
            scr.text(yy, x + w - 8, "o open", C_DIM)
            self.hit(yy, x, x + w, click=lambda mx: self.open_panel("timer"))
            col = C_PEACH if p["running"] else C_DIM
            scr.text(yy + 1, x + 1, self.pomo_label()[: w - 2], col, at=1 if p["running"] else 0)
            self.hit(yy + 1, x, x + w, click=lambda mx: self.pomo_toggle())
            every = self.timer["every"] or 4
            done = p["round"] % every if self.timer["every"] else p["round"]
            dots = "".join("●" if i < done else "○" for i in range(min(every, w - 14)))
            scr.text(yy + 2, x + 1, f"{dots}  round {p['round'] + (p['phase'] == 'focus')}"[: w - 2], (120, 112, 150))
        if "mood" in sections:
            for j, s in enumerate(lines):
                scr.text(bottom - len(lines) + j, x + 1, s, C_DIM, at=2)

    def draw_status(self, scr, y, x, w):
        e = self.eng
        st, chord, section, *_ = self.now_playing()
        secs = e.played // SR
        bp = int(e.beat_pos)
        beats = " ".join("●" if i == bp and not e.paused else "○" for i in range(4))
        state = "❚❚ paused" if e.paused else ("tuning…" if e.muted else "♫ on air")
        scr.text(y, x, state, C_PEACH, at=1)
        self.hit(y, x, x + 9, click=lambda mx: setattr(e, "paused", not e.paused))
        vol = int(round(e.volume * 10))
        vs = "vol " + "▮" * vol + "▯" * (10 - vol)
        right = x + w - len(vs)
        timer = ("  " + self.pomo_label()) if self.pomo_active() else ""
        segs = [f"  {st['name']}", f"  ·  {key_name(st)}", f"  ·  {chord:<6}", f"  ·  {st['bpm']} bpm", "  ·  "]
        while segs and x + 10 + sum(map(len, segs)) + 7 + 9 + len(timer) + len(vs) + 2 > x + w:
            segs.pop(1 if len(segs) > 2 else 0)
        info = "".join(segs)
        cx = x + 10
        scr.text(y, cx, info, C_WHITE)
        self.hit(y, cx, cx + len(info), click=lambda mx: setattr(self, "card", not self.card))
        cx += len(info)
        scr.text(y, cx, beats, C_PINK)
        cx += 7
        scr.text(y, cx, f"  {mmss(secs)}", C_DIM)
        cx += 7
        if timer and cx + len(timer) < right - 1:
            scr.text(y, cx, timer, C_PEACH if self.pomo["running"] else C_DIM)
            self.hit(y, cx + 2, cx + len(timer), click=lambda mx: self.open_panel("timer"))
        if right > cx:
            scr.text(y, right, vs, C_LAV)

            def setvol(mx, right=right):
                e.volume = float(np.clip((mx - right - 3) / 10, 0, 1))
            self.hit(y, right, x + w, click=setvol, drag=True,
                     wheel=lambda d: setattr(e, "volume", float(np.clip(round(e.volume - d * 0.05, 2), 0, 1))))
        hint = "↑↓⏎ tune  n/p skip  ←→ visual  g scenes  ␣ pause  a mixer  o focus  i now playing  z screensaver  , settings  q quit"
        scr.text(y + 1, x, hint[:w], C_DIM)
