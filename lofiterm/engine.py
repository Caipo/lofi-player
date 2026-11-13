import math
import queue
from functools import partial
import threading
import time

import numpy as np

from .composer import compose_song
from .dsp import SYNTHS, _t, add_note, fftconv, fir, lowpass, make_kit, syn_bass, syn_bell, syn_shaku
from .stations import STATIONS
from .theory import PC, PENTA, SR, TAIL, TAU, hz, parse_chord, voice_chord


# ════════════════════════════════════════════════════════════════════════════
#  Engine: renders bars in a worker thread, streams them from the audio callback
# ════════════════════════════════════════════════════════════════════════════
class Engine:
    def __init__(self):
        self.rng = np.random.default_rng()
        self.kit = make_kit(self.rng)
        t = _t(int(1.8 * SR))
        pre = np.zeros(int(0.018 * SR))
        self.ir = []
        for _ in range(2):
            ir = np.concatenate([pre, lowpass(self.rng.standard_normal(len(t)), 5000) * np.exp(-t / 0.5)])
            self.ir.append(ir / np.sqrt((ir ** 2).sum()))
        self.rain_bp = fir(2600, 255) - fir(350, 255)
        t = _t(int(0.012 * SR))
        self.ping = np.sin(TAU * 3100 * t) * np.exp(-t / 0.003)
        self.crk_k = np.exp(-np.arange(40) / 5.0)

        self.q = queue.Queue(maxsize=2)
        self.cur, self.pos, self.meta = None, 0, None
        self.gain, self.volume = 0.0, 0.7
        self.amb_names = ["rain", "vinyl", "fire", "waves", "cafe", "storm"]
        self.amb_on = dict(rain=True, vinyl=True, fire=False, waves=False, cafe=False, storm=False)
        self.amb_g = {k: 0.0 for k in self.amb_names}
        self.amb_lvl = {k: 1.0 for k in self.amb_names}       # mixer slider per layer, 0…1
        self.fx_in = queue.SimpleQueue()                       # one-shot sounds (timer chimes) from the UI
        self.fx = []                                           # …playing, owned by the audio thread: [buf, pos]
        self.apos, self.pend = 0, {}
        self.next_thunder = int(3 * SR)
        self.flash_at, self.flash_power = -1e9, 0.0
        self.bp_fire = fir(4200, 255) - fir(1100, 255)
        self.bp_wash = fir(5500, 255) - fir(700, 255)
        self.bp_storm = fir(1900, 255) - fir(160, 255)
        self.bp_static = fir(3800, 255) - fir(320, 255)       # AM-radio band for the tuning static
        t = _t(int(0.008 * SR))
        self.fire_k = self.rng.standard_normal(len(t)) * np.exp(-t / 0.0015)
        self.vowel_a = (fir(1250, 255) - fir(520, 255)) + 0.5 * (fir(2900, 255) - fir(2300, 255))   # "ah"
        self.vowel_b = (fir(460, 255) - fir(220, 255)) + 0.7 * (fir(2700, 255) - fir(1900, 255))    # "ee"
        t = _t(int(0.7 * SR))
        ir = lowpass(self.rng.standard_normal(len(t)), 3500) * np.exp(-t / 0.18)
        self.room_ir = ir / np.sqrt((ir ** 2).sum())
        self.voices = [dict(f0=f, gain=g, talk=False, end=int(SR * self.rng.uniform(0, 3)),
                            into=self.rng.uniform(0.4, 1.4), ph=self.rng.uniform(0, TAU))
                       for f, g in [(105, 1.0), (125, 0.8), (190, 0.9), (215, 0.7), (150, 0.6), (240, 0.5), (98, 0.6)]]
        self.rg, self.cg = 0.0, 0.0
        self.an_k = fir(2900, 63).astype(np.float32)          # vinyl "analog" tone
        self.an_state = np.zeros((62, 2), np.float32)
        self.paused, self.muted = False, True
        self.played = 0
        self.ring = np.zeros((8192, 2), np.float32)
        self.ri = 0
        self.req = None
        self.running = True
        self.station_idx = 0
        self.song_at = {}                                      # station → the song it's up to (in this world)
        self._reset(0)

    def _flag(name):
        return property(lambda self: self.amb_on[name], lambda self, v: self.amb_on.__setitem__(name, v))

    rain_on, vinyl_on, fire_on = _flag("rain"), _flag("vinyl"), _flag("fire")
    waves_on, cafe_on, storm_on = _flag("waves"), _flag("cafe"), _flag("storm")

    @property
    def wet(self):
        return self.amb_on["rain"] or self.amb_on["storm"]

    # ── control (UI thread) ────────────────────────────────────────────────
    def tune(self, i):
        self.req = i

    def chime(self, notes, gap=0.16):
        """Queue a soft bell arpeggio (MIDI notes) that plays over the music,
        even while it is paused. Used by the focus timer."""
        rng = np.random.default_rng()
        buf = np.zeros(int((len(notes) * gap + 2.4) * SR))
        for i, n in enumerate(notes):
            add_note(buf, i * gap * SR, syn_bell, hz(n), 0.5, 0.32, rng, rel=1.6)
        st = np.stack([buf, buf], -1).astype(np.float32)
        st[:, 1] = np.roll(st[:, 1], int(0.012 * SR))          # a touch of width
        self.fx_in.put([st, 0])

    # ── worker ─────────────────────────────────────────────────────────────
    def recompose(self):
        """The world changed: forget every station's songs and start this one's first song afresh."""
        if getattr(self, "th", None) is None:
            self.song_at = {}
            self._reset(self.station_idx)                       # not playing yet: no fade needed
        else:
            self.fresh = True                                   # the worker clears song_at when it switches
            self.tune(self.station_idx)

    def _reset(self, i, advance=False):
        """Switch to station i. With `advance` (tuning in while playing) it moves on to the station's next
        song rather than replaying the one it was on."""
        self.station_idx = i
        self.rain_on = STATIONS[i]["rain"]      # each station has its own ambience preset
        self.bar, self.gpos = 0, 0
        self.tail = np.zeros((TAIL, 2))
        self.prev_note = 76
        self.played = 0
        k = self.song_at.get(i)
        self._start_song(0 if k is None else k + 1 if advance else k)
        self.st = self.wst                      # the UI's view; from here on it follows playback (see _callback)

    def static(self, dur=1.4):
        """The sound of turning a radio dial: hiss that wavers as it sweeps past stations, crackle, a
        heterodyne whistle and a garbled scrap of somebody talking. Plays over the fade as you tune."""
        now = time.monotonic()
        if now < getattr(self, "static_until", 0.0) - 0.5:  # still crackling from the last turn of the dial
            return
        self.static_until = now + dur
        rng = np.random.default_rng()                        # (self.rng belongs to the worker thread)
        n = int(dur * SR)
        t = _t(n)
        knots = np.arange(0, dur + 0.05, 0.045)
        sweep = np.interp(t, knots, rng.random(len(knots))) ** 2                      # stations drifting past
        hiss = fftconv(rng.standard_normal(n), self.bp_static) * (0.3 + 0.7 * sweep)
        imp = (rng.random(n) < 0.0025) * rng.standard_normal(n) * rng.uniform(1, 4, n)
        crackle = np.convolve(imp, self.crk_k, "same")
        f = rng.uniform(1400, 2600) * np.exp(-t / rng.uniform(0.3, 0.6)) + 260 + 90 * np.sin(TAU * 5.5 * t)
        whistle = np.sin(np.cumsum(TAU * f / SR)) * np.exp(-((t - rng.uniform(0.2, 0.45)) / 0.16) ** 2)
        a = rng.uniform(0.55, 0.85)                         # a voice between stations: a few syllables
        pulses = np.zeros(n)
        f0 = rng.uniform(110, 190)
        pulses[np.arange(int(a * SR), int((a + 0.32) * SR), max(1, int(SR / f0)))] = 1.0
        voice = fftconv(pulses, self.vowel_a if rng.random() < 0.5 else self.vowel_b)
        voice *= (np.sin(TAU * rng.uniform(5, 8) * t) > -0.2) * np.clip(np.sin(np.pi * (t - a) / 0.32), 0, 1)
        sig = (hiss / (np.abs(hiss).max() + 1e-9) + 0.5 * crackle / (np.abs(crackle).max() + 1e-9)
               + 0.18 * whistle + 0.45 * voice / (np.abs(voice).max() + 1e-9))
        env = np.minimum(1, t / 0.03) * np.clip((dur - t) / 0.5, 0, 1)
        sig *= env * 0.17 / (np.abs(sig).max() + 1e-9)
        st = np.stack([sig, np.roll(sig, int(0.0007 * SR))], -1).astype(np.float32)
        self.fx_in.put([st, 0])

    def _start_song(self, k):
        """Compose song k for the current station. `wst` is what the worker plays from: the station merged
        with its song. The audio callback hands each bar's `st` to the UI as that bar starts playing."""
        i = self.station_idx
        self.song_at[i] = k
        self.wst = {**STATIONS[i], **compose_song(i, k)}
        self.sb = 0                             # bar within the song
        self.motifs, self.phrase_notes = {}, {}

    def _worker(self):
        while self.running:
            if self.req is not None:
                new = self.req
                self.muted = True
                t0 = time.time()
                while self.gain > 0.001 and time.time() - t0 < 1.5:
                    time.sleep(0.01)
                while not self.q.empty():
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        break
                self.cur = None
                if getattr(self, "fresh", False):
                    self.song_at, self.fresh = {}, False
                self._reset(new, advance=True)
                self.req = None
                self.q.put(self.render_bar())
                self.muted = False
                continue
            try:
                self.q.put(self.render_bar(), timeout=0.1) if not self.q.full() else time.sleep(0.03)
            except queue.Full:
                pass

    def start(self):
        import sounddevice as sd
        self.q.put(self.render_bar())
        self.muted = False
        self.th = threading.Thread(target=self._worker, daemon=True)
        self.th.start()
        self.stream = sd.OutputStream(samplerate=SR, channels=2, dtype="float32",
                                      blocksize=1024, latency="high", callback=self._callback)
        self.stream.start()

    def stop(self):
        self.running = False
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass

    # ── one bar of music ───────────────────────────────────────────────────
    def render_bar(self):
        st, rng = self.wst, self.rng
        plan = st["plan"][self.sb]                   # what this bar of the song does
        sec, part = plan["sec"], plan["part"]
        prog = st["prog"] if part == "A" else st["progB"]
        beat = 60.0 / st["bpm"]
        step = beat / 4
        bar_n = int(round(4 * beat * SR))
        N = bar_n + TAIL
        keys = np.zeros(N)
        lead = np.zeros(N)
        bass = np.zeros(N)
        drums = np.zeros(N)
        riff = np.zeros(N)

        cname = prog[plan["step"] % len(prog)]
        halves = cname.split("|")                    # a bar may hold two chords (a ii–V), two beats each
        hl = 4 / len(halves)
        voiced = [voice_chord(c) for c in halves]
        self.evs = evs = []                          # what the bar plays, for the visuals: (sample, kind, value)
        for j, c in enumerate(halves):
            pc, ints = parse_chord(c)
            evs.append((int(j * hl * beat * SR), "c", pc + (0.5 if 3 in ints and 4 not in ints else 0.0)))
        bnote, cnotes = voiced[0]
        sty = st["style"]
        rootless = sty.get("rootless") and st.get("walk")
        swung = st["drums"].get("feel") == "brush"

        # chords (gently strummed, humanized). Over a walking bass, jazz piano leaves the root to the bass
        syn = SYNTHS[st["keys"]]
        for j, (c, (_, cn)) in enumerate(zip(halves, voiced)):
            h0 = j * hl
            if rootless and len(cn) >= 4:
                pc = parse_chord(c)[0]
                cn = [n for n in cn if (n - pc) % 12] or cn
            vel = 0.42 / math.sqrt(len(cn)) * (0.62 if st.get("riff") or st.get("arp") else 1.0)   # under a riff the keys step back
            hits = [(b0, dur) for b0, dur in st["comp"] if h0 <= b0 < h0 + hl] or [(h0, hl * 0.8)]
            for b0, dur in hits:
                lag = st["swing"] * step if swung and b0 % 1 == 0.5 else 0.0   # jazz comping sits on the swing
                s = (b0 * beat + lag + rng.uniform(0, 0.012)) * SR
                for k, n in enumerate(cn):
                    add_note(keys, s + k * rng.uniform(0.006, 0.02) * SR, syn, hz(n),
                             min(dur, h0 + hl - b0 + 0.25) * beat * 0.95, vel * rng.uniform(0.8, 1.05), rng)

        # broken chords (an Alberti bass, a prelude's figure) on harpsichord or lute, over the held chord
        if st.get("arp") and sec != "turn":
            asyn = SYNTHS[st["arp_inst"]]
            for j, (_, cn) in enumerate(voiced):
                ns = sorted(cn)[:4]
                if len(ns) < 4:
                    ns.append(ns[0] + 12)
                for e in range(int(j * hl * 2), int((j + 1) * hl * 2)):
                    n = ns[min(st["arp"][e % 8], len(ns) - 1)]
                    on = e * beat / 2 + (st["swing"] * step if e % 2 else 0) + rng.normal(0, 0.004)
                    evs.append((int(max(0.0, on) * SR), "p", n))
                    add_note(keys, on * SR, asyn, hz(n), beat * 0.6, 0.2 * (1.1 if e % 4 == 0 else 0.9) * rng.uniform(0.9, 1),
                             rng, rel=0.4)

        # bass: a walking line on jazzy songs (chord tones, then a half step into the next chord), else the root
        if st.get("walk"):
            nxt = voice_chord(prog[(plan["step"] + 1) % len(prog)].split("|")[0])[0]
            line_ = []
            for j, (c, (bn, _)) in enumerate(zip(halves, voiced)):
                ivs = [i for i in parse_chord(c)[1] if i < 12]
                target = voiced[j + 1][0] if j + 1 < len(halves) else nxt
                approach = target + int(rng.choice([-1, 1, -1, 2]))         # chromatic, or a step from above
                if len(halves) == 1:
                    line_ += [bn, bn + int(rng.choice(ivs[1:] or [7])), bn + int(rng.choice([7, 12, ivs[-1]])), approach]
                else:
                    line_ += [bn, approach]
            for j, n in enumerate(line_):
                n = n - 12 if n > 50 else n + 12 if n < 28 else n
                add_note(bass, (j * beat + rng.uniform(0, 0.008)) * SR, syn_bass, hz(n), beat * 0.85,
                         0.5 * (1.0 if j % 2 == 0 else 0.85), rng, rel=0.08)
            if rng.random() < 0.3:                      # a swung skip note into beat 4
                n = line_[2] - 12 if line_[2] > 50 else line_[2]
                add_note(bass, (2 * beat + beat * 0.66) * SR, syn_bass, hz(n), beat * 0.25, 0.25, rng, rel=0.05)
        else:
            bsyn = SYNTHS.get(sty.get("bass"), syn_bass)
            bowed = bsyn is not syn_bass                 # a cello holds the note out under the bar
            for j, (bn, _) in enumerate(voiced):
                add_note(bass, (j * hl * beat + rng.uniform(0, 0.01)) * SR, bsyn, hz(bn), beat * hl * (0.92 if bowed else 0.675),
                         0.55 * (0.8 if bowed else 1.0), rng, rel=0.2 if bowed else 0.12)
            if not bowed and rng.random() < 0.65:
                nb = voiced[-1][0] + rng.choice([0, 7, 12])
                add_note(bass, 3 * beat * SR, syn_bass, hz(nb), beat * 0.8, 0.4, rng, rel=0.1)

        # drums: none in the intro, hats only in breakdowns and the turnaround between songs
        kicks = []
        full = sec in ("groove", "melody")
        hats_only = sec in ("breakdown", "turn")
        pat = st["drums"]
        brush = pat.get("feel") == "brush"
        wa = st.get("wa")
        taiko_fill = rng.random() < 0.6
        if wa and self.sb == 0:                         # hyoshigi: the wooden clappers that open a show
            for k, tt in enumerate((0.0, 0.26, 0.44)):
                self._hit(drums, int(tt * SR), "w", 0.3 - 0.05 * k)
        for s in range(16):
            delay = 0.0
            if s % 4 == 2:
                delay = st["swing"] * step
            elif s % 2 == 1:
                delay = st["swing"] * step * 0.5
            pos = int(max(0.0, s * step + delay + rng.normal(0, 0.004)) * SR)
            if brush:                                   # jazz: ride, brushes, a feathered kick, no pumping
                if full and s % 4 == 0:
                    self._hit(drums, pos, "k", (0.5 if pat["k"][s] == "x" else 0.2) * rng.uniform(0.85, 1.0))
                elif full and pat["k"][s] == "x":
                    self._hit(drums, pos, "k", 0.4 * rng.uniform(0.85, 1.0))
                if full and s in (4, 12):
                    self._hit(drums, pos, "b", 0.3 * rng.uniform(0.85, 1.0))
                    self._hit(drums, pos, "h", 0.09)    # the hi-hat pedal on 2 and 4
                if full and pat["s"][s] == "x" and rng.random() < 0.8:
                    self._hit(drums, pos, "s", 0.14 * rng.uniform(0.7, 1.0))
                elif full and s % 4 == 2 and rng.random() < 0.08:
                    self._hit(drums, pos, "s", 0.08)
                if (full or hats_only) and pat["h"][s] == "x":
                    acc = 1.0 if s in (4, 12) else 0.8 if s % 4 == 0 else 0.6
                    self._hit(drums, pos, "r", 0.2 * acc * rng.uniform(0.8, 1.0))
                continue
            if full and pat["k"][s] == "x":
                self._hit(drums, pos, "k", 0.85 * rng.uniform(0.9, 1.0))
                kicks.append(pos)
            if wa and full:                             # Japanese hand drums over the kit
                if s == 0 and wa["taiko"] and plan["i"] % 2 == 0:
                    self._hit(drums, pos, "T", 0.5 * rng.uniform(0.9, 1.0))
                elif s in (12, 14) and plan["i"] % 4 == 3 and taiko_fill:     # don-don into the next phrase
                    self._hit(drums, pos, "T", (0.3 if s == 12 else 0.4) * rng.uniform(0.9, 1.0))
            if wa and (full or hats_only) and wa["w"][s] == "x" and rng.random() > 0.1:
                self._hit(drums, pos, "w", 0.11 * rng.uniform(0.75, 1.0))
            if full and pat["s"][s] == "x":
                self._hit(drums, pos, "s", 0.5 * rng.uniform(0.85, 1.0))
            elif full and s in (7, 15) and rng.random() < 0.15:
                self._hit(drums, pos, "s", 0.13)         # ghost note
            if (full or hats_only) and pat["h"][s] == "x" and rng.random() > 0.07:
                acc = 1.0 if s % 4 == 0 else 0.7
                self._hit(drums, pos, "h", 0.16 * acc * rng.uniform(0.7, 1.0))

        root = PC[st["key"][0]]
        scale = sty.get("scale") or PENTA
        scale_notes = {root + i + o for i in scale[st["key"][1]] for o in (60, 72, 84, 96)}
        # wind chimes (furin) on a breeze now and then
        if sty.get("chimes") and sec != "intro" and rng.random() < sty["chimes"]:
            hi = sorted(n for n in scale_notes if 88 <= n <= 100)
            t0 = rng.uniform(0, 2.5)
            for k in range(int(rng.integers(2, 5))):
                add_note(lead, (t0 + k * rng.uniform(0.12, 0.3)) * beat * SR, syn_bell, hz(int(rng.choice(hi))),
                         0.4, 0.035 * rng.uniform(0.6, 1), rng, rel=0.8)

        # the koto / shamisen ostinato that carries a Japanese song from the intro to the last bar. It climbs
        # the scale from each chord's root, so it follows the changes without leaving the scale
        if st.get("riff") and sec != "turn":
            rf = st["riff"] if part == "A" else st["riffB"]
            rsyn = SYNTHS[st["riff_inst"]]
            ladder = sorted(root + i + o for i in scale[st["key"][1]] for o in range(36, 108, 12))
            rv = (0.14 if sec == "melody" else 0.22) * (0.85 if sec == "breakdown" else 1.0)
            nxt = st["plan"][self.sb + 1] if self.sb + 1 < len(st["plan"]) else None
            phrase_end = nxt is None or (nxt["sec"], nxt["part"]) != (sec, part) or plan["i"] % 4 == 3
            ren = phrase_end and rng.random() < sty.get("gliss", 0.0)
            ai = 0
            for j, c in enumerate(halves):
                h0 = j * hl
                b = 55 + (parse_chord(c)[0] - 55) % 12
                ai = max(i for i, n in enumerate(ladder) if n <= b)
                for e, stp, ln in rf:
                    if not h0 <= e / 2 < h0 + hl or (ren and e >= 7):
                        continue
                    n, on = ladder[ai + stp], e * beat / 2 + (st["swing"] * step if e % 2 else 0) + rng.normal(0, 0.005)
                    vel, hold = rv * (1.0 if e % 4 == 0 else 0.8) * rng.uniform(0.85, 1), ln * beat / 2 * 0.95
                    lo = ladder[ai + stp - 1]
                    evs.append((int(max(0.0, on) * SR), "p", n))
                    if ln >= 2 and n - lo <= 2 and rng.random() < sty.get("bends", 0.0):   # oshide: pluck low, press up
                        add_note(riff, on * SR, partial(rsyn, bend=float(n - lo)), hz(lo), hold, vel, rng, rel=0.3)
                    else:
                        add_note(riff, on * SR, rsyn, hz(n), hold, vel, rng, rel=0.3)
            if ren:                                     # a ren: a sweep across the strings into the next phrase
                up = rng.random() < 0.65
                run = [ladder[ai + k] for k in (range(0, 7) if up else range(8, 1, -1))]
                t0 = 4 * beat - len(run) * 0.05 - 0.03
                for k, n in enumerate(run):
                    add_note(riff, (t0 + k * 0.05) * SR, rsyn, hz(n), 0.25, rv * (0.55 + 0.07 * k), rng, rel=0.35)
        # a rin (singing bowl) rings in each new section
        if st.get("wa") and plan["i"] == 0 and sec != "turn" and rng.random() < 0.7:
            add_note(riff, 0, SYNTHS["rin"], hz(root + 72 if root + 72 < 80 else root + 60), 3.0, 0.07, rng, rel=1.5)

        # lead melody in the melody sections. Each part (A, B) gets its own rhythm and its notes are kept, so
        # when a part comes back its tune comes back too (with the odd note changed)
        melody = sec == "melody"
        if melody:
            motif = self.motifs.get(part)
            if motif is None:
                motif = [(e, int(rng.integers(1, 4))) for e in range(8) if rng.random() < st["lead_p"]] or [(0, 3)]
                self.motifs[part] = motif
            cand = set(scale_notes)
            if not sty.get("strict"):                    # strict: stay on the scale (the Japanese in / yo scales)
                cand |= {n + o for _, cn in voiced for n in cn for o in (12, 24)}
            cand = np.array(sorted(c for c in cand if 67 <= c <= 88))
            notes = self.phrase_notes.get((part, plan["i"]))
            if notes is None or len(notes) != len(motif):
                notes = []
                for _ in motif:
                    w = np.exp(-np.abs(cand - self.prev_note) / 2.5)
                    self.prev_note = int(rng.choice(cand, p=w / w.sum()))
                    notes.append(self.prev_note)
                bop = sty.get("bebop", 0.0)              # bebop: offbeat notes slide in a half step from the next
                for j in range(len(notes) - 1):
                    if motif[j][0] % 2 and rng.random() < bop:
                        notes[j] = notes[j + 1] + int(rng.choice([-1, 1]))
                self.phrase_notes[(part, plan["i"])] = notes
            elif rng.random() < 0.2:
                j = int(rng.integers(len(notes)))
                notes[j] = int(cand[np.abs(cand - notes[j]).argsort()[int(rng.integers(1, 3))]])
            self.prev_note = notes[-1]
            lname = st["lead"]
            lsyn = SYNTHS[lname]
            grace = sty.get("grace", 0.0) if lname in ("koto", "shami") else 0.0
            bends = sty.get("bends", 0.0)
            for j, ((e, d), note) in enumerate(zip(motif, notes)):
                on = e * beat / 2 + (st["swing"] * step if e % 2 else 0) + rng.normal(0, 0.006)
                if lname == "shaku":                     # long breaths: legato tones, meri bends, falling ends
                    last = j == len(motif) - 1
                    hold = max(d, min((motif[j + 1][0] if not last else 9) - e, 5)) * beat / 2
                    syn = partial(syn_shaku, bend=1.0 if rng.random() < bends * 1.5 else 0.0,
                                  fall=2.0 if last and rng.random() < 0.5 else 0.0, hold=hold * 0.75,
                                  air=2.2 if rng.random() < 0.25 else 1.0)
                    add_note(lead, on * SR, syn, hz(note), hold, 0.2 * rng.uniform(0.8, 1), rng, rel=0.25)
                    continue
                above = cand[cand > note]
                if d >= 2 and len(above) and rng.random() < sty.get("trill", 0.0):   # a mordent or a short trill
                    up, k = int(above[0]), 0
                    for k in range(int(rng.choice([2, 2, 4]))):
                        add_note(lead, (on + k * 0.065) * SR, lsyn, hz(up if k % 2 == 0 else note), 0.06, 0.16, rng, rel=0.03)
                    add_note(lead, (on + (k + 1) * 0.065) * SR, lsyn, hz(note), max(0.1, d * beat / 2 * 0.9 - 0.2),
                             0.2 * rng.uniform(0.8, 1), rng, rel=0.3)
                    continue
                below = cand[cand < note]
                if d >= 2 and len(below) and note - below[-1] <= 2 and rng.random() < bends:   # oshide
                    add_note(lead, on * SR, partial(lsyn, bend=float(note - below[-1])), hz(int(below[-1])),
                             d * beat / 2 * 0.9, 0.2 * rng.uniform(0.8, 1), rng, rel=0.3)
                    continue
                if grace and rng.random() < grace:       # a koto flick: the string pressed up from a step below
                    below = cand[cand < note]
                    gn = int(below[-1]) if len(below) else note - 2
                    add_note(lead, max(0.0, on - 0.07) * SR, lsyn, hz(gn), 0.05, 0.12, rng, rel=0.04)
                add_note(lead, on * SR, lsyn, hz(note), d * beat / 2 * 0.9, 0.2 * rng.uniform(0.8, 1), rng, rel=0.3)
            for (e, d), note in zip(motif, notes):
                evs.append((int(max(0.0, e * beat / 2 + (st["swing"] * step if e % 2 else 0)) * SR), "n", note))

        # sidechain pump from the kick
        duck = np.ones(N)
        for p in kicks:
            L = min(N - p, int(0.3 * SR))
            duck[p:p + L] = np.minimum(duck[p:p + L], 1 - 0.32 * np.exp(-np.arange(L) / (0.11 * SR)))

        tonal = keys + lead * 0.9 + riff * 0.8
        wetL = fftconv(tonal + lead * 0.6, self.ir[0]) * 0.32
        wetR = fftconv(tonal + lead * 0.6, self.ir[1]) * 0.32
        L = (keys + lead * 1.0 + riff * 0.75 + bass + wetL) * duck     # the riff sits right, the lead left
        R = (keys + lead * 0.7 + riff * 1.0 + bass + wetR) * duck
        L, R = self._wow(L, st["wow"]), self._wow(R, st["wow"])
        out = np.empty((N, 2))
        out[:, 0] = lowpass(L + drums, st["cutoff"])
        out[:, 1] = lowpass(R + drums, st["cutoff"])

        out[:TAIL] += self.tail
        chunk, self.tail = out[:bar_n], out[bar_n:].copy()
        music = (np.tanh(chunk * 0.6) * 0.95).astype(np.float32)   # gentle tape saturation
        layers, strikes = self._ambience(bar_n)
        self.gpos += bar_n
        self.bar += 1
        sb = self.sb
        self.sb += 1
        if self.sb >= len(st["plan"]):                 # the song's over: compose the next one
            self._start_song(st["song"] + 1)
        evs.sort(key=lambda ev: ev[0])
        return music, layers, dict(chord=cname, bar=sb, strikes=strikes, section=sec, step=plan["step"],
                                   prog=prog, part=part, st=st, evs=evs)

    def _hit(self, buf, pos, name, vel):
        self.evs.append((pos, name, vel))
        smp = self.kit[name]
        n = min(len(smp), len(buf) - pos)
        if n > 0:
            buf[pos:pos + n] += smp[:n] * vel

    def _wow(self, x, depth):
        """Tape wow & flutter as a time-varying delay (continuous across bars)."""
        i = np.arange(len(x), dtype=float)
        tg = (self.gpos + i) / SR
        d = depth * SR * (1 + np.sin(TAU * 0.33 * tg)) + 0.00012 * SR * (1 + np.sin(TAU * 5.7 * tg))
        return np.interp(i - d, i, x, left=0.0)

    def _ambience(self, n):
        rng = self.rng
        K = len(self.rain_bp)
        tg = (self.gpos + np.arange(n)) / SR
        swell = 0.8 + 0.2 * np.sin(TAU * 0.07 * tg)
        rain = np.empty((n, 2))
        for c in range(2):
            w = rng.standard_normal(n + K - 1)
            drops = np.zeros(n)
            k = rng.poisson(28 * n / SR)
            np.add.at(drops, rng.integers(0, n, k), rng.uniform(0.05, 0.4, k))
            rain[:, c] = np.convolve(w, self.rain_bp, "valid") * swell * 0.9 + np.convolve(drops, self.ping)[:n]
        rain *= 0.09

        imp = np.zeros(n)
        k = rng.poisson(9 * n / SR)
        np.add.at(imp, rng.integers(0, n, k), rng.uniform(0, 1, k) ** 4 * rng.choice([-1, 1], k) * 0.5)
        k = rng.poisson(0.35 * n / SR)
        np.add.at(imp, rng.integers(0, n, k), rng.uniform(0.4, 0.8, k) * rng.choice([-1, 1], k))
        crk = np.convolve(imp, self.crk_k)[:n] * 0.35
        hiss_k = fir(6000, 63)
        crackle = np.empty((n, 2))
        for c in range(2):
            hiss = np.convolve(rng.standard_normal(n + 62), hiss_k, "valid") * 0.006
            crackle[:, c] = crk * (0.8 + 0.2 * c) + hiss
        tga = (self.apos + np.arange(n)) / SR
        layers = dict(rain=rain, vinyl=crackle, fire=self._amb_fire(n, tga),
                      waves=self._amb_waves(n, tga), cafe=self._amb_cafe(n))
        layers["storm"], strikes = self._amb_storm(n)
        self.apos += n
        return {k: v.astype(np.float32) for k, v in layers.items()}, strikes

    # ── extra ambience layers (always rendered so toggles respond instantly) ──
    def _noise(self, n, kern):
        return np.convolve(self.rng.standard_normal(n + len(kern) - 1), kern, "valid")

    def _ev_buf(self, name, n, spill):
        """Work buffer that already holds sound spilling over from earlier bars."""
        buf = np.zeros((n + spill, 2))
        prev = self.pend.get(name)
        if prev is not None:
            m = min(len(prev), len(buf))
            buf[:m] += prev[:m]
        return buf

    def _ev_keep(self, name, buf, n):
        self.pend[name] = buf[n:].copy()
        return buf[:n]

    def _amb_fire(self, n, tg):
        rng = self.rng
        flick = 0.7 + 0.3 * np.sin(TAU * 0.13 * tg) * np.sin(TAU * 0.31 * tg + 1)
        roar = self._noise(n, fir(260, 255)) * flick * 1.4
        hiss = self._noise(n, self.bp_fire) * 0.16 * (0.6 + 0.4 * np.sin(TAU * 1.7 * tg))
        imp = np.zeros(n)
        k = rng.poisson(16 * n / SR)
        np.add.at(imp, rng.integers(0, n, k), rng.uniform(0, 1, k) ** 2 * rng.choice([-1, 1], k))
        k = rng.poisson(0.7 * n / SR)                       # the occasional loud snap
        np.add.at(imp, rng.integers(0, n, k), rng.uniform(1.5, 3.0, k) * rng.choice([-1, 1], k))
        crack = np.convolve(imp, self.fire_k)[:n] * 0.45
        out = np.empty((n, 2))
        out[:, 0] = roar + hiss + crack
        out[:, 1] = (roar + hiss) * 0.8 + crack * 0.6       # hearth sits a little to the left
        return out * 0.16

    def _amb_waves(self, n, tg):
        out = np.empty((n, 2))
        for c in range(2):
            t2 = tg + c * 1.7                                # waves arrive a bit later on the right
            s1 = 0.5 * (1 - np.cos(TAU * t2 / 9.0))
            s2 = 0.5 * (1 - np.cos(TAU * t2 / 13.7 + 1))
            env = 0.12 + 0.88 * (0.6 * s1 ** 2 + 0.4 * s2 ** 2)
            low = self._noise(n, fir(480, 255))
            wash = self._noise(n, self.bp_wash)
            out[:, c] = low * env * 1.5 + wash * env ** 3 * 0.55
        return out * 0.22

    def _amb_cafe(self, n):
        rng, a0 = self.rng, self.apos
        imp_a, imp_b, breath = np.zeros(n), np.zeros(n), np.zeros(n)
        for v in self.voices:
            env, vow = np.zeros(n), np.zeros(n)
            p = 0
            while p < n:
                if v["end"] <= a0 + p:                       # start / stop talking
                    v["talk"] = not v["talk"]
                    v["end"] = a0 + p + int(SR * (rng.uniform(1.2, 4.5) if v["talk"] else rng.uniform(0.6, 3.5)))
                seg_end = min(n, v["end"] - a0)
                q = p
                while v["talk"] and q < seg_end:            # syllables
                    L = int(SR * rng.uniform(0.09, 0.22))
                    e = min(n, q + L)
                    env[q:e] = np.maximum(env[q:e], rng.uniform(0.3, 1.0) * np.hanning(L)[: e - q])
                    vow[q:e] = rng.random()
                    q += int(L * rng.uniform(0.9, 1.4))
                p = seg_end
            if not env.any():
                continue
            f0 = v["f0"] * (1 + 0.1 * np.sin(TAU * v["into"] * (a0 + np.arange(n)) / SR + v["ph"]))
            idx = np.nonzero(np.diff(np.floor(np.cumsum(f0) / SR)))[0]
            imp_a[idx] += env[idx] * vow[idx] * v["gain"]
            imp_b[idx] += env[idx] * (1 - vow[idx]) * v["gain"]
            breath += env * v["gain"]
        voices = np.convolve(imp_a, self.vowel_a, "same") + np.convolve(imp_b, self.vowel_b, "same")
        voices += self._noise(n, self.vowel_a) * breath * 0.05
        buf = self._ev_buf("cafe", n, len(self.room_ir) + 2 * SR)
        k = rng.poisson(0.35 * n / SR)                       # cups and spoons
        for pos in rng.integers(0, n, k):
            t = _t(int(0.35 * SR))
            f = rng.uniform(2400, 3600)
            clink = (np.sin(TAU * f * t) + 0.5 * np.sin(TAU * f * 1.51 * t)) * np.exp(-t / 0.07) * rng.uniform(0.03, 0.08)
            buf[pos:pos + len(t), rng.integers(0, 2)] += clink
        if rng.random() < n / SR / 28:                       # espresso machine steaming milk
            pos, L = rng.integers(0, n), int(SR * rng.uniform(1.5, 2.6))
            t = _t(L)
            steam = self._noise(L, self.bp_wash) * np.minimum(1, t / 0.15) * np.minimum(1, (L / SR - t) / 0.3) * 0.05
            buf[pos:pos + L, 0] += steam
            buf[pos:pos + L, 1] += steam * 0.7
        dry = lowpass(voices, 2800) * 0.5
        wet = fftconv(np.concatenate([dry, np.zeros(len(self.room_ir) - 1)]), self.room_ir)   # room tail spills over
        for c, (gw, gd) in enumerate(((0.55, 0.35), (0.5, 0.45))):
            buf[: len(wet), c] += wet * gw
            buf[:n, c] += dry * gd
        return self._ev_keep("cafe", buf, n) * 2.0

    def _amb_storm(self, n):
        rng = self.rng
        buf = self._ev_buf("storm", n, int(9 * SR))
        for c in range(2):                                   # heavy, darker downpour
            buf[:n, c] += self._noise(n, self.bp_storm) * 0.09 * (0.85 + 0.15 * np.sin(TAU * 0.05 * (self.apos + np.arange(n)) / SR + c))
        strikes = []
        while self.next_thunder < self.apos + n:
            off = max(0, int(self.next_thunder - self.apos))
            self.next_thunder += int(SR * rng.uniform(8, 22))
            near = rng.random() < 0.45
            L = int(SR * rng.uniform(4.5, 7.5))
            t = _t(L)
            x = rng.standard_normal(L)
            for _ in range(3):                               # deep rumble (smoothed noise)
                x = np.convolve(x, np.ones(220) / 220, "same")
            x /= np.abs(x).max() + 1e-9
            env = np.zeros(L)
            for _ in range(rng.integers(3, 6)):              # rolling bumps
                c0, wd = rng.uniform(0.1, 3.2), rng.uniform(0.3, 1.1)
                env += rng.uniform(0.4, 1.0) * np.exp(-((t - c0) / wd) ** 2)
            env *= np.exp(-t / 3.0) * np.minimum(1, t / 0.08)
            thunder = x * env * (0.42 if near else 0.24)
            if near:                                         # the sharp crack of a close strike
                cr = self._noise(L, self.bp_wash) * np.exp(-t / 0.07) * 0.18
                thunder += cr
            pan = rng.uniform(0.6, 1.0)
            buf[off:off + L, 0] += thunder * pan
            buf[off:off + L, 1] += thunder * (1.6 - pan)
            delay = rng.uniform(0.05, 0.25) if near else rng.uniform(0.6, 2.0)   # light before sound
            strikes.append((max(0, off - int(delay * SR)), 1.0 if near else 0.45))
        return self._ev_keep("storm", buf, n), strikes

    # ── audio callback (realtime thread) ───────────────────────────────────
    def _callback(self, outdata, frames, time_info, status):
        target = 0.0 if (self.paused or self.muted) else self.volume
        g0 = self.gain
        stp = frames / SR / 0.3
        g1 = min(target, g0 + stp) if target > g0 else max(target, g0 - stp)
        self.gain = g1
        for name in self.amb_names:                     # smooth fades for every ambience toggle / slider
            tgt = self.amb_lvl[name] if self.amb_on[name] else 0.0
            self.amb_g[name] += float(np.clip(tgt - self.amb_g[name], -0.03, 0.03))
        self.cg = self.amb_g["vinyl"]
        if g0 <= 0 and g1 <= 0:
            outdata.fill(0)
            self._mix_fx(outdata)
            self._push(outdata)
            return
        out = np.zeros((frames, 2), np.float32)
        amb = np.zeros((frames, 2), np.float32)
        filled = 0
        while filled < frames:
            if self.cur is None or self.pos >= len(self.cur[0]):
                try:
                    self.cur = self.q.get_nowait()
                except queue.Empty:
                    self.cur = None
                    break
                self.pos, self.meta = 0, self.cur[2]
                self.st = self.meta["st"]
            mus, layers, meta = self.cur
            k = min(frames - filled, len(mus) - self.pos)
            sl = slice(self.pos, self.pos + k)
            out[filled:filled + k] = mus[sl]
            a = amb[filled:filled + k]
            for name, arr in layers.items():
                g = self.amb_g[name]
                if g > 0.0:
                    a += g * arr[sl]
            if self.amb_g["storm"] > 0.05:                   # lightning, in sync with the audio
                for off, power in meta["strikes"]:
                    if self.pos <= off < self.pos + k:
                        self.flash_at, self.flash_power = time.monotonic(), power
            self.pos += k
            filled += k
        out = self._analog(out) + amb
        self.played += filled
        out *= np.linspace(g0, g1, frames, dtype=np.float32)[:, None]
        self._mix_fx(out)
        outdata[:] = out
        self._push(out)

    def _mix_fx(self, out):
        while not self.fx_in.empty():
            self.fx.append(self.fx_in.get_nowait())
        if not self.fx:
            return
        keep = []
        for item in self.fx:
            buf, pos = item
            n = min(len(out), len(buf) - pos)
            out[:n] += buf[pos:pos + n] * max(0.25, self.volume)
            item[1] = pos + n
            if item[1] < len(buf):
                keep.append(item)
        self.fx = keep

    def _analog(self, x):
        """Record-player colour on the music, blended in by the vinyl toggle:
        rolled-off highs, narrower stereo, a touch of saturation and a 33⅓-rpm warp.
        Always runs (dry path is delayed to match) so toggling never clicks."""
        frames = len(x)
        ext = np.concatenate([self.an_state, x])
        self.an_state = ext[-(len(self.an_k) - 1):]
        d = (len(self.an_k) - 1) // 2
        dry = ext[d:d + frames]
        m = self.cg
        if m < 0.005:
            return dry
        wet = np.empty_like(dry)
        for c in range(2):
            wet[:, c] = np.convolve(ext[:, c], self.an_k, "valid")
        wet = 0.7 * wet + 0.3 * wet.mean(axis=1, keepdims=True)
        wet = np.tanh(wet * 1.3) / 1.3
        ph = TAU * 0.555 * (self.played + np.arange(frames)) / SR
        wet *= (1 - 0.03 * (0.5 + 0.5 * np.sin(ph)))[:, None]
        return (dry * (1 - m) + wet * m).astype(np.float32)

    def _push(self, block):
        n, size = len(block), len(self.ring)
        i = self.ri
        if i + n <= size:
            self.ring[i:i + n] = block
        else:
            a = size - i
            self.ring[i:] = block[:a]
            self.ring[: n - a] = block[a:]
        self.ri = (i + n) % size

    def snapshot(self, n):
        idx = (self.ri - n + np.arange(n)) % len(self.ring)
        return self.ring[idx]

    @property
    def beat_pos(self):
        cur = self.cur
        if cur is None:
            return 0.0
        return 4.0 * self.pos / max(1, len(cur[0]))
