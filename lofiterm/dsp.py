import math

import numpy as np

from .theory import SR, TAU


# ════════════════════════════════════════════════════════════════════════════
#  DSP helpers
# ════════════════════════════════════════════════════════════════════════════
_FIR = {}


def fir(cut, taps=127):
    key = (cut, taps)
    if key not in _FIR:
        m = np.arange(taps) - (taps - 1) / 2
        h = np.sinc(2 * cut / SR * m) * np.hamming(taps)
        _FIR[key] = h / h.sum()
    return _FIR[key]


def lowpass(x, cut, taps=127):
    return np.convolve(x, fir(cut, taps), mode="same")


def fftconv(x, ir):
    n = len(x) + len(ir) - 1
    N = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, N) * np.fft.rfft(ir, N), N)[: len(x)]


def _t(n):
    return np.arange(n) / SR


# ── instruments: (freq, t, rng) -> waveform ────────────────────────────────
def syn_rhodes(f, t, rng):
    mi = 1.4 * np.exp(-t / 0.22)
    y = np.sin(TAU * f * t + mi * np.sin(TAU * f * t))
    y += 0.2 * np.sin(TAU * 2 * f * t) * np.exp(-t / 0.35)
    return y * np.exp(-t / 2.2) * (1 + 0.1 * np.sin(TAU * 4.3 * t + rng.uniform(0, TAU)))


def syn_piano(f, t, rng):
    y = np.zeros_like(t)
    for k in range(1, 8):
        fk = f * k * math.sqrt(1 + 0.0004 * k * k)
        if fk > 9000:
            break
        y += np.sin(TAU * fk * t + rng.uniform(0, TAU)) / k ** 1.4 * np.exp(-t * (0.5 + 0.55 * k))
    return y * 0.85


def syn_pad(f, t, rng):
    y = np.zeros_like(t)
    for det in (-0.006, 0.0, 0.007):
        for k in range(1, 6):
            y += np.sin(TAU * f * (1 + det) * k * t + rng.uniform(0, TAU)) / k
    return y * np.minimum(1, t / 0.7) * 0.24 * (1 + 0.15 * np.sin(TAU * 0.3 * t))


def syn_bell(f, t, rng):
    return (np.sin(TAU * f * t) * np.exp(-t / 1.1)
            + 0.35 * np.sin(TAU * 2.76 * f * t) * np.exp(-t / 0.3)
            + 0.12 * np.sin(TAU * 5.4 * f * t) * np.exp(-t / 0.09))


def syn_pluck(f, t, rng):
    y = np.zeros_like(t)
    for k in range(1, 6):
        y += np.sin(TAU * f * k * t) / k * np.exp(-t * (1.5 + 2.5 * k))
    return y * 1.2


def syn_bass(f, t, rng):
    return (np.sin(TAU * f * t) + 0.25 * np.sin(TAU * 2 * f * t) * np.exp(-t / 0.3)) * np.exp(-t / 1.6)


def _bend(t, semi, at=0.09, dur=0.12):
    """Pitch ratio for a bend of `semi` semitones that starts `at` seconds in and takes `dur` (smoothstep)."""
    if not semi:
        return 1.0
    u = np.clip((t - at) / dur, 0, 1)
    return 2 ** (semi * u * u * (3 - 2 * u) / 12)


def syn_koto(f, t, rng, bend=0.0):
    """A plucked silk string: a nail click, a bright, quickly dulling attack, a slight pitch drop as the plucked
    string settles and a long thin ring. `bend` presses the string behind the bridge (oshide), raising the
    pitch that many semitones just after the pluck."""
    ph = TAU * np.cumsum(f * (1 + 0.012 * np.exp(-t / 0.04)) * _bend(t, bend)) / SR
    y = np.zeros_like(t)
    for k in range(1, 9):
        y += np.sin(k * ph + rng.uniform(0, 0.3)) / k ** 0.85 * np.exp(-t * (0.8 + 1.5 * k))
    m = min(len(t), int(0.006 * SR))
    click = np.zeros_like(t)
    n = rng.standard_normal(m)
    click[:m] = np.diff(n, prepend=0) * np.exp(-t[:m] / 0.0015) * 0.35       # the tsume (ivory nail) on silk
    yuri = 1 + 0.05 * np.clip((t - 0.3) / 0.3, 0, 1) * np.sin(TAU * 5.0 * t)
    return (y * yuri + click) * 0.9


def syn_shami(f, t, rng, bend=0.0):
    """Shamisen: a bachi (plectrum) slapping the skin as it strikes the string, then a twangy, buzzing tone
    (the sawari buzz of the low string on the neck) that dies away fast."""
    ph = TAU * np.cumsum(f * (1 + 0.008 * np.exp(-t / 0.025)) * _bend(t, bend, 0.06, 0.08)) / SR
    y = np.zeros_like(t)
    for k in range(1, 13):
        if f * k > 9000:
            break
        y += np.sin(k * ph + rng.uniform(0, 0.4)) / k ** 0.75 * np.exp(-t * (2.0 + 0.8 * k))
    buzz = np.tanh(3.5 * y) * np.exp(-t / 0.22) * 0.35                       # sawari
    m = min(len(t), int(0.05 * SR))
    hit = np.zeros_like(t)
    tm = t[:m]
    hit[:m] = (0.5 * np.sin(TAU * 175 * tm) * np.exp(-tm / 0.035)          # the skin under the bachi
               + 0.3 * np.diff(rng.standard_normal(m), prepend=0) * np.exp(-tm / 0.004))
    return (y * 0.55 + buzz + hit) * np.exp(-t / 0.9)


def syn_rin(f, t, rng):
    """A rin (temple singing bowl): inharmonic partials that shimmer as close pairs beat, ringing for seconds."""
    y = np.zeros_like(t)
    for r, a, d in ((1.0, 1.0, 3.2), (2.71, 0.45, 1.6), (5.12, 0.22, 0.7), (8.2, 0.1, 0.3)):
        y += a * (np.sin(TAU * f * r * t + rng.uniform(0, TAU)) + np.sin(TAU * f * r * 1.0025 * t)) * 0.5 * np.exp(-t / d)
    return y * np.minimum(1, t / 0.002)


def syn_vibes(f, t, rng):
    """Vibraphone: a sine bar with its 4th partial and the motor's tremolo."""
    y = np.sin(TAU * f * t) + 0.22 * np.sin(TAU * 4 * f * t) * np.exp(-t / 0.15) + 0.06 * np.sin(TAU * 10 * f * t) * np.exp(-t / 0.03)
    return y * np.exp(-t / 1.8) * (1 - 0.3 * (0.5 + 0.5 * np.sin(TAU * 5.5 * t + rng.uniform(0, TAU))))


def syn_shaku(f, t, rng, bend=0.0, fall=0.0, hold=1.0, air=1.0):
    """Shakuhachi: a breathy bamboo flute that scoops up into the note and blooms into a slow vibrato.
    `bend` starts the note that many semitones flat and lifts it (meri → kari), `fall` lets the pitch drop
    away after `hold` seconds as the breath runs out, `air` scales the breath (a muraiki attack when high)."""
    scoop = 1 - 0.035 * np.exp(-t / 0.09)
    vib = 1 + 0.006 * np.clip((t - 0.25) / 0.4, 0, 1) * np.sin(TAU * 5.0 * t)
    lift = _bend(t, bend, 0.08, 0.22) * 2 ** (-bend / 12) if bend else 1.0
    drop = 2 ** (-fall * np.clip((t - hold) / 0.4, 0, 1) ** 2 / 12) if fall else 1.0
    ph = TAU * np.cumsum(f * scoop * vib * lift * drop) / SR
    y = np.sin(ph) + 0.18 * np.sin(2 * ph) + 0.06 * np.sin(3 * ph)
    k = max(1, int(SR / (f * 2)))
    n = np.convolve(rng.standard_normal(len(t)), np.ones(k) / k, "same")
    breath = n * (0.6 * air * np.exp(-t / (0.08 * air)) + 0.2) * (1 + np.sin(ph))   # air riding on the tone
    env = np.minimum(1, t / 0.07) * (1 + 0.15 * np.exp(-((t - 0.05) / 0.04) ** 2))
    return (y + breath) * env * 0.7


def syn_horn(f, t, rng):
    """A soft tenor-sax-like reed: a scoop into the pitch, a bright attack that mellows, late vibrato, breath."""
    bend = 1 - 0.02 * np.exp(-t / 0.05)
    vib = 1 + 0.005 * np.clip((t - 0.2) / 0.3, 0, 1) * np.sin(TAU * 5.3 * t)
    ph = TAU * np.cumsum(f * bend * vib) / SR
    bright = 0.55 + 0.45 * np.exp(-t / 0.25)
    y = np.zeros_like(t)
    for k, a in ((1, 1.0), (2, 0.55), (3, 0.42), (4, 0.25), (5, 0.18), (6, 0.1), (7, 0.07)):
        if f * k > 7000:
            break
        y += a * bright ** (k - 1) * np.sin(k * ph)
    n = np.convolve(rng.standard_normal(len(t)), np.ones(8) / 8, "same")
    return (y * 0.6 + n * 0.08 * (0.4 + np.exp(-t / 0.06))) * np.minimum(1, t / 0.035)


def syn_harpsi(f, t, rng):
    """Harpsichord: a quill plucking two slightly detuned choirs of wire strings. Bright and nasal (the pluck
    point near the string's end), no dynamics, a quick bloom of overtones and a clean decay."""
    y = np.zeros_like(t)
    for det, a in ((1.0, 1.0), (1.0019, 0.55)):
        for k in range(1, 18):
            if f * k * det > 10000:
                break
            y += a * abs(math.sin(math.pi * k * 0.13)) / k ** 0.6 * np.sin(TAU * f * det * k * t + rng.uniform(0, TAU)) \
                * np.exp(-t * (0.7 + 0.28 * k))
    m = min(len(t), int(0.004 * SR))
    y[:m] += rng.standard_normal(m) * np.exp(-t[:m] / 0.001) * 0.6               # the quill catching the string
    return y * 0.75


def syn_lute(f, t, rng):
    """Lute: a gut string plucked by the thumb, warm and soft, ringing in a hollow pear-shaped body."""
    y = np.zeros_like(t)
    for k in range(1, 9):
        y += abs(math.sin(math.pi * k * 0.22)) / k ** 1.2 * np.sin(TAU * f * k * t + rng.uniform(0, 0.3)) * np.exp(-t * (1.0 + 1.1 * k))
    y += 0.08 * np.sin(TAU * f * 2.003 * t) * np.exp(-t / 0.5)                     # the paired course, a hair off
    return y * 1.1


def _bowed(f, t, rng, bright, att, vib, n=1, spread=0.0):
    """A bowed string section (n players, detuned by `spread`): a sawtooth-ish tone shaped by `bright`,
    a bow attack of `att` seconds, vibrato that fades in, and a little bow hiss."""
    y = np.zeros_like(t)
    for j in range(n):
        d = 1 + spread * (j - (n - 1) / 2)
        rate = 5.4 + rng.uniform(-0.4, 0.4)
        v = 1 + vib * np.clip((t - 0.18) / 0.35, 0, 1) * np.sin(TAU * rate * t + rng.uniform(0, TAU))
        ph = TAU * np.cumsum(f * d * v) / SR + rng.uniform(0, TAU)
        for k in range(1, 9 if n == 1 else 7):         # an ensemble needs fewer partials: they blur anyway
            if f * k > 6500:
                break
            y += (bright ** (k - 1)) / k * np.sin(k * ph)
    hiss = np.convolve(rng.standard_normal(len(t)), np.ones(6) / 6, "same") * 0.04
    swell = np.minimum(1, t / att) * (1 + 0.08 * np.sin(TAU * 0.4 * t))
    return (y / n + hiss) * swell


def syn_strings(f, t, rng):
    """A string section's sustained chord tone: two bows slightly apart, soft-edged, slowly swelling."""
    return _bowed(f, t, rng, 0.72, 0.3, 0.004, n=2, spread=0.005) * 0.9


def syn_violin(f, t, rng):
    """A solo violin: a brighter bow, a quicker attack and a singing vibrato."""
    return _bowed(f, t, rng, 0.84, 0.07, 0.007) * 0.55


def syn_cello(f, t, rng):
    """A cello bowing the bass line: dark, round, a gentle bow attack."""
    return _bowed(f, t, rng, 0.6, 0.06, 0.003) * 0.85


def syn_recorder(f, t, rng):
    """A wooden recorder: almost a pure tone, a breathy chiff at the start and a whisper of air."""
    vib = 1 + 0.003 * np.clip((t - 0.3) / 0.4, 0, 1) * np.sin(TAU * 4.6 * t)
    ph = TAU * np.cumsum(f * vib) / SR
    y = np.sin(ph) + 0.09 * np.sin(2 * ph) + 0.05 * np.sin(3 * ph)
    k = max(1, int(SR / (f * 3)))
    n = np.convolve(rng.standard_normal(len(t)), np.ones(k) / k, "same")
    chiff = n * np.exp(-t / 0.025) * 0.9 + n * 0.06
    return (y + chiff) * np.minimum(1, t / 0.03) * 0.7


SYNTHS = dict(rhodes=syn_rhodes, piano=syn_piano, pad=syn_pad, bell=syn_bell, pluck=syn_pluck, koto=syn_koto,
              vibes=syn_vibes, shaku=syn_shaku, horn=syn_horn, shami=syn_shami, rin=syn_rin,
              harpsi=syn_harpsi, lute=syn_lute, strings=syn_strings, violin=syn_violin, cello=syn_cello,
              recorder=syn_recorder)


def add_note(buf, start, syn, f, hold_s, vel, rng, rel=0.35):
    start = max(0, int(start))
    if start >= len(buf):
        return
    n = min(len(buf) - start, int((hold_s + rel * 5) * SR))
    t = _t(n)
    y = syn(f, t, rng) * np.minimum(1, t / 0.005) * vel
    hold = int(hold_s * SR)
    if hold < n:
        y[hold:] *= np.exp(-np.arange(n - hold) / SR / rel)
    buf[start:start + n] += y


def make_kit(rng):
    t = _t(int(0.5 * SR))
    f = 46 + 95 * np.exp(-t / 0.035)
    kick = np.sin(TAU * np.cumsum(f) / SR) * np.exp(-t / 0.3) * np.minimum(1, t / 0.001)
    t = _t(int(0.35 * SR))
    noise = lowpass(rng.standard_normal(len(t)), 4500)
    snare = 0.5 * np.sin(TAU * 190 * t) * np.exp(-t / 0.05) + noise / np.abs(noise).max() * np.exp(-t / 0.13)
    t = _t(int(0.09 * SR))
    n = rng.standard_normal(len(t))
    hat = (n - lowpass(n, 6500)) * np.exp(-t / 0.022)
    t = _t(int(1.4 * SR))                                   # ride cymbal: inharmonic partials, a ping, a wash
    ride = sum(a * np.sin(TAU * f * t + rng.uniform(0, TAU))
               for f, a in ((2150, 0.5), (2870, 0.4), (3460, 0.35), (4230, 0.3), (5050, 0.25), (6100, 0.2)))
    ride = ride * np.exp(-t / 0.55) * (1 + 1.5 * np.exp(-t / 0.02))
    n = rng.standard_normal(len(t))
    ride += 0.6 * (n - lowpass(n, 3000)) * np.exp(-t / 0.18)
    t = _t(int(0.4 * SR))                                   # a brush swept across the snare
    brush = lowpass(rng.standard_normal(len(t)), 5000) * np.minimum(1, t / 0.06) * np.exp(-t / 0.12)
    t = _t(int(0.9 * SR))                                   # taiko: a big hide drum, pitch sagging, a skin slap
    f = 68 + 55 * np.exp(-t / 0.06)
    taiko = np.sin(TAU * np.cumsum(f) / SR) * np.exp(-t / 0.32) + 0.3 * np.sin(TAU * np.cumsum(f * 2.3) / SR) * np.exp(-t / 0.09)
    taiko += 0.5 * lowpass(rng.standard_normal(len(t)), 900) * np.exp(-t / 0.025)
    t = _t(int(0.14 * SR))                                  # hyoshigi / woodblock: a dry hardwood clack
    wood = (np.sin(TAU * 1180 * t) * np.exp(-t / 0.02) + 0.6 * np.sin(TAU * 2630 * t + 1) * np.exp(-t / 0.011)
            + 0.4 * rng.standard_normal(len(t)) * np.exp(-t / 0.0015))
    norm = lambda x: x / np.abs(x).max()
    return dict(k=norm(kick), s=norm(snare), h=norm(hat), r=norm(ride), b=norm(brush), T=norm(taiko), w=norm(wood))
