import numpy as np


SR = 48000
TAU = 2 * np.pi
TAIL = int(2.6 * SR)          # room for reverb + note releases past the bar line

# ════════════════════════════════════════════════════════════════════════════
#  Music theory
# ════════════════════════════════════════════════════════════════════════════
PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
      "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
QUAL = {
    "": [0, 4, 7], "m": [0, 3, 7],
    "maj7": [0, 4, 7, 11], "maj9": [0, 4, 7, 11, 14], "6/9": [0, 4, 9, 14],
    "m7": [0, 3, 7, 10], "m9": [0, 3, 7, 10, 14], "m11": [0, 3, 10, 14, 17],
    "7": [0, 4, 7, 10], "9": [0, 4, 10, 14], "13": [0, 4, 10, 14, 21],
    "7b9": [0, 4, 10, 13], "sus9": [0, 5, 10, 14], "m7b5": [0, 3, 6, 10],
    "7sus4": [0, 5, 7, 10], "add9": [0, 4, 7, 14], "madd9": [0, 3, 7, 14], "m6": [0, 3, 7, 9], "dim7": [0, 3, 6, 9],
    "sus4": [0, 5, 7], "dim": [0, 3, 6],
}
PENTA = {"maj": [0, 2, 4, 7, 9], "min": [0, 3, 5, 7, 10]}


def parse_chord(name):
    root, rest = name[0], name[1:]
    if rest[:1] in ("#", "b"):
        root, rest = root + rest[0], rest[1:]
    return PC[root], QUAL[rest]


def voice_chord(name):
    """Bass note + a warm close voicing between G3 and Eb5."""
    pc, ints = parse_chord(name)
    bass = 36 + pc
    if bass > 43:
        bass -= 12
    notes = []
    for i in ints:
        n = 48 + pc + i
        while n < 55:
            n += 12
        while n > 75:
            n -= 12
        notes.append(n)
    return bass, sorted(set(notes))


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)
