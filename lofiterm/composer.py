"""The composer: writes an endless run of songs for each station, in that station's style.

A song is a key, an A and a B chord progression (one chord per bar), a form that says what every bar does,
a drum pattern, a chord rhythm, instruments, mix settings and a title. Everything is drawn from the world
seed, so a world always plays the same music (LOFI_SEED reproduces it) and `x` re-rolls it. The engine
plays a song's bars in order and asks for the next song when it runs out; only humanisation (timing,
velocity, which melody notes) is left to chance at play time.
"""
from .stations import STATIONS
from .world import wrng

NOTE = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SHARP_KEYS = {False: {7, 2, 9, 4, 11, 6}, True: {4, 11, 6, 1, 8}}       # G D A E B F# major; E B F# C# G# minor


def names(root, minor):
    """Note names that read right in this key: sharps in sharp keys, flats otherwise."""
    return SHARP if root in SHARP_KEYS[minor] else NOTE
MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]

# chord quality per scale degree: (plain, lush) choices. "4m" = the minor iv borrowed into a major key,
# "5D" = the dominant V7 in a minor key (harmonic minor's leading tone)
Q_MAJ = {1: (["maj7"], ["maj9", "6/9"]), 2: (["m7"], ["m9", "m11"]), 3: (["m7"], ["m9"]), 4: (["maj7"], ["maj9"]),
         5: (["7"], ["9", "13", "sus9"]), 6: (["m7"], ["m9"]), 7: (["m7b5"], ["m7b5"]), "4m": (["m7"], ["m9"])}
Q_MIN = {1: (["m7"], ["m9", "m11"]), 2: (["m7b5"], ["m7b5"]), 3: (["maj7"], ["maj9"]), 4: (["m7"], ["m9", "m11"]),
         5: (["m7"], ["m9"]), 6: (["maj7"], ["maj9"]), 7: (["7"], ["9", "13"]), "5D": (["7b9"], ["7b9", "13"])}
ROOT = {"4m": 5, "5D": 7}

# progressions that sound like lofi, as scale degrees…
TEMPLATES = {
    False: [[2, 5, 1, 6], [4, 3, 2, 1], [1, 6, 2, 5], [4, 5, 3, 6], [2, 5, 1, 1], [1, 3, 4, "4m"], [6, 4, 1, 5],
            [4, 4, 3, 6], [1, 4, 3, 6], [2, 3, 4, 5], [4, 3, 6, 5], [1, "4m", 1, "4m"], [4, "4m", 3, 6], [6, 2, 5, 1]],
    True: [[1, 4, 7, 3], [1, 6, 3, 7], [4, 5, 1, 1], [1, 4, "5D", 1], [6, 7, 1, 1], [2, "5D", 1, 6], [1, 7, 6, "5D"],
           [4, 7, 3, 6], [1, 4, 1, 6], [6, 4, 1, "5D"], [4, 7, 1, 1]],
}
# …and where a chord likes to go next, for progressions grown by a random walk
WALK = {
    False: {1: {4: 3, 6: 3, 2: 2, 3: 1, 5: 1}, 2: {5: 5, 4: 1, 3: 1}, 3: {6: 3, 4: 3, 2: 1},
            4: {5: 3, 1: 2, 2: 2, 3: 2, "4m": 1}, 5: {1: 5, 6: 2, 3: 1}, 6: {2: 3, 4: 3, 5: 1, 3: 1}, 7: {3: 3, 1: 1},
            "4m": {1: 4, 3: 2}},
    True: {1: {4: 3, 6: 3, 7: 2, 3: 1, 2: 1}, 2: {"5D": 5, 5: 2}, 3: {6: 3, 4: 3, 7: 1},
           4: {7: 3, "5D": 2, 1: 2, 5: 1}, 5: {1: 4, 6: 2}, "5D": {1: 5, 6: 2}, 6: {7: 3, 4: 2, 3: 2, "5D": 1},
           7: {3: 4, 1: 3, 6: 1}},
}

# chord rhythms: (beat, length in beats) per bar
COMP = {
    "sustain": lambda r: [(0, 4)],
    "halves": lambda r: r.choice([[(0, 2.5), (2.5, 1.5)], [(0, 2), (2, 2)], [(0, 3), (3, 1)]]),
    "charleston": lambda r: r.choice([[(0, 1.5), (1.5, 1.25), (3, 1)], [(0, 1.5), (1.5, 2.5)],
                                      [(0, 0.75), (1.5, 1), (2.75, 1.25)]]),
    "stabs": lambda r: _onsets(r, [0.5, 0.75, 1.5, 2, 2.5, 2.75, 3.25, 3.5], r.randint(2, 3)),
    "busy": lambda r: _onsets(r, [0.75, 1, 1.5, 2, 2.5, 3, 3.25, 3.5], r.randint(3, 5)),
}

# where extra kicks like to land, per groove feel (16th-note step → chance)
KICKS = {"boom": {7: 0.55, 10: 0.65, 8: 0.2, 3: 0.2, 13: 0.2, 15: 0.1}, "lazy": {10: 0.7, 7: 0.2, 13: 0.25, 6: 0.15},
         "sparse": {10: 0.35, 7: 0.15}, "shuffle": {3: 0.5, 10: 0.6, 13: 0.5, 7: 0.2},
         "bounce": {6: 0.6, 10: 0.6, 13: 0.5, 3: 0.2, 8: 0.2}, "brush": {10: 0.35, 6: 0.2, 14: 0.15}}

ADJ = ["slow", "quiet", "faded", "late", "warm", "sleepy", "soft", "blue", "golden", "hazy", "little", "distant",
       "paper", "velvet", "lazy", "dusty", "half-awake", "lost", "borrowed", "tiny"]
NOUN = ["letters", "rooms", "clouds", "trains", "tapes", "streets", "lights", "rivers", "windows", "mornings",
        "static", "postcards", "orbits", "daydreams", "gardens", "footsteps", "polaroids", "echoes", "sweaters"]
WHEN = ["at 2am", "at dawn", "at dusk", "after school", "on repeat", "in june", "in november", "till morning"]


def _onsets(r, grid, n):
    on = sorted({0.0} | set(r.sample(grid, n)))
    out = []
    for j, b in enumerate(on):
        nxt = on[j + 1] if j + 1 < len(on) else 4.0
        out.append((b, max(0.4, (nxt - b) * r.uniform(0.6, 0.9))))
    return out


def station_bpm(i):
    """This station's tempo in the current world (steady from song to song)."""
    return wrng("bpm", i).randint(*STATIONS[i]["style"]["bpm"])


def song_key(i, k):
    """Song k's key: songs drift to related keys (up a fourth or fifth, the relative major/minor, a step)."""
    sty = STATIONS[i]["style"]
    r = wrng("key", i)
    root, minor = r.randrange(12), r.random() < sty["minor"]
    for j in range(1, k + 1):
        r = wrng("keystep", i, j)
        move = r.choice(["4th", "5th", "rel", "rel", "step", "stay"])
        if move == "rel":
            root, minor = (root + 3) % 12 if minor else (root + 9) % 12, not minor
        else:
            root = (root + {"4th": 5, "5th": 7, "step": r.choice((2, -2)), "stay": 0}[move]) % 12
        if r.random() < 0.25:                                     # now and then just pick the mode afresh
            minor = r.random() < sty["minor"]
    return root, minor


def _chord(root, deg, minor, lush, r):
    q = (Q_MIN if minor else Q_MAJ)[deg]
    pc = root + (ROOT[deg] if deg in ROOT else (MINOR if minor else MAJOR)[deg - 1])
    return names(root, minor)[pc % 12] + r.choice(q[1] if r.random() < lush else q[0])


def _degrees(r, minor):
    if r.random() < 0.55:
        return list(r.choice(TEMPLATES[minor]))
    walk = WALK[minor]
    for _ in range(40):
        seq = [r.choice([1, 1, 4, 6, 2] if not minor else [1, 1, 4, 6])]
        while len(seq) < 4:
            opts = walk[seq[-1]]
            seq.append(r.choices(list(opts), weights=list(opts.values()))[0])
        if seq[0] in walk[seq[-1]] and len(set(seq)) >= 3:      # it has to loop back round
            return seq
    return list(r.choice(TEMPLATES[minor]))


def pool_progression(r, root, minor, sty):
    """One of the station's own progressions (style "pool": "offset:quality" per bar, offsets in semitones
    from the tonic; "a:q|b:q" splits a bar into two chords of two beats)."""
    nm = names(root, minor)
    spec = r.choice(sty["pool"]["min" if minor else "maj"])
    return ["|".join(nm[(root + int(o)) % 12] + q for o, q in (p.split(":") for p in bar.split("|"))) for bar in spec]


def progression(r, root, minor, sty):
    """Four chords (one per bar) with this station's colour: richer extensions when lush, and when jazzy a
    secondary dominant leading into the next chord, or a ii–V turnaround back to the top. Stations with
    their own pool of progressions (Japanese pop changes, jazz standards' changes) mostly draw from it."""
    if sty.get("pool") and r.random() < sty.get("pool_p", 0.0):
        return pool_progression(r, root, minor, sty)
    degs = _degrees(r, minor)
    chords = [_chord(root, d, minor, sty["lush"], r) for d in degs]
    if r.random() < sty["jazz"] * 0.6:                          # V7 of the next chord
        j = r.randrange(1, 4)
        if degs[j] != 1:
            tgt = root + (ROOT.get(degs[(j + 1) % 4]) or (MINOR if minor else MAJOR)[degs[(j + 1) % 4] - 1])
            chords[j] = names(root, minor)[(tgt + 7) % 12] + r.choice(["7b9", "13", "9"])
    if r.random() < sty["jazz"] * 0.35 and degs[0] == 1:          # ii–V back to the tonic
        chords[2] = names(root, minor)[(root + 2) % 12] + ("m7b5" if minor else r.choice(["m9", "m7"]))
        chords[3] = names(root, minor)[(root + 7) % 12] + ("7b9" if minor else r.choice(["13", "9"]))
    return chords


def make_drums(r, feel, busy, half):
    """A one-bar groove as 16-step strings: kick, snare, hats."""
    k, s, h = ["."] * 16, ["."] * 16, ["."] * 16
    k[0] = "x"
    if half:
        s[8] = "x"
    else:
        s[4] = s[12] = "x"
    if feel == "shuffle" and r.random() < 0.6:
        s[15] = "x"
    for step, p in KICKS[feel].items():
        if s[step] != "x" and r.random() < p * (0.6 + 0.8 * busy):
            k[step] = "x"
    if feel == "brush":                                 # a jazz ride: ding, ding-a ding, ding-a
        for step in (0, 4, 6, 8, 12, 14):
            h[step] = "x"
        if r.random() < busy:
            h[r.choice((2, 10))] = "x"
        s = ["."] * 16                                  # no backbeat: the snare comps on swung offbeats
        for step in r.sample([2, 6, 10, 14, 7, 15], 1 + int(r.random() < busy)):
            s[step] = "x"
    elif feel == "sparse":
        for step in (2, 6, 10, 14):
            h[step] = "x" if r.random() < 0.9 else "."
    else:
        for step in range(0, 16, 2):
            h[step] = "." if feel == "lazy" and step % 8 == 6 and r.random() < 0.6 else "x"
        for step in range(1, 16, 2):
            if r.random() < busy * 0.55:
                h[step] = "x"
    return dict(k="".join(k), s="".join(s), h="".join(h), feel=feel)


# broken chords, as indices into the chord's notes from the bottom (3 = the top note), one per eighth:
# an Alberti bass, a rising and falling arpeggio, and the even figure of a Bach prelude
ARPS = [[0, 3, 1, 3, 0, 3, 1, 3], [0, 1, 2, 3, 2, 1, 0, 1], [0, 1, 2, 3, 0, 1, 2, 3], [0, 2, 3, 2, 1, 2, 3, 2],
        [0, 2, 1, 3, 0, 2, 1, 3]]

RIFF_RHYTHMS = [[0, 2, 3, 4, 6], [0, 1, 2, 4, 5, 6], [0, 2, 4, 5, 6, 7], [0, 3, 4, 6], [0, 1, 3, 4, 6, 7],
                [0, 2, 3, 5, 6], [0, 3, 6, 7], [0, 1, 2, 3, 4, 6]]


def make_riff(r):
    """A one-bar koto / shamisen ostinato, the hook of a Japanese lofi track: (eighth, scale step, length in
    eighths). Steps count notes of the station's pentatonic scale up from the chord's root (5 steps = an
    octave), so the riff follows the changes without ever leaving the scale."""
    on = r.choice(RIFF_RHYTHMS)
    steps, cur = [], 0
    for j in range(len(on)):
        if j:
            cur = max(-2, min(7, cur + r.choice([1, 2, 3, 5, -1, -2, -3, 4])))
        steps.append(cur)
    if r.random() < 0.5:
        steps[-1] = r.choice([0, 3, 5])                   # land back home: the root, the fifth or the octave
    return [(e, st, (on[j + 1] if j + 1 < len(on) else 8) - e) for j, (e, st) in enumerate(zip(on, steps))]


def make_wa(r):
    """Japanese hand percussion for one song: woodblock steps, and whether a taiko marks the bars."""
    hits = set(r.sample([3, 6, 10, 14, 15, 7], r.randint(1, 3)))
    return dict(w="".join("x" if s in hits else "." for s in range(16)), taiko=r.random() < 0.7)


def make_form(r, first):
    """What each bar does: intro / turn (hats only) / groove / melody / breakdown, over A and B phrases."""
    plan = [("intro", "A")] * 2 if first else [("turn", "A")]
    form = r.choice([["A", "A", "B", "A"], ["A", "B", "A", "B"], ["A", "A", "B", "B", "A"], ["A", "B", "A"],
                     ["A", "B", "B", "A"]])
    plen = r.choice([4, 8, 8])
    if plen * len(form) < 24:                           # keep songs from ending before they've settled in
        plen = 8
    for j, part in enumerate(form):
        if j == len(form) - 1 and j > 1 and r.random() < 0.6:
            plan += [("breakdown", part)] * 2
        mel = j > 0 and r.random() < (0.7 if part == "B" else 0.5)
        plan += [("melody" if mel else "groove", part)] * plen
    if not any(sec == "melody" for sec, _ in plan):
        plan[-plen:] = [("melody", plan[-1][1])] * plen
    out, run = [], 0                                    # number the bars of each run: which chord, which melody bar
    for j, (sec, part) in enumerate(plan):
        run = run + 1 if j and plan[j - 1] == (sec, part) else 0
        out.append(dict(sec=sec, part=part, i=run, step=run % 4))
    return out


def title(r, words):
    w = r.choice(words)
    return r.choice([
        f"{r.choice(ADJ)} {r.choice(NOUN)}", f"{w} & {r.choice(NOUN)}", f"{r.choice(NOUN)} {r.choice(WHEN)}",
        f"{r.choice(ADJ)} {w}", f"before the {w}", f"{w}, again", f"{r.choice(NOUN)} for {w}", f"{w} {r.choice(WHEN)}",
    ])


def compose_song(i, k):
    """Song k of station i in the current world."""
    sty = STATIONS[i]["style"]
    r = wrng("song", i, k)
    root, minor = song_key(i, k)
    A = progression(r, root, minor, sty)
    B = A
    for _ in range(12):
        B = progression(r, root, minor, sty)
        if B != A:
            break
    keys = r.choice(sty["keys"])
    riff = sty.get("riff", 0.0) and r.random() < sty["riff"]
    rinst = r.choice(sty["riff_inst"]) if riff else None
    lead = r.choice([x for x in sty["lead"] if x not in (keys, rinst)] or sty["lead"])
    arp = sty.get("arp", 0.0) and r.random() < sty["arp"]
    extra = dict(arp=r.choice(ARPS) if arp else None, arp_inst=r.choice(sty["arp_inst"]) if arp else None,
                 riff=make_riff(r) if riff else None, riffB=make_riff(r) if riff else None, riff_inst=rinst,
                 wa=make_wa(r) if sty.get("wa") else None)
    dr = sty["drums"]
    plan = make_form(r, k == 0)
    return dict(title=title(r, sty["words"]), song=k, key=(names(root, minor)[root], "min" if minor else "maj"), bpm=station_bpm(i),
                prog=A, progB=B, keys=keys, lead=lead, comp=COMP[r.choice(sty["comp"])](r),
                drums=make_drums(r, dr["feel"], dr["busy"], r.random() < dr["half"]),
                swing=r.uniform(*sty["swing"]), cutoff=int(r.uniform(*sty["cutoff"])), wow=r.uniform(*sty["wow"]),
                lead_p=r.uniform(*sty["lead_p"]), walk=r.random() < sty.get("walk", 0.0), plan=plan, **extra)
