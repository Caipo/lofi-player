"""The stations. Each one has a fixed identity (name, mood, weather, ambience preset, and through its index its
look in every scene) and a musical *style*: ranges and pools the composer (composer.py) draws whole songs from.

style fields
  bpm        tempo range; one tempo is picked per station per world, so beat-synced visuals stay steady
  minor      chance a song is in a minor key
  keys/lead  instrument pools for the chords and the melody (see dsp.SYNTHS)
  comp       chord-rhythm styles to pick from (see composer.COMP)
  drums      groove feel (see composer.make_drums), how busy it gets 0…1, and the chance of a half-time feel
  swing, cutoff, wow, lead_p   ranges for swing, the tape low-pass, wow & flutter, melody density
  lush       0…1, how far chords stretch (7ths → 9ths/11ths/13ths)
  jazz       0…1, how often progressions get ii–V turnarounds, secondary dominants and borrowed chords
  words      flavour words for the song titles
optional
  scale      melody notes per mode {"maj": [...], "min": [...]} instead of the plain pentatonic
  strict     the melody stays on `scale` (no chord tones off it)
  pool       the station's own progressions per mode, "offset:quality" per bar ("a:q|b:q" = two chords in
             a bar); pool_p is how often a song uses one instead of the generated kind
  walk       chance a song gets a walking bass line (a note on every beat) instead of the held root
  rootless   over a walking bass, the keys leave the root out (jazz piano voicings)
  bebop      chance an offbeat melody note becomes a chromatic approach into the next one
  grace      chance a koto note gets a grace note flicked up from the scale step below
  chimes     chance per bar of a few wind-chime (furin) notes
  riff       chance a song is built on a koto / shamisen ostinato (riff_inst picks the instrument) that
             plays the whole song, intro included; gliss is the chance of a ren (a glissando across the
             strings) sweeping into a new phrase; bends is the chance a long note is pressed up (oshide)
  wa         Japanese hand percussion: taiko on the bar, a woodblock, a rin bowl at each new section
  shaku      a shakuhachi lead plays legato long tones with meri bends, breath accents and falling ends
  arp        chance a song's chords are broken into arpeggios (Alberti bass, preludes) on arp_inst, with
             the keys holding the chord softly underneath
  trill      chance a long melody note is ornamented (a mordent or a short trill from the note above)
  bass       the bass instrument (default the round synth bass; "cello" bows it)
  drums feel "brush" plays a jazz kit: ride cymbal, brushes, feathered kick, hi-hat on 2 and 4
station flags
  theme      "japan" / "jazz" / "english": scenes add that station's own props (torii, lanterns; records,
             saxophones, ♪; castles, spires, thatched cottages, gas lamps, candles)
"""

STATIONS = [
    dict(name="Rainy Window", mood="rain taps on the glass. warm rhodes, slow tea.", rain=True, weather="rain",
         style=dict(bpm=(68, 76), minor=0.3, keys=["rhodes", "rhodes", "piano"], lead=["bell", "rhodes"],
                    comp=["sustain", "halves"], drums=dict(feel="boom", busy=0.35, half=0.15), swing=(0.18, 0.25),
                    cutoff=(3100, 3700), wow=(0.0009, 0.0013), lead_p=(0.3, 0.4), lush=0.7, jazz=0.4,
                    words=["rain", "window", "tea", "glass", "puddles", "umbrella", "grey"])),
    dict(name="Midnight Study", mood="desk lamp on, 2am, pages turning.", rain=False, weather="rain",
         style=dict(bpm=(76, 84), minor=0.4, keys=["piano", "piano", "rhodes"], lead=["rhodes", "piano"],
                    comp=["halves", "charleston", "sustain"], drums=dict(feel="boom", busy=0.4, half=0.1),
                    swing=(0.15, 0.22), cutoff=(2800, 3300), wow=(0.0008, 0.0011), lead_p=(0.25, 0.35),
                    lush=0.8, jazz=0.8, words=["notes", "lamp", "2am", "pages", "ink", "margins", "coffee"])),
    dict(name="Sunday Coffee", mood="sunlight, oat milk, nowhere to be.", rain=False, weather="rain",
         style=dict(bpm=(84, 92), minor=0.15, keys=["rhodes", "rhodes", "piano"], lead=["pluck", "bell"],
                    comp=["charleston", "stabs", "halves"], drums=dict(feel="shuffle", busy=0.6, half=0.0),
                    swing=(0.12, 0.18), cutoff=(4000, 4600), wow=(0.0006, 0.0008), lead_p=(0.4, 0.5),
                    lush=0.6, jazz=0.5, words=["sunday", "oat milk", "croissant", "sunlight", "slow", "brunch"])),
    dict(name="Snowfall Tape", mood="quiet streets under snow. warped cassette.", rain=False, weather="snow",
         style=dict(bpm=(60, 68), minor=0.35, keys=["pad", "pad", "rhodes"], lead=["bell", "bell", "pluck"],
                    comp=["sustain", "sustain", "halves"], drums=dict(feel="sparse", busy=0.2, half=0.5),
                    swing=(0.08, 0.15), cutoff=(2300, 2900), wow=(0.0015, 0.0021), lead_p=(0.25, 0.35),
                    lush=0.9, jazz=0.2, words=["snow", "frost", "cassette", "scarf", "streetlight", "hush"])),
    dict(name="Neon Bus Ride", mood="city lights smear across the window.", rain=True, weather="rain",
         style=dict(bpm=(88, 96), minor=0.55, keys=["rhodes", "rhodes", "piano"], lead=["pluck", "rhodes"],
                    comp=["stabs", "stabs", "charleston"], drums=dict(feel="lazy", busy=0.55, half=0.1),
                    swing=(0.16, 0.23), cutoff=(3600, 4200), wow=(0.0007, 0.0009), lead_p=(0.35, 0.45),
                    lush=0.7, jazz=0.6, words=["neon", "night bus", "city", "headlights", "last stop", "transit"])),
    dict(name="Moonlit Garden", mood="crickets, paper lanterns, a slow walk home.", rain=False, weather="rain",
         style=dict(bpm=(66, 74), minor=0.7, keys=["piano", "piano", "pad"], lead=["bell", "bell", "pluck"],
                    comp=["sustain", "halves"], drums=dict(feel="boom", busy=0.3, half=0.3), swing=(0.2, 0.27),
                    cutoff=(2900, 3400), wow=(0.001, 0.0014), lead_p=(0.3, 0.4), lush=0.8, jazz=0.35,
                    words=["moon", "lanterns", "crickets", "garden", "koi", "petals", "walk home"])),
    dict(name="Lazy Noon", mood="bright sky, lemonade, bikes in the park.", rain=False, weather="rain", noon=True,
         style=dict(bpm=(98, 108), minor=0.1, keys=["rhodes", "piano"], lead=["pluck", "pluck", "bell"],
                    comp=["busy", "stabs", "charleston"], drums=dict(feel="bounce", busy=0.75, half=0.0),
                    swing=(0.08, 0.15), cutoff=(5000, 5800), wow=(0.0004, 0.0006), lead_p=(0.5, 0.6),
                    lush=0.5, jazz=0.3, words=["lemonade", "bikes", "park", "sunshine", "picnic", "noon"])),
    dict(name="Tokyo Cassette", mood="sakura on wet asphalt, a vending machine hums.", rain=False, weather="rain",
         theme="japan",
         style=dict(bpm=(72, 82), minor=0.6, keys=["rhodes", "piano", "rhodes"], lead=["shaku", "shaku", "koto", "shami"],
                    comp=["sustain", "halves", "sustain"], drums=dict(feel="boom", busy=0.35, half=0.3),
                    swing=(0.1, 0.18), cutoff=(3800, 4400), wow=(0.0009, 0.0012), lead_p=(0.3, 0.42),
                    lush=0.4, jazz=0.1, scale={"maj": [0, 2, 5, 7, 9], "min": [0, 1, 5, 7, 8]}, strict=True,
                    grace=0.3, chimes=0.18, pool_p=0.85, riff=0.9, riff_inst=["koto", "koto", "shami"],
                    gliss=0.5, bends=0.3, wa=True,
                    pool={"maj": [["5:maj7", "7:7", "4:m7", "9:m7"],                 # the royal road (ōdō shinkō)
                                  ["5:maj7", "7:7", "4:7", "9:m7"],
                                  ["5:maj9", "4:7", "9:m9", "7:m7|0:7"],             # marunouchi sadistic
                                  ["9:m7", "5:maj7", "7:7sus4", "0:add9"],            # komuro: vi–IV–V–I
                                  ["5:maj7", "7:7sus4", "9:m7", "0:add9"]],
                          "min": [["8:maj7", "10:7sus4", "0:m9", "0:m9"],              # ♭VI–♭VII–i
                                  ["5:m7", "7:m7", "0:madd9", "3:maj7"],               # iv–v–i–♭III
                                  ["8:maj7", "7:7", "0:m7", "3:maj7"],
                                  ["0:madd9", "8:maj7", "10:7sus4", "0:madd9"],
                                  ["5:m9", "10:7", "3:maj7", "8:maj7|7:7"],            # the circle of fifths, J-pop style
                                  ["0:m9", "10:7sus4", "8:maj7", "7:7"],               # i–♭VII–♭VI–V7, anime-ballad descent
                                  ["8:maj7", "7:7", "0:m7", "10:m7|3:7"]]},
                    words=["sakura", "tokyo", "kissaten", "shrine", "last train", "vending machine", "yukata",
                           "hanami", "tatami", "shibuya"])),
    dict(name="Smoky Jazz Club", mood="brushed snare, upright bass, a tenor in the corner.", rain=False,
         weather="rain", theme="jazz",
         style=dict(bpm=(84, 100), minor=0.4, keys=["piano", "piano", "rhodes"], lead=["vibes", "horn", "horn"],
                    comp=["charleston", "stabs", "charleston"], drums=dict(feel="brush", busy=0.55, half=0.0),
                    swing=(0.55, 0.66), cutoff=(4300, 5200), wow=(0.0005, 0.0008), lead_p=(0.5, 0.68),
                    lush=1.0, jazz=1.0, walk=1.0, rootless=True, bebop=0.35,
                    scale={"maj": [0, 2, 3, 4, 7, 9], "min": [0, 3, 5, 6, 7, 10]},          # the blues scales
                    pool_p=0.8,
                    pool={"maj": [["0:maj9", "9:7b9", "2:m9", "7:13"],                  # I–VI7–ii–V
                                  ["4:m7", "9:7b9", "2:m9", "1:9"],                    # iii–VI7–ii–♭II7 (tritone sub)
                                  ["2:m9", "7:13", "0:maj9", "0:6/9|9:7b9"],
                                  ["0:maj9", "5:9", "4:m7|9:7b9", "2:m9|7:13"],
                                  ["5:maj9", "5:m6", "4:m7|9:7b9", "2:m9|7:13"],       # IV–iv–iii VI–ii V
                                  ["0:6/9", "1:dim7", "2:m9", "3:dim7"]],              # passing diminished chords
                          "min": [["0:m9", "0:m6", "2:m7b5", "7:7b9"],
                                  ["0:m9", "5:m9", "10:13", "3:maj9"],
                                  ["0:m9", "8:9", "2:m7b5", "7:7b9"],
                                  ["5:m9", "10:13", "3:maj9", "2:m7b5|7:7b9"]]},
                    words=["blue", "velvet", "bourbon", "last set", "brushes", "smoke", "standards", "upright",
                           "after hours", "neon"])),
    dict(name="Candlelit Manor", mood="harpsichord by candlelight, fog over the moor.", rain=False,
         weather="rain", theme="english",
         style=dict(bpm=(70, 80), minor=0.5, keys=["strings", "strings", "piano"], lead=["recorder", "violin", "recorder"],
                    comp=["sustain", "sustain", "halves"], drums=dict(feel="sparse", busy=0.3, half=0.45),
                    swing=(0.04, 0.1), cutoff=(3600, 4300), wow=(0.0008, 0.0011), lead_p=(0.4, 0.55),
                    lush=0.0, jazz=0.0, arp=0.85, arp_inst=["harpsi", "harpsi", "lute"], trill=0.3, bass="cello",
                    scale={"maj": [0, 2, 4, 5, 7, 9, 11], "min": [0, 2, 3, 5, 7]}, pool_p=0.85,
                    pool={"maj": [["0:|7:", "9:m|4:m", "5:|0:", "5:|7:"],                 # Pachelbel's canon
                                  ["0:", "5:", "7:7", "0:"],                            # I–IV–V–I, a hymn's cadence
                                  ["0:", "9:m", "5:", "7:7"],
                                  ["0:", "2:m", "7:sus4|7:7", "0:"],
                                  ["0:", "4:m", "5:", "2:m|7:7"],
                                  ["0:|5:", "0:|7:", "9:m|5:", "7:sus4|7:"]],             # a galliard's lilt
                          "min": [["0:m", "3:", "10:", "7:m"],                          # Greensleeves
                                  ["0:m", "3:", "10:|7:", "0:m"],
                                  ["0:m|7:7", "0:m|10:", "3:|10:", "0:m|7:7"],          # La Folia
                                  ["0:m", "10:", "8:", "7:7"],                          # the lament (Andalusian) descent
                                  ["0:m", "10:", "0:m", "5:"],                          # Dorian, Scarborough Fair
                                  ["5:m", "0:m", "2:dim|7:7", "0:m"]]},                 # iv–i–ii°–V–i
                    words=["candle", "manor", "moor", "greensleeves", "harpsichord", "minuet", "lantern", "abbey",
                           "parchment", "hedgerow", "tea", "fog"])),
]
