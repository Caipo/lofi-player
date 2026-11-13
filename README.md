# ☾ lofi.term

A terminal lofi radio. The music is composed and synthesized live: every song (key, chords, form,
groove, melody, instruments, even its title) is procedurally generated, then played with dusty swung
drums, bass, tape wobble, vinyl crackle and rain. It needs no audio files, and every station plays an
endless run of new songs.

## Run

```sh
./lofi                 # or: .venv/bin/python lofi.py
```

First-time setup on another machine:

```sh
python3 -m venv .venv && .venv/bin/pip install numpy sounddevice
```

## Stations

Each station is a style, not a playlist. Its songs are written on the fly within that style:

| Station        | BPM    | Style                                                        |
|----------------|--------|--------------------------------------------------------------|
| Rainy Window   | 68–76  | rhodes and bells, mostly major, boom-bap, rain on            |
| Midnight Study | 76–84  | felt piano, jazzy: ii–V turnarounds and secondary dominants  |
| Sunday Coffee  | 84–92  | shuffle groove, charleston comping, plucked melodies         |
| Snowfall Tape  | 60–68  | warm pads, sparse and often half-time, heavy tape wobble, snow |
| Neon Bus Ride  | 88–96  | syncopated rhodes stabs, lazy kicks, often minor, rain on    |
| Moonlit Garden | 66–74  | piano and bells, mostly minor                                |
| Lazy Noon      | 98–108 | bright and busy: bouncy drums and comping, a midday sky in every scene |
| Tokyo Cassette | 72–82  | Japanese lofi: a koto or shamisen riff through every song, shakuhachi melodies, taiko, woodblock and singing bowl, J-pop changes on the in and yo scales |
| Smoky Jazz Club| 84–100 | triplet swing, ii–Vs and tritone subs, rootless piano over a walking bass, ride and brushes, vibes or a tenor-like horn playing bebop lines |
| Candlelit Manor| 70–80  | old English classical: harpsichord or lute arpeggios over strings, a bowed cello bass, recorder or violin melodies with trills, Pachelbel, Greensleeves and La Folia changes |

### How the music is generated

- **Tempo**: one per station per world, picked from the station's range, so beat-synced visuals stay steady.
- **Key**: each song moves to a related key (up a fourth or fifth, the relative major/minor, or a step).
- **Chords**: an A and a B progression of four chords each. They come from progressions that sound like
  lofi, or grow by a random walk over chord moves that sound right. Jazzier stations add secondary
  dominants, borrowed chords and ii–V turnarounds. Lusher stations stretch the chords to 9ths, 11ths and 13ths.
  *Tokyo Cassette* mostly plays Japanese pop changes: the royal road (IV–V–iii–vi), the Marunouchi
  Sadistic (IV–III7–vi–v I7), Komuro's vi–IV–V–I, and in minor ♭VI–♭VII–i, the circle of fifths
  and the anime-ballad descent i–♭VII–♭VI–V7. *Candlelit Manor* plays plain triads in classical and
  English folk progressions: Pachelbel's canon, hymn cadences, Greensleeves, La Folia, the lament
  descent and Dorian Scarborough Fair. *Smoky Jazz Club* mostly
  plays standards-style changes: I–VI7–ii–V, tritone substitutions, passing diminished chords, and
  ii–V pairs packed into a single bar.
- **Form**: an intro (or a one-bar turnaround between songs), then A and B phrases of 4 or 8 bars in
  patterns like AABA or ABAB. There's sometimes a breakdown before the last phrase, and some phrases
  carry the melody. A song lasts about one to three minutes, then the next one is written.
- **Melody**: each part gets its own rhythm and tune. When A comes back, so does its melody (with the odd
  note changed). Melodies use the pentatonic scale, except on *Tokyo Cassette*, whose melodies stay on
  the Japanese in scale (minor songs) and yo scale (major songs), with koto grace notes and now and then
  a few wind-chime notes; *Smoky Jazz Club*, whose lines use the blues scales and slide into notes
  from a half step away, bebop-style; and *Candlelit Manor*, whose melodies use the full major scale
  (or minor without its sixth and seventh) and decorate long notes with mordents and short trills.
- **Bass**: the root note, held, with a pickup note now and then. On *Smoky Jazz Club* it walks instead:
  a note on every beat, chord tones, then a half step into the next chord, with the odd swung skip note.
  The piano leaves the root to the bass there and plays rootless voicings.
- **Tokyo Cassette** is built the way Japanese lofi tracks are: most songs ride on a one-bar koto or
  shamisen ostinato that plays from the intro to the last bar, climbing the scale from each chord's root.
  A ren (a glissando across the strings) sweeps into new phrases, and long notes are pluck-and-press
  bends (oshide). The shakuhachi plays long legato breaths that lift from a semitone flat (meri), with
  breathy muraiki accents and falling phrase ends. A taiko marks the bars and rolls into new phrases, a
  woodblock knocks on the offbeats, a rin (singing bowl) rings in each section, and wooden clappers
  (hyoshigi) open each song. The Rhodes steps back to a quiet bed under the riff.
- **Candlelit Manor** sounds like a lofi take on a Baroque chamber piece: most songs break their chords
  into a harpsichord or lute figure (an Alberti bass, a Bach-prelude pattern, rising and falling
  arpeggios) over softly held strings, with a cello bowing the bass and a recorder or violin on top.
  The drums stay sparse and often half-time, and the swing is nearly straight.
- **Groove and sound**: kick, snare and hat patterns are generated from the station's feel (boom-bap,
  lazy, sparse, shuffle, bounce, or jazz: a ride cymbal, brushes, a feathered kick and the hi-hat on 2
  and 4), along with the chord rhythm, instruments (rhodes, piano, pad, bell, pluck, koto, shamisen, shakuhachi,
  vibraphone, a tenor-like horn, harpsichord, lute, strings, violin, cello, recorder), swing, tape filter and wow.
- **Title**: every song gets one, like “window at dawn” or “polaroids for margins”.

Everything comes from the world seed. `LOFI_SEED` replays the same songs, and `x` writes new ones. Only
the humanising (timing, velocity, a melody note here and there) changes from listen to listen. When you
tune back to a station, it carries on with its next song.

## Visualizers

1. **spectrum**: pastel gradient bars with falling peak caps, under a starry sky with a far ridge on the horizon, standing on a grid floor that rolls toward you with the beat
2. **rainy city**: skyline with flickering windows; rain (or snow) falls harder when the music gets louder; puddle reflections
3. **forest**: hazy mountains and two rows of trees with moonlit and shaded sides: a far row up the slope behind the mist, and a front row that sways with the volume, small beside the fire and towering at the edges so the clearing frames the camp. The trees suit the station: pines, with autumn maples on *Sunday Coffee*, green broadleaves on *Lazy Noon* and *Moonlit Garden*, cherry blossom on *Tokyo Cassette* and dark oaks on *Candlelit Manor*. There is also a campsite with a tent, a bench and a fire that flares on the bass and sends up a stream of sparks, an owl, fireflies, and grass that grows from the spectrum. On *Snowfall Tape* the snow slowly builds up until the ground, branches and tent turn white (it melts when you switch stations)
4. **coffee shop**: rain on the window over blurry city lights, a sleepy cat flicking its tail on the beat, pendant lamps that glow with the bass, a radio with a live EQ, a steaming mug, and a flickering neon sign
5. **ocean**: a little sailboat bobs on the swell and rolls with the slope of the waves (with a wake and a lantern that pulses on the beat), moonlight shimmering on the water, a lighthouse with a sweeping beam, gulls and drifting clouds. The sea's roughness follows the tempo: slow stations get a long, glassy swell, fast ones a short, choppy sea with whitecaps. Louder music makes the waves bigger still; with rain on, the clouds thicken, the stars and gulls hide, and drops ripple on the swell
6. **cherry blossom**: a blossom tree grown procedurally, branch by branch (a trunk splitting into limbs, branches and twigs that bend and taper, with blossom along the young wood), so every world has a different tree. It stands in a meadow of grass blades, small flowers and a stepping-stone path leading from a stone lantern, coloured for each station: a spring meadow at noon, autumn grass at sunset, snow drifts on *Snowfall Tape*. Petals tumble down on the wind (more when the music swells, with gusts on heavy bass), and pile up pink on the ground. Mount Fuji rises behind the hills, its snowcap catching the moonlight (or the noon sun). Paper lanterns and a stone lantern glow on the beat. Rain knocks the petals down faster
7. **lava lamp**: a glowing lamp in a dark room. Wax blobs heat up at the bulb, rise, cool off at the top and sink, stretching and merging as they go. The bass turns up the heat (so does the fireplace toggle), and every station has its own colors
8. **sleepy cat**: a cat sleeps curled up in its bed, breathing slowly in time with the music, with its tail tip twitching on the beat and an ear flicking now and then. Every station has its own cat (grey tabby, black, orange tabby, white, tuxedo, siamese, cream tabby, black and white, russian blue, british brown tabby) and its own room: a moonlit night, a sunset, a snowy evening, neon, or a sunny noon with the cat napping in a sunbeam. A dream bubble shows what it's dreaming about (a fish, a mouse, a butterfly or a ball of yarn), and the dream changes every few bars. Fairy lights on the curtain rod twinkle on the beat, dust drifts in the light from the window, and a wall clock shows your real time, its pendulum swinging with the music. Rain runs down the glass, snow settles on the roofs outside, and the fireplace toggle adds a warm glow
9. **koi pond** (←/→ or `v`): a garden pond seen from above. Kohaku, sanke, showa, tancho, ogon, chagoi and asagi koi (some with long butterfly fins) wander around, their bodies swaying as they swim and their shadows falling on the pond floor. They swim faster when the music gets louder, one darts off on each heavy bass hit, and now and then one rises to gulp at the surface, leaving a ring. Caustic light shimmers on the water with the music, lotus flowers on the lily pads bloom brighter with the bass, and stone lanterns glow on the beat (warmer with the fireplace on). Rain and storms dimple the water with rings, petals or maple leaves drift on a slow current, and on *Snowfall Tape* the banks slowly turn white. Wide windows get a chain of pools joined by little red bridges
10. **night train** (key `0`): the view from a carriage at night. Streaky clouds and moonlit mountain peaks with snowcaps barely move. Wooded hills, villages with lit windows and the odd church steeple drift by, a patchwork of fields slides past faster the nearer it is, copses of trees swing by the line, lakes mirror the moon, and the hedges and telegraph poles whip past in a blur (a neon skyline on *Neon Bus Ride*). Now and then the train dives into a tunnel: the view goes dark, tunnel lamps streak past, and the carriage shows up reflected in the glass. The carriage rocks on the rail joints in time with the beat and the tea trembles in its mug. The train runs faster when the music is louder and eases to a halt when you pause. Inside are wood panelling, brass sconces, luggage on the rack, high-backed seats with lace covers, and a green banker's lamp and a ticket on the table. Rain and snow streak back across the glass, the ocean-waves toggle routes the line along the coast, and wider windows show up to three windows in a row
11. **aquarium**: a fish tank glowing in a dark room. Light rays and caustics play across the sand, plants sway (harder on the bass), a school of tetras and some goldfish, angelfish and bettas swim about and dart on heavy bass hits, bubbles rise from the air stone, and a treasure chest pops open on the bass and lets out a puff of bubbles. The waves toggle stirs up the surface
12. **rooftop**: a city rooftop. String lights chase the beat, laundry flaps in the wind (gusts on the bass), a black cat sits on the brick parapet looking out over the city, its tail swaying down the wall and potted plants sway. On the roof there's a stairwell hut with a lit door, an AC unit whose fan spins with the music, steam puffing from a vent, and a glowing skylight. A striped deck chair sits next to a crate with a radio (live EQ) and a steaming mug, and a telescope points at the moon. Clouds drift by, a plane blinks across and the odd shooting star falls. Rain leaves puddles that hold the lights, snow whitens the roof, and the fireplace toggle lights a fire bowl
13. **planet**: a procedurally generated planet that turns once every 8 bars, so it spins faster on upbeat stations (*Lazy Noon*, around 100 bpm) and slower on sleepy ones (*Snowfall Tape*, around 64), and eases to a stop when you pause. Its terrain, axial tilt, moons and sometimes a ring are generated from the world seed, and every station repaints it as a different kind of world: an ocean world, a city planet whose night side glitters with lights, a ringed desert with canyons, an ice world with polar auroras, a banded neon gas giant with a great storm, a jungle that glows on its night side with the bass, and a blue-green garden world. Sunlight wraps around it with a glowing terminator and atmosphere, the sun glints off the oceans, the ring casts a shadow (and the planet shadows the ring), clouds drift a little faster than the ground and thicken in the rain, and lightning lights up the cloud tops in a storm

### Playing along

The scenes don't just listen to the sound: the engine tells them every drum hit, melody note and chord
change it plays, so they move exactly in time with what you hear. Kicks, snares and hats each have
their own effect, every note of the melody leaves a mark (higher notes sit higher or further right), each
chord has its own colour (its root's place on the circle of fifths sets the hue, and minor chords are
cooler), and the busier the section is (intro, breakdown, groove, melody), the livelier the scene gets.

| scene          | kick                               | snare / hats                          | melody notes                                | chord changes                       |
|----------------|------------------------------------|---------------------------------------|---------------------------------------------|-------------------------------------|
| spectrum       | the bars' glow and the floor line throb | the reflection ripples; spotlights flare | sparks rise off the bars at the note's pitch | the nebula takes the chord's colour |
| rainy city     | some windows flash, antennas blink | neon signs flare, the puddles ripple  | a window lights up, higher for higher notes | lights go on and off; the city glow changes colour |
| forest         | the campfire glows                 | a log cracks with a burst of sparks; fireflies twinkle on the hats | a firefly flares at the note's height | the sky changes colour, the owl blinks |
| coffee shop    | regulars nod, the radio speaker thumps, the lamps brighten | the other regulars nod on the backbeat, the neon flares | notes float off the radio's aerial | the chord on the chalkboard flashes |
| ocean          | n/a                                | moonlight on the water glitters on the hats | a star glints, higher for higher notes | the horizon changes colour, the lighthouse flares |
| cherry blossom | the stone lantern glows            | the blossoms glint on the hats        | a bright petal shakes loose, left to right by pitch | a puff of petals; the sky changes colour |
| lava lamp      | the wax throbs, the lamp glows     | more little bubbles on the hats       | a bubble rises, across the glass by pitch   | the lamp's halo changes colour      |
| sleepy cat     | fairy lights and the dream bubble pulse | an ear twitches at the backbeat  | sparkles rise in the dream bubble           | the fairy lights blush to the chord's colour |
| koi pond       | caustics flash, lanterns glow, a koi sometimes darts off | fireflies blink on the hats | a drop rings the water, across by pitch | the shallows change colour |
| night train    | the carriage bumps, the tea shivers, tunnel lamps flare | pole lamps flare, village windows twinkle | a star winks, higher for higher notes | the night outside changes colour |
| aquarium       | light rays brighten, the chest lid lifts | a fish darts; the surface shimmers on the hats | a bubble rises from the air stone (higher notes are smaller) | the school of tetras wheels round; the water changes colour |
| rooftop        | the string lights throb            | the laundry snaps in a gust           | notes float off the radio's aerial          | a colour washes through the lights; sometimes a shooting star |
| planet         | aurora and night-side jungle glow pulse, the atmosphere brightens | city lights twinkle on the hats | a star flares, higher for higher notes | the nebula changes colour |

On the Japanese stations the koto and shamisen riff counts as well: lanterns brighten with each pluck,
and the plucks drop rings on the koi pond.

### Rare events

Every couple of minutes or so, something small happens in the scene you're watching. Each scene has its own
event (they freeze while paused, and `e` brings one on right away):

| Scene          | Rare event |
|----------------|------------|
| spectrum       | a little cat strolls along the tops of the bars, hopping up as they rise |
| rainy city     | fireworks burst over the skyline (at noon, hot-air balloons drift over instead) |
| forest         | a deer steps into the clearing, grazes, looks up at the camp (eyes catching the firelight) and wanders off |
| coffee shop    | a city bus pulls up outside, idles at the stop with its windows lit, and pulls away |
| ocean          | a whale surfaces past the boat, blows a plume of spray, and lifts its flukes as it dives |
| cherry blossom | a V of red-crowned cranes flies over |
| lava lamp      | a moth flutters in and circles the lamp, its shadow swooping huge across the wall |
| sleepy cat     | a mouse scurries along the skirting board and stops to sniff (the cat's ear twitches) |
| koi pond       | a dragonfly hovers over the pond and dips to the water, leaving a ring |
| night train    | a train on the other line rushes past the other way in a blur of lit windows |
| aquarium       | a pufferfish potters across the tank and blows up into a spiky ball on a heavy bass hit |
| rooftop        | a hot-air balloon drifts across the sky, its burner flaring on the bass |
| planet         | a comet sails past behind the planet, its tails streaming away from the sun |

### Every launch is a new world

Scenes are generated from a world seed that changes every time you start, so the layout of every
scene is new each time: the city's skyline (stepped tops, spires, domes, window grids), the forest's
ranges, tree rows and where the campsite sits (the tent can be on either side of the fire), the
café's menu, shelf, regulars and lamps, the ocean's island, lighthouse colours, moon and sailboat,
the blossom tree (grown branch by branch), the planet (terrain, tilt, moons and ring), the lava lamps' shapes and wax, the cat's room (window, houses outside,
painting, plant), the koi pond, the train's route and the rooftop's skyline and neighbour. The cats
on the café sill and the rooftop parapet are generated too, as are the moon's craters and every star
field and texture, and so is the music (see above). Press `x` for a new world at any time. `LOFI_SEED=1234` pins one world (handy with
`--preview`).

### Picking a scene

Type its number like on a TV remote: `1`–`9` switch at once (`0` is 10), and a second digit typed within a second makes a two-digit number (`1` then `3` → planet). `g` (or clicking the scene name in the top bar) opens a scene picker that lists every scene. Moving through it previews each scene live, and Enter keeps one. ←/→ and `v` step through them, and the sidebar lists them all, numbered.

### Every station has its own look

Changing station restyles every scene to match it:

| Station        | City           | Forest        | Ocean        | Blossom   | Café sign     | Cat     |
|----------------|----------------|---------------|--------------|-----------|---------------|---------|
| Rainy Window   | purple dusk    | misty night   | moonlit      | sakura    | ☾ open late   | grey tabby |
| Midnight Study | deep midnight  | dark midnight | midnight     | white plum| study hall    | black   |
| Sunday Coffee  | golden sunset  | autumn dusk   | sunset       | red maple | fresh brew    | orange tabby |
| Snowfall Tape  | snowy evening  | winter        | winter grey  | snowy     | warm drinks   | white   |
| Neon Bus Ride  | neon           | aurora        | neon         | wisteria  | open 24/7     | tuxedo  |
| Moonlit Garden | teal night     | firefly garden| teal moon    | moonlit sakura | night tea | siamese |
| Lazy Noon      | bright midday  | summer day    | blue noon    | sakura in sun | lunch special | cream tabby |
| Tokyo Cassette | plum dusk      | shrine path   | sakura dusk  | deep sakura | kissa · open | black and white |
| Smoky Jazz Club| smoky amber    | amber night   | bourbon night| magnolia  | live jazz     | russian blue |
| Candlelit Manor| foggy gaslight | misty moor    | grey channel | apple blossom | tea room  | british brown tabby |

The spectrum, lava lamp, train, aquarium and rooftop change colors to match, and the planet becomes a different kind of world. On *Lazy Noon*, every scene with a sky gets a high midday sun instead of the moon.

The three themed stations also add their own props to every scene:

| Scene          | Tokyo Cassette                                          | Smoky Jazz Club                                          | Candlelit Manor                                        |
|----------------|---------------------------------------------------------|----------------------------------------------------------|--------------------------------------------------------|
| spectrum       | a striped rising sun over a snow-capped peak            | spotlights, and a keyboard whose keys light up under the bars | a castle on a hill under the moon, fog at its foot     |
| rainy city     | a lit lattice tower, strings of red paper lanterns      | a club marquee with chasing bulbs, notes drifting out    | a medieval old town: timber-framed houses with jettied floors, steep gables and candlelit windows, wall lanterns and tavern signs over a cobbled street, a gothic cathedral rising behind |
| forest         | a torii gate and a stone lantern                        | a gramophone on the bench, playing to the fire           | a thatched cottage up the clearing, chimney smoking           |
| coffee shop    | noren curtains, a beckoning cat waving on the beat      | blinds half down, an upright bass behind the counter     | leaded diamond-pane windows, a candelabra and a teapot |
| ocean          | Mount Fuji, a torii standing in the water               | a showboat with every window lit                         | a chalk headland running in from the edge, a castle on its brow |
| cherry blossom | a distant pagoda on the far hill, in the open sky       | a bench under a street lamp with a saxophone on it       | apple blossom, a village church on the far hill        |
| lava lamp      | shoji screens                                           | records and sleeves pinned to the wall                   | damask wallpaper, a gilt-framed portrait, a candelabra |
| sleepy cat     | a pagoda and Fuji outside, a paper lantern lamp         | a record player spinning on the bedside table            | a church spire and a castle outside, a candle on old books |
| koi pond       | paper lanterns floating on the water                    | candles floating on amber water                          | candles and white petals floating on the water         |
| night train    | Fuji in the distance, torii and pagodas passing by      | a neon skyline, a cocktail glass on the table            | castles on mounds drifting past on the far hills                 |
| aquarium       | a torii and a pagoda on the sand                        | a sunken saxophone                                       | the classic aquarium castle, bubbling                  |
| rooftop        | paper lanterns instead of bulbs, a little shrine        | a gramophone and a stack of records                      | a smoking chimney stack rising behind the parapet       |
| planet         | a sakura world                                          | a banded giant whose ring is a grooved record            | a green, cloudy isle world                             |

### Scene transitions

Changing scene dissolves from the old picture into the new one through a different shape each time: a
diagonal wipe, an iris opening from the middle, venetian blinds, and an ordered-dither dissolve, with a
soft band of the station's colour along the moving edge. Turn them off in settings (`,`).

## Keys

| Key            | Action                  |
|----------------|-------------------------|
| ↑/↓ or j/k     | move station cursor     |
| Enter          | tune in                 |
| n / p          | next / previous station |
| ←/→, v         | next / previous scene   |
| 1–9, 0, 10–13  | scene by number (two digits: type them quickly) |
| g              | scene picker |
| space          | pause / resume          |
| r              | rain: rain sound plus rain in the city, forest, café window and ocean |
| c              | vinyl: crackle and a warm record-player tone, plus a worn-film look (dust, scratches, a rolling tracking bar, dimmed corners) on every visualizer |
| f              | fireplace: crackling logs. The forest campfire grows and a fireplace appears in the coffee shop |
| w              | ocean waves: surf that rolls in and crashes in stereo. Bigger, foamier waves in the ocean scene |
| m              | café chatter: a murmur of voices, cups clinking, the espresso machine steaming. The coffee shop fills with regulars and people walk past the window |
| t              | thunderstorm: a heavy downpour and rolling thunder. Lightning flashes in time with each thunderclap, with bolts across the sky in the outdoor scenes |
| a              | ambience mixer: a level slider for every layer (plus master volume) |
| o              | focus timer panel (pomodoro). `O` starts / pauses the timer from anywhere |
| ,              | settings: screensaver, pop-ups, tuning static, scene transitions, mouse |
| i              | now-playing card: song title, key, chord progression (current chord lit), section, bar |
| z              | screensaver                |
| e              | bring on the scene's [rare event](#rare-events) now |
| x              | new world: regenerate every scene and all the music |
| tab or s       | hide / show sidebar     |
| + / -          | volume                  |
| q              | quit                    |

Inside a panel: ↑/↓ pick a row, ←/→ adjust, Enter toggles or acts, Tab switches panels, Esc closes.

### Mouse

Click a station to tune in, click a visual to show it, and click an ambience toggle to switch it.
Click the status bar to pause or open the now-playing card. Drag the mixer sliders or the volume
bar, and scroll anywhere for volume (or over a slider to nudge it). You can turn the mouse off in
settings if you want your terminal's text selection back.

### Focus timer

A pomodoro timer: focus, then a short break, with a long break after every few rounds. Set the
lengths with ←/→, or type the minutes straight in and press Enter (focus 1–180 min, short break
1–60, long break 1–90, long break every 1–12 rounds or never). When a phase ends, a soft bell chime
plays over the music and a pop-up tells you what's next. The next phase starts on its own unless you
turn off auto-start. The countdown sits in the sidebar, the status bar and the screensaver.

### Screensaver

`z`, or automatically after 1–60 idle minutes. The scene fills the whole terminal with a small clock,
the now-playing line and the focus timer. It can move to the next scene every few minutes. Any key
or click wakes it.

### Tuning static

Changing station sounds like turning an old radio dial: hiss that wavers as it sweeps past other
stations, crackle, a whistle and a garbled scrap of somebody talking, while the picture tears and snows
for a moment like a TV between channels. Turn it off in settings (`,`).

### Now playing

The status bar shows the station, key, current chord, tempo and beat. A pop-up names the station and
its song when you tune in, and another announces each new song. The terminal tab's title follows the
station. `i` opens a card with the song's title and number, the instruments, the progression of the part
that's playing, the section (intro, groove, melody, breakdown, turn) with its part (A/B), and the bar count.

### Remembered settings

Station, visual, volume, sidebar, every ambience layer's on/off state and level, the timer lengths,
screensaver options, pop-ups, tuning static, scene transitions and mouse are saved to `~/.config/lofi/settings.json` (or
`$XDG_CONFIG_HOME/lofi/`) as you change them, and restored the next time you start. Delete the file
to reset everything.

Every scene is painted as pixel art: each character cell holds two pixels (drawn with `▀`), in
24-bit color where the terminal supports it (iTerm2, WezTerm, Ghostty, VS Code, or any terminal that
sets `COLORTERM=truecolor`), otherwise quantised to the 256-color palette. Force either mode with
`LOFI_COLOR=true` or `LOFI_COLOR=256`. Any window shape works, from a skinny 50-column split pane up to an ultra-wide monitor. Scenes lay
themselves out for the space: a wide café gets more windows, lamps and regulars along the counter; the
lava lamp becomes a row of lamps; the cherry blossom grows a grove across the meadow; the cat's room gets
a bookshelf. The sidebar shows at 84+ columns (tab hides it).

- `--selftest` composes a song for every station (title, key, both progressions, the groove), renders a
  few bars offline and reports CPU use and levels (`LOFI_SEED` picks the world).
- `--preview DIR` renders every visualizer to PNG files in `DIR` without audio or a terminal
  (`LOFI_SIZE=40x120`, `LOFI_STATION=3` and `LOFI_VINYL=1` adjust the size, the station and the film look;
  `LOFI_EVENT=0.4` shows every scene's rare event, 40% of the way through, and `LOFI_GROOVE=1` pins the
  drum, note and chord signals at full so you can see what each scene does with them).
