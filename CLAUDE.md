# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`lofi.term`: a terminal lofi radio. The music is composed procedurally and synthesized live with numpy, and
every visual scene is procedural pixel art drawn in the terminal. The code is the `lofiterm/` package.
`lofi.py` is only the entry point (`./lofi` runs it with the bundled venv). `README.md` is the user-facing
feature reference; keep it in sync when scenes, stations, keys or the music generation change.

## Commands

```sh
python3 -m venv .venv && .venv/bin/pip install numpy sounddevice   # first-time setup
./lofi                                   # run (needs an interactive TTY and an audio device)
.venv/bin/python lofi.py --selftest      # compose + render every station offline: song info, ms/bar, levels; writes selftest.wav
.venv/bin/python lofi.py --preview DIR   # render every scene to DIR/<n>-<name>.png and print ms/frame
.venv/bin/python -m pyflakes lofiterm    # the only lint; keep it clean
```

There is no test suite. To check changes, use `--preview` for visuals (look at the PNGs), `--selftest` for
audio, and pyflakes. Env vars: `LOFI_SEED=1234` (fixed world, which also fixes the songs),
`LOFI_SIZE=40x120` (rows×cols), `LOFI_STATION=3`, `LOFI_VINYL=1` (film look), `LOFI_EVENT=0.4` (pin every
scene's rare event at that point in `--preview`), `LOFI_GROOVE=1` (pin the music signals below at that level
in `--preview`, firing every few frames), and `LOFI_COLOR=true|256`.

To check one scene across all stations without rendering every scene, build `Engine()` and
`App(fake_term, eng)`, set `app.viz = VIZ_NAMES.index("...")`, and for each station call `eng._reset(i)`,
push one `render_bar()` into the engine (see `tools.preview`), then `app.analyze()` and
`app.render_scene(rows, cols)`.

Run Python from the project directory. Running the venv's python with the scratchpad as the working
directory hangs in this environment.

## Architecture

Module layers, low to high (imports only go downward):
`theory`/`world`/`stations` → `composer`, `dsp` → `engine`; `term` → `paint` → `scenes/*` → `app` →
`tools`/`main`.

**Music**
- `stations.py`: each station has a fixed identity (name, mood, weather, ambience preset). Its index also
  picks its look in every scene. Each station has a `style`: ranges and pools the composer draws from. The
  module docstring documents the fields.
- `composer.py`: `compose_song(i, k)` writes song k for station i: key (drifting to related keys via
  `song_key`), A/B progressions, a per-bar `plan`, drums, comping, instruments, mix values and a title.
  - Chord names must parse with `theory.parse_chord`, i.e. a root plus a quality in `QUAL`.
  - `station_bpm(i)` is fixed per world, so visuals can rely on `st["bpm"]`.
  - It is all seeded through `wrng`, so it is deterministic per world.
- `engine.py` (`Engine`): three threads.
  - A worker renders one bar at a time (`render_bar`) into a 2-slot queue. It plays from `self.wst`, the
    station merged with the current song, and walks `plan` with `self.sb`. When the plan runs out it calls
    `_start_song(k + 1)`.
  - The sounddevice callback mixes bars with the ambience layers, one-shot `fx_in` sounds and the analog
    effects. Whenever a bar starts playing it sets `self.st = meta["st"]`, so **the UI's `e.st` and
    `e.meta` follow what you hear**, not what the worker is composing two bars ahead. UI code should
    read `e.st`/`e.meta`, never `e.wst`.
  - `tune(i)` → the worker fades, flushes and runs `_reset(i, advance=True)`, which moves on to that
    station's next song. `recompose()` (called by `App.new_world`) restarts the music for a new world.
  - The beat clock is `played` (samples since tune-in) with `st["bpm"]`. `flash_at`/`flash_power` time
    lightning to the thunder.
  - `render_bar` lists what the bar plays in `meta["evs"]` as `(sample, kind, value)`: every `_hit` (drum
    kind + velocity), lead notes `"n"` and riff/arp plucks `"p"` (MIDI pitch), and chord changes `"c"`
    (root pitch class, +0.5 if minor). A new instrument part should add its events here.

**UI**: `app.py` (`App`) runs a 30 fps loop: keys → `analyze()` (FFT, `self.bass`, `band_levels`, then
`groove()`, which turns the events that just played into music signals, see Conventions) →
`draw()` (sidebar, status bar, panels, toasts, now-playing card). `term.py` has raw terminal I/O, the colour
palette and `Screen`. Settings are saved as JSON (`settings.py`, `~/.config/lofi/settings.json`). The saved
scene is stored by name.

**Scenes** (`scenes/`): each scene is a mixin class with a `v_<name>(cv, P, h, w)` method, and
`App(*MIXINS)` inherits them all.
- `P` is a float32 pixel array of shape `(2h, w, 3)`. Pixels are roughly square: two per cell, drawn as
  `▀`. `cv` is a `paint.Canvas` for text and glyph overlays. `compose()` merges the two, then `analog_fx`
  adds the vinyl look.
- **To add a scene:** create the module with its mixin, then add it to `SCENES` (name, method,
  description) and `MIXINS` in `scenes/__init__.py`. Give it a rare event too (see below). The scene's number is its position. Also update the
  README scene list and keys table, and the scene range in the `lofiterm/__init__.py` docstring.
- A scene's per-station look table (`FOREST`, `ROOF`, `PLANET`, …) lives in its own module and is indexed by
  station via `self.var(TABLE)`. **A new station needs an entry in every one of these tables**, plus a
  `style`.
- **Rare events**: every scene calls `ev = self.rare(dur)` once per frame and, while it returns `(u, r)`,
  draws its event in a `_rare_<scene>` method. `u` runs 0→1 over `dur` seconds of `self.rclock` (scene time
  that stops on pause, so use it, not `time.time()`, for the event's motion). `r` is a `random.Random`
  reseeded identically every frame of the event, so make every `r` draw unconditionally and in the same
  order. `self.rare_mem` is scratch state for one event. The scheduler (one event every 2–5 minutes, `e`
  summons one) lives in `SharedScene.rare`; README has the list of events.
- `scenes/shared.py` has helpers that several scenes use, as `SharedScene` methods: `weather`, `pine`,
  `broadleaf` (a round-crowned tree; forest and train mix it with pines per station),
  `flames`, `sky_body`, `stars`, `lightning`, `analog_fx`, `balloon`, `quadruped`, `tune_fx` (the
  picture's static while tuning; the sound is `Engine.static`), and `transition` (the wipe / iris /
  blinds / dissolve between scenes, started by `App.set_viz` from the last frame in `App._last`). It also
  holds shared data (cats, `CITY`, `COATS`, …). `self.sky = (x0, x1, y0, y1)` tells `lightning()` where it
  may draw bolts.
- Station themes: a station may set `theme` ("japan", "jazz", "english"). `self.theme()` returns it, and
  every scene draws that theme's props with the shared helpers `torii`, `chochin` (paper lantern),
  `pagoda`, `record`, `sax`, `notes` (drifting ♪ glyphs), `castle`, `spire`, `cottage`, `gaslamp` and
  `candles`. A new themed station needs its props in every scene.

**World seed and caching**
- `world.WORLD["seed"]` changes on launch and on `x`. Every generated layout and every musical choice
  must come from `wrng(*key)` or `wseed(*key)`, never from the global `random`, so a seed reproduces its
  world.
- `self.tex(name, key, fn)` caches arrays. Include `(h, w)` and the station index in `key` when the result
  depends on them. `self.layout(name, h, w, fn)` caches generated parameters.
- `new_world` clears both caches and every attribute named in `App.SCENE_STATE` (per-scene mutable state),
  so add new scene-state attributes there.
- `paint.py` has the drawing helpers: `glow`, `disc`, `poly`, `line`, `rect`, gradients, `ramp`, `blur`,
  `noise_tex`, `vnoise3`/`fbm3` (3D noise for spheres), stars, `moon` and `mix`.

**Performance**: scenes take about 1–5 ms/frame at 40×120, and a bar renders in 8–20% of real time (pads
are the heaviest). Use vectorized numpy, and build static things once through `self.tex`.

## Conventions

- Scenes react to the music only through `self.bass`, `band_levels()`, `self.rms`, the beat clock and the
  groove signals from `App.groove()`. Those signals are `self.kick`/`snare`/`hat`/`note`/`pluck`/`change`
  (0…1, they jump on a hit and decay), `self.pitch`/`ppitch` (the last note, 0 low…1 high), `self.tint` (the
  current chord's colour), `self.energy` (how busy the section is, eased) and `self.fired` (the kinds that
  hit this frame, for spawning things). Every scene uses them, so keep a new scene in step. The shared
  helpers are `note_sparks` (a glint per note) and `chord_wash` (tint rows with the chord). Scenes also
  react to ambience flags (`eng.amb_on`, `eng.wet`, `eng.amb_g["storm"]`) and station flags
  (`weather == "snow"`, `noon`). Anything that moves with time should stop or ease to a halt when
  `eng.paused` is set.
- Layouts adapt to window size, from 50 columns up to ultra-wide. Extra content in wide windows comes from
  the same seeded generators.
- `cat` is aliased to `batcat` in this shell and `timeout` isn't installed. Use the Read tool, or `sed -n`,
  to view files.
