#!/usr/bin/env python3
"""
lofi.term — a terminal lofi radio.

Everything you hear is synthesized live: jazzy chords, dusty drums, bass,
little melodies, tape wobble, vinyl crackle and rain. No audio files needed.

Keys
  ↑/↓ or j/k   move station cursor        enter   tune in
  n / p        next / previous station    space   pause / resume
  ←/→ or v     cycle visualizer           1-13    scene by number (type fast)
  g            scene picker
  r            rain on/off                c       vinyl crackle + analog look
  f            fireplace                  w       ocean waves
  m            café chatter               t       thunderstorm
  a            ambience mixer             o       focus (pomodoro) timer
  O            start / pause the timer    ,       settings
  i            now-playing card           z       screensaver
  tab or s     hide / show the sidebar    x       new world (regenerate every scene)
  + / -        volume                     q       quit
The mouse works too: click stations, visuals and toggles, drag the sliders,
scroll for volume. Settings are remembered in ~/.config/lofi/settings.json.
"""
from lofiterm.main import main

if __name__ == "__main__":
    main()
