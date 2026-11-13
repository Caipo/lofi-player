import os
import time

import numpy as np

from .app import App
from .engine import Engine
from .scenes import VIZ_NAMES
from .stations import STATIONS
from .theory import SR
from .world import WORLD


# ════════════════════════════════════════════════════════════════════════════
def selftest():
    """Render a few bars of every station offline and report timing/levels."""
    import wave
    if os.environ.get("LOFI_SEED", "").isdigit():
        WORLD["seed"] = int(os.environ["LOFI_SEED"])
    eng = Engine()
    eng.recompose()
    for i, st in enumerate(STATIONS):
        eng._reset(i)
        song = eng.wst
        print(f"{song['name']}: “{song['title']}” · {song['key'][0]} {song['key'][1]} · {song['bpm']} bpm · "
              f"{song['keys']} + {song['lead']}{' + ' + song['riff_inst'] + ' riff' if song.get('riff_inst') else ''}{' + ' + song['arp_inst'] + ' arpeggios' if song.get('arp_inst') else ''} · {len(song['plan'])} bars\n"
              f"    A {' '.join(song['prog'])}   B {' '.join(song['progB'])}   "
              f"kick {song['drums']['k']}  snare {song['drums']['s']}")
        t0 = time.time()
        bars = [eng.render_bar() for _ in range(8)]
        dt = (time.time() - t0) / 8
        mus = np.concatenate([b[0] for b in bars])
        st = eng.st
        print(f"{st['name']:<16} render {dt * 1000:6.1f} ms/bar ({dt / (240 / st['bpm']) * 100:4.1f}% realtime)  "
              f"peak {np.abs(mus).max():.2f}  rms {np.sqrt((mus ** 2).mean()):.3f}")
        if i == 0:
            mix = mus + np.concatenate([sum(b[1][k] for k in ("rain", "vinyl")) for b in bars])
            with wave.open("selftest.wav", "wb") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(SR)
                wf.writeframes((np.clip(mix * 0.7, -1, 1) * 32767).astype("<i2").tobytes())
    print("wrote selftest.wav")


def _write_png(path, img):
    import struct
    import zlib
    h, w = img.shape[:2]
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                 + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def preview(outdir, cols=120, rows=40):
    """--preview DIR: render every visualizer offline to PNGs (no audio, no terminal)
    and print frame timings. Overlay glyphs are drawn as small blobs."""
    rows, cols = map(int, os.environ.get("LOFI_SIZE", f"{rows}x{cols}").split("x"))
    os.makedirs(outdir, exist_ok=True)
    eng = Engine()
    eng.cg = float(os.environ.get("LOFI_VINYL", "0"))
    fake = type("T", (), {"size": lambda s: (rows, cols)})()
    app = App(fake, eng)
    if os.environ.get("LOFI_EVENT"):                 # pin every scene's rare event at this point (0…1)
        app.rare_force = float(os.environ["LOFI_EVENT"])
    if os.environ.get("LOFI_GROOVE"):                # pin the music signals (kick, snare, note…) at this level
        app.groove_force = float(os.environ["LOFI_GROOVE"])
    for vi, name in enumerate(VIZ_NAMES):
        eng._reset(int(os.environ.get("LOFI_STATION", "0")))
        mus, layers, meta = eng.render_bar()
        eng._push(np.concatenate([mus, mus])[:8192])
        eng.cur, eng.pos, eng.meta = (mus, layers, meta), len(mus) // 3, meta
        app.viz = vi
        times = []
        for _ in range(45):
            t0 = time.time()
            app.analyze()
            ch, fg, bg = app.render_scene(rows - 5, cols - 2)
            times.append(time.time() - t0)
        h, w = ch.shape
        img = np.zeros((h * 12, w * 6, 3), np.uint8)
        for y in range(h):
            for x in range(w):
                c = ch[y, x]
                f, b_ = np.clip(fg[y, x], 0, 255), np.clip(bg[y, x], 0, 255)
                blk = img[y * 12:(y + 1) * 12, x * 6:(x + 1) * 6]
                if c == "▀":
                    blk[:6], blk[6:] = f, b_
                elif c == " ":
                    blk[:] = b_
                else:
                    blk[:] = b_
                    if c == "█":
                        blk[:] = f
                    elif c == "▄":
                        blk[6:] = f
                    else:
                        blk[3:9, 2:4] = f
        _write_png(os.path.join(outdir, f"{vi + 1}-{name.replace(' ', '_')}.png"), img)
        print(f"{vi + 1} {name:<15} {np.mean(times[5:]) * 1000:6.1f} ms/frame")
