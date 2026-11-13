import locale
import sys

from .app import App
from .engine import Engine
from .settings import load_settings
from .term import Term
from .tools import preview, selftest


def main():
    if "--selftest" in sys.argv:
        selftest()
        return
    if "--preview" in sys.argv:
        i = sys.argv.index("--preview")
        preview(sys.argv[i + 1] if i + 1 < len(sys.argv) else "preview")
        return
    locale.setlocale(locale.LC_ALL, "")
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("lofi.term needs to run in an interactive terminal.")
        sys.exit(1)
    eng = Engine()
    app = App(None, eng, load_settings())        # restores station, volume, ambience… before playback
    try:
        eng.start()
    except Exception as ex:
        print(f"Could not open an audio output device: {ex}")
        sys.exit(1)
    try:
        with Term() as term:
            app.term = term
            app.run()
    except KeyboardInterrupt:
        pass
    finally:
        eng.stop()
        app.autosave(force=True)
    print("☾ thanks for listening.")
