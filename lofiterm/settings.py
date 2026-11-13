import json
import os


# ════════════════════════════════════════════════════════════════════════════
#  Settings, remembered between runs in ~/.config/lofi/settings.json
# ════════════════════════════════════════════════════════════════════════════
TIMER_DEFAULT = dict(focus=25, short=5, long=15, every=4, auto=True, chime=True)
TIMER_RANGE = dict(focus=(1, 180, 5), short=(1, 60, 1), long=(1, 90, 5), every=(0, 12, 1))   # lo, hi, arrow step
SAVER_DEFAULT = dict(idle=0, cycle=0, clock=True)
IDLE_CHOICES = [0, 1, 2, 5, 10, 15, 30, 60]
CYCLE_CHOICES = [0, 1, 2, 3, 5, 10, 15, 30]


def settings_path():
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "lofi", "settings.json")


def load_settings():
    try:
        with open(settings_path(), encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(d):
    p = settings_path()
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=2)
        os.replace(tmp, p)
    except OSError:
        pass


def _num(v, lo, hi, default):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
        return default
    return min(hi, max(lo, v))
