"""The world seed. Everything generated (scene layouts, textures, creatures and the music) is drawn from it, so
the same seed gives the same world. It changes on every launch and on `x`; LOFI_SEED pins it."""
import random
import zlib

WORLD = {"seed": 0}


def wseed(*key):
    """A stable integer seed for `key` in the current world."""
    return zlib.crc32(repr((WORLD["seed"],) + key).encode()) & 0x7FFFFFFF


def wrng(*key):
    """A random.Random for `key` in the current world (same world → same layout)."""
    return random.Random(wseed(*key))
