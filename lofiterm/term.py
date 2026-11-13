import os
import select
import sys
import termios
import tty

import numpy as np


# ════════════════════════════════════════════════════════════════════════════
#  Colour + terminal output
#  Every scene is painted into an RGB pixel buffer two pixels per character
#  cell (drawn with "▀": top pixel = foreground, bottom = background), with a
#  character overlay on top for text, rain, sparks, petals and braille lines.
#  Truecolor when the terminal supports it, else quantised to the 256 palette.
# ════════════════════════════════════════════════════════════════════════════
(C_PINK, C_PEACH, C_LAV, C_PURPLE, C_TEAL, C_BLUE, C_YELLOW, C_DIM, C_DARK,
 C_RAIN, C_WHITE, C_GREEN, C_SEL, C_FOREST, C_MOSS, C_BROWN, C_WOOD, C_FIRE,
 C_PINE, C_FAR, C_BACK, C_SAKURA, C_ROSE) = range(1, 24)
PAL = [(225, 222, 235)] * 24
for _cid, _rgb in {
        C_PINK: (255, 168, 208), C_PEACH: (255, 208, 168), C_LAV: (206, 182, 255), C_PURPLE: (160, 128, 240),
        C_TEAL: (128, 222, 214), C_BLUE: (118, 168, 232), C_YELLOW: (255, 236, 158), C_DIM: (132, 126, 154),
        C_DARK: (72, 66, 102), C_RAIN: (172, 192, 232), C_WHITE: (246, 244, 255), C_GREEN: (172, 232, 158),
        C_SEL: (206, 182, 255), C_FOREST: (72, 112, 82), C_MOSS: (122, 162, 112), C_BROWN: (172, 122, 86),
        C_WOOD: (112, 72, 52), C_FIRE: (255, 142, 82), C_PINE: (26, 62, 56), C_FAR: (62, 62, 98),
        C_BACK: (72, 82, 122), C_SAKURA: (255, 216, 236), C_ROSE: (222, 132, 172)}.items():
    PAL[_cid] = _rgb
PAL = list(PAL)
_PAL_ID = {}
_PALA = [None]
BG = (16, 14, 28)            # app background
PANEL = (21, 18, 36)         # sidebar
LINE = (64, 58, 96)          # frame lines


def rgb(r, g, b):
    """Palette id for an arbitrary colour (for overlay text)."""
    key = (int(r), int(g), int(b))
    i = _PAL_ID.get(key)
    if i is None:
        i = _PAL_ID[key] = len(PAL)
        PAL.append(key)
    return i


def pal_array():
    if _PALA[0] is None or len(_PALA[0]) != len(PAL):
        _PALA[0] = np.array(PAL, np.float32)
    return _PALA[0]


def _rgbof(c):
    return PAL[c] if isinstance(c, int) else c


_CUBE = np.array([0, 95, 135, 175, 215, 255])


def q256(a):
    """Nearest xterm-256 colour index for an (…, 3) array."""
    a = a.astype(np.int32)
    ci = np.argmin(np.abs(a[..., None] - _CUBE), axis=-1)
    cube = 16 + 36 * ci[..., 0] + 6 * ci[..., 1] + ci[..., 2]
    dc = ((a - _CUBE[ci]) ** 2).sum(-1)
    gi = np.clip(np.round((a.mean(-1) - 8) / 10), 0, 23).astype(np.int32)
    dg = ((a - (8 + 10 * gi)[..., None]) ** 2).sum(-1)
    return np.where(dg < dc, 232 + gi, cube)


class Screen:
    """Whole-terminal frame: a char, fg, bg and attribute per cell."""

    def __init__(self, h, w):
        self.h, self.w = h, w
        self.ch = np.full((h, w), " ", "<U1")
        self.fg = np.zeros((h, w, 3), np.uint8)
        self.bg = np.empty((h, w, 3), np.uint8)
        self.bg[:] = BG
        self.at = np.zeros((h, w), np.uint8)

    def text(self, y, x, s, fg, bg=None, at=0):
        if not (0 <= y < self.h) or x >= self.w or not s:
            return
        if x < 0:
            s, x = s[-x:], 0
        s = s[: self.w - x]
        n = len(s)
        if not n:
            return
        self.ch[y, x:x + n] = list(s)
        self.fg[y, x:x + n] = _rgbof(fg)
        if bg is not None:
            self.bg[y, x:x + n] = _rgbof(bg)
        self.at[y, x:x + n] = at

    def fill(self, y0, y1, x0, x1, bg):
        self.bg[y0:y1, x0:x1] = bg

    def box(self, y, x, h, w, bg, border):
        """Filled panel with a rounded frame, drawn over whatever is there."""
        y1, x1 = min(self.h, y + h), min(self.w, x + w)
        self.ch[y:y1, x:x1] = " "
        self.bg[y:y1, x:x1] = bg
        self.at[y:y1, x:x1] = 0
        self.text(y, x, "╭" + "─" * (w - 2) + "╮", border)
        self.text(y + h - 1, x, "╰" + "─" * (w - 2) + "╯", border)
        for yy in range(y + 1, y + h - 1):
            self.text(yy, x, "│", border)
            self.text(yy, x + w - 1, "│", border)

    def blit(self, y, x, ch, fg, bg):
        h, w = ch.shape
        self.ch[y:y + h, x:x + w] = ch
        self.fg[y:y + h, x:x + w] = np.clip(fg, 0, 255)
        self.bg[y:y + h, x:x + w] = np.clip(bg, 0, 255)


class Term:
    """Raw-mode terminal: alternate screen, key reading, diffed truecolor output."""

    def __init__(self):
        mode = os.environ.get("LOFI_COLOR", "").lower()
        ct = os.environ.get("COLORTERM", "").lower()
        prog = os.environ.get("TERM_PROGRAM", "")
        if mode in ("256", "8bit"):
            self.truecolor = False
        elif mode in ("true", "truecolor", "24bit"):
            self.truecolor = True
        else:
            self.truecolor = ct in ("truecolor", "24bit") or prog in ("iTerm.app", "WezTerm", "ghostty", "vscode", "WarpTerminal")
        self.fd = sys.stdin.fileno()
        self.out = sys.stdout
        self.buf = ""
        self.prev = None
        self.mouse = False
        self.title = None

    def __enter__(self):
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.out.write("\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[2J")
        self.out.flush()
        return self

    def __exit__(self, *exc):
        self.set_mouse(False)
        self.out.write("\x1b[0m\x1b[?7h\x1b[?25h\x1b[?1049l")
        if self.title is not None:
            self.out.write("\x1b]0;\x07")
        self.out.flush()
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def set_mouse(self, on):
        """SGR mouse reporting: clicks, drags (button-event tracking) and the wheel."""
        if on != self.mouse:
            self.mouse = on
            self.out.write("\x1b[?1000h\x1b[?1002h\x1b[?1006h" if on else "\x1b[?1006l\x1b[?1002l\x1b[?1000l")
            self.out.flush()

    def set_title(self, s):
        if s != self.title:
            self.title = s
            self.out.write(f"\x1b]0;{s}\x07")
            self.out.flush()

    def size(self):
        s = os.get_terminal_size(self.out.fileno())
        return s.lines, s.columns

    def keys(self):
        while select.select([self.fd], [], [], 0)[0]:
            data = os.read(self.fd, 1024)
            if not data:
                break
            self.buf += data.decode("utf-8", "ignore")
        s, out, i = self.buf, [], 0
        self.buf = ""
        while i < len(s):
            c = s[i]
            if c == "\x1b":
                if i + 1 >= len(s):
                    out.append("ESC")
                    i += 1
                    continue
                if s[i + 1] in "[O":
                    j = i + 2
                    while j < len(s) and not ("@" <= s[j] <= "~"):
                        j += 1
                    if j >= len(s):                      # incomplete: wait for the rest
                        self.buf = s[i:]
                        break
                    if s[i + 1] == "[" and j == i + 2 and s[j] == "M":    # legacy X10 mouse: skip its 3 raw bytes
                        if j + 3 >= len(s):                  # not all here yet
                            self.buf = s[i:]
                            break
                        i = j + 4
                        continue
                    if s[i + 1] == "[" and s[i + 2:i + 3] == "<" and s[j] in "Mm":   # SGR mouse
                        try:
                            b, mx, my = (int(v) for v in s[i + 3:j].split(";"))
                            out.append(("MOUSE", b, mx - 1, my - 1, s[j] == "M"))
                        except ValueError:
                            pass
                        i = j + 1
                        continue
                    k = {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT", "Z": "BTAB"}.get(s[j])
                    if k:
                        out.append(k)
                    i = j + 1
                    continue
                out.append("ESC")
                i += 1
                continue
            out.append({"\r": "ENTER", "\n": "ENTER", "\t": "TAB"}.get(c, c))
            i += 1
        return out

    def present(self, s):
        if self.truecolor:                      # drop the lowest bits: invisible, but fewer repaints
            f, b = s.fg.astype(np.int32) & 0xFC, s.bg.astype(np.int32) & 0xFC
            fk = (f[..., 0] << 16) | (f[..., 1] << 8) | f[..., 2]
            bk = (b[..., 0] << 16) | (b[..., 1] << 8) | b[..., 2]
        else:
            fk, bk = q256(s.fg), q256(s.bg)
        p = self.prev
        if p is None or p[0].shape != fk.shape:
            changed = np.ones(fk.shape, bool)
            parts = ["\x1b[0m\x1b[2J"]
        else:
            changed = (fk != p[0]) | (bk != p[1]) | (s.ch != p[2]) | (s.at != p[3])
            parts = []
        self.prev = (fk, bk, s.ch.copy(), s.at.copy())
        ys, xs = np.nonzero(changed)
        if not len(ys):
            return
        tc = self.truecolor
        lf = lb = la = None
        py, px = -1, -1
        for y, x, fv, bv, c, a in zip(ys.tolist(), xs.tolist(), fk[ys, xs].tolist(), bk[ys, xs].tolist(),
                                      s.ch[ys, xs].tolist(), s.at[ys, xs].tolist()):
            if y != py:
                parts.append(f"\x1b[{y + 1};{x + 1}H")
            elif x != px:
                parts.append(f"\x1b[{x - px}C" if x - px > 1 else f"\x1b[{x + 1}G")
            if a != la:                          # attribute change resets the colours too
                parts.append("\x1b[0" + (";1" if a & 1 else "") + (";3" if a & 2 else "") + "m")
                la, lf, lb = a, None, None
            if c == " ":
                fv = lf if lf is not None else fv  # a blank only needs its background
            if fv != lf:
                parts.append(f"\x1b[38;2;{fv >> 16};{(fv >> 8) & 255};{fv & 255}m" if tc else f"\x1b[38;5;{fv}m")
                lf = fv
            if bv != lb:
                parts.append(f"\x1b[48;2;{bv >> 16};{(bv >> 8) & 255};{bv & 255}m" if tc else f"\x1b[48;5;{bv}m")
                lb = bv
            parts.append(c)
            py, px = y, x + 1
        self.out.write("".join(parts))
        self.out.flush()
