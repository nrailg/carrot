from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Key:
    label: str
    qcode: str
    x: float
    y: float


def keyboard_layout() -> list[Key]:
    rows = [
        [(c, c) for c in "1234567890"] + [("-", "minus"), ("=", "equal")],
        [(c, c) for c in "qwertyuiop"] + [("[", "bracket_left"), ("]", "bracket_right")],
        [(c, c) for c in "asdfghjkl"] + [(";", "semicolon"), ("'", "apostrophe")],
        [(c, c) for c in "zxcvbnm"],
    ]
    rows[3] += [(".", "dot"), (",", "comma"), ("/", "slash"), ("BS", "backspace")]
    rows.append([("SPACE", "spc"), ("ENTER", "ret")])
    keys = [
        Key(label, code, 0.60 - r * 0.032, (len(row) - 1) * 0.016 - c * 0.032)
        for r, row in enumerate(rows)
        for c, (label, code) in enumerate(row)
    ]
    keys.extend([Key("SHIFT", "shift", 0.35, 0.30), Key("CTRL", "ctrl", 0.31, 0.30)])
    return keys


def typing_chord(char: str) -> tuple[str, bool]:
    """Return the physical base key and whether the expert must hold the Shift key.

    Parameters
    ----------
    char : str
        One US-layout character. This planner never emits guest HID events.

    Returns
    -------
    tuple[str, bool]
        Key label and physical Shift requirement.
    """
    shifted = dict(zip('!@#$%^&*()_+{}:"<>?', "1234567890-=[];',./", strict=True))
    if char in shifted:
        return shifted[char], True
    if "A" <= char <= "Z":
        return char.lower(), True
    return {"\n": "ENTER", " ": "SPACE", "\b": "BS"}.get(char, char), False


class KeySwitches:
    def __init__(self, count: int, press: float = 0.0015, release: float = 0.0006) -> None:
        assert 0 <= release < press, "key release threshold must be below press threshold"
        self.press = press
        self.release = release
        self.down = np.zeros(count, dtype=bool)

    def update(self, depth: np.ndarray, contact: np.ndarray) -> list[tuple[int, bool]]:
        assert depth.shape == contact.shape == self.down.shape
        assert np.isfinite(depth).all() and np.isfinite(contact).all()
        next_down = np.where(
            self.down, depth > self.release, (depth >= self.press) & (contact > 0.01)
        )
        events = [(int(i), bool(next_down[i])) for i in np.flatnonzero(next_down != self.down)]
        self.down = next_down
        return events

    def reset(self) -> list[tuple[int, bool]]:
        events = [(int(i), False) for i in np.flatnonzero(self.down)]
        self.down[:] = False
        return events


DEFAULT_STROKES = [
    [[0.20, 0.48], [0.20, 0.82], [0.75, 0.82], [0.75, 0.48]],
    [[0.12, 0.48], [0.47, 0.16], [0.83, 0.48], [0.12, 0.48]],
    [[0.39, 0.82], [0.39, 0.60], [0.56, 0.60], [0.56, 0.82]],
]


class PaintCanvas:
    def __init__(self, size: int = 384, width: int = 9) -> None:
        assert size >= 32 and 1 <= width < size
        self.size = size
        self.width = width
        self.image = Image.new("RGB", (size, size), "white")
        self.last: tuple[int, int] | None = None
        self.contact_samples = 0

    def touch(self, uv: tuple[float, float], normal_force: float) -> bool:
        assert np.isfinite([*uv, normal_force]).all()
        if normal_force <= 0.02 or not all(0 <= v <= 1 for v in uv):
            self.last = None
            return False
        point = tuple(int(round(v * (self.size - 1))) for v in uv)
        draw = ImageDraw.Draw(self.image)
        color = (30, 75, 155)
        radius = self.width / 2
        if self.last is not None:
            draw.line([self.last, point], fill=color, width=self.width)
        draw.ellipse(
            (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
            fill=color,
        )
        self.last = point
        self.contact_samples += 1
        return True

    def reset(self) -> None:
        self.image.paste("white", (0, 0, self.size, self.size))
        self.last = None
        self.contact_samples = 0

    def score(self, reference: Image.Image) -> float:
        assert reference.size == self.image.size
        painted = np.asarray(self.image).min(axis=2) < 200
        target = np.asarray(reference.convert("RGB")).min(axis=2) < 200
        union = np.count_nonzero(painted | target)
        return float(np.count_nonzero(painted & target) / union) if union else 1.0


def reference_image(strokes: list, size: int = 384, width: int = 9) -> Image.Image:
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    for stroke in strokes:
        points = np.asarray(stroke, dtype=float)
        assert points.ndim == 2 and points.shape[0] >= 2 and points.shape[1] == 2
        assert np.isfinite(points).all() and ((points >= 0) & (points <= 1)).all()
        pixels = [tuple(np.rint(point * (size - 1)).astype(int)) for point in points]
        draw.line(pixels, fill=(30, 75, 155), width=width, joint="curve")
        for x, y in pixels:
            r = width / 2
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(30, 75, 155))
    return image
