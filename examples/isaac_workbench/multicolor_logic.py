from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

COLORS = ((220, 40, 40), (35, 165, 65), (40, 80, 220))
COLOR_NAMES = ("red", "green", "blue")


@dataclass(frozen=True)
class Stroke:
    color: int
    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        assert 0 <= self.color < len(COLORS)
        points = np.asarray(self.points)
        assert points.ndim == 2 and points.shape[0] >= 2 and points.shape[1] == 2
        assert np.isfinite(points).all() and ((points >= 0) & (points <= 1)).all()


@dataclass(frozen=True)
class PaintTask:
    seed: int
    strokes: tuple[Stroke, ...]
    rack_colors: tuple[int, ...]


def generate_task(seed: int, num_colors: int = 3) -> PaintTask:
    assert seed >= 0 and 1 <= num_colors <= 3
    rng = np.random.default_rng(seed)
    colors = rng.permutation(3)[:num_colors]
    strokes = []
    for i, color in enumerate(colors):
        center = np.array([0.5, (i + 1) / (num_colors + 1)])
        angle = rng.uniform(-0.3, 0.3)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        points = np.array([[-0.27, 0], [0, rng.uniform(-0.06, 0.06)], [0.27, 0]])
        points = points * rng.uniform(0.75, 1.1) @ rotation.T + center
        points += rng.uniform(-0.035, 0.035, size=2)
        strokes.append(Stroke(int(color), tuple(tuple(p) for p in points)))
    return PaintTask(seed, tuple(strokes), tuple(int(c) for c in rng.permutation(3)))


def render_strokes(strokes: tuple[Stroke, ...], size: int = 384, width: int = 9) -> Image.Image:
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    for stroke in strokes:
        points = [tuple(np.rint(np.asarray(p) * (size - 1)).astype(int)) for p in stroke.points]
        color = COLORS[stroke.color]
        draw.line(points, fill=color, width=width, joint="curve")
        for x, y in points:
            radius = width / 2
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return image


def color_iou(actual: Image.Image, target: Image.Image) -> tuple[float, np.ndarray]:
    assert actual.size == target.size
    actual_pixels, target_pixels = np.asarray(actual), np.asarray(target)
    assert actual_pixels.shape == target_pixels.shape and actual_pixels.shape[-1] == 3
    scores = np.zeros(3, dtype=np.float32)
    present = np.zeros(3, dtype=bool)
    for i, color in enumerate(COLORS):
        painted = np.all(actual_pixels == color, axis=-1)
        reference = np.all(target_pixels == color, axis=-1)
        union = np.count_nonzero(painted | reference)
        present[i] = union > 0
        scores[i] = np.count_nonzero(painted & reference) / union if union else 0
    return (float(scores[present].mean()) if present.any() else 1.0), scores


class ColorCanvas:
    def __init__(self, size: int = 384, width: int = 9) -> None:
        assert size >= 32 and 1 <= width < size
        self.size, self.width = size, width
        self.image = Image.new("RGB", (size, size), "white")
        self.last: list[tuple[int, int] | None] = [None] * 3
        self.contact_samples = 0

    def touch(self, color: int, uv: tuple[float, float], normal_force: float) -> bool:
        assert 0 <= color < 3 and np.isfinite([*uv, normal_force]).all()
        if normal_force <= 0.02 or not all(0 <= value <= 1 for value in uv):
            self.last[color] = None
            return False
        point = tuple(int(round(value * (self.size - 1))) for value in uv)
        draw = ImageDraw.Draw(self.image)
        if self.last[color] is not None:
            draw.line([self.last[color], point], fill=COLORS[color], width=self.width)
        radius = self.width / 2
        x, y = point
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=COLORS[color])
        self.last[color] = point
        self.contact_samples += 1
        return True

    def reset(self) -> None:
        self.image.paste("white", (0, 0, self.size, self.size))
        self.last = [None] * 3
        self.contact_samples = 0


class PaintSchedule:
    """Oracle subtask order; no robot actions or object poses are exposed to the actor.

    Parameters
    ----------
    task : PaintTask
        Generated reference program, used only by the environment and demonstration expert.
    """

    def __init__(self, task: PaintTask) -> None:
        self.task = task
        self.stroke_index = 0
        self.phase = "pick"

    @property
    def stroke(self) -> Stroke:
        return self.task.strokes[min(self.stroke_index, len(self.task.strokes) - 1)]

    @property
    def prompt(self) -> str:
        color = COLOR_NAMES[self.stroke.color]
        return {
            "pick": f"pick up the {color} pen from the rack",
            "draw": f"draw the {color} stroke shown in the reference image",
            "place": f"put the {color} pen back in its rack slot and release it",
            "done": "move the empty gripper away from the canvas",
        }[self.phase]

    @property
    def key(self) -> int:
        return self.stroke_index * 3 + {"pick": 0, "draw": 1, "place": 2, "done": 0}[self.phase]

    def advance(self, *, held: bool, drawn: bool, returned: bool) -> tuple[float, bool]:
        old_key = self.key
        milestone = 0.0
        if self.phase == "pick" and held:
            self.phase = "draw"
            milestone = 0.1
        elif self.phase == "draw" and drawn:
            self.phase = "place"
        elif self.phase == "place" and returned:
            milestone = 0.1
            self.stroke_index += 1
            self.phase = "done" if self.stroke_index == len(self.task.strokes) else "pick"
        return milestone, self.key != old_key
