import numpy as np
from scene import CANVAS_TOP, CANVAS_X, CANVAS_Y, KEY_TOP, TIP_RADIUS, Workbench


class ScriptedPolicy:
    """Privileged waypoint baseline; never sends HID or changes canvas pixels directly."""

    def __init__(
        self, env: Workbench, program: str | None = None, strokes: list | None = None
    ) -> None:
        self.env = env
        self.waypoints = []
        self.expected_keys = []
        if env.task == "computer":
            assert program is not None
            mapping = {key.label: i for i, key in enumerate(env.keys)}
            mapping.update({"\n": mapping["ENTER"], " ": mapping["SPACE"], "\b": mapping["BS"]})
            assert all(c in mapping for c in program), "program contains unsupported symbols"
            for char in program:
                index = mapping[char]
                self.expected_keys.append(index)
                key = env.keys[index]
                hover = np.array([key.x, key.y, KEY_TOP + TIP_RADIUS + 0.035])
                press = np.array([key.x, key.y, KEY_TOP + TIP_RADIUS - 0.0035])
                self.waypoints += [(hover, 0.18, 12), (press, 0.06, 12), (hover, 0.08, 10)]
        else:
            assert strokes is not None
            for stroke in strokes:
                points = [
                    np.array(
                        [
                            CANVAS_X[0] + u * (CANVAS_X[1] - CANVAS_X[0]),
                            CANVAS_Y[1] - v * (CANVAS_Y[1] - CANVAS_Y[0]),
                            CANVAS_TOP + TIP_RADIUS - 0.001,
                        ]
                    )
                    for u, v in stroke
                ]
                self.waypoints += [(points[0] + [0, 0, 0.035], 0.18, 12), (points[0], 0.04, 12)]
                self.waypoints += [(p, 0.075, 1) for p in points[1:]]
                self.waypoints += [(points[-1] + [0, 0, 0.035], 0.06, 12)]
        self.index = 0
        self.progress = 0
        self.start = None
        self.finished = False
        self.last_action = env.home.copy()

    def action(self) -> np.ndarray:
        if self.finished:
            return self.last_action.copy()
        target, speed, settle = self.waypoints[self.index]
        if self.start is None:
            self.start = self.env.tool_pose()[0].copy()
            self.travel_steps = max(
                1, int(np.ceil(np.linalg.norm(target - self.start) / (speed / 60)))
            )
        self.progress += 1
        fraction = min(1, self.progress / self.travel_steps)
        self.last_action = self.env.ik_action(self.start + fraction * (target - self.start))
        if self.progress >= self.travel_steps + settle:
            self.index += 1
            self.start = None
            self.progress = 0
            self.finished = self.index == len(self.waypoints)
        return self.last_action.copy()
