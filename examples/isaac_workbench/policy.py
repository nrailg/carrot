import numpy as np
from logic import typing_chord
from scene import CANVAS_TOP, CANVAS_X, CANVAS_Y, KEY_TOP, TIP_RADIUS, Workbench


class ScriptedPolicy:
    """Privileged waypoint baseline; never sends HID or changes canvas pixels directly."""

    def __init__(
        self, env: Workbench, program: str | None = None, strokes: list | None = None
    ) -> None:
        self.env = env
        self.waypoints = []
        self.expected_keys = []
        self.expected_chords = []
        if env.task == "computer":
            assert program is not None
            mapping = {key.label: i for i, key in enumerate(env.keys)}
            shift_index = mapping["SHIFT"]
            shift = env.keys[shift_index]
            shift_hover = np.array([shift.x, shift.y, KEY_TOP + TIP_RADIUS + 0.035])
            shift_press = shift_hover - [0, 0, 0.0385]
            park = np.array([0.35, 0.55, 0.30])
            for char in program:
                label, shifted = typing_chord(char)
                assert label in mapping, f"unsupported character: {char!r}"
                index = mapping[label]
                key = env.keys[index]
                hover = np.array([key.x, key.y, KEY_TOP + TIP_RADIUS + 0.035])
                press = np.array([key.x, key.y, KEY_TOP + TIP_RADIUS - 0.0035])
                self.waypoints.append((0, hover, 0.18, 12))
                if shifted:
                    self.expected_keys.append(shift_index)
                    self.expected_chords.append(index)
                    self.waypoints += [(1, shift_hover, 0.18, 12), (1, shift_press, 0.06, 12)]
                self.expected_keys.append(index)
                self.waypoints += [(0, press, 0.06, 12), (0, hover, 0.08, 10)]
                if shifted:
                    self.waypoints.append((1, shift_hover, 0.08, 12))
                    self.waypoints.append((1, park, 0.18, 12))
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
                self.waypoints += [
                    (0, points[0] + [0, 0, 0.035], 0.18, 12),
                    (0, points[0], 0.04, 12),
                ]
                self.waypoints += [(0, p, 0.075, 1) for p in points[1:]]
                self.waypoints += [(0, points[-1] + [0, 0, 0.035], 0.06, 12)]
        self.index = 0
        self.progress = 0
        self.start = None
        self.finished = False
        self.last_action = env.home.copy()
        self.targets = [None] * env.num_arms

    def action(self) -> np.ndarray:
        if self.finished:
            return self.last_action.copy()
        arm, target, speed, settle = self.waypoints[self.index]
        if self.start is None:
            self.start = self.env.tool_pose(arm)[0].copy()
            self.travel_steps = max(
                1, int(np.ceil(np.linalg.norm(target - self.start) / (speed / 60)))
            )
        self.progress += 1
        fraction = min(1, self.progress / self.travel_steps)
        self.targets[arm] = self.start + fraction * (target - self.start)
        for active_arm, position in enumerate(self.targets):
            if position is not None:
                self.last_action[9 * active_arm : 9 * (active_arm + 1)] = self.env.ik_action(
                    position, active_arm
                )
        if self.progress >= self.travel_steps + settle:
            self.index += 1
            self.start = None
            self.progress = 0
            self.finished = self.index == len(self.waypoints)
        return self.last_action.copy()
