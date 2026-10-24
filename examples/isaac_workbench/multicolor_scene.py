from pathlib import Path
from typing import override

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.types import ArticulationActions
from isaacsim.sensors.camera import Camera
from multicolor_controls import EEF_OFFSET, EEF_ROTATION, command_target, libero_state
from multicolor_logic import (
    COLORS,
    ColorCanvas,
    PaintSchedule,
    PaintTask,
    color_iou,
    render_strokes,
)
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade
from scene import CANVAS_TOP, CANVAS_X, CANVAS_Y, DT, Surface, Workbench, cube
from scipy.spatial.transform import Rotation

PEN_TIP_OFFSET = np.array([0.0, 0.0, -0.066])
RACK_POSITIONS = np.array([[0.29, y, 0.23] for y in (-0.23, 0, 0.23)])
DOWN = Rotation.from_quat([1, 0, 0, 0])


class MulticolorWorkbench(Workbench):
    """Physical three-pen Panda scene with LIBERO-shaped Cartesian controls.

    Parameters
    ----------
    robot_usd : Path
    task : PaintTask
    world : World
        All vector slots share one simulator; only reset may set rigid body poses.
    index : int
    """

    def __init__(self, robot_usd: Path, task: PaintTask, world: World, index: int) -> None:
        super().__init__("paint", robot_usd, render_strokes(task.strokes), world, index, (224, 224))
        self.stage.RemovePrim(self.root + "/MonitorBody")
        self.stage.RemovePrim(self.root + "/Monitor")
        cube(
            self.stage,
            self.root + "/ReferenceBoardBody",
            (0.50, 0.38, 0.46),
            (0.42, 0.02, 0.34),
            (0.04, 0.05, 0.07),
        )
        self.monitor = Surface(
            self.stage,
            self.root + "/ReferenceBoard",
            [(0.30, 0.369, 0.62), (0.70, 0.369, 0.62), (0.70, 0.369, 0.30), (0.30, 0.369, 0.30)],
            render_strokes((task.strokes[0],)),
        )
        position, target = np.array([0.10, -0.85, 0.78]), np.array([0.43, 0, 0.30])
        back = position - target
        back /= np.linalg.norm(back)
        right = np.cross([0, 0, 1], back)
        right /= np.linalg.norm(right)
        up = np.cross(back, right)
        quaternion = Rotation.from_matrix(np.column_stack((right, up, back))).as_quat()
        self.camera_position = position + self.origin
        self.camera_orientation = quaternion[[3, 0, 1, 2]]
        hand = self.root + "/Robot0/panda_hand"
        self.stage.RemovePrim(hand + "/BrushHandle")
        self.stage.RemovePrim(hand + "/ToolTip")
        self.canvas = ColorCanvas()
        self.schedule = PaintSchedule(task)
        self.target = render_strokes(task.strokes)
        self.wrist = Camera(hand + "/WristCamera", resolution=(224, 224))
        self._pens()
        self.grip_command = -1.0
        self.held_ticks = np.zeros(3, dtype=int)
        self.return_ticks = np.zeros(3, dtype=int)
        self.ever_held = np.zeros(3, dtype=bool)
        self.drop_ticks = np.zeros(3, dtype=int)
        self.dropped = np.zeros(3, dtype=bool)
        self.controls = 0
        self.score = 0.0
        self.pick_count = 0
        self.return_count = 0

    def _pens(self) -> None:
        material = UsdShade.Material.Define(self.stage, self.root + "/PenMaterial")
        friction = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        friction.CreateStaticFrictionAttr(1.5)
        friction.CreateDynamicFrictionAttr(1.2)
        friction.CreateRestitutionAttr(0)
        tip_material = UsdShade.Material.Define(self.stage, self.root + "/TipMaterial")
        tip_friction = UsdPhysics.MaterialAPI.Apply(tip_material.GetPrim())
        tip_friction.CreateStaticFrictionAttr(0.12)
        tip_friction.CreateDynamicFrictionAttr(0.1)
        tip_friction.CreateRestitutionAttr(0)
        PhysxSchema.PhysxMaterialAPI.Apply(tip_material.GetPrim()).CreateFrictionCombineModeAttr(
            "min"
        )
        handles, tips = [], []
        for color in range(3):
            slot = self.schedule.task.rack_colors.index(color)
            position = RACK_POSITIONS[slot]
            path = f"{self.root}/Pen{color}"
            handle = UsdGeom.Xform.Define(self.stage, path)
            handle.AddTranslateOp().Set(Gf.Vec3d(*position))
            barrel = cube(
                self.stage,
                path + "/Barrel",
                (0, 0, 0),
                (0.014, 0.014, 0.12),
                tuple(np.asarray(COLORS[color]) / 255),
            )
            UsdPhysics.RigidBodyAPI.Apply(handle.GetPrim())
            UsdPhysics.MassAPI.Apply(handle.GetPrim()).CreateMassAttr(0.025)
            UsdShade.MaterialBindingAPI.Apply(barrel.GetPrim()).Bind(
                material, materialPurpose="physics"
            )
            tip_path = path + "Tip"
            tip = UsdGeom.Sphere.Define(self.stage, tip_path)
            tip.CreateRadiusAttr(0.004)
            tip.AddTranslateOp().Set(Gf.Vec3d(*(position + PEN_TIP_OFFSET)))
            tip.CreateDisplayColorAttr([Gf.Vec3f(*(np.asarray(COLORS[color]) / 255))])
            UsdPhysics.CollisionAPI.Apply(tip.GetPrim())
            UsdShade.MaterialBindingAPI.Apply(tip.GetPrim()).Bind(
                tip_material, materialPurpose="physics"
            )
            contact = PhysxSchema.PhysxCollisionAPI.Apply(tip.GetPrim())
            contact.CreateContactOffsetAttr(0.0005)
            contact.CreateRestOffsetAttr(0.0)
            UsdPhysics.RigidBodyAPI.Apply(tip.GetPrim())
            UsdPhysics.MassAPI.Apply(tip.GetPrim()).CreateMassAttr(0.002)
            joint = UsdPhysics.FixedJoint.Define(self.stage, path + "TipJoint")
            joint.CreateBody0Rel().SetTargets([path])
            joint.CreateBody1Rel().SetTargets([tip_path])
            # Joint frames are in unscaled rigid-body coordinates, not cube mesh coordinates.
            joint.CreateLocalPos0Attr(Gf.Vec3f(*PEN_TIP_OFFSET))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
            handles.append(path)
            tips.append(tip_path)
        for slot, position in enumerate(RACK_POSITIONS):
            cube(
                self.stage,
                f"{self.root}/Rack{slot}Base",
                (position[0], position[1], 0.145),
                (0.052, 0.052, 0.03),
                (0.25, 0.28, 0.3),
            )
            for side, (dx, dy, sx, sy) in enumerate(
                (
                    (-0.016, 0, 0.01, 0.042),
                    (0.016, 0, 0.01, 0.042),
                    (0, -0.016, 0.022, 0.01),
                    (0, 0.016, 0.022, 0.01),
                )
            ):
                cube(
                    self.stage,
                    f"{self.root}/Rack{slot}Side{side}",
                    (position[0] + dx, position[1] + dy, 0.1875),
                    (sx, sy, 0.055),
                    (0.25, 0.28, 0.3),
                )
        self.pens = self.world.scene.add(
            RigidPrim(handles, name=f"pens_{self.index}", reset_xform_properties=False)
        )
        self.tips = self.world.scene.add(
            RigidPrim(
                tips,
                name=f"tips_{self.index}",
                reset_xform_properties=False,
                track_contact_forces=True,
                contact_filter_prim_paths_expr=[[self.root + "/CanvasCollision"] for _ in tips],
                max_contact_count=256,
            )
        )
        fingers = [self.root + "/Robot0/panda_" + side + "finger" for side in ("left", "right")]
        assert all(self.stage.GetPrimAtPath(p).IsValid() for p in fingers)
        self.fingers = self.world.scene.add(
            RigidPrim(
                fingers,
                name=f"fingers_{self.index}",
                reset_xform_properties=False,
                track_contact_forces=True,
                contact_filter_prim_paths_expr=[handles, handles],
                max_contact_count=256,
            )
        )

    @override
    def initialize(self) -> None:
        super().initialize()
        self.camera.set_focal_length(0.021)
        assert self.robot.get_body_disable_gravity().all(), (
            "arm gravity compensation must be active"
        )
        self.robot.set_gains(
            kps=np.array([[800] * 7 + [1200, 1200]]),
            kds=np.array([[60] * 7 + [40, 40]]),
        )
        self.home[7:] = 0.04
        self.robot.set_joint_positions(self.home[None])
        self.robot.set_joint_position_targets(self.home[None])
        self.wrist.initialize()
        self.wrist.set_local_pose(
            np.array([0.06, 0, 0.025]), np.array([0, 1, 0, 0]), camera_axes="usd"
        )
        self.wrist.set_focal_length(0.012)
        self.wrist.set_horizontal_aperture(0.024)
        self.wrist.set_clipping_range(0.01, 2)

    def eef_pose(self) -> tuple[np.ndarray, Rotation]:
        positions, quaternions = self.hand.get_world_poses()
        rotation = Rotation.from_quat(quaternions[0, [1, 2, 3, 0]])
        return positions[0] + rotation.apply(EEF_OFFSET) - self.origin, rotation * EEF_ROTATION

    def command(self, action: np.ndarray) -> None:
        """Set a Cartesian goal held over twelve 240 Hz ticks (one 20 Hz action).

        Parameters
        ----------
        action : numpy.ndarray
            Shape (7,). Clipped input scales to 5 cm, 0.5 rad; positive grip closes.
        """
        position, rotation = self.eef_pose()
        self.goal_position, self.goal_rotation, self.grip_command = command_target(
            position, rotation, self.grip_command, action
        )

    def servo(self) -> None:
        position, rotation = self.eef_pose()
        error = np.concatenate(
            (self.goal_position - position, (self.goal_rotation * rotation.inv()).as_rotvec())
        )
        jacobian = self.robot.get_jacobians()[0, self.hand_index, :, :7].copy()
        jacobian[:3] += np.cross(jacobian[3:].T, rotation.apply(EEF_OFFSET)).T
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 0.02**2 * np.eye(6), error)
        targets = self.robot.get_joint_positions()[0].copy()
        targets[:7] += np.clip(delta, -0.04, 0.04)
        targets[7:] = 0.02 * (1 - self.grip_command)
        targets = np.clip(targets, self.limits[:, 0], self.limits[:, 1])
        self.robot.apply_action(ArticulationActions(joint_positions=targets[None]))

    @override
    def read_contacts(self) -> None:
        self.physics_step += 1
        forces = self.tips.get_contact_force_matrix(dt=DT)
        positions, _ = self.tips.get_world_poses()
        assert forces.shape == (3, 1, 3) and np.isfinite(forces).all()
        for color, (tip, force) in enumerate(
            zip(positions - self.origin, forces[:, 0], strict=True)
        ):
            uv = (
                (tip[0] - CANVAS_X[0]) / np.ptp(CANVAS_X),
                (CANVAS_Y[1] - tip[1]) / np.ptp(CANVAS_Y),
            )
            self.canvas.touch(color, uv, float(abs(force[2])))

    def finish_control(self) -> tuple[float, bool, bool, dict]:
        self.controls += 1
        positions, quaternions = self.pens.get_world_poses()
        positions = positions - self.origin
        tip_positions = self.tips.get_world_poses()[0] - self.origin
        rotation = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]])
        force = np.linalg.norm(self.fingers.get_contact_force_matrix(dt=DT), axis=-1)
        assert force.shape == (2, 3)
        rack = RACK_POSITIONS[[self.schedule.task.rack_colors.index(c) for c in range(3)]]
        eef, _ = self.eef_pose()
        near_hand = np.linalg.norm(positions - eef, axis=1) < 0.045
        grasped = (force > 0.02).all(axis=0) & near_hand
        lifted = grasped & (positions[:, 2] > 0.30)
        self.held_ticks = np.where(lifted, self.held_ticks + 1, 0)
        self.ever_held |= self.held_ticks >= 3
        upright = rotation.apply(np.tile([0, 0, 1], (3, 1)))[:, 2] > 0.95
        in_rack = (np.linalg.norm(positions - rack, axis=1) < 0.018) & upright
        slow = np.linalg.norm(self.pens.get_linear_velocities(), axis=1) < 0.05
        released = (force < 0.02).all(axis=0) & (np.linalg.norm(positions - eef, axis=1) > 0.065)
        returned = in_rack & slow & released
        self.return_ticks = np.where(returned, self.return_ticks + 1, 0)
        dropped = self.ever_held & ~grasped & ~in_rack & slow
        self.drop_ticks = np.where(dropped, self.drop_ticks + 1, 0)
        self.dropped |= self.drop_ticks >= 10
        score, scores = color_iou(self.canvas.image, self.target)
        color = self.schedule.stroke.color
        phase = self.schedule.phase
        milestone, changed = self.schedule.advance(
            held=bool(self.held_ticks[color] >= 3),
            drawn=bool(scores[color] >= 0.72 and tip_positions[color, 2] > CANVAS_TOP + 0.02),
            returned=bool(self.return_ticks[color] >= 3),
        )
        self.pick_count += int(phase == "pick" and changed)
        self.return_count += int(phase == "place" and changed)
        success = bool(
            self.schedule.phase == "done"
            and score >= 0.72
            and (self.return_ticks >= 3).all()
            and (self.robot.get_joint_positions()[0, 7:] > 0.03).all()
            and eef[2] > 0.30
        )
        failure = bool((positions[:, 2] < 0.12).any() or self.dropped.any())
        reward = score - self.score + milestone + float(success)
        self.score = score
        if changed:
            self.monitor.update(render_strokes((self.schedule.stroke,)))
        return (
            reward,
            success or failure,
            changed,
            {
                "seed": self.schedule.task.seed,
                "iou": score,
                "color_iou": scores.tolist(),
                "success": success,
                "picks": self.pick_count,
                "returns": self.return_count,
                "drops": int(self.dropped.sum()),
                "controls": self.controls,
                "phase": self.schedule.phase,
                "contact_samples": self.canvas.contact_samples,
            },
        )

    def actor_observation(self) -> dict:
        position, rotation = self.eef_pose()
        frame = self.wrist.get_rgba()
        assert frame.shape == (224, 224, 4), f"wrist camera not ready: {frame.shape}"
        return {
            "observation/image": self.rgb(),
            "observation/wrist_image": frame[:, :, :3].copy(),
            "observation/state": libero_state(
                position, rotation, self.robot.get_joint_positions()[0, 7:]
            ),
            "prompt": self.schedule.prompt,
        }

    def reset_task(self, task: PaintTask) -> None:
        self.reset_state()
        self.schedule = PaintSchedule(task)
        self.target = render_strokes(task.strokes)
        positions = RACK_POSITIONS[[task.rack_colors.index(c) for c in range(3)]] + self.origin
        quaternions = np.tile([1, 0, 0, 0], (3, 1))
        self.pens.set_world_poses(positions, quaternions)
        self.tips.set_world_poses(positions + PEN_TIP_OFFSET, quaternions)
        for bodies in (self.pens, self.tips):
            bodies.set_linear_velocities(np.zeros((3, 3)))
            bodies.set_angular_velocities(np.zeros((3, 3)))
        self.grip_command = -1.0
        self.held_ticks[:] = self.return_ticks[:] = self.drop_ticks[:] = 0
        self.ever_held[:] = self.dropped[:] = False
        self.controls = self.pick_count = self.return_count = 0
        self.score = 0.0
        self.monitor.update(render_strokes((self.schedule.stroke,)))


class PaintExpert:
    def __init__(self, env: MulticolorWorkbench, *, event_driven: bool = False) -> None:
        self.env = env
        self.event_driven = event_driven
        self.key = -1
        self.waypoints: list[tuple[np.ndarray, float, int]] = []
        self.index, self.dwell = 0, 0
        self.release_position: np.ndarray | None = None
        self.release_rotation: Rotation | None = None

    def action(self) -> np.ndarray:
        env = self.env
        eef, rotation = env.eef_pose()
        if self.key != env.schedule.key:
            self.key = env.schedule.key
            self.index = self.dwell = 0
            self.release_position = None
            self.release_rotation = None
            color = env.schedule.stroke.color
            rack = RACK_POSITIONS[env.schedule.task.rack_colors.index(color)]
            rack = rack + [0, 0, 0.025]
            above = rack + [0, 0, 0.13]
            if env.schedule.phase == "pick":
                self.waypoints = [
                    (above, -1, 220),
                    (rack.copy(), -1, 1),
                    (rack.copy(), 1, 220),
                    (above, 1, 8),
                ]
            elif env.schedule.phase == "place":
                self.waypoints = [
                    (np.array([eef[0], eef[1], 0.38]), 1, 1),
                    (above, 1, 1),
                    (rack.copy(), 1, 8),
                    (rack.copy(), -1, 220),
                    (above, -1, 8),
                ]
            elif env.schedule.phase == "draw":
                points = [
                    np.array(
                        [
                            CANVAS_X[0] + x * np.ptp(CANVAS_X),
                            CANVAS_Y[1] - y * np.ptp(CANVAS_Y),
                            CANVAS_TOP + 0.0035,
                        ]
                    )
                    for x, y in env.schedule.stroke.points
                ]
                path = [points[0]]
                for start, end in zip(points[:-1], points[1:], strict=True):
                    count = max(2, int(np.linalg.norm(end - start) / 0.0015))
                    path.extend(np.linspace(start, end, count)[1:])
                self.waypoints = [(points[0] + [0, 0, 0.06], 1, 1)]
                self.waypoints += [(p, 1, 1) for p in path]
                self.waypoints += [(points[-1] + [0, 0, 0.07], 1, 8)]
            else:
                self.waypoints = [(np.array([0.4, 0, 0.4]), -1, 220)]
        target, grip, ticks = self.waypoints[self.index]
        goal_rotation = DOWN
        tilt = 0.0
        if env.schedule.phase == "draw":
            color = env.schedule.stroke.color
            tip = env.tips.get_world_poses()[0][color] - env.origin
            target = target + eef - tip
        elif env.schedule.phase == "place":
            if self.index <= 2:
                positions, quaternions = env.pens.get_world_poses()
                color = env.schedule.stroke.color
                pen_rotation = Rotation.from_quat(quaternions[color, [1, 2, 3, 0]])
                axis = pen_rotation.apply([0, 0, 1])
                correction = np.cross(axis, [0, 0, 1])
                sine = np.linalg.norm(correction)
                tilt = np.arctan2(sine, axis[2])
                if sine > 1e-8:
                    goal_rotation = Rotation.from_rotvec(correction * (tilt / sine)) * rotation
                else:
                    goal_rotation = rotation
            if self.index in (1, 2):
                pen = positions[color] - env.origin
                target = target + eef - pen - [0, 0, 0.025]
            elif self.index == 3:
                if self.release_position is None:
                    self.release_position = eef.copy()
                    self.release_rotation = rotation
                target = self.release_position
                goal_rotation = self.release_rotation
        distance = np.linalg.norm(target - eef)
        tolerance = 0.006 if env.schedule.phase in ("pick", "place") else 0.001
        if self.event_driven:
            fingers = env.robot.get_joint_positions()[0, 7:]
            color = env.schedule.stroke.color
            forces = np.linalg.norm(env.fingers.get_contact_force_matrix(dt=DT), axis=-1)
            ready = distance < tolerance and tilt < 0.04
            if env.schedule.phase == "pick":
                if self.index <= 1:
                    ready &= bool((fingers > 0.035).all())
                elif self.index == 2:
                    ready &= bool((forces[:, color] > 0.02).all())
            elif env.schedule.phase == "place" and self.index == 3:
                ready &= bool((fingers > 0.035).all() and (forces[:, color] < 0.02).all())
            if ready and self.index < len(self.waypoints) - 1:
                self.index += 1
        elif distance < tolerance and tilt < 0.04:
            self.dwell += 1
            if self.dwell >= ticks and self.index < len(self.waypoints) - 1:
                self.index += 1
                self.dwell = 0
        speed = 0.1 if env.schedule.phase == "place" else 0.3
        translation = np.clip((target - eef) / 0.05, -speed, speed)
        rotvec = (goal_rotation * rotation.inv()).as_rotvec() / 0.5
        return np.concatenate((translation, np.clip(rotvec, -0.2, 0.2), [grip])).astype(np.float32)
