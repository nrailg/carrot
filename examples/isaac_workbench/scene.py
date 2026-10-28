from pathlib import Path

import numpy as np
import omni.ui as ui
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, RigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.types import ArticulationActions
from isaacsim.sensors.camera import Camera
from logic import KeySwitches, PaintCanvas, keyboard_layout
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation

DT = 1 / 240
TIP_OFFSET = np.array([0.0, 0.0, 0.16])
TIP_RADIUS = 0.004
KEY_TOP = 0.165
CANVAS_TOP = 0.15
CANVAS_X = (0.37, 0.65)
CANVAS_Y = (-0.17, 0.17)


def cube(
    stage: Usd.Stage, path: str, position: tuple, size: tuple, color: tuple, collision: bool = True
) -> UsdGeom.Cube:
    shape = UsdGeom.Cube.Define(stage, path)
    shape.CreateSizeAttr(1)
    shape.AddTranslateOp().Set(Gf.Vec3d(*position))
    shape.AddScaleOp().Set(Gf.Vec3f(*size))
    shape.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collision:
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        contact = PhysxSchema.PhysxCollisionAPI.Apply(shape.GetPrim())
        contact.CreateContactOffsetAttr(0.001)
        contact.CreateRestOffsetAttr(0.0)
    return shape


class Surface:
    def __init__(
        self, stage: Usd.Stage, path: str, corners: list, image: Image.Image, emissive: bool = False
    ) -> None:
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in corners])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateDoubleSidedAttr(True)
        uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        )
        uv.Set([(0, 1), (1, 1), (1, 0), (0, 0)])
        material = UsdShade.Material.Define(stage, path + "Material")
        shader = UsdShade.Shader.Define(stage, path + "Material/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        texture = UsdShade.Shader.Define(stage, path + "Material/Texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath("dynamic://" + path.replace("/", "_"))
        )
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        reader = UsdShade.Shader.Define(stage, path + "Material/UV")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            texture.ConnectableAPI(), "rgb"
        )
        if emissive:
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                texture.ConnectableAPI(), "rgb"
            )
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        self.provider = ui.DynamicTextureProvider(path.replace("/", "_"))
        self.update(image)

    def update(self, image: Image.Image) -> None:
        pixels = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        self.provider.set_bytes_data(pixels.ravel().tolist(), list(image.size))


class Workbench:
    """Isaac workbench with one painting arm or two typing arms.

    Parameters
    ----------
    task : str
        computer or paint; a fresh instance owns its scene and contact state.
    robot_usd : Path
        Locally resolved Isaac Lab FrankaEmika/panda_instanceable.usd asset.
    reference : PIL.Image.Image
        Image displayed on the monitor; paint target or guest framebuffer.
    """

    def __init__(
        self,
        task: str,
        robot_usd: Path,
        reference: Image.Image,
        world: World | None = None,
        index: int = 0,
        resolution: tuple[int, int] = (960, 720),
    ) -> None:
        assert task in ("computer", "paint") and robot_usd.is_file()
        self.task = task
        self.world = (
            world
            if world is not None
            else World(physics_dt=DT, rendering_dt=1 / 60, stage_units_in_meters=1.0)
        )
        self.stage = get_current_stage()
        self.index = index
        self.root = f"/World/Env{index}"
        self.origin = np.array([0, index * 3.5, 0])
        self.num_arms = 2 if task == "computer" else 1
        self.action_dim = self.num_arms * 9
        self.resolution = resolution
        root = UsdGeom.Xform.Define(self.stage, self.root)
        root.AddTranslateOp().Set(Gf.Vec3d(*self.origin))
        cube(self.stage, self.root + "/Floor", (0, 0, -0.025), (1.5, 1.8, 0.05), (0.21, 0.25, 0.29))
        print("WORKBENCH scene: world created", flush=True)
        light = UsdLux.DomeLight.Define(self.stage, "/World/Light")
        light.CreateIntensityAttr(1100)
        cube(
            self.stage, self.root + "/Desk", (0.50, 0, 0.08), (0.65, 0.65, 0.10), (0.12, 0.17, 0.22)
        )
        cube(
            self.stage,
            self.root + "/MonitorBody",
            (0.82, 0, 0.40),
            (0.025, 0.52, 0.34),
            (0.04, 0.05, 0.07),
        )
        self.monitor = Surface(
            self.stage,
            self.root + "/Monitor",
            [
                (0.805, 0.245, 0.55),
                (0.805, -0.245, 0.55),
                (0.805, -0.245, 0.25),
                (0.805, 0.245, 0.25),
            ],
            reference,
            emissive=True,
        )
        robot_paths = [f"{self.root}/Robot{arm}" for arm in range(self.num_arms)]
        hand_paths = [path + "/panda_hand" for path in robot_paths]
        for arm, (path, hand_path) in enumerate(zip(robot_paths, hand_paths, strict=True)):
            add_reference_to_stage(str(robot_usd.resolve()), path)
            robot_prim = self.stage.GetPrimAtPath(path)
            if arm:
                UsdGeom.Xformable(robot_prim).AddTranslateOp(opSuffix="baseOffset").Set(
                    Gf.Vec3d(0, 0.65, 0)
                )
            assert self.stage.GetPrimAtPath(hand_path).IsValid(), "expected Panda hand body missing"
            rod = UsdGeom.Cylinder.Define(self.stage, hand_path + "/BrushHandle")
            rod.CreateRadiusAttr(0.005)
            rod.CreateHeightAttr(0.12)
            rod.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.10))
            rod.CreateDisplayColorAttr([Gf.Vec3f(0.65, 0.26, 0.08)])
            tip = UsdGeom.Sphere.Define(self.stage, hand_path + "/ToolTip")
            tip.CreateRadiusAttr(TIP_RADIUS)
            tip.AddTranslateOp().Set(Gf.Vec3d(*TIP_OFFSET))
            tip.CreateDisplayColorAttr([Gf.Vec3f(0.03, 0.10, 0.24)])
            UsdPhysics.CollisionAPI.Apply(tip.GetPrim())
            PhysxSchema.PhysxCollisionAPI.Apply(tip.GetPrim()).CreateContactOffsetAttr(0.0005)
            PhysxSchema.PhysxCollisionAPI(tip.GetPrim()).CreateRestOffsetAttr(0.0)
        print("WORKBENCH scene: robot loaded", flush=True)
        for prim in self.stage.Traverse():
            if any(
                prim.GetPath().HasPrefix(Sdf.Path(path)) for path in robot_paths
            ) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)
        self.robot = self.world.scene.add(Articulation(robot_paths, name=f"robot_{index}"))
        self.keys = keyboard_layout()
        self.switches = KeySwitches(len(self.keys))
        self.canvas = PaintCanvas()
        self.key_view = None
        self.canvas_surface = None
        if task == "computer":
            self._keyboard()
            filters = [f"{self.root}/Keys/Key{i:02d}" for i in range(len(self.keys))]
        else:
            cube(
                self.stage,
                self.root + "/CanvasCollision",
                (0.51, 0, 0.14),
                (0.28, 0.34, 0.02),
                (1, 1, 1),
            )
            self.canvas_surface = Surface(
                self.stage,
                self.root + "/Canvas",
                [
                    (CANVAS_X[0], CANVAS_Y[1], CANVAS_TOP + 0.0001),
                    (CANVAS_X[1], CANVAS_Y[1], CANVAS_TOP + 0.0001),
                    (CANVAS_X[1], CANVAS_Y[0], CANVAS_TOP + 0.0001),
                    (CANVAS_X[0], CANVAS_Y[0], CANVAS_TOP + 0.0001),
                ],
                self.canvas.image,
            )
            filters = [self.root + "/CanvasCollision"]
        self.hand = self.world.scene.add(
            RigidPrim(
                hand_paths,
                name=f"hand_{index}",
                reset_xform_properties=False,
                track_contact_forces=True,
                contact_filter_prim_paths_expr=[filters for _ in hand_paths],
                max_contact_count=512,
            )
        )
        position = np.array([0.12, -1.05, 0.85]) if task == "paint" else np.array([0.0, -1.15, 1.1])
        target = np.array([0.45, 0, 0.24]) if task == "paint" else np.array([0.4, 0.15, 0.25])
        back = position - target
        back /= np.linalg.norm(back)
        right = np.cross([0, 0, 1], back)
        right /= np.linalg.norm(right)
        up = np.cross(back, right)
        quat = Rotation.from_matrix(np.column_stack((right, up, back))).as_quat()
        self.camera_position = position + self.origin
        self.camera_orientation = quat[[3, 0, 1, 2]]
        self.camera = Camera(self.root + "/Camera", resolution=resolution)
        if world is None:
            self.world.reset()
            self.initialize()
            for _ in range(60):
                self.world.step(render=True)

    def initialize(self) -> None:
        self.camera.initialize()
        self.camera.set_world_pose(self.camera_position, self.camera_orientation, camera_axes="usd")
        # Camera setters use stage units (metres), including both focal length and sensor width.
        self.camera.set_focal_length(0.016)
        self.camera.set_horizontal_aperture(0.024)
        self.camera.set_clipping_range(0.01, 2.4 if self.task == "computer" else 1.85)
        self.home = np.tile([0, -0.569, 0, -2.810, 0, 3.037, 0.741, 0.008, 0.008], self.num_arms)
        assert self.robot.dof_names == [f"panda_joint{i}" for i in range(1, 8)] + [
            "panda_finger_joint1",
            "panda_finger_joint2",
        ], self.robot.dof_names
        self.robot.set_joint_positions(self.home.reshape(self.num_arms, 9))
        self.robot.set_joint_position_targets(self.home.reshape(self.num_arms, 9))
        self.robot.set_gains(
            kps=np.tile([800] * 7 + [300, 300], (self.num_arms, 1)),
            kds=np.tile([60] * 7 + [20, 20], (self.num_arms, 1)),
        )
        self.limits = self.robot.get_dof_limits()[0]
        self.hand_index = self.robot.body_names.index("panda_hand") - 1
        self.physics_step = 0
        self.events: list[tuple[int, int, bool]] = []
        self.contact_events: list[dict] = []
        self.chord_samples: list[dict] = []
        self.shift_index = next(i for i, key in enumerate(self.keys) if key.label == "SHIFT")
        self.arm_key_forces = np.zeros((self.num_arms, len(self.keys)))
        self.samples: list[dict] = []
        self.last_force = 0.0
        self.last_depth = np.zeros(len(self.keys))

    def reset_state(self) -> None:
        self.robot.set_joint_positions(self.home.reshape(self.num_arms, 9))
        self.robot.set_joint_velocities(np.zeros((self.num_arms, 9)))
        self.robot.set_joint_position_targets(self.home.reshape(self.num_arms, 9))
        if self.key_view is not None:
            positions = np.array([[key.x, key.y, KEY_TOP - 0.005] for key in self.keys])
            orientations = np.tile([1.0, 0.0, 0.0, 0.0], (len(self.keys), 1))
            self.key_view.set_world_poses(positions + self.origin, orientations)
            self.key_view.set_linear_velocities(np.zeros_like(positions))
            self.key_view.set_angular_velocities(np.zeros_like(positions))
        self.switches.reset()
        self.canvas.reset()
        self.last_force = 0.0
        self.last_depth[:] = 0
        self.physics_step = 0
        self.events.clear()
        self.contact_events.clear()
        self.chord_samples.clear()
        self.arm_key_forces[:] = 0
        self.samples.clear()
        if self.task == "paint":
            self.update_canvas()

    def _keyboard(self) -> None:
        cube(
            self.stage,
            self.root + "/KeyboardBase",
            (0.48, 0.06, 0.14),
            (0.38, 0.52, 0.012),
            (0.07, 0.08, 0.1),
        )
        paths = []
        for i, key in enumerate(self.keys):
            path = f"{self.root}/Keys/Key{i:02d}"
            paths.append(path)
            body = UsdGeom.Xform.Define(self.stage, path)
            body.AddTranslateOp().Set(Gf.Vec3d(key.x, key.y, KEY_TOP - 0.005))
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(0.01)
            PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim()).CreateDisableGravityAttr(True)
            cube(self.stage, path + "/Cap", (0, 0, 0), (0.027, 0.027, 0.010), (0.09, 0.12, 0.16))
            label = Image.new("RGB", (96, 96), (24, 31, 41))
            draw = ImageDraw.Draw(label)
            font = ImageFont.load_default(size=30 if len(key.label) < 3 else 18)
            draw.text((48, 48), key.label, font=font, anchor="mm", fill="white")
            # Keep providers alive: the renderer owns the URI, not the Python upload provider.
            surface = Surface(
                self.stage,
                path + "/Label",
                [
                    (-0.013, 0.013, 0.0051),
                    (-0.013, -0.013, 0.0051),
                    (0.013, -0.013, 0.0051),
                    (0.013, 0.013, 0.0051),
                ],
                label,
            )
            if i == 0:
                self.key_surfaces = []
            self.key_surfaces.append(surface)
            joint = UsdPhysics.PrismaticJoint.Define(self.stage, path + "/Spring")
            joint.CreateBody1Rel().SetTargets([path])
            joint.CreateAxisAttr("Z")
            joint.CreateLocalPos0Attr(
                Gf.Vec3f(*(np.array([key.x, key.y, KEY_TOP - 0.005]) + self.origin))
            )
            joint.CreateLowerLimitAttr(-0.004)
            joint.CreateUpperLimitAttr(0)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(150)
            drive.CreateDampingAttr(1)
            drive.CreateTargetPositionAttr(0)
        self.key_view = self.world.scene.add(
            RigidPrim(paths, name=f"keys_{self.index}", reset_xform_properties=False)
        )

    def tool_pose(self, arm: int = 0) -> tuple[np.ndarray, np.ndarray]:
        positions, orientations = self.hand.get_world_poses()
        quat = orientations[arm]
        matrix = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        return positions[arm] + matrix @ TIP_OFFSET - self.origin, matrix

    def ik_action(self, target: np.ndarray, arm: int = 0) -> np.ndarray:
        tip, matrix = self.tool_pose(arm)
        desired = Rotation.from_quat([1, 0, 0, 0]).as_matrix()
        error = np.concatenate((target - tip, Rotation.from_matrix(desired @ matrix.T).as_rotvec()))
        jacobian = self.robot.get_jacobians()[arm, self.hand_index, :, :7].copy()
        offset = matrix @ TIP_OFFSET
        jacobian[:3] += np.cross(jacobian[3:].T, offset).T
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 0.002**2 * np.eye(6), error)
        action = self.robot.get_joint_positions()[arm].copy()
        action[:7] += np.clip(delta, -0.04, 0.04)
        action[7:] = 0.008
        return np.clip(action, self.limits[:, 0] + 0.001, self.limits[:, 1] - 0.001)

    def step(self, action: np.ndarray) -> dict:
        """Advance 4 physics ticks; return proprioception, tool pose and fresh contact state.

        Parameters
        ----------
        action : numpy.ndarray
            Shape (18,) for computer, (9,) for paint; each arm has 7 radian and 2 metre targets.

        Returns
        -------
        dict
            RGB is available separately through rgb(); no OS text or target key is observed.
        """
        self.apply(action)
        for tick in range(4):
            self.world.step(render=tick == 3)
            self.read_contacts()
        return self.observation(action)

    def apply(self, action: np.ndarray) -> None:
        assert action.shape == (self.action_dim,) and np.isfinite(action).all()
        targets = action.reshape(self.num_arms, 9)
        assert ((targets >= self.limits[:, 0]) & (targets <= self.limits[:, 1])).all()
        self.robot.apply_action(ArticulationActions(joint_positions=targets))

    def read_contacts(self) -> None:
        self.physics_step += 1
        forces = self.hand.get_contact_force_matrix(dt=DT)
        assert np.isfinite(forces).all()
        tip, _ = self.tool_pose()
        self.last_force = float(np.abs(forces[:, :, 2]).sum())
        if self.task == "computer":
            positions, _ = self.key_view.get_world_poses()
            self.last_depth = KEY_TOP - 0.005 - positions[:, 2]
            self.arm_key_forces = np.linalg.norm(forces, axis=2)
            for key, down in self.switches.update(self.last_depth, self.arm_key_forces.sum(axis=0)):
                self.events.append((self.physics_step, key, down))
                self.contact_events.append(
                    {
                        "physics_step": self.physics_step,
                        "key": key,
                        "down": down,
                        "depth": float(self.last_depth[key]),
                        "arm_forces": self.arm_key_forces[:, key].tolist(),
                    }
                )
            shift = self.shift_index
            if self.switches.down[shift] and self.arm_key_forces[1, shift] > 0.01:
                for key in np.flatnonzero(self.switches.down):
                    if key != shift and self.arm_key_forces[0, key] > 0.01:
                        self.chord_samples.append(
                            {
                                "physics_step": self.physics_step,
                                "key": int(key),
                                "character_force": float(self.arm_key_forces[0, key]),
                                "shift_force": float(self.arm_key_forces[1, shift]),
                            }
                        )
        else:
            uv = (
                (tip[0] - CANVAS_X[0]) / (CANVAS_X[1] - CANVAS_X[0]),
                (CANVAS_Y[1] - tip[1]) / (CANVAS_Y[1] - CANVAS_Y[0]),
            )
            self.canvas.touch(uv, self.last_force)

    def observation(self, action: np.ndarray) -> dict:
        tip, _ = self.tool_pose()
        sample = {
            "joint_positions": self.robot.get_joint_positions().reshape(-1).copy(),
            "action": action.copy(),
            "tool_position": tip.copy(),
            "normal_force": self.last_force,
            "physics_step": self.physics_step,
        }
        self.samples.append(sample)
        return sample

    def rgb(self) -> np.ndarray:
        frame = self.camera.get_rgba()
        assert frame.shape == (self.resolution[1], self.resolution[0], 4), (
            f"camera not ready: {frame.shape}"
        )
        return frame[:, :, :3].copy()

    def update_canvas(self) -> None:
        assert self.canvas_surface is not None
        self.canvas_surface.update(self.canvas.image)

    def close(self) -> None:
        self.world.stop()
