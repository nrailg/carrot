"""Replace box-shaped collision meshes with equivalent USD cubes in a reference overlay."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _xyz(point) -> tuple[float, float, float]:
    return tuple(float(value) for value in point)


def _bbox(points: list[tuple[float, float, float]]) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(function(point[axis] for point in points) for axis in range(3))
        for function in (min, max)
    )


def _corners(minimum: tuple[float, ...], maximum: tuple[float, ...]):
    return list(itertools.product(*zip(minimum, maximum, strict=True)))


def _require_box(points: list[tuple[float, float, float]], tolerance: float):
    minimum, maximum = _bbox(points)
    if any(high - low <= tolerance for low, high in zip(minimum, maximum, strict=True)):
        raise ValueError("Collider box must have positive extent on all axes")
    expected = _corners(minimum, maximum)
    if len(points) != 8 or any(
        not any(
            max(abs(a - b) for a, b in zip(point, corner, strict=True)) <= tolerance
            for corner in expected
        )
        for point in points
    ):
        raise ValueError("Collision mesh is not exactly the eight corners of a box")
    return minimum, maximum


def normalize_box_colliders(source: Path, output: Path, tolerance: float = 1e-6) -> dict:
    """Author an overlay that preserves a source asset but replaces exact box meshes with cubes."""
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    source = source.resolve(strict=True)
    source_stage = Usd.Stage.Open(str(source))
    source_root = source_stage.GetDefaultPrim()
    if not source_root:
        raise ValueError(f"Source USD has no default prim: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/Asset", "Xform")
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source_stage))
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source_stage))
    root.GetReferences().AddReference(str(source))

    candidates = [
        prim
        for prim in stage.Traverse()
        if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if not candidates:
        raise ValueError(f"No collision meshes found in {source}")

    converted = []
    for composed_prim in candidates:
        mesh = UsdGeom.Mesh(composed_prim)
        points = [_xyz(point) for point in mesh.GetPointsAttr().Get()]
        try:
            minimum, maximum = _require_box(points, tolerance)
        except ValueError as error:
            raise ValueError(f"{composed_prim.GetPath()}: {error}") from error
        original_local = UsdGeom.Xformable(composed_prim).GetLocalTransformation()
        center = tuple((low + high) / 2 for low, high in zip(minimum, maximum, strict=True))
        half = tuple((high - low) / 2 for low, high in zip(minimum, maximum, strict=True))
        box_local = Gf.Matrix4d(1.0)
        box_local.SetScale(Gf.Vec3d(*half))
        box_local.SetTranslateOnly(Gf.Vec3d(*center))
        replacement_local = box_local * original_local
        expected_parent = [_xyz(original_local.Transform(Gf.Vec3d(*point))) for point in points]
        actual_parent = [
            _xyz(replacement_local.Transform(Gf.Vec3d(*corner)))
            for corner in itertools.product((-1.0, 1.0), repeat=3)
        ]
        expected_bbox = _bbox(expected_parent)
        actual_bbox = _bbox(actual_parent)
        error = max(
            abs(expected - actual)
            for expected_side, actual_side in zip(expected_bbox, actual_bbox, strict=True)
            for expected, actual in zip(expected_side, actual_side, strict=True)
        )
        if error > tolerance:
            raise AssertionError(
                f"Cube replacement changed bounds for {composed_prim.GetPath()}: {error}"
            )

        collision_enabled = UsdPhysics.CollisionAPI(composed_prim).GetCollisionEnabledAttr().Get()
        prim = stage.OverridePrim(composed_prim.GetPath())
        prim.SetTypeName("Cube")
        cube = UsdGeom.Cube(prim)
        cube.CreateSizeAttr(2.0)
        transform = UsdGeom.Xformable(prim)
        transform.ClearXformOpOrder()
        transform.AddTransformOp().Set(replacement_local)
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        collision = UsdPhysics.CollisionAPI.Apply(prim)
        collision.CreateCollisionEnabledAttr(
            True if collision_enabled is None else bool(collision_enabled)
        )
        converted.append(
            {
                "path": str(prim.GetPath()),
                "point_bbox_local": [minimum, maximum],
                "cube_center_local": center,
                "cube_half_size_local": half,
                "parent_bbox_before": expected_bbox,
                "parent_bbox_after": actual_bbox,
                "max_bbox_error": error,
            }
        )

    stage.GetRootLayer().Export(str(output))
    result = {
        "source": str(source),
        "source_sha256": _sha256(source),
        "output": str(output.resolve()),
        "tolerance": tolerance,
        "converted": converted,
    }
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    print(json.dumps(normalize_box_colliders(args.source, args.output, args.tolerance), indent=2))
