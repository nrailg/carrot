"""Create an independent USD reference without editing the downloaded scene."""

import argparse
import json
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


def extract_fixture(
    source: Path, prim_path: str, output: Path, *, static_body: bool = False
) -> dict:
    """Reference one complete fixture subtree in meters at the origin.

    Parameters
    ----------
    source : Path
        Local scene, including its referenced assets and textures.
    prim_path : str
        Complete fixture subtree containing its rigid bodies and joints.
    output : Path
        New USD file. Existing output is rejected.
    static_body : bool
        Wrap a static collision subtree in a kinematic rigid body.

    Returns
    -------
    dict
        Source provenance and output rigid-body / joint paths.
    """
    if output.exists():
        raise FileExistsError(output)
    source = source.resolve(strict=True)
    source_stage = Usd.Stage.Open(str(source))
    if not source_stage.GetPrimAtPath(prim_path):
        raise ValueError(f"Missing fixture prim {prim_path} in {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/Asset", "Xform")
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source_stage))
    root.GetReferences().AddReference(str(source), prim_path)
    root.SetActive(True)
    transform = UsdGeom.Xformable(root)
    transform.ClearXformOpOrder()
    transform.AddTransformOp().Set(Gf.Matrix4d(1.0))
    if static_body:
        if any(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in stage.Traverse()):
            raise ValueError("Static wrapper cannot contain existing rigid bodies")
        if not any(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in stage.Traverse()):
            raise ValueError("Static wrapper requires collision geometry")
        UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    bodies, joints = [], []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies.append(str(prim.GetPath()))
        if prim.IsA(UsdPhysics.Joint):
            joints.append(str(prim.GetPath()))
            joint = UsdPhysics.Joint(prim)
            if prim.IsA(UsdPhysics.FixedJoint) and not joint.GetBody0Rel().GetTargets():
                # The source scene's world anchor must not survive fixture relocation.
                joint.GetLocalPos0Attr().Set(Gf.Vec3f(0.0))
                joint.GetLocalRot0Attr().Set(Gf.Quatf(1.0))
            for rel_name in ("physics:body0", "physics:body1"):
                for target in prim.GetRelationship(rel_name).GetTargets():
                    if not target.HasPrefix(Sdf.Path("/Asset")):
                        raise ValueError(f"Fixture has external body reference: {target}")
    if not bodies:
        raise ValueError(f"No rigid bodies in {prim_path}")
    stage.GetRootLayer().Export(str(output))
    source_transform = Gf.Transform(
        UsdGeom.Xformable(source_stage.GetPrimAtPath(prim_path)).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
    )
    source_quat = source_transform.GetRotation().GetQuat()
    result = {
        "source_position": list(source_transform.GetTranslation()),
        "source_rotation_xyzw": [*source_quat.GetImaginary(), source_quat.GetReal()],
        "source_scale": list(source_transform.GetScale()),
        "source": str(source),
        "prim": prim_path,
        "output": str(output),
        "bodies": bodies,
        "joints": joints,
    }
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prim", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(extract_fixture(args.source, args.prim, args.output), indent=2))
