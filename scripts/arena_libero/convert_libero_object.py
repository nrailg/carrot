import argparse
import hashlib
import json
import math
import shutil
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def convert_object(source: Path, output: Path) -> dict:
    """Convert a LIBERO textured OBJ and its MJCF collision boxes to native USD.

    Parameters
    ----------
    source : Path
        Object MJCF with one visual mesh and box collision geoms.
    output : Path
        New USD path; the texture is copied beside it. Existing files are rejected.

    Returns
    -------
    dict
        Input hashes, metric visual bounds and collision count.
    """
    if output.exists():
        raise FileExistsError(output)
    xml = ET.parse(source).getroot()
    meshes = xml.findall("asset/mesh")
    if len(meshes) != 1:
        raise ValueError("Expected one visual mesh")
    mesh_xml = meshes[0]
    scale = tuple(map(float, mesh_xml.attrib["scale"].split()))
    if len(scale) != 3:
        raise ValueError("Expected three mesh scale components")
    obj_path = source.with_suffix(".obj")
    points, texcoords, indices, face_counts, uv_indices = [], [], [], [], []
    for line in obj_path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            points.append(
                tuple(
                    float(value) * factor for value, factor in zip(fields[1:4], scale, strict=True)
                )
            )
        elif fields[0] == "vt":
            texcoords.append(tuple(map(float, fields[1:3])))
        elif fields[0] == "f":
            face_counts.append(len(fields) - 1)
            for item in fields[1:]:
                vertex, uv, *_ = item.split("/")
                if not vertex or not uv or int(vertex) <= 0 or int(uv) <= 0:
                    raise ValueError("Expected positive vertex and UV indices")
                indices.append(int(vertex) - 1)
                uv_indices.append(int(uv) - 1)
    if (
        not points
        or not indices
        or max(indices) >= len(points)
        or max(uv_indices) >= len(texcoords)
    ):
        raise ValueError("Invalid textured OBJ geometry")
    bounds = [tuple(fn(point[axis] for point in points) for axis in range(3)) for fn in (min, max)]
    msh_path = source.parent / mesh_xml.attrib["file"]
    msh = msh_path.read_bytes()
    vertices, normals, uvs, faces = struct.unpack_from("<4i", msh)
    if len(msh) != 16 + 12 * (vertices + normals + faces) + 8 * uvs:
        raise ValueError("Invalid MuJoCo binary mesh dimensions")
    values = struct.unpack_from(f"<{vertices * 3}f", msh, 16)
    for bound, fn in zip(bounds, (min, max), strict=True):
        expected = tuple(fn(values[axis::3]) * scale[axis] for axis in range(3))
        if any(abs(a - b) > 1e-6 for a, b in zip(bound, expected, strict=True)):
            raise ValueError("OBJ geometry does not match the MJCF visual mesh")
    collisions = xml.findall("worldbody/body/body/geom[@group='0']")
    if not collisions or any(geom.attrib["type"] != "box" for geom in collisions):
        raise ValueError("Only explicit MJCF box colliders are supported")
    texture_xml = xml.find("asset/texture")
    if texture_xml is None:
        raise ValueError("Missing texture")
    texture = source.parent / texture_xml.attrib["file"]
    texture_output = output.with_name(f"{output.stem}_texture.png")
    if texture_output.exists():
        raise FileExistsError(texture_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    visual = UsdGeom.Mesh.Define(stage, "/Asset/Visual")
    visual.CreatePointsAttr(points)
    visual.CreateFaceVertexCountsAttr(face_counts)
    visual.CreateFaceVertexIndicesAttr(indices)
    visual.CreateSubdivisionSchemeAttr("none")
    uv = UsdGeom.PrimvarsAPI(visual).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying"
    )
    uv.Set(texcoords)
    uv.SetIndices(uv_indices)
    material = UsdShade.Material.Define(stage, "/Asset/Material")
    surface = UsdShade.Shader.Define(stage, "/Asset/Material/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    sampler = UsdShade.Shader.Define(stage, "/Asset/Material/Texture")
    sampler.CreateIdAttr("UsdUVTexture")
    sampler.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_output.name))
    sampler.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    reader = UsdShade.Shader.Define(stage, "/Asset/Material/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    sampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result"
    )
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        sampler.ConnectableAPI(), "rgb"
    )
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(material)
    for index, geom in enumerate(collisions):
        half = tuple(map(float, geom.attrib["size"].split()))
        position = tuple(map(float, geom.attrib["pos"].split()))
        rotation = tuple(map(float, geom.attrib["quat"].split()))
        norm = math.sqrt(sum(value * value for value in rotation))
        rotation = tuple(value / norm for value in rotation)
        cube = UsdGeom.Cube.Define(stage, f"/Asset/Collision{index}")
        cube.CreateSizeAttr(2.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*position))
        cube.AddOrientOp().Set(Gf.Quatf(rotation[0], Gf.Vec3f(*rotation[1:])))
        cube.AddScaleOp().Set(Gf.Vec3f(*half))
        cube.CreatePurposeAttr("guide")
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    # MuJoCo includes the non-colliding visual mesh in this asset's authored inertia.
    model = mujoco.MjModel.from_xml_path(str(source))
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if body < 0:
        raise ValueError("Missing source object body")
    mass = float(model.body_mass[body])
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr(mass)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*(float(value) for value in model.body_ipos[body])))
    mass_api.CreateDiagonalInertiaAttr(
        Gf.Vec3f(*(float(value) for value in model.body_inertia[body]))
    )
    inertial_quat = model.body_iquat[body]
    mass_api.CreatePrincipalAxesAttr(
        Gf.Quatf(float(inertial_quat[0]), Gf.Vec3f(*(float(value) for value in inertial_quat[1:])))
    )
    shutil.copy2(texture, texture_output)
    stage.GetRootLayer().Export(str(output))
    result = {
        "source": "LIBERO stable_hope_objects; native conversion, not a Lightwheel USD",
        "inputs": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (source, obj_path, msh_path, texture)
        },
        "minimum": bounds[0],
        "maximum": bounds[1],
        "collision_count": len(collisions),
        "mass": mass,
    }
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(convert_object(args.source, args.output), indent=2))
