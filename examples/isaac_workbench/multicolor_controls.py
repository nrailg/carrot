import os

import numpy as np
from scipy.spatial.transform import Rotation

EEF_OFFSET = np.array([0.0, 0.0, 0.097])
EEF_ROTATION = Rotation.from_euler("z", -90, degrees=True)
LIBERO_BASE = np.array([-0.66, 0.0, 0.912])


def remove_sensitive_environment() -> None:
    """Remove inherited credentials before Kit can include its environment in startup diagnostics.

    Notes
    -----
    This local-assets simulator requires no service credentials. Only the current subprocess
    environment is modified; the Ray actor and launcher environments are unaffected.
    """
    markers = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "TICKET", "COOKIE", "AUTH", "_KEY")
    for name in tuple(os.environ):
        if any(marker in name.upper() for marker in markers):
            del os.environ[name]


def command_target(
    position: np.ndarray, rotation: Rotation, grip: float, action: np.ndarray
) -> tuple[np.ndarray, Rotation, float]:
    """Decode the robosuite OSC delta scales and incremental Panda gripper convention.

    Parameters
    ----------
    position : numpy.ndarray
        Current local end-effector position in metres.
    rotation : Rotation
        Current end-effector orientation in the LIBERO gripper frame.
    grip : float
        Accumulated command in [-1, 1]; positive is closed.
    action : numpy.ndarray
        Shape (7,); position, axis-angle rotation and gripper commands, clipped to [-1, 1].

    Returns
    -------
    tuple[numpy.ndarray, Rotation, float]
        Absolute local goal and updated gripper command, held for one control period.
    """
    assert action.shape == (7,) and np.isfinite(action).all()
    action = np.clip(action, -1, 1)
    return (
        position + 0.05 * action[:3],
        Rotation.from_rotvec(0.5 * action[3:6]) * rotation,
        float(np.clip(grip + 0.01 * np.sign(action[6]), -1, 1)),
    )


def libero_state(position: np.ndarray, rotation: Rotation, fingers: np.ndarray) -> np.ndarray:
    """Express source proprioception in LIBERO's fixed bench frame and signed finger convention.

    Parameters
    ----------
    position : numpy.ndarray
        Metres relative to the Panda base; vector environment offsets are already removed.
    rotation : Rotation
    fingers : numpy.ndarray
        Two nonnegative Isaac finger displacements in metres.

    Returns
    -------
    numpy.ndarray
        Shape (8,), matching LIBERO position, quaternion-to-axis-angle and signed qpos semantics.
        The fixed base translation is the benchmark's MountedPanda calibration, not object state.
    """
    assert position.shape == (3,) and fingers.shape == (2,)
    quat = rotation.as_quat()
    w = np.clip(quat[3], -1, 1)
    denominator = np.sqrt(1 - w * w)
    rotvec = (
        np.zeros(3) if np.isclose(denominator, 0) else quat[:3] * 2 * np.arccos(w) / denominator
    )
    return np.concatenate((position + LIBERO_BASE, rotvec, fingers * [1, -1])).astype(np.float32)
