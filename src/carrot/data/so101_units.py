"""Model joints use radians (or legacy normalized positions) and gripper fractions."""

import numpy as np


def to_model_units(values: np.ndarray, *, use_degrees: bool) -> np.ndarray:
    """Copy driver values into model units.

    Parameters
    ----------
    values : numpy.ndarray
        Shape (..., 6); first five components are degrees when use_degrees is true,
        otherwise normalized positions. The last component is gripper percentage.
    use_degrees : bool

    Returns
    -------
    numpy.ndarray
        Independent float32 array; radians when requested, otherwise normalized
        positions; gripper fractions in [0, 1].
    """
    result = np.asarray(values, dtype=np.float32).copy()
    assert result.ndim >= 1 and result.shape[-1] == 6 and np.isfinite(result).all(), (
        "expected finite SO101 joints"
    )
    if use_degrees:
        result[..., :5] = np.deg2rad(result[..., :5])
    result[..., 5] /= 100
    return result


def to_robot_units(values: np.ndarray, *, use_degrees: bool) -> np.ndarray:
    """Copy model values into driver units.

    Parameters
    ----------
    values : numpy.ndarray
        Shape (..., 6); radian angles (or normalized positions) and gripper in [0, 1].
    use_degrees : bool

    Returns
    -------
    numpy.ndarray
        Independent float32 array; degree angles when requested and gripper percentage.
    """
    result = np.asarray(values, dtype=np.float32).copy()
    assert result.ndim >= 1 and result.shape[-1] == 6 and np.isfinite(result).all(), (
        "expected finite SO101 joints"
    )
    if use_degrees:
        result[..., :5] = np.rad2deg(result[..., :5])
    result[..., 5] *= 100
    return result


def model_stats(stats: dict, *, use_degrees: bool) -> dict:
    """Scale six-axis location/spread statistics into model units.

    Parameters
    ----------
    stats : dict
        Driver-unit mean/std/extrema/quantiles; count is preserved without scaling.
    use_degrees : bool

    Returns
    -------
    dict
        Converted statistics with the same fields.
    """
    return {
        key: value if key == "count" else to_model_units(value, use_degrees=use_degrees).tolist()
        for key, value in stats.items()
    }
