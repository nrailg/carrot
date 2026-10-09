from numbers import Integral

import numpy as np


# 固定几何与七段笔画避免字体、平台或依赖版本改变编码。
_SEGMENTS = (
    (4, 0, 36, 6), (36, 4, 42, 42), (36, 46, 42, 84),
    (4, 82, 36, 88), (0, 46, 6, 84), (0, 4, 6, 42), (4, 41, 36, 47),
)
_DIGITS = ("abcdef", "bc", "abdeg", "abcdg", "bcfg", "acdfg", "acdefg",
           "abc", "abcdefg", "abcdfg")


def render_frame_index(index: int) -> np.ndarray:
    """Render a zero-based frame index as four fixed seven-segment decimal digits.

    Parameters
    ----------
    index : int
        Integer in [0, 9999]; booleans are rejected.

    Returns
    -------
    np.ndarray
        Deterministic RGB uint8 array of shape (224, 224, 3), with no input besides index.

    Raises
    ------
    AssertionError
        If index is not an integer in the supported range.
    """
    assert isinstance(index, Integral) and not isinstance(index, (bool, np.bool_)), (
        "frame index must be an integer, not boolean"
    )
    assert 0 <= index <= 9999, "frame index must be in [0, 9999]"
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    for position, digit in enumerate(f"{int(index):04d}"):
        left, top = 16 + 50 * position, 68
        for segment in _DIGITS[int(digit)]:
            x0, y0, x1, y1 = _SEGMENTS[ord(segment) - ord("a")]
            image[top + y0:top + y1, left + x0:left + x1] = 255
    return image
