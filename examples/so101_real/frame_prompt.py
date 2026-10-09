from numbers import Integral


def format_frame_prompt(prompt: str, index: int) -> str:
    """Append an episode-local frame condition to the original task prompt.

    Parameters
    ----------
    prompt : str
        Non-empty original task text; callers must not pass an already conditioned prompt.
    index : int
        Integer in [0, 9999]; booleans are rejected.

    Returns
    -------
    str
        Original task with trailing whitespace removed, followed by `` Frame: NNNN.``.

    Raises
    ------
    AssertionError
        If the task is empty or index is not a supported integer.
    """
    assert isinstance(prompt, str) and prompt.strip(), "frame prompt requires non-empty task text"
    assert isinstance(index, Integral) and not isinstance(index, bool), (
        "frame index must be an integer, not boolean"
    )
    assert 0 <= index <= 9999, "frame index must be in [0, 9999]"
    return f"{prompt.rstrip()} Frame: {int(index):04d}."
