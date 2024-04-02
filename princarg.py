from typing import Any

import numpy as np


def princarg(x: Any) -> Any:
    """
    Wraps normalized angles `x`, e.g. divided by 2π, to the interval [-0.5, +0.5).
    """

    return np.remainder(x + 0.5, 1) - 0.5
