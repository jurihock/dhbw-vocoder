from numpy.typing import ArrayLike, NDArray

import numpy as np
import resampy


def resample(x: ArrayLike, q: int) -> NDArray:
    """
    Interpolates the 1D array `x` according to the scaling factor `q`
    by using the band-limited sinc interpolation method.
    """

    assert q > 0

    if q == 1:
        return np.copy(x)

    x = np.atleast_1d(x)
    assert x.ndim == 1

    return resampy.resample(x, q, 1)
