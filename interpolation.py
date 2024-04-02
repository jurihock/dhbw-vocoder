from numpy.typing import ArrayLike, NDArray

import numpy as np


def interpolate(x: ArrayLike, q: float) -> NDArray:
    """
    Interpolates the ND array `x` according to the scaling factor `q`
    by using the linear interpolation method.
    """

    assert q > 0

    if q == 1:
        return np.copy(x)

    s = np.shape(x)

    x = np.atleast_2d(x)
    y = np.zeros_like(x)

    n = np.shape(x)[-1]
    m = int(n * q)

    i = np.arange(min(n, m))
    k = i * n / m

    j = np.trunc(k).astype(int)
    k = k - j

    ok = (0 <= j) & (j < n - 1)

    i, j, k = i[ok], j[ok], k[ok]

    y[..., i] = k * x[..., j + 1] + (1 - k) * x[..., j]

    return np.reshape(y, s)
