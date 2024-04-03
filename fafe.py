from typing import Union
from numpy.typing import ArrayLike, NDArray

import numpy as np


class FAFE:
    '''
    Fast, Accurate Frequency Estimator according to [1,2].

    [1] Eric Jacobsen and Peter Kootsookos
        Fast, Accurate Frequency Estimators
        IEEE Signal Processing Magazine (2007)
        https://ieeexplore.ieee.org/document/4205098

    [2] Rick Lyons
        Streamlining Digital Signal Processing
        IEEE Press (2012)
        https://ieeexplore.ieee.org/book/6241055
    '''

    def __init__(self, samplerate: int, mode: Union[str, None]) -> NDArray:

        self.samplerate = samplerate
        self.mode = mode

    def __call__(self, dfts: ArrayLike) -> NDArray:

        dfts = np.atleast_2d(dfts)
        assert dfts.ndim == 2

        l = np.roll(dfts, +1, axis=-1)
        m = dfts
        r = np.roll(dfts, -1, axis=-1)

        if self.mode is None:

            l = np.abs(l)
            m = np.abs(m)
            r = np.abs(r)

            with np.errstate(all='ignore'):
                drifts = (r - l) / (4 * m - 2 * r - 2 * l)

        elif str(self.mode).lower() == 'jacobsen':

            # Jacobsen "On Local Interpolation of DFT Outputs" 1994

            with np.errstate(all='ignore'):
                drifts = -np.real((r - l) / (2 * m - r - l))

        elif str(self.mode).lower() == 'quinn':

            # Quinn "Estimating frequency by interpolation using Fourier coefficients" 1994

            drifts = np.zeros_like(dfts, float)

            with np.errstate(all='ignore'):

                alpha1 = np.real(l / m)
                alpha2 = np.real(r / m)

                beta1 = +alpha1 / (1 - alpha1)
                beta2 = -alpha2 / (1 - alpha2)

                mask = (beta1 > 0) & (beta2 > 0)

                drifts[mask]  = beta2[mask]
                drifts[~mask] = beta1[~mask]

        elif str(self.mode).lower() == 'macleod':

            # MacLeod "Fast nearly ML estimation of the parameters of real or complex single tones or resolved multiple tones" 1998

            raise NotImplementedError('TODO')

        elif str(self.mode).lower() == 'gradke':

            # Gradke "Interpolation Algorithms for Discrete Fourier Transforms of Weighted Signals" 1983

            l = np.abs(l)
            m = np.abs(m)
            r = np.abs(r)

            drifts = np.zeros_like(dfts, float)

            with np.errstate(all='ignore'):

                alpha1 = m / l
                alpha2 = r / m

                beta1 = (alpha1 - 2) / (alpha1 + 1)
                beta2 = (2 * alpha2 - 1) / (alpha2 + 1)

                mask = l > r
                drifts[mask] = beta1[mask]

                mask = l < r
                drifts[mask] = beta2[mask]

        elif str(self.mode).lower() == 'hawkes':

            # Hawkes "Bin Interpolation" 1990

            p = 1.36  # hann window function

            l = np.abs(l)
            m = np.abs(m)
            r = np.abs(r)

            with np.errstate(all='ignore'):
                drifts = p * (r - l) / (m + r + l)

        elif str(self.mode).lower() == 'lyons':

            # Lyons "Private Communication" 2006

            q = 0.55  # hann window function

            with np.errstate(all='ignore'):
                drifts = -np.real(q * (r - l) / (2 * m + r + l))

        else:

            drifts = np.zeros_like(dfts, float)

        drifts[~np.isfinite(drifts)] = 0

        drifts[...,  0] = 0
        drifts[..., -1] = 0

        samplerate = self.samplerate
        dftsize    = dfts.shape[-1]
        framesize  = dftsize * 2 - 2

        bins  = np.arange(dftsize) + drifts
        freqs = bins * samplerate / framesize

        return freqs
