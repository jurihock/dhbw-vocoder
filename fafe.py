import numpy as np


class FAFE:
    '''
    Fast, Accurate Frequency Estimator according to [1].

    [1] Eric Jacobsen and Peter Kootsookos
        Fast, Accurate Frequency Estimators
        IEEE Signal Processing Magazine (2007)
        https://ieeexplore.ieee.org/document/4205098
    '''

    def __init__(self, samplerate, *, mode=None):

        self.samplerate = samplerate
        self.mode = mode

    def __call__(self, dfts):

        dfts = np.atleast_2d(dfts)
        assert dfts.ndim == 2

        l = np.roll(dfts, +1, axis=-1)
        m = dfts
        r = np.roll(dfts, -1, axis=-1)

        if self.mode is None:

            with np.errstate(all='ignore'):
                drifts = -np.real((r - l) / (2 * m - r - l))

        elif str(self.mode).lower() == 'p':

            p = 1.36  # TODO: Hawkes "Bin Interpolation" 1990

            l = np.abs(l)
            m = np.abs(m)
            r = np.abs(r)

            with np.errstate(all='ignore'):
                drifts = p * (r - l) / (m + r + l)

        elif str(self.mode).lower() == 'q':

            q = 0.55  # TODO: Lyons "Private Communication" 2006

            with np.errstate(all='ignore'):
                drifts = -np.real(q * (r - l) / (2 * m + r + l))

        else:

            drifts = np.zeros_like(dfts)

        drifts[~np.isfinite(drifts)] = 0

        drifts[...,  0] = 0
        drifts[..., -1] = 0

        samplerate = self.samplerate
        dftsize    = dfts.shape[-1]
        framesize  = dftsize * 2 - 2

        bins  = np.arange(dftsize) + drifts
        freqs = bins * samplerate / framesize

        return freqs
