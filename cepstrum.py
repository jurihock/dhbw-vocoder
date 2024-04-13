from numpy.typing import ArrayLike, NDArray

import numpy as np


class Cepstrum:

    def __init__(self, samplerate: int) -> None:

        assert samplerate > 0

        self.samplerate = samplerate

    def lifter(self, x: ArrayLike, *, quefrency: float) -> NDArray:
        """
        Performs cepstral lowpass liftering on DFT matrix `x`
        according to the specified cutoff `quefrency` in seconds and
        returns the resulting spectral envelope of the same shape as `x`.
        """

        x = np.atleast_2d(x)
        assert x.ndim == 2

        epsilon    = np.finfo(x.dtype).eps
        samplerate = self.samplerate
        cutoff     = int(quefrency * samplerate)

        spectrum = np.log10(np.abs(x) + epsilon)
        cepstrum = np.fft.irfft(spectrum, axis=-1)

        cepstrum[..., 1:cutoff] *= 2
        cepstrum[..., cutoff+1:] = 0

        envelope = np.fft.rfft(cepstrum, axis=-1)
        y = np.power(10, np.real(envelope))

        return y
