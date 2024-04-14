from typing import Tuple, Union
from numpy.typing import ArrayLike, NDArray

import numpy as np

from scipy.signal import medfilt2d
from stft import STFT


class HPSS:
    """
    Harmonic Percussive Source Separation (HPSS).
    """

    def __init__(self, samplerate: int, *, order: int = 10, overlap: int = 16, dense: int = 1):
        """
        Creates a new HPSS processor instance for the specified `samplerate` in hertz,
        FFT vector size `1 << order`, and STFT hop size `(2 << order) // overlap`.
        Parameter `dense` increases the FFT bin density by zero-padding in the time domain.
        """

        assert samplerate > 0
        assert order > 0
        assert overlap > 0
        assert dense > 0

        self.samplerate = samplerate
        self.framesize  = 2 << order
        self.hopsize    = self.framesize // overlap
        self.padsize    = (2 << (order + dense - 1)) - self.framesize

        self.stft = STFT(self.framesize, hopsize=self.hopsize, padsize=self.padsize)

    def __call__(self, x: ArrayLike,
                 kernel: Union[int, Tuple[int, int]],
                 threshold: Union[str, float] = 'hard') -> Tuple[NDArray, NDArray, NDArray]:
        """
        Performs the harmonic percussive source separation on `x`,
        which can be a time-domain array or a frequency-domain DFT matrix.

        Returns a tuple of (harmonic, percussive, residual) estimates,
        each of the same shape as the input `x`.

        If the input is a time-domain array, then each output will be a time-domain array too.
        If the input is a DFT matrix, then each output will be a DFT matrix,
        containing the estimated magnitudes, but the original phase values.

        The `kernel` parameter controls the size of the spectral median filter.
        If the specified `kernel` value is a scalar,
        the same mask size will be used for each estimate.
        Otherwise, if the `kernel` is a tuple containing two elements,
        the first element will be used for the harmonic mask and
        the second one for the percussive mask.

        The `threshold` value can be either a "hard" or "soft" string keyword
        or a scalar value, e.g. 1.
        """

        shape = np.shape(x)

        if not np.any(np.iscomplex(x)):

            istft = True

            x = np.atleast_1d(x)
            assert x.ndim == 1

            X = self.stft.stft(x)
            assert X.ndim == 2

        else:

            istft = False

            X = np.atleast_2d(x)
            assert X.ndim == 2
        
        epsilon = np.finfo(X.dtype).eps

        abs = np.abs(X)
        arg = np.exp(1j * np.angle(X))

        kernel1 = np.ravel(kernel)[0]
        kernel2 = np.ravel(kernel)[-1]

        # apply horizontal median filter
        median1 = medfilt2d(abs, (kernel1, 1))

        # apply vertical median filter
        median2 = medfilt2d(abs, (1, kernel2))

        if str(threshold).lower() == 'hard':

            mask1 = (median1 >= median2).astype(float)
            mask2 = (median2 >= median1).astype(float)

        elif str(threshold).lower() == 'soft':

            median1 = np.square(median1)
            median2 = np.square(median2)

            mask1 = median1 / (median1 + median2 + epsilon)
            mask2 = median2 / (median1 + median2 + epsilon)

        else:

            threshold = float(threshold)

            mask1 = (median1 / (median2 + epsilon)) >  threshold
            mask2 = (median2 / (median1 + epsilon)) >= threshold

        mask3 = 1 - (mask1 + mask2)

        abs1 = abs * mask1
        abs2 = abs * mask2
        abs3 = abs * mask3

        y1 = abs1 * arg
        y2 = abs2 * arg
        y3 = abs3 * arg

        if istft:
            
            y1 = self.stft.istft(y1).resize(shape)
            y2 = self.stft.istft(y2).resize(shape)
            y3 = self.stft.istft(y3).resize(shape)

        assert y1.shape == shape
        assert y2.shape == shape
        assert y3.shape == shape

        return y1, y2, y3
