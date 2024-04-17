from typing import Tuple, Union
from numpy.typing import ArrayLike, NDArray

import numpy as np

from scipy.signal import medfilt2d
from stft import STFT
from settings import Settings


class HPSS:
    """
    Harmonic Percussive Source Separation (HPSS).
    """

    def __init__(self, samplerate: int, settings: Union[Settings, None] = None):
        """
        Creates a new HPSS processor instance
        for the specified `samplerate` in hertz
        and customized STFT `settings`.
        """

        assert samplerate > 0

        self.samplerate = samplerate
        self.settings   = settings or Settings()

        self.stft = STFT(framesize=self.settings.framesize,
                         hopsize=self.settings.hopsize,
                         padsize=self.settings.padsize)

    def __call__(self, x: ArrayLike,
                 kernel: Union[int, Tuple[int, int]],
                 threshold: Union[str, float] = 'hard') -> Tuple[NDArray, NDArray, NDArray]:
        """
        Performs the harmonic percussive source separation on `x`,
        which can be a time-domain array or a frequency-domain DFT matrix.

        Returns a tuple of (harmonic, percussive, noise) estimates,
        each of the same shape as the input `x`.

        If the input is a time-domain array, then each output will be a time-domain array too.
        If the input is a DFT matrix, then each output still will be a DFT matrix,
        containing the estimated magnitudes, but the original phase values.

        The `kernel` parameter controls the size of the spectral median filter.
        If the specified `kernel` value is a scalar,
        the same mask size will be used for each estimate.
        Otherwise, if the `kernel` is a tuple containing two elements,
        the first element will be used for the harmonic mask and
        the second one for the percussive mask.

        The `threshold` value can be a scalar or one of the following string keywords:
        "hard", "soft", "fuzzy", "proto".
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

        median1 = medfilt2d(abs, (kernel1, 1))
        median2 = medfilt2d(abs, (1, kernel2))

        median1 = np.square(median1)
        median2 = np.square(median2)

        if str(threshold).lower() == 'hard':

            # Harmonic/Percussive Separation Using Median Filtering (FitzGerald)
            # http://dafx10.iem.at/papers/DerryFitzGerald_DAFx10_P15.pdf

            mask1 = (median1 >= median2).astype(float)
            mask2 = (median2 >= median1).astype(float)

        elif str(threshold).lower() == 'soft':

            # Harmonic/Percussive Separation Using Median Filtering (FitzGerald)
            # http://dafx10.iem.at/papers/DerryFitzGerald_DAFx10_P15.pdf

            temp1 = median1 / (median1 + median2 + epsilon)
            temp2 = median2 / (median1 + median2 + epsilon)

            mask1 = temp1
            mask2 = temp2
        
        elif str(threshold).lower() == 'fuzzy':

            # Enhanced Fuzzy Decomposition of Sound Into Sines, Transients and Noise (Fierro)
            # https://arxiv.org/pdf/2210.14041.pdf

            temp1 = median1 / (median1 + median2 + epsilon)
            temp2 = median2 / (median1 + median2 + epsilon)
            temp3 = 1 - np.sqrt(np.abs(temp1 - temp2))

            mask1 = temp1 - 0.5 * temp3
            mask2 = temp2 - 0.5 * temp3

        elif str(threshold).lower() == 'proto':

            # Enhanced Fuzzy Decomposition of Sound Into Sines, Transients and Noise (Fierro)
            # https://arxiv.org/pdf/2210.14041.pdf

            temp0 = median1 / (median1 + median2 + epsilon)
            temp1 = np.square(np.sin(temp0 * np.pi + np.pi))
            temp2 = np.square(np.sin(temp0 * np.pi - np.pi))

            mask1 = temp1 * (temp0 >= 0.5)
            mask2 = temp2 * (temp0 <= 0.5)

        else:

            # Extending harmonic-percussive separation of audio signals (Driedger)
            # https://archives.ismir.net/ismir2014/paper/000127.pdf

            threshold = float(threshold)

            temp1 = median1 / (median1 + median2 + epsilon)
            temp2 = median2 / (median1 + median2 + epsilon)

            mask1 = (temp1 / (temp2 + epsilon)) > threshold
            mask2 = (temp2 / (temp1 + epsilon)) > threshold

        mask3 = 1 - (mask1 + mask2)

        abs1 = abs * mask1
        abs2 = abs * mask2
        abs3 = abs * mask3

        y1 = abs1 * arg
        y2 = abs2 * arg
        y3 = abs3 * arg

        if istft:
            
            y1 = self.stft.istft(y1)
            y2 = self.stft.istft(y2)
            y3 = self.stft.istft(y3)

            y1.resize(shape)
            y2.resize(shape)
            y3.resize(shape)

        assert y1.shape == shape
        assert y2.shape == shape
        assert y3.shape == shape

        return y1, y2, y3
