from typing import Union
from numpy.typing import ArrayLike, NDArray

import numpy as np

from numpy.lib.stride_tricks import sliding_window_view


class STFT:
    """
    Short-Time Fourier Transform (STFT).
    """

    def __init__(self, framesize: int, *, hopsize: Union[int, None] = None, padsize: int = 0, shift: bool = False):
        """
        Create a new STFT plan.

        Parameters
        ----------
        framesize: int
            Time domain segment length in samples.
        hopsize: int, optional
            Distance between consecutive segments in samples.
            Defaults to `framesize // 4`.
        padsize: int, optional
            Number of zeros to pad the segments with.
        shift: bool, optional
            Enable circular shift of segments.
        """

        self.framesize = framesize
        self.hopsize   = hopsize or (self.framesize // 4)
        self.padsize   = padsize
        self.shift     = shift
        self.window    = np.hanning(self.framesize + 1)[:-1]

    def stft(self, samples: ArrayLike) -> NDArray:
        """
        Estimates the DFT matrix for the given sample array.

        Parameters
        ----------
        samples: ndarray
            Array of time domain signal values.

        Returns
        -------
        dfts: ndarray
            Estimated DFT matrix of shape (samples, frequencies).
        """

        samples = np.atleast_1d(samples)

        assert samples.ndim == 1, f'Expected 1D array (samples,), got {samples.shape}!'

        frames = sliding_window_view(samples, self.framesize, writeable=False)[::self.hopsize]
        dfts   = self.fft(frames)

        return dfts

    def istft(self, dfts: ArrayLike) -> NDArray:
        """
        Synthesizes the sample array from the given DFT matrix.

        Parameters
        ----------
        dfts: ndarray
            DFT matrix of shape (samples, frequencies).

        Returns
        -------
        samples: ndarray
            Synthesized array of time domain signal values.
        """

        dfts = np.atleast_2d(dfts)

        assert dfts.ndim == 2, f'Expected 2D array (samples, frequencies), got {dfts.shape}!'

        gain = self.hopsize / np.sum(np.square(self.window))
        size = dfts.shape[0] * self.hopsize + self.framesize

        samples = np.zeros(size, float)

        frames0 = sliding_window_view(samples, self.framesize, writeable=True)[::self.hopsize]
        frames1 = self.ifft(dfts) * gain

        for i in range(min(len(frames0), len(frames1))):

            frames0[i] += frames1[i]

        return samples

    def fft(self, data: ArrayLike) -> NDArray:
        """
        Performs the forward FFT.
        """

        assert len(np.shape(data)) == 2

        data = np.atleast_2d(data) * self.window

        if self.padsize:

            data = np.pad(data, ((0, 0), (0, self.padsize)))

        if self.shift:

            data = np.fft.fftshift(data, axes=-1)

        return np.fft.rfft(data, axis=-1, norm='forward')

    def ifft(self, data: ArrayLike) -> NDArray:
        """
        Performs the backward FFT.
        """

        assert len(np.shape(data)) == 2

        data = np.fft.irfft(data, axis=-1, norm='forward')

        if self.shift:

            data = np.fft.ifftshift(data, axes=-1)

        if self.padsize:

            data = data[..., self.framesize]

        data *= self.window

        return data