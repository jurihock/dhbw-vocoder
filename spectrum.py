from typing import Tuple
from numpy.typing import ArrayLike, NDArray

import matplotlib.pyplot as plot
import numpy as np

from stft import STFT


class Spectrum:
    """
    Wrapper around the STFT processor providing `spectrogram`, `cepstrogram`, and `phasogram` plots
    in addition to `analyze` (stft) and `synthesize` (istft) procedures.
    """

    def __init__(self, samplerate: int, *, order: int = 10, overlap: int = 16, dense: int = 1):
        """
        Creates a new spectrum processor instance for the specified `samplerate` in hertz,
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

    def freqs(self) -> NDArray:
        """
        Returns an array of DFT bin center frequency values in hertz.
        """

        return self.stft.freqs(self.samplerate)

    def analyze(self, x: ArrayLike) -> NDArray:
        """
        Performs the STFT procedure on the specified time-domain array `x`.
        Returns the frequency-domain DFT matrix `y`.
        """

        return self.stft.stft(x)

    def synthesize(self, x: ArrayLike) -> NDArray:
        """
        Performs the ISTFT procedure on the specified frequency-domain DFT matrix `x`.
        Returns the time-domain array `y`.
        """

        return self.stft.istft(x)

    def spectrogram(self, x: ArrayLike, *,
                    name: str = 'Spectrogram',
                    xlim: Tuple[float, float] = (None, None),
                    ylim: Tuple[float, float] = (None, None),
                    clim: Tuple[float, float] = (-120, 0)):
        """
        Creates the spectrogram plot of `x`,
        which can be a time-domain array
        or a frequency-domain DFT matrix.
        Call the `show` function in order
        to display the created plot.
        """

        if not np.any(np.iscomplex(x)):

            x = np.atleast_1d(x)
            assert x.ndim == 1

            X = self.analyze(x)
            assert X.ndim == 2

        else:

            X = np.atleast_2d(x)
            assert X.ndim == 2

        samplerate = self.samplerate
        framesize  = self.framesize
        hopsize    = self.hopsize

        epsilon = np.finfo(X.dtype).eps

        spectrum = np.abs(X)
        spectrum = 20 * np.log10(spectrum + epsilon)

        frequencies = np.fft.rfftfreq(framesize, 1 / samplerate)
        timestamps  = np.arange(len(X)) * hopsize / samplerate

        extent = (timestamps[0], timestamps[-1], frequencies[0], frequencies[-1])
        args   = dict(aspect='auto', cmap='inferno', extent=extent, interpolation='nearest', origin='lower')

        plot.figure(name)
        plot.imshow(spectrum.T, **args)
        colorbar = plot.colorbar()

        plot.xlabel('time [s]')
        plot.ylabel('frequency [Hz]')
        colorbar.set_label('magnitude [dB]')

        plot.xlim(*xlim)
        plot.ylim(*ylim)
        plot.clim(*clim)

        return self

    def cepstrogram(self, x: ArrayLike, *,
                    name: str = 'Cepstrogram',
                    xlim: Tuple[float, float] = (None, None),
                    ylim: Tuple[float, float] = (None, None),
                    clim: Tuple[float, float] = (0, 0.1)):
        """
        Creates the cepstrogram plot of `x`,
        which can be a time-domain array
        or a frequency-domain DFT matrix.
        Call the `show` function in order
        to display the created plot.
        """

        if not np.any(np.iscomplex(x)):

            x = np.atleast_1d(x)
            assert x.ndim == 1

            X = self.analyze(x)
            assert X.ndim == 2

        else:

            X = np.atleast_2d(x)
            assert X.ndim == 2

        samplerate = self.samplerate
        framesize  = self.framesize
        hopsize    = self.hopsize

        epsilon = np.finfo(X.dtype).eps

        spectrum = np.abs(X)
        spectrum = np.log10(spectrum + epsilon)

        cepstrum = np.fft.irfft(spectrum, axis=-1)

        quefrencies = np.arange(framesize) / samplerate / 1e-3
        timestamps  = np.arange(len(X)) * hopsize / samplerate

        # for symmetry reasons, plot only the half of the real cepstrum
        assert quefrencies.size == cepstrum.shape[-1]
        quefrencies = quefrencies[:quefrencies.size//2]
        cepstrum    = cepstrum[..., :quefrencies.size]

        extent = (timestamps[0], timestamps[-1], quefrencies[0], quefrencies[-1])
        args   = dict(aspect='auto', cmap='binary', extent=extent, interpolation='nearest', origin='lower')

        plot.figure(name)
        plot.imshow(cepstrum.T, **args)
        colorbar = plot.colorbar()

        plot.xlabel('time [s]')
        plot.ylabel('quefrency [ms]')
        colorbar.set_label('normalized cepstral amplitude')

        plot.xlim(*xlim)
        plot.ylim(*ylim)
        plot.clim(*clim)

        return self

    def phasogram(self, x: ArrayLike, *,
                  name: str = 'Phasogram',
                  xlim: Tuple[float, float] = (None, None),
                  ylim: Tuple[float, float] = (None, None),
                  clim: Tuple[float, float] = (-np.pi, +np.pi)):
        """
        Creates the phasogram plot of `x`,
        which can be a time-domain array
        or a frequency-domain DFT matrix.
        Call the `show` function in order
        to display the created plot.
        """

        if not np.any(np.iscomplex(x)):

            x = np.atleast_1d(x)
            assert x.ndim == 1

            X = self.analyze(x)
            assert X.ndim == 2

        else:

            X = np.atleast_2d(x)
            assert X.ndim == 2

        samplerate = self.samplerate
        framesize  = self.framesize
        hopsize    = self.hopsize

        values = np.angle(X)

        frequencies = np.fft.rfftfreq(framesize, 1 / samplerate)
        timestamps  = np.arange(len(X)) * hopsize / samplerate

        extent = (timestamps[0], timestamps[-1], frequencies[0], frequencies[-1])
        args   = dict(aspect='auto', cmap='twilight', extent=extent, interpolation='nearest', origin='lower')

        plot.figure(name)
        plot.imshow(values.T, **args)
        colorbar = plot.colorbar()

        plot.xlabel('time [s]')
        plot.ylabel('frequency [Hz]')
        colorbar.set_label('phase [rad]')

        plot.xlim(*xlim)
        plot.ylim(*ylim)
        plot.clim(*clim)

        return self

    def show(self):
        """
        Displays the plot previously created by the
        `spectrogram`, `cepstrogram`, or `phasogram` functions.
        """

        plot.tight_layout()
        plot.show()
