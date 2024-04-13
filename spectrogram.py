from typing import Tuple
from numpy.typing import ArrayLike, NDArray

import matplotlib.pyplot as plot
import numpy as np

from stft import STFT


class Spectrogram:

    def __init__(self, samplerate: int, *, order: int = 10, overlap: int = 16, dense: int = 1):

        assert samplerate > 0
        assert order > 0
        assert overlap > 0

        self.samplerate = samplerate
        self.framesize  = 2 << order
        self.hopsize    = self.framesize // overlap
        self.padsize    = (2 << (order + dense - 1)) - self.framesize

    def analyze(self, x: ArrayLike) -> NDArray:

        framesize = self.framesize
        hopsize   = self.hopsize
        padsize   = self.padsize

        stft = STFT(framesize, hopsize=hopsize, padsize=padsize)
        
        y = stft.stft(x)

        return y

    def spectrogram(self, x: ArrayLike,
                    xlim: Tuple[float, float] = (None, None),
                    ylim: Tuple[float, float] = (None, None),
                    clim: Tuple[float, float] = (-120, 0)):

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

        plot.figure('Spectrogram')
        plot.imshow(spectrum.T, **args)
        colorbar = plot.colorbar()

        plot.xlabel('time [s]')
        plot.ylabel('frequency [Hz]')
        colorbar.set_label('magnitude [dB]')

        plot.xlim(*xlim)
        plot.ylim(*ylim)
        plot.clim(*clim)

        return self

    def cepstrogram(self, x: ArrayLike,
                    xlim: Tuple[float, float] = (None, None),
                    ylim: Tuple[float, float] = (None, None),
                    clim: Tuple[float, float] = (0, 0.1)):

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

        extent = (timestamps[0], timestamps[-1], quefrencies[0], quefrencies[-1])
        args   = dict(aspect='auto', cmap='binary', extent=extent, interpolation='nearest', origin='lower')

        plot.figure('Cepstrogram')
        plot.imshow(cepstrum.T, **args)
        colorbar = plot.colorbar()

        plot.xlabel('time [s]')
        plot.ylabel('quefrency [ms]')
        colorbar.set_label('normalized cepstral amplitude')

        plot.xlim(*xlim)
        plot.ylim(*ylim)
        plot.clim(*clim)

        return self

    def phasogram(self, x: ArrayLike,
                  xlim: Tuple[float, float] = (None, None),
                  ylim: Tuple[float, float] = (None, None),
                  clim: Tuple[float, float] = (-np.pi, +np.pi)):

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

        plot.figure('Phasogram')
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

        plot.tight_layout()
        plot.show()
