from typing import Union
from numpy.typing import ArrayLike, NDArray

import numpy as np

from princarg import princarg
from interpolation import interpolate
from resampling import resample
from stft import STFT
from settings import Settings


class Vocoder:
    """
    Collection of phase vocoder based routines for time-scale and pitch-shifting modifications.
    """

    def __init__(self, samplerate: int, settings: Union[Settings, None] = None):
        """
        Creates a new phase vocoder instance
        for the specified `samplerate` in hertz
        and customized STFT `settings`.
        """

        assert samplerate > 0

        self.samplerate = samplerate
        self.settings   = settings or Settings()

    def tsm(self, x: ArrayLike, *, timefactor: float = 1, shiftpitch: bool = False) -> NDArray:
        """
        Performs time-scale modification (TSM) to `x` according to the specified
        time-scaling factor `timefactor` and optional pitch-shifting `shiftpitch`
        by resampling the time-scaling result.
        """

        samplerate = self.samplerate
        framesize  = self.settings.framesize
        hopsize    = self.settings.hopsize
        padsize    = self.settings.padsize

        hopsizeA = hopsize
        hopsizeS = int(hopsizeA * timefactor)

        stft  = STFT(framesize, hopsize=hopsizeA, padsize=padsize, center=True)
        istft = STFT(framesize, hopsize=hopsizeS, padsize=padsize, center=True)

        # load and analyze the input file 'x'

        x = np.atleast_1d(x)
        X = stft.stft(x)

        ω = stft.freqs() * samplerate

        ΔtA = hopsizeA / samplerate
        ΔtS = hopsizeS / samplerate

        # preprocess phase values

        φA  = np.angle(X) / (2 * np.pi)
        ΔφA = np.diff(φA, axis=0, prepend=0)

        # perform time scaling

        εA = princarg(ΔφA - ω * ΔtA)
        εS = εA * timefactor  # = εA * (ΔtS / ΔtA)

        # postprocess phase values

        ΔφS = εS + ω * ΔtS
        φS  = np.cumsum(ΔφS, axis=0) * (2 * np.pi)

        # synthesize and save the output file 'y'

        Y = np.abs(X) * np.exp(1j * φS)
        y = istft.istft(Y)

        if shiftpitch:

            y = resample(y, timefactor)

        return y

    def psm(self, x: ArrayLike, *, pitchfactor: float = 1) -> NDArray:
        """
        Performs pitch-shifting modification (PSM) to `x` according
        to the specified pitch-shifting factor `pitchfactor`.
        """

        samplerate = self.samplerate
        framesize  = self.settings.framesize
        hopsize    = self.settings.hopsize
        padsize    = self.settings.padsize

        stft = STFT(framesize, hopsize=hopsize, padsize=padsize, center=False)

        # load and analyze the input file 'x'

        x = np.atleast_1d(x)
        X = stft.stft(x)

        ω  = stft.freqs() * samplerate
        Δt = hopsize / samplerate

        # preprocess phase values

        φA  = np.angle(X) / (2 * np.pi)
        ΔφA = np.diff(φA, axis=0, prepend=0)

        # manipulate instantaneous frequencies

        εA = princarg(ΔφA - ω * Δt)

        λA = εA / Δt + ω  # = (εA + ω * Δt) / Δt
        λS = interpolate(λA, pitchfactor) * pitchfactor

        εS = λS * Δt  # = λS * Δt - ω * Δt

        # postprocess phase values

        ΔφS = εS  # = εS + ω * Δt
        φS  = np.cumsum(ΔφS, axis=0) * (2 * np.pi)

        # manipulate magnitudes

        rA = np.abs(X)
        rS = interpolate(rA, pitchfactor)

        rS[(λS <= 0) | (λS >= samplerate / 2)] = 0

        # synthesize and save the output file 'y'

        Y = rS * np.exp(1j * φS)
        y = stft.istft(Y)

        return y

    def ptm(self, x: ArrayLike, *, pitchfactor: float = 1, timefactor: float = 1) -> NDArray:
        """
        Performs combined pitch-shifting and time-scale modification (PTM)
        to `x` according to the specified pitch-shifting factor `pitchfactor`
        and time-scaling factor `timefactor` as well.
        """

        samplerate = self.samplerate
        framesize  = self.settings.framesize
        hopsize    = self.settings.hopsize
        padsize    = self.settings.padsize

        hopsizeA = hopsize
        hopsizeS = int(hopsizeA * timefactor)

        stft  = STFT(framesize, hopsize=hopsizeA, padsize=padsize, center=True)
        istft = STFT(framesize, hopsize=hopsizeS, padsize=padsize, center=True)

        # load and analyze the input file 'x'

        x = np.atleast_1d(x)
        X = stft.stft(x)

        ω  = stft.freqs() * samplerate

        ΔtA = hopsizeA / samplerate
        ΔtS = hopsizeS / samplerate

        # preprocess phase values

        φA  = np.angle(X) / (2 * np.pi)
        ΔφA = np.diff(φA, axis=0, prepend=0)

        # manipulate instantaneous frequencies

        εA = princarg(ΔφA - ω * ΔtA)

        λA = εA / ΔtA + ω  # = (εA + ω * ΔtA) / ΔtA
        λS = interpolate(λA, pitchfactor) * pitchfactor

        εS = λS * ΔtS  # = λS * ΔtS - ω * ΔtS

        # postprocess phase values

        ΔφS = εS  # = εS + ω * ΔtS
        φS  = np.cumsum(ΔφS, axis=0) * (2 * np.pi)

        # manipulate magnitudes

        rA = np.abs(X)
        rS = interpolate(rA, pitchfactor)

        rS[(λS <= 0) | (λS >= samplerate / 2)] = 0

        # synthesize and save the output file 'y'

        Y = rS * np.exp(1j * φS)
        y = istft.istft(Y)

        return y
