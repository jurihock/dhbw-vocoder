from numpy.typing import ArrayLike, NDArray

import numpy as np

from princarg import princarg
from interpolation import interpolate
from resampling import resample
from stft import STFT
from fafe import FAFE


class Vocoder:
    """
    Collection of phase vocoder based routines for time-scale and pitch-shifting modifications.
    """

    def __init__(self, samplerate: int, *, order: int = 10, overlap: int = 16, dense: int = 1):
        """
        Creates a new phase vocoder instance for the specified `samplerate` in hertz,
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

    def tsm(self, x: ArrayLike, *, timefactor: float = 1, shiftpitch: bool = False) -> NDArray:
        """
        Performs time-scale modification (TSM) to `x` according to the specified
        time-scaling factor `timefactor` and optional pitch-shifting `shiftpitch`
        by resampling the time-scaling result.
        """

        samplerate = self.samplerate
        framesize  = self.framesize
        hopsize    = self.hopsize
        padsize    = self.padsize

        hopsizeA = hopsize
        hopsizeS = int(hopsizeA * timefactor)

        stft  = STFT(framesize, hopsize=hopsizeA, padsize=padsize, shift=True)
        istft = STFT(framesize, hopsize=hopsizeS, padsize=padsize, shift=True)

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
        framesize  = self.framesize
        hopsize    = self.hopsize
        padsize    = self.padsize

        stft = STFT(framesize, hopsize=hopsize, padsize=padsize, shift=False)

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
        framesize  = self.framesize
        hopsize    = self.hopsize
        padsize    = self.padsize

        hopsizeA = hopsize
        hopsizeS = int(hopsizeA * timefactor)

        stft  = STFT(framesize, hopsize=hopsizeA, padsize=padsize, shift=True)
        istft = STFT(framesize, hopsize=hopsizeS, padsize=padsize, shift=True)

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

    def experimental(self, x: ArrayLike, *, pitchfactor: float = 1, timefactor: float = 1, phase_vs_magnitude: float = 1) -> NDArray:
        """
        Performs combined pitch-shifting and time-scale modification (PTM)
        to `x` according to the specified pitch-shifting factor `pitchfactor`
        and time-scaling factor `timefactor` as well.

        Use `phase_vs_magnitude` parameter to balance between the phase based `-1`
        and the magnitude based `+1` instantaneous frequency estimate.
        """

        samplerate = self.samplerate
        framesize  = self.framesize
        hopsize    = self.hopsize
        padsize    = self.padsize

        hopsizeA = hopsize
        hopsizeS = int(hopsizeA * timefactor)

        stft  = STFT(framesize, hopsize=hopsizeA, padsize=padsize, shift=True)
        istft = STFT(framesize, hopsize=hopsizeS, padsize=padsize, shift=True)

        fafe = FAFE(samplerate, mode='p')

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

        λA0 = εA / ΔtA + ω  # = (εA + ω * ΔtA) / ΔtA
        λA1 = fafe(X)

        β = np.clip(phase_vs_magnitude / np.array([-2, +2]) + 0.5, 0, 1)

        λA = λA0 * β[0] + λA1 * β[1]
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
