from pathlib import Path
from numpy.typing import NDArray

from subprocess import run
from hpss import HPSS
from vocoder import Vocoder

import numpy as np
import soundfile


def tsm(x: NDArray, vocoder: Vocoder) -> NDArray:

    y = np.vstack([
        vocoder.tsm(data, timefactor=1, shiftpitch=False)
        for data in x
    ])

    return y


def psm(x: NDArray, vocoder: Vocoder) -> NDArray:

    y = np.vstack([
        vocoder.psm(data, pitchfactor=1)
        for data in x
    ])

    return y


def ptm(x: NDArray, vocoder: Vocoder) -> NDArray:

    y = np.vstack([
        vocoder.ptm(data, pitchfactor=1, timefactor=1)
        for data in x
    ])

    return y


if __name__ == '__main__':

    # Specify input and output file paths:

    cwd = Path().cwd()
    src = cwd / 'x.wav'
    dst = cwd / 'y.wav'

    # Load source file as `x`:

    print(f'Reading {src.resolve()}')
    x, samplerate = soundfile.read(src, always_2d=True)
    x = x.T  # make it appear as (channels, samples)

    # Preprocessing:

    # optionally try HPSS to extract and bypass the transients
    # hpss = HPSS(samplerate, order=10, overlap=16, dense=1)
    # harm, perc, misc = hpss(x[0], 42)

    # Setup and customize the phase vocoder:

    vocoder = Vocoder(samplerate, order=10, overlap=16, dense=1)

    # Uncomment one of the following procedures:

    y = x
    # y = tsm(x, vocoder)  # time-scale modification
    # y = psm(x, vocoder)  # pitch-shifting modification
    # y = ptm(x, vocoder)  # pitch-shifting and time-scale modification

    # Write `y` to destination file:

    print(f'Writing {dst.resolve()}')
    soundfile.write(dst, np.squeeze(y.T), samplerate)

    # This feature requires sox.sourceforge.net
    # to be included in the PATH:
    # run(['play', dst])
