from pathlib import Path
from numpy.typing import NDArray

from subprocess import run
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

    cwd = Path().cwd()
    src = cwd / '.data' / 'voice' / 'vocals.wav'
    dst = src.parent.with_suffix(src.suffix)

    # Load source file as `x`:

    print(f'Reading {src.resolve()}')
    x, samplerate = soundfile.read(src, always_2d=True)

    # Setup and customize the phase vocoder:

    vocoder = Vocoder(samplerate, order=10, overlap=16)

    # Uncomment one of the following procedures:

    # y = tsm(x.T, vocoder).T  # time-scale modification
    # y = psm(x.T, vocoder).T  # pitch-shifting modification
    # y = ptm(x.T, vocoder).T  # pitch-shifting and time-scale modification

    # Write `y` to destination file:

    print(f'Writing {dst.resolve()}')
    soundfile.write(dst, np.squeeze(y), samplerate)

    # This feature requires sox.sourceforge.net
    # to be included in the PATH:
    run(['play', dst], check=False)
