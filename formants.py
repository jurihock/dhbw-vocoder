from pathlib import Path

from cepstrum import Cepstrum
from spectrum import Spectrum

import numpy as np
import soundfile


if __name__ == '__main__':

    # Specify input and output file paths:

    cwd = Path().cwd()
    src = cwd / 'x.wav'
    dst = cwd / 'y.wav'

    # Load source file as `x`:

    print(f'Reading {src.resolve()}')
    x, samplerate = soundfile.read(src, always_2d=True)
    x = x.T   # make it appear as (channels, samples)
    x = x[0]  # select a single channel

    # Analyze input `x`:

    spectrum = Spectrum(samplerate)
    cepstrum = Cepstrum(samplerate)

    X = spectrum.analyze(x)

    # Extract spectral envelope and residual:

    envelope = cepstrum.lifter(X, quefrency=1e-3)
    residual = np.abs(X) / envelope
    phase    = np.exp(1j * np.angle(X))

    Y = [envelope * phase, residual * phase]

    spectrum.cepstrogram(X, name='Origin cepstrogram')
    spectrum.spectrogram(X, name='Origin spectrogram')
    spectrum.spectrogram(Y[0], name='Envelope spectrogram')
    spectrum.spectrogram(Y[1] / Y[1].shape[-1], name='Residual spectrogram')
    spectrum.show()

    # Synthesize output `y`:

    y = np.asarray([spectrum.synthesize(Y[i])
                    for i in range(len(Y))])

    y /= np.abs(y).max(axis=-1)[..., None]

    # Write `y` to destination file:

    print(f'Writing {dst.resolve()}')
    soundfile.write(dst, np.squeeze(y.T), samplerate)
