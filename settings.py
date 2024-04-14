from dataclasses import dataclass


@dataclass
class Settings:
    """
    Collection of common STFT related settings.

    order: int
        Total number of DFT bins expressed as `1 << order`.
    overlap: int
        STFT frame overlap level, equivalent to a hop size of `(2 << order) // overlap`.
    dense: int
        Increase of the DFT bin density by zero padding in the time domain, if > 1.
    """

    order:   int = 10
    overlap: int = 16
    dense:   int = 1

    @property
    def framesize(self) -> int:
        assert self.order > 0
        return 2 << self.order

    @property
    def hopsize(self) -> int:
        assert self.overlap > 0
        return self.framesize // self.overlap

    @property
    def padsize(self) -> int:
        assert self.dense > 0
        return (2 << (self.order + self.dense - 1)) - self.framesize
