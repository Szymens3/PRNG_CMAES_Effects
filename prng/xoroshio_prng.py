"""Module with Xoroshiro number generator class"""

from numpy.random import Generator
from randomgen import Xoroshiro128
from .prng import Prng


class XoroshiroPrng(Prng):
    """XOR, rotate, shift, rotate number generator"""

    name = "xoroshiro"

    def __init__(self, seed, dim, logger=None):
        self._prng = Generator(Xoroshiro128(seed, plusplus=True))

    def __str__(self) -> str:
        return f"xoroshiro_{super().__str__()}"

    def std_normal(self, dim: int, n: int = 1):
        return self._prng.standard_normal(dim)
