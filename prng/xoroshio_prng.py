from .prng import PRNG
from numpy.random import Generator
from randomgen import Xoroshiro128
class XOROSHIRO_PRNG(PRNG):
    def __init__(self, seed):
        self._prng = Generator(Xoroshiro128(1234, plusplus=True))

    def std_normal(self, dim: int):
        return self._prng.standard_normal(dim)