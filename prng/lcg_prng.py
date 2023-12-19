from numpy.random import Generator
from randomgen import LCG128Mix

from .prng import PRNG


class LCG_PRNG(PRNG):
    def __init__(self, seed):
        lcg_mult = 0x1DA942042E4DD58B5
        dxsm_mult = 0xff37f1f758180525
        self._prng = Generator(LCG128Mix(multiplier=lcg_mult, dxsm_multiplier=dxsm_mult, output="dxsm", seed=seed))

    def __str__(self) -> str:
        return f"lcg_{super().__str__()}"

    def std_normal(self, dim: int):
        return self._prng.standard_normal(dim)
