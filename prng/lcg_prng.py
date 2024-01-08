"""Module with lcg number generator"""
from numpy.random import Generator
from randomgen import LCG128Mix

from .prng import Prng


class LcgPrng(Prng):
    """Linear congruential generator"""

    name = "lcg"

    def __init__(self, seed, dim, logger):
        lcg_mult = 0x1DA942042E4DD58B5
        dxsm_mult = 0xFF37F1F758180525
        self._prng = Generator(
            LCG128Mix(
                multiplier=lcg_mult, dxsm_multiplier=dxsm_mult, output="dxsm", seed=seed
            )
        )

    def __str__(self) -> str:
        return f"lcg_{super().__str__()}"

    def std_normal(self, dim: int, n: int = 1):
        return self._prng.standard_normal(dim)
