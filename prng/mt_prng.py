"""Module with mt number generator"""

import numpy as np

from .prng import Prng


class MtPrng(Prng):
    """Mersenne Twister number generator class"""

    name = "mt"

    def __init__(self, seed, dim, logger=None):
        self._prng = np.random.RandomState(seed)

    def __str__(self) -> str:
        return f"mt_{super().__str__()}"

    def std_normal(self, dim: int, n: int = 1):
        return self._prng.randn(dim)
