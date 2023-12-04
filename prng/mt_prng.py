import numpy as np

from .prng import PRNG

class MT_PRNG(PRNG):
    def __init__(self, seed):
        self._prng = np.random.RandomState(seed)

    def std_normal(self, dim: int):
        return self._prng.randn(dim)