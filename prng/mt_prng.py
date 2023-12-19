import numpy as np

from .prng import PRNG

class MT_PRNG(PRNG):
    def __init__(self, seed):
        self._prng = np.random.RandomState(seed)

    def __str__(self) -> str:
        return f"mt_prng_{super().__str__()}"

    def std_normal(self, dim: int):
        return self._prng.randn(dim)