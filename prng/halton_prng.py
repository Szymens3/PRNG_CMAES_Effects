import numpy as np
from scipy.stats import qmc
from .mocking_prng import MOCKING_PRNG


class HALTON_PRNG(MOCKING_PRNG):
    name='halton'
    def __init__(self, seed, dim, max_FES_coef=10_000, chunk_size=2**20)-> None:
        self._prng = qmc.Halton(d=dim, scramble=True, seed=seed)
        super().__init__(seed, dim, max_FES_coef=max_FES_coef, chunk_size=chunk_size)
        

    def __str__(self) -> str:
        return f"halton_{super().__str__()}"

    def _gen_uniform(self, dim: int, n=1):
        return self._prng.random(n).astype(np.float32).reshape(-1)