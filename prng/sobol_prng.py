"""Module with Sobol Generator class"""

import numpy as np
from scipy.stats import qmc
from .mocking_prng import MockingPrng


class SobolPrng(MockingPrng):
    """Sobol psudo random numbers generator - pregenerates file and reads it"""

    name = "sobol"

    def __init__(self, seed, dim, max_fes_coef=10_000, chunk_size=2**20, logger=None) -> None:
        self._prng = qmc.Sobol(d=dim, scramble=True, seed=seed)
        super().__init__(seed, dim, max_fes_coef=max_fes_coef, chunk_size=chunk_size, logger=logger)

    def __str__(self) -> str:
        return f"sobol_{super().__str__()}"

    def _gen_uniform(self, dim: int, n=1):
        return self._prng.random(n).astype(np.float32).reshape(-1)
