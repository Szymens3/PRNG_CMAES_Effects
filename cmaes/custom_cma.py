from typing import Optional, cast

import numpy as np

from .cma import CMA
from prng.prng import PRNG


class CustomCMA(CMA):
    def __init__(
            self,
            mean: np.ndarray,
            sigma: float,
            rng: PRNG,
            bounds: Optional[np.ndarray] = None,
            n_max_resampling: int = 100,
            population_size: Optional[int] = None,
            cov: Optional[np.ndarray] = None,
            lr_adapt: bool = False,
    ):
        super().__init__(mean=mean, sigma=sigma, bounds=bounds, n_max_resampling=n_max_resampling,
                         population_size=population_size, cov=cov, lr_adapt=lr_adapt)

        self._rng = rng

    def reseed_rng(self, seed: int) -> None:
        raise AttributeError("CustomCMA does not support reseeding the RNG.")

    def _sample_solution(self) -> np.ndarray:
        B, D = self._eigen_decomposition()
        z = self._rng.std_normal(self._n_dim)  # ~ N(0, I)
        y = cast(np.ndarray, B.dot(np.diag(D))).dot(z)  # ~ N(0, C)
        x = self._mean + self._sigma * y  # ~ N(m, σ^2 C)
        return x

    def anchor(self):
        return self._mean
