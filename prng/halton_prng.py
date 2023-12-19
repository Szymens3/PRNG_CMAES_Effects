from scipy.stats import qmc, norm
from .prng import PRNG


class HALTON_PRNG(PRNG):
    def __init__(self, seed) -> None:
        self._prng = qmc.Halton(d=10, seed=seed)
        self._dim = 10
        self._seed = seed

    def __str__(self) -> str:
        return f"halton_{super().__str__()}"

    def std_normal(self, dim: int):
        if self._dim != dim:
            self._dim = dim
            self._prng = qmc.Halton(d=dim, seed=self._seed)
        samples = self._prng.random(n=1)
        return norm.ppf(samples)[0]
