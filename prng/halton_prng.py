from scipy.stats import qmc, norm
from .prng import PRNG


class HALTON_PRNG(PRNG):
    def __init__(self, seed) -> None:
        self._prng = qmc.Halton(d=1, seed=seed)

    def std_normal(self, dim: int):
        samples = self._prng.random(n=dim)
        return norm.ppf(samples).reshape((1, dim))[0]