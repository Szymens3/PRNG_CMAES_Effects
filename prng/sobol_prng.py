from chaospy import Normal
from .prng import PRNG


class SOBOL_PRNG(PRNG):
    def __init__(self, seed) -> None:
        self._distr = Normal(0, 1)
        self._seed = seed

    def __str__(self) -> str:
        return f"sobol_{super().__str__()}"

    def std_normal(self, dim: int):
        samples = self._distr.sample(dim, rule="sobol", seed=self._seed)
        return samples
