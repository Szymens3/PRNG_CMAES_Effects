import numpy as np
from scipy.stats import qmc
from .mocking_prng import MOCKING_PRNG



class SOBOL_PRNG(MOCKING_PRNG):
    def __init__(self, seed) -> None:
        super().__init__(seed)
        self._prng = qmc.Sobol(d=1, scramble=False, seed=self._seed)
        _ = self._prng.fast_forward(5) # First value in sequence is 0 so mapping to std_dist yields -inf. Skipping first few values prevents that

    def __str__(self) -> str:
        return f"sobol_{super().__str__()}"
    
    def _gen_uniform(self, dim: int):
        return self._prng.random(dim).astype(np.float32)
    



    

