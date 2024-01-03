import numpy as np
from scipy.stats import qmc
from .mocking_prng import MOCKING_PRNG



class SOBOL_PRNG(MOCKING_PRNG):
    name='sobol'
    def __init__(self, seed) -> None:
        self._prng = qmc.Sobol(d=1, scramble=True, seed=seed)
        super().__init__(seed)
        

    def __str__(self) -> str:
        return f"sobol_{super().__str__()}"
    
    def _gen_uniform(self, dim: int):
        return self._prng.random(dim).astype(np.float32)
    



    

