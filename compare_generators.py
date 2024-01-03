import time

from scipy.special import erfinv
from scipy.stats import qmc, norm
import numpy as np

from prng.prng import PRNG
from prng.lcg_prng import LCG_PRNG
from prng.mt_prng import MT_PRNG
from prng.xoroshio_prng import XOROSHIRO_PRNG
from prng.urandom_prng import URANDOM_PRNG
from prng.halton_prng import HALTON_PRNG
from prng.sobol_prng import SOBOL_PRNG

seed=1000

def uniform_to_std_normal(uniform_numbers):
    normal_numbers = np.sqrt(2) * erfinv(2 * uniform_numbers - 1)
    return normal_numbers


class SOBOL_NO_FILES(PRNG):
    def __init__(self, seed):
        self._prng = qmc.Sobol(d=1, seed=seed)

    def __str__(self) -> str:
        return f"sobol_no_files{super().__str__()}"
    
    def std_normal(self, dim: int):
        return uniform_to_std_normal(self._prng.random(dim))
    
class SOBOL_PRNG_RE(PRNG):
    def __init__(self, seed) -> None:
        self._prng = qmc.Sobol(d=1, seed=seed)
        self._seed = seed

    def __str__(self) -> str:
        return f"sobol_re{super().__str__()}"

    def std_normal(self, dim: int):
        samples = self._prng.random(n=dim)
        return norm.ppf(samples)
    
class HALTON_NO_FILES_1_worker(PRNG):
    def __init__(self, seed):
        self._prng = qmc.Halton(d=1, seed=seed)

    def __str__(self) -> str:
        return f"halton_no_files{super().__str__()}"
    
    def std_normal(self, dim: int):
        return uniform_to_std_normal(self._prng.random(dim))
    
import cProfile

gens = [XOROSHIRO_PRNG, MT_PRNG, LCG_PRNG, SOBOL_PRNG, SOBOL_NO_FILES, HALTON_PRNG, HALTON_NO_FILES_1_worker, URANDOM_PRNG]
#gens = [XOROSHIRO_PRNG, MT_PRNG, LCG_PRNG, SOBOL_PRNG, HALTON_PRNG, URANDOM_PRNG]
n = 1_000_000
dim = 100

def test_gen_i(gen_i, n, dim):
    start = time.process_time()
    for i in range(n):
        v = gen_i.std_normal(dim)
    end = time.process_time()
    print(f"Gen: {gen_i} took: {end-start} s to generate {n} times {dim} values from std_norm distribution")


for gen in gens:
    gen_i = gen(seed)
    cProfile.run(f'test_gen_i(gen_i, n={n}, dim={dim})', sort='cumulative')



