import pytest
import time
from scipy.special import erfinv
from scipy.stats import qmc
import numpy as np


from prng.lcg_prng import LCG_PRNG
from prng.mt_prng import MT_PRNG
from prng.xoroshio_prng import XOROSHIRO_PRNG
from prng.urandom_prng import URANDOM_PRNG
from prng.halton_prng import HALTON_PRNG
from prng.sobol_prng import SOBOL_PRNG

prngs = [LCG_PRNG, MT_PRNG, XOROSHIRO_PRNG, URANDOM_PRNG, HALTON_PRNG, SOBOL_PRNG]
seed = 1000


@pytest.mark.parametrize("dim", [10,30,50,100])
@pytest.mark.parametrize("prng", prngs)
def test_generate_n_samples_time_in_one_call(dim, prng):
    instance = prng(seed, dim)
    start_time = time.process_time()
    samples = instance.std_normal(dim)
    end_time = time.process_time()
    assert len(samples) == dim
    assert end_time - start_time < 1.0

@pytest.mark.parametrize("n_time_pair", [(1000, 0.1),(10_000, 0.5),(100_000, 1)])
@pytest.mark.parametrize("dim", [10,30,50,100])
@pytest.mark.parametrize("prng", prngs)
def test_generate_dim_samples_time_in_n_calls(n_time_pair,dim, prng):
    n, exp_time = n_time_pair
    instance = prng(seed, dim)
    test_start = time.process_time()
    for i in range(n):
        samples = instance.std_normal(dim)
        assert len(samples) == dim
    test_end = time.process_time()
    assert test_end - test_start < exp_time


def test_time_in_10_000_000_calls_loop_over_entire_file():
    # Loop 
    seed = 1000
    dim = 100
    g = URANDOM_PRNG(seed, dim)
    for i in range(10_000_000):
        samples = g.std_normal(dim)
    assert True

def test_time_in_10_000_000_calls_loop_over_entire_file_halton():
    # Loop 
    seed = 1000
    dim = 100
    g = HALTON_PRNG(seed, dim)
    for i in range(10_000_000):
        samples = g.std_normal(dim)
    assert True

def test_time_in_10_000_000_calls_loop_over_entire_file_sobol():
    # Loop 
    seed = 1000
    dim = 100
    g = SOBOL_PRNG(seed, dim)
    for i in range(10_000_000):
        samples = g.std_normal(dim)
    assert True

def test_time_in_1_000_000_calls_loop_over_entire_file_sobol():
    # Loop 
    seed = 1000
    dim = 100
    g = SOBOL_PRNG(seed, dim)
    for i in range(1_000_000):
        samples = g.std_normal(dim)
    assert True

def test_time_in_1_000_000_calls_loop_over_entire_file_sobol_generated():
    def uniform_to_std_normal(uniform_numbers):
        normal_numbers = np.sqrt(2) * erfinv(2 * uniform_numbers - 1)
        return normal_numbers
    # Loop 
    seed = 1000
    dim = 100
    g = qmc.Sobol(d=1, scramble=True, seed=seed)
    for i in range(1_000_000):
        samples = uniform_to_std_normal(g.random(dim).astype(np.float32))
    assert True


