"""Test generators speed"""

import time
import pytest

from prng.lcg_prng import LcgPrng
from prng.mt_prng import MtPrng
from prng.xoroshio_prng import XoroshiroPrng
from prng.urandom_prng import UrandomPrng
from prng.halton_prng import HaltonPrng
from prng.sobol_prng import SobolPrng

prngs = [LcgPrng, MtPrng, XoroshiroPrng, UrandomPrng, HaltonPrng, SobolPrng]
DEFAULT_SEED = 1000


@pytest.mark.parametrize("dim", [10, 30, 50, 100])
@pytest.mark.parametrize("prng", prngs)
def test_generate_point_time(dim, prng):
    """
    Test speed of generating one point -
    Regardless of dim it should be performed in a fraction of a second
    """
    instance = prng(DEFAULT_SEED, dim)
    start_time = time.process_time()
    samples = instance.std_normal(dim)
    end_time = time.process_time()
    assert len(samples) == dim
    assert end_time - start_time < 0.001


@pytest.mark.parametrize("n_time_pair", [(1000, 0.1), (10_000, 0.5), (100_000, 1)])
@pytest.mark.parametrize("dim", [10, 30, 50, 100])
@pytest.mark.parametrize("prng", prngs)
def test_generate_dim_samples_time_in_n_calls(n_time_pair, dim, prng):
    """
    Test speed of n points
    For each number of points there is specified expected time
    """
    n, exp_time = n_time_pair
    instance = prng(DEFAULT_SEED, dim)
    test_start = time.process_time()
    for _ in range(n):
        samples = instance.std_normal(dim)
        assert len(samples) == dim
    test_end = time.process_time()
    assert test_end - test_start < exp_time


@pytest.mark.parametrize("prng", [UrandomPrng, SobolPrng, HaltonPrng])
def test_time_urandom_10_000_000_points_dim_100(prng):
    """
    Test file using Generators
    10_000_000 is 10 x more than max_FES for dim 100 (10_000 * 100)
    """
    dim = 100
    g = prng(DEFAULT_SEED, dim)
    start = time.process_time()
    for _ in range(10_000_000):
        _ = g.std_normal(dim)
    stop = time.process_time()
    assert stop - start < 10
