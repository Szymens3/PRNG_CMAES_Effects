"""Module for testing batch mode for retrieving several points at the time from generators"""
import pytest
import numpy as np

from prng.sobol_prng import SobolPrng

def test_retrieve_one_point_from_2_dims():
    seed = 1000
    dim = 2

    file_path = "prng/sobol_prng_files/1000_2"
    vals = np.array(range(10),dtype=np.float32)
    vals.tofile(file_path)

    s = SobolPrng(seed, dim)

    assert len(s.buffered_values) == 10
    assert isinstance(s.buffered_values[0], np.float32)
    assert np.array_equal(s.buffered_values, vals)

    assert np.array_equal(s.std_normal(2,1),np.array(range(2), dtype=np.float32))

