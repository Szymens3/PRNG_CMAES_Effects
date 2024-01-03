import pytest
import time
import numpy as np
from prng.urandom_prng import URANDOM_PRNG
from prng.halton_prng import HALTON_PRNG
from prng.sobol_prng import SOBOL_PRNG
from scipy.stats import qmc
from scipy.special import erfinv


def test_urandom_creation():
    seed = 1000
    g = URANDOM_PRNG(seed)
    assert type(g) == URANDOM_PRNG

def test_urandom_std_normal_return_type():
    seed = 1000
    g = URANDOM_PRNG(seed)
    assert type(g) == URANDOM_PRNG
    v = g.std_normal(1)
    assert type(v) == np.ndarray
    v_1 = v[0]
    assert type(v_1) == np.float32

def test_speed_std_normal_size_10():
    seed = 1000
    dim = 10
    g = URANDOM_PRNG(seed)

    start_time = time.time()
    samples = g.std_normal(dim)
    end_time = time.time()

    assert len(samples) == dim
    assert end_time - start_time < 1.0

def test_get_next_chunk():
    seed = 1000
    dim = 1
    g = URANDOM_PRNG(seed, chunk_size=2)

    assert len(g.buffered_values)==2
    assert g._current_idx == 0
    samples = g.std_normal(dim)
    assert len(samples) == 1
    assert g._current_idx == 1
    samples = g.std_normal(dim)
    assert len(samples) == 1
    assert g._current_idx == 2
    samples = g.std_normal(dim)
    assert len(samples) == 1
    assert g._current_idx == 1
    
def test_speed_std_normal_size_10_000():
    seed = 1000
    dim = 10_000
    g = URANDOM_PRNG(seed)

    start_time = time.time()
    samples = g.std_normal(dim)
    end_time = time.time()

    assert len(samples) == dim
    assert end_time - start_time < 1.0


def test_time_in_1000000_calls_dim_100():
    seed = 1000
    dim = 100
    g = URANDOM_PRNG(seed)
    for i in range(1000000):
        samples = g.std_normal(dim)
    assert True

def test__dim_eq_chunk_size():
    seed = 1000
    dim = 2
    g = URANDOM_PRNG(seed, chunk_size=2)
    assert len(g.buffered_values)==2
    assert g._current_idx == 0
    samples = g.std_normal(dim)
    assert len(samples) == 2
    assert g._current_idx == 2
    samples = g.std_normal(dim)
    assert len(samples) == 2
    assert g._current_idx == 2
    samples = g.std_normal(dim)
    assert len(samples) == 2
    assert g._current_idx == 2

def test__dim_2_chunk_size_3():
    seed = 1000
    dim = 2
    g = URANDOM_PRNG(seed, chunk_size=3)
    assert len(g.buffered_values)==3
    assert g._current_idx == 0
    samples = g.std_normal(dim)
    assert len(samples) == 2
    assert g._current_idx == 2
    samples = g.std_normal(dim)
    assert len(samples) == 2
    assert g._current_idx == 1
    samples = g.std_normal(dim)
    assert len(samples) == 2
    assert g._current_idx == 3
        
def test__dim_2xx20_default_chunk_size():
    seed = 1000
    dim = 2**20
    g = URANDOM_PRNG(seed)
    assert len(g.buffered_values)==2**20
    assert g._current_idx == 0
    samples = g.std_normal(dim)
    assert len(samples) == 2**20
    assert g._current_idx == 2**20
    samples = g.std_normal(dim)
    assert len(samples) == 2**20
    assert g._current_idx == 2**20


def test_time_in_10_000_000_calls_loop_over_entire_file():
    # Loop 
    seed = 1000
    dim = 100
    g = URANDOM_PRNG(seed)
    for i in range(10_000_000):
        samples = g.std_normal(dim)
    assert True

def test_time_in_10_000_000_calls_loop_over_entire_file_halton():
    # Loop 
    seed = 1000
    dim = 100
    g = HALTON_PRNG(seed)
    for i in range(10_000_000):
        samples = g.std_normal(dim)
    assert True

def test_time_in_10_000_000_calls_loop_over_entire_file_sobol():
    # Loop 
    seed = 1000
    dim = 100
    g = SOBOL_PRNG(seed)
    for i in range(10_000_000):
        samples = g.std_normal(dim)
    assert True

def test_time_in_1_000_000_calls_loop_over_entire_file_sobol():
    # Loop 
    seed = 1000
    dim = 100
    g = SOBOL_PRNG(seed)
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    with open('./prng/urandom_prng_files/1000', 'rb') as file:
        data = np.fromfile(file, dtype=np.float32)
        plt.hist(data, bins=100, alpha=0.7)  # Adjust the number of bins as needed
        plt.title('Histogram of Data')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()