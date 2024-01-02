import pytest
import time
from prng.lcg_prng import LCG_PRNG
from prng.mt_prng import MT_PRNG
from prng.xoroshio_prng import XOROSHIRO_PRNG
from prng.urandom_prng import URANDOM_PRNG
from prng.halton_prng import HALTON_PRNG
from prng.sobol_prng import SOBOL_PRNG

seed = 1000

instances_to_test = [
    LCG_PRNG(seed),
    MT_PRNG(seed),
    XOROSHIRO_PRNG(seed),
    HALTON_PRNG(seed),
    SOBOL_PRNG(seed),
    URANDOM_PRNG(seed)
]

@pytest.mark.parametrize("n", [10,30,50,100])
@pytest.mark.parametrize("instance", instances_to_test)
def test_generate_n_samples_time_in_one_call(n, instance):
    start_time = time.time()
    samples = instance.std_normal(n)
    end_time = time.time()
    assert len(samples) == n
    assert end_time - start_time < 1.0

@pytest.mark.parametrize("dim", [10,30,50,100])
@pytest.mark.parametrize("instance", instances_to_test)
def test_generate_dim_samples_time_in_10000_calls(dim, instance):
    test_start = time.time()
    for i in range(10_000):
        start_time = time.time()
        samples = instance.std_normal(dim)
        end_time = time.time()
        assert len(samples) == dim
        assert end_time - start_time < 1.0
    test_end = time.time()
    assert test_end - test_start < 10.0
    print(f"{str(instance)}: {test_end - test_start}")


if __name__ == "__main__":
    for dim in [10,30,50,100]:
        for instance in instances_to_test:
            test_generate_dim_samples_time_in_10000_calls(dim, instance)
