import pytest
import tempfile
from itertools import permutations
import numpy as np

from prng.sobol_prng import SobolPrng


def compare_binary_files(file1_path, file2_path):
    chunk_size = 2**20
    with open(file1_path, 'rb') as file1, open(file2_path, 'rb') as file2:
        while True:
            chunk1 = file1.read(chunk_size)
            chunk2 = file2.read(chunk_size)
            if chunk1 != chunk2:
                return False
            if not chunk1:
                return True
            
def test_compare_binary_files_empty():
    with tempfile.NamedTemporaryFile() as temp_file1, tempfile.NamedTemporaryFile() as temp_file2:
        temp_file1.write(b'')
        temp_file1.flush()
        file1_path = temp_file1.name
        temp_file2.write(b'')
        temp_file2.flush()
        file2_path = temp_file2.name
        assert compare_binary_files(file1_path, file2_path)

def test_compare_binary_files_one_empty_other_not_empty():
    with tempfile.NamedTemporaryFile() as temp_file1, tempfile.NamedTemporaryFile() as temp_file2:
        temp_file1.write(b'Some content')
        temp_file1.flush()
        file1_path = temp_file1.name
        temp_file2.write(b'')
        temp_file2.flush()
        file2_path = temp_file2.name
        assert not compare_binary_files(file1_path, file2_path)

def test_compare_binary_files_both_one_char():
    with tempfile.NamedTemporaryFile() as temp_file1, tempfile.NamedTemporaryFile() as temp_file2:
        temp_file1.write(b'a')
        temp_file1.flush()
        file1_path = temp_file1.name
        temp_file2.write(b'a')
        temp_file2.flush()
        file2_path = temp_file2.name
        assert compare_binary_files(file1_path, file2_path)

def test_compare_binary_files_one_one_char_second_two_chars():
    with tempfile.NamedTemporaryFile() as temp_file1, tempfile.NamedTemporaryFile() as temp_file2:
        temp_file1.write(b'a')
        temp_file1.flush()
        file1_path = temp_file1.name
        temp_file2.write(b'ab')
        temp_file2.flush()
        file2_path = temp_file2.name
        assert not compare_binary_files(file1_path, file2_path)

def test_compare_binary_files_both_same_two_chars():
    with tempfile.NamedTemporaryFile() as temp_file1, tempfile.NamedTemporaryFile() as temp_file2:
        temp_file1.write(b'ab')
        temp_file1.flush()
        file1_path = temp_file1.name
        temp_file2.write(b'ab')
        temp_file2.flush()
        file2_path = temp_file2.name
        assert compare_binary_files(file1_path, file2_path)

def test_compare_binary_files_same_files_bigger_than_chunk_size():
    with tempfile.NamedTemporaryFile() as temp_file1, tempfile.NamedTemporaryFile() as temp_file2:
        temp_file1.write(b'ab'*2**20)
        temp_file1.flush()
        file1_path = temp_file1.name
        temp_file2.write(b'ab'*2**20)
        temp_file2.flush()
        file2_path = temp_file2.name
        assert compare_binary_files(file1_path, file2_path)

def test_compare_binary_files_diff_files_bigger_than_chunk_size():
    with tempfile.NamedTemporaryFile() as temp_file1, tempfile.NamedTemporaryFile() as temp_file2:
        temp_file1.write(b'ab'*2**20)
        temp_file1.flush()
        file1_path = temp_file1.name
        temp_file2.write(b'ab'*2**20+ b'c')
        temp_file2.flush()
        file2_path = temp_file2.name
        assert not compare_binary_files(file1_path, file2_path)

# def test_compare_sobol_seed_file_1000_SL_and_WL():
#     file1_path = 'sobol_prng_files/1000'
#     file2_path = 'prng/sobol_prng_files/1000'

#     assert compare_binary_files(file1_path, file2_path)

# Change file_path in MockingPrng and n_values_per_file and chunk_size
from prng.lcg_prng import LcgPrng
from prng.mt_prng import MtPrng
from prng.xoroshio_prng import XoroshiroPrng
from prng.urandom_prng import UrandomPrng
from prng.halton_prng import HaltonPrng
from prng.sobol_prng import SobolPrng

prngs = [LcgPrng, MtPrng, XoroshiroPrng, UrandomPrng, HaltonPrng, SobolPrng]

@pytest.mark.parametrize("nonMockingGensClass", [LcgPrng, MtPrng, XoroshiroPrng])
def test_nonMockingGensClass_prng_consistancy_when_generator_initialised_in_different_order(nonMockingGensClass):
    seed = 1000
    dim = 10
    all_permutations = permutations(prngs)

    last = None

    for perm in all_permutations:
        gen_is = [perm[i](seed, dim) for i in range(len(perm))]
        for gen_i in gen_is:
            if isinstance(gen_i, nonMockingGensClass):
                if last is None:
                    last = gen_i.std_normal(dim)
                else:
                    new_last = gen_i.std_normal(dim)
                    assert np.array_equal(last, new_last)
                    last = new_last


@pytest.mark.parametrize("sobol_or_halton_Class", [SobolPrng, HaltonPrng])
def test_SOBOL_HALTONClass_prng_consistancy_when_generator_initialised_in_different_order(sobol_or_halton_Class):
    seed = 1000
    dim = 10
    all_permutations = permutations(prngs)

    last = None

    for perm in all_permutations:
        gen_is = [perm[i](seed, dim) for i in range(len(perm))]
        for gen_i in gen_is:
            if isinstance(gen_i, sobol_or_halton_Class):
                if last is None:
                    last = gen_i._gen_uniform(dim)
                else:
                    new_last = gen_i._gen_uniform(dim)
                    assert np.array_equal(last, new_last)
                    last = new_last

def test_urandomClass_prng_consistancy_when_generator_initialised_in_different_order():
    seed = 1000
    dim = 10
    all_permutations = permutations(prngs)

    last = None

    for perm in all_permutations:
        gen_is = [perm[i](seed, dim) for i in range(len(perm))]
        for gen_i in gen_is:
            if isinstance(gen_i, UrandomPrng):
                if last is None:
                    last = gen_i._gen_uniform(dim)
                else:
                    new_last = gen_i._gen_uniform(dim)
                    assert not np.array_equal(last, new_last) # URANDOM can not be seed-ed when not reading from file.
                    last = new_last