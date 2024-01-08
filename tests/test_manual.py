import os
import numpy as np
from scipy.special import erfinv


def uniform_to_std_normal(uniform_numbers):
    normal_numbers = np.sqrt(2) * erfinv(2 * uniform_numbers - 1)
    return normal_numbers


def test_sqrt_2_type():
    x = np.sqrt(2)
    assert x > 1
    assert x < 2
    assert type(x) == np.float64


def test_type_when_casting():
    seed = 1000
    dim = 10
    gen = np.random.RandomState(seed)
    sample = gen.randn(dim)
    sample = sample.astype(np.float32)
    assert sample.shape == (dim,)
    assert sample.dtype == np.float32


def test_non_zero_and_non_one():
    seed = 1000
    dim = 10
    gen = np.random.RandomState(seed)
    sample = gen.randn(dim)
    assert np.all(sample)
    assert np.sum(sample == 1) == 0
    sample = sample.astype(np.float32)
    assert np.all(sample)
    assert np.sum(sample == 1) == 0


def test_threshold():
    numbers = np.array(range(0, 4), dtype=np.float32)
    left_threshold = np.float32(2.5)
    numbers[numbers < left_threshold] = left_threshold
    expected = np.array([np.float32(2.5), np.float32(2.5), np.float32(2.5), 3])
    assert np.array_equal(numbers, expected)


def test_chunk_smaller_than_file():
    chunk_size = 2
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
    file_name = "data_test"
    data.tofile(file_name)
    assert os.path.getsize(file_name) == len(data) * 4
    with open("data_test", "rb") as file:
        chunk = file.read(chunk_size * 4)
        buffered_values = np.frombuffer(chunk, dtype=np.float32)
        assert np.array_equal(buffered_values, np.array(range(1, 3), dtype=np.float32))


def test_chunk_exactly_the_same_size_as_file():
    chunk_size = 10
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
    file_name = "data_test"
    data.tofile(file_name)
    assert os.path.getsize(file_name) == len(data) * 4
    with open("data_test", "rb") as file:
        chunk = file.read(chunk_size * 4)
        buffered_values = np.frombuffer(chunk, dtype=np.float32)
        assert np.array_equal(buffered_values, np.array(range(1, 11), dtype=np.float32))


def test_chunk_slighthly_bigger_size_than_file():
    chunk_size = 11
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
    file_name = "data_test"
    data.tofile(file_name)
    assert os.path.getsize(file_name) == len(data) * 4
    with open("data_test", "rb") as file:
        chunk = file.read(chunk_size * 4)
        buffered_values = np.frombuffer(chunk, dtype=np.float32)
        assert np.array_equal(buffered_values, np.array(range(1, 11), dtype=np.float32))


def test_chunk_size_90_values_bigger_size_than_file():
    chunk_size = 100
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
    file_name = "data_test"
    data.tofile(file_name)
    assert os.path.getsize(file_name) == len(data) * 4
    with open("data_test", "rb") as file:
        chunk = file.read(chunk_size * 4)
        buffered_values = np.frombuffer(chunk, dtype=np.float32)
        assert np.array_equal(buffered_values, np.array(range(1, 11), dtype=np.float32))


def test_chunk_size_90_values_bigger_size_than_file():
    chunk_size = 100
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
    file_name = "data_test"
    data.tofile(file_name)
    assert os.path.getsize(file_name) == len(data) * 4
    with open("data_test", "rb") as file:
        chunk = file.read(chunk_size * 4)
        buffered_values = np.frombuffer(chunk, dtype=np.float32)
        assert np.array_equal(buffered_values, np.array(range(1, 11), dtype=np.float32))


import scipy.stats as stats

def probability_less_than_threshold(mu, sigma, threshold):
    # Create a normal distribution with mean mu and standard deviation sigma
    dist = stats.norm(mu, sigma)
    
    # Calculate the probability of getting a value less than the threshold
    probability = dist.cdf(threshold)
    
    return probability

# # Example usage:
# mean = 0  # Mean of the normal distribution
# std_dev = 1  # Standard deviation of the normal distribution
# threshold_value = -5  # Threshold value

# prob_less_than_threshold = probability_less_than_threshold(mean, std_dev, threshold_value)
# print(f"The probability of getting a value less than {threshold_value} is {prob_less_than_threshold}")
