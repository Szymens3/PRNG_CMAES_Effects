"""Module for comparing all implemented pseudo random number generators"""

import time
import os
from typing import List

from scipy.special import erfinv
from scipy.stats import qmc
import numpy as np
import matplotlib.pyplot as plt

from prng.prng import Prng
from prng.lcg_prng import LcgPrng
from prng.mt_prng import MtPrng
from prng.xoroshio_prng import XoroshiroPrng
from prng.urandom_prng import UrandomPrng
from prng.halton_prng import HaltonPrng
from prng.sobol_prng import SobolPrng

DEFAULT_SEED = 1000
DEFAULT_DIM = 10
NR_OF_VALUES_TO_PLOT = 10**6
CORRECT_DIMS = [10, 30, 50, 100]


def uniform_to_std_normal(uniform_numbers):
    """
    Maps [0,1) to aprox. [-5.5, 5.5] using inverse of error function.
    Limited range because numerical errors
    """
    normal_numbers = np.sqrt(2) * erfinv(2 * uniform_numbers - 1)
    return normal_numbers


# print(uniform_to_std_normal(np.array([np.float32(1.4901163e-08), np.float32(0.9999999)])))


class SobolGenOnTheGo(Prng):
    """Class analogous to SobolPrng but generates values on the go"""

    def __init__(self, seed, dim):
        self._prng = qmc.Sobol(d=dim, seed=seed)

    def __str__(self) -> str:
        return f"sobol_no_files{super().__str__()}"

    def std_normal(self, dim: int, n: int = 1):
        return uniform_to_std_normal(
            self._prng.random(1).astype(np.float32).reshape(-1)
        )


class HaltonGenOnTheGo(Prng):
    """Class analogous to HaltonPrng but generates values on the go"""

    def __init__(self, seed, dim):
        self._prng = qmc.Halton(d=dim, seed=seed)

    def __str__(self) -> str:
        return f"halton_no_files{super().__str__()}"

    def std_normal(self, dim: int, n: int = 1):
        return uniform_to_std_normal(
            self._prng.random(n).astype(np.float32).reshape(-1)
        )


class HaltonGenOnTheGoDimDepended(Prng):
    """
    Class analogous to HaltonPrng but generates values on the go.
    Values in each dim are interdepended
    """

    def __init__(self, seed, dim):
        self._prng = qmc.Halton(d=1, seed=seed)

    def __str__(self) -> str:
        return f"halton_with_dependent_values_per_dim_{super().__str__()}"

    def std_normal(self, dim: int, n=1):
        return uniform_to_std_normal(self._prng.random(n).astype(np.float32))


def test_gen_i(gen_i, n, dim):
    """Tests the speed of generating n times dim values for generator instance"""
    start = time.process_time()
    for _ in range(n):
        _ = gen_i.std_normal(dim)
    end = time.process_time()
    print(
        f"Gen: {gen_i} took: {end-start} s to generate {n} times \
            {dim} values from std_norm distribution"
    )


all_gens: List(Prng) = [
    XoroshiroPrng,
    MtPrng,
    LcgPrng,
    SobolPrng,
    SobolGenOnTheGo,
    HaltonPrng,
    HaltonGenOnTheGo,
    UrandomPrng,
]

# pylint: disable=W0102
def compare_speed(n=1_000_000, dim=100, gens: List(Prng) = all_gens, seed=DEFAULT_SEED):
    """Tests the speed of generating n times dim values"""
    for gen in gens:
        gen_i = gen(seed, dim)
        # cProfile.run(f'test_gen_i(gen_i, n={n}, dim={dim})', sort='cumulative')
        test_gen_i(gen_i, n=n, dim=dim)


def visualize_gen_instance_values_distribution_per_coordinate(
    gen_i: Prng, dim=DEFAULT_DIM, total_nr_sampled_values=100_000_000
):
    """Plot dim histograms of values returned by generator instance"""
    plt.figure(figsize=(24, 12))
    values = []
    for _ in range(total_nr_sampled_values // dim):
        values.append(gen_i.std_normal(dim))
    values = np.concatenate(values, axis=0)

    for d in range(dim):
        plt.subplot(((d) // 10) + 1, 10, d + 1)
        every_dim = values[d::dim]
        plt.hist(every_dim, bins=20, alpha=0.7)
        plt.title(f"Dim {d+1}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
    plt.suptitle(gen_i, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_x_numbers_from_file(file_path, n_of_numbers, bin_count=100):
    """Plot first n_of_numbers np.float32 values from binary file"""
    with open(file_path, "rb") as file:
        start = time.process_time()
        file_size = os.path.getsize(file_path)
        print(f"File size: {file_size} B")
        buff = file.read(min(n_of_numbers * 4, file_size))
        values = np.frombuffer(buff, dtype=np.float32)
        print("Getting values time:", time.process_time() - start)
        print("Plotting ", len(values), " values")
        plot_numbers(values, file_path, bin_count)


def plot_numbers(values, title, bin_count=100):
    """Plot values"""
    plt.figure(figsize=(36, 12))
    plt.hist(values, bins=bin_count, alpha=0.7)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()


if __name__ == "__main__":
    gens_used_in_experiments = [
        SobolPrng,
        HaltonPrng,
        UrandomPrng,
        XoroshiroPrng,
        MtPrng,
        LcgPrng,
    ]
    for dim in CORRECT_DIMS:
        for gen in gens_used_in_experiments:
            gen_i = gen(DEFAULT_SEED, dim)
            visualize_gen_instance_values_distribution_per_coordinate(
                gen_i, dim, NR_OF_VALUES_TO_PLOT
            )

    TOTAL_NR_VALUES_TO_SAMPLE = 10_000_000
    halton_no_files_dim_depended = HaltonGenOnTheGo(DEFAULT_SEED, DEFAULT_DIM)
    visualize_gen_instance_values_distribution_per_coordinate(
        halton_no_files_dim_depended, DEFAULT_DIM, TOTAL_NR_VALUES_TO_SAMPLE
    )

    compare_speed()

    plot_x_numbers_from_file("prng/sobol_prng_files/1001_10", 2**30)

    slow_gens = [SobolGenOnTheGo, HaltonGenOnTheGo]
    seeds = list(range(1000, 1051))
    MAX_FES_COEF = 10_000
    for gen in slow_gens:
        for dim in [10, 30, 50, 100]:
            st_s = time.process_time()
            for seed in seeds:
                gen_i = gen(seed, dim)
                st = time.process_time()
                for i in range(MAX_FES_COEF * dim):
                    point = gen_i.std_normal(dim)
                print(
                    f"{gen_i} seed: {seed} generated {MAX_FES_COEF*dim} points \
                        of dim: {dim} in {time.process_time()-st} s"
                )
            print(
                f"{gen_i} generated vals for all seeds for dim: {dim} \
                    in {time.process_time()-st_s} s"
            )

    file_using_gens = [HaltonPrng, SobolPrng, UrandomPrng]
    for gen in file_using_gens:
        for dim in CORRECT_DIMS:
            st = time.process_time()
            for seed in seeds:
                gen_i = gen(seed, dim)
            print(
                f"Time to generate all 51 files for gen: {gen.name} \
                    for dim: {dim} generated in: {time.process_time()-st} s"
            )


    h = HaltonPrng(seed, dim)
    NR_OF_POINTS = 1000_000
    values = []
    for i in range(NR_OF_POINTS):
        values.append(h.std_normal(None))
    values = np.concatenate(values, axis=0)
    filtered_values = values[np.logical_or(values > 5, values < -5)]
    print(h, filtered_values)
    plot_numbers(values, h)
