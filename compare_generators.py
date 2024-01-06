import time, os

from scipy.special import erfinv
from scipy.stats import qmc, norm
import numpy as np
import matplotlib.pyplot as plt

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

# print(uniform_to_std_normal(np.array([np.float32(1.4901163e-08), np.float32(0.9999999)])))


class SOBOL_NO_FILES(PRNG):
    def __init__(self, seed, dim):
        self._prng = qmc.Sobol(d=dim, seed=seed)

    def __str__(self) -> str:
        return f"sobol_no_files{super().__str__()}"
    
    def std_normal(self, dim: int):
        return uniform_to_std_normal(self._prng.random(1).astype(np.float32).reshape(-1))
    
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
    def __init__(self, seed, dim):
        self._prng = qmc.Halton(d=dim, seed=seed)

    def __str__(self) -> str:
        return f"halton_no_files{super().__str__()}"
    
    def std_normal(self, dim: int):
        return uniform_to_std_normal(self._prng.random(1).astype(np.float32).reshape(-1))
    
class HALTON_NO_FILES_depend(PRNG):
    def __init__(self, seed, dim):
        self._prng = qmc.Halton(d=1, seed=seed)

    def __str__(self) -> str:
        return f"halton_with_dependent_values_per_dim_{super().__str__()}"
    
    def std_normal(self, dim: int, n=1):
        return uniform_to_std_normal(self._prng.random(n).astype(np.float32))
    
import cProfile


#gens = [XOROSHIRO_PRNG, MT_PRNG, LCG_PRNG, SOBOL_PRNG, HALTON_PRNG, URANDOM_PRNG]

def test_gen_i(gen_i, n, dim):
    start = time.process_time()
    for i in range(n):
        v = gen_i.std_normal(dim)
    end = time.process_time()
    print(f"Gen: {gen_i} took: {end-start} s to generate {n} times {dim} values from std_norm distribution")


def compare_speed(n=1_000_000, dim=100, gens=[XOROSHIRO_PRNG, MT_PRNG, LCG_PRNG, SOBOL_PRNG, SOBOL_NO_FILES, HALTON_PRNG, HALTON_NO_FILES_1_worker, URANDOM_PRNG]):
    for gen in gens:
        gen_i = gen(seed, dim)
        # cProfile.run(f'test_gen_i(gen_i, n={n}, dim={dim})', sort='cumulative')
        test_gen_i(gen_i, n=n, dim=dim)

def visualize_gen_instance_values_distribution_per_coordinate(gen_i: PRNG, dim=10, total_nr_sampled_values=100_000_000):
    plt.figure(figsize=(24, 12))
    values = []
    for i in range(total_nr_sampled_values//dim):
        values.append(gen_i.std_normal(dim))
    values = np.concatenate(values, axis=0)

    for d in range(dim):
        plt.subplot(((d)// 10)+1, 10, d+1)
        every_dim = values[d::dim]
        plt.hist(every_dim, bins=20, alpha=0.7)
        plt.title(f"Dim {d+1}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
    plt.suptitle(gen_i, fontsize=16)
    plt.tight_layout()
    plt.show()

class HALTON_WITH_DIM(PRNG):
    def __init__(self, seed, dim):
        self._dim = dim
        self._prng = qmc.Halton(d=dim, scramble=True, seed=seed)

    def __str__(self) -> str:
        return f"halton_with_dim_{self._dim}_{super().__str__()}"
    
    def std_normal(self, dim: int):
        return uniform_to_std_normal(self._prng.random(1).reshape(-1))

def plot_x_numbers_from_file(file_path, n_of_numbers, bin_count=100):

    with open(file_path, 'rb') as file:
        start = time.process_time()
        file_size = os.path.getsize(file_path)
        print(f"File size: {file_size} B")
        buff = file.read(min(n_of_numbers * 4, file_size))
        values = np.frombuffer(buff, dtype=np.float32)
        print("Getting values time:", time.process_time()-start)
        print("Plotting ", len(values), " values")
        plot_numbers(values, file_path)

def plot_numbers(values, title ,bin_count=100):
        plt.figure(figsize=(36, 12))
        plt.hist(values, bins=bin_count, alpha=0.7)
        plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()



# if __name__ == "__main__":
#     # gens = [SOBOL_PRNG, HALTON_PRNG] #, URANDOM_PRNG, XOROSHIRO_PRNG, MT_PRNG, LCG_PRNG]
#     # seed = 1001
#     # dim = 100
#     # nr_values_generated = 10**6
#     # for dim in [10,30,50,100]:
#     #     for gen in gens:
#     #         gen_i = gen(seed, dim)
#     #         # visualize_gen_instance_values_distribution_per_coordinate(gen_i, dim, nr_values_generated)


#     # visualize_gen_instance_values_distribution_per_coordinate(HALTON_NO_FILES_depend(seed, 10), 10, 10_000_000)


#     # compare_speed()


    
#     # plot_x_numbers_from_file('prng/sobol_prng_files/1001_10',2**30)

#     gens = [SOBOL_NO_FILES, HALTON_NO_FILES_1_worker]
#     seeds =list(range(1000,1051))
#     max_FES_coef = 10_000
#     for gen in gens:
#         for dim in [10,30,50,100]:
#             st_s = time.process_time()
#             for seed in seeds:
#                 gen_i = gen(seed, dim)
#                 st = time.process_time()
#                 for i in range(max_FES_coef*dim):
#                     point = gen_i.std_normal(dim)
#                 print(f"{gen_i} seed: {seed} generated {max_FES_coef*dim} points of dim: {dim} in {time.process_time()-st} s")
#             print(f"{gen_i} generated vals for all seeds for dim: {dim} in {time.process_time()-st_s} s")


#     # 
#     # for gen in gens:
#     #     for dim in [10,30,50,100]:
#     #         st = time.process_time()
#     #         for seed in seeds:
#     #             gen_i = gen(seed, dim)
#     #         print(f"Time to generate all 51 files for gen: {gen.name} for dim: {dim} generated in: {time.process_time()-st} s")


#     # h = HALTON_WITH_DIM(seed, dim)
#     # n_of_points = 1000_000
#     # values = []
#     # for i in range(n_of_points):
#     #     values.append(h.std_normal(None))
#     # values = np.concatenate(values, axis=0)
#     # filtered_values = values[np.logical_or(values > 5, values < -5)]
#     # print(h, filtered_values)
#     # plot_numbers(values, h)



