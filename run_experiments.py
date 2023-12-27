import opfunu
import numpy as np
import os
import argparse

from prng.lcg_prng import LCG_PRNG
from prng.mt_prng import MT_PRNG
from prng.xoroshio_prng import XOROSHIRO_PRNG
from prng.urandom_prng import URANDOM_PRNG
from prng.halton_prng import HALTON_PRNG
from prng.sobol_prng import SOBOL_PRNG
from cmaes.custom_cma import CustomCMA


def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def run_experiment(directory_path, func_i, func, dim, algorithm_name, rng, N_EXPERIMENTS_PER_FUNC_PER_DIM=51, max_FES_coef = 10000):

    # if func_i in [5,17,29]:
    #     print("Problematic functions")

    max_FES = max_FES_coef * dim
    bounds = np.array([[-100, 100]] * dim)

    create_directory_if_not_exists(directory_path)

    file_path = f"{directory_path}/{algorithm_name}_{func_i+1}_{dim}"
    problem = func(ndim=dim)

    x_global = problem.x_global
    y_global = problem.evaluate(x_global)

    run_checkpoints = max_FES * np.array([0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    experiments_results = np.zeros((len(run_checkpoints), N_EXPERIMENTS_PER_FUNC_PER_DIM))


    for experiment_i in range(N_EXPERIMENTS_PER_FUNC_PER_DIM):

        initial_mean_vector = rng.std_normal(dim)
        optimizer = CustomCMA(mean=initial_mean_vector, sigma=1.3, bounds=bounds, rng=rng)

        run_results = np.zeros(len(run_checkpoints))
        checkpoint_pointer = 0
        x = initial_mean_vector
        value = problem.evaluate(x)
        FES = 1
        while abs(value - y_global) > 10**(-8) and FES < max_FES:
            solutions = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()
                value = problem.evaluate(x)
                solutions.append((x, value))
            optimizer.tell(solutions)
            FES += optimizer.population_size
            if FES >= run_checkpoints[checkpoint_pointer]:
                run_results[checkpoint_pointer] = abs(value - y_global)
                checkpoint_pointer += 1

        experiments_results[:, experiment_i] = run_results
    np.savetxt(file_path, experiments_results, fmt='%d', delimiter=' ')
    print(f"Experiments for: function number {func_i+1}, dimension: {dim} has been completed and saved!")

if __name__ == "__main__":

    algorithm_name = "cmaes"
    print(f"Starting experiments for Studying the Impact of Pseudo-Random Number Generator Selection on CMA-ES Algorithm.")

    all_funcs_2017 = [
    opfunu.cec_based.F12017,
    opfunu.cec_based.F22017,
    opfunu.cec_based.F32017,
    opfunu.cec_based.F42017,
    opfunu.cec_based.F52017,
    opfunu.cec_based.F62017,
    opfunu.cec_based.F72017,
    opfunu.cec_based.F82017,
    opfunu.cec_based.F92017,
    opfunu.cec_based.F102017,
    opfunu.cec_based.F112017,
    opfunu.cec_based.F122017,
    opfunu.cec_based.F132017,
    opfunu.cec_based.F142017,
    opfunu.cec_based.F152017,
    opfunu.cec_based.F162017,
    opfunu.cec_based.F172017,
    opfunu.cec_based.F182017,
    opfunu.cec_based.F192017,
    opfunu.cec_based.F202017,
    opfunu.cec_based.F212017,
    opfunu.cec_based.F222017,
    opfunu.cec_based.F232017,
    opfunu.cec_based.F242017,
    opfunu.cec_based.F252017,
    opfunu.cec_based.F262017,
    opfunu.cec_based.F272017,
    opfunu.cec_based.F282017,
    opfunu.cec_based.F292017,
]

    parser = argparse.ArgumentParser(prog="CMA-ES trainer")
    parser.add_argument('problem_dim', type=int)
    args = parser.parse_args()

    dim = args.problem_dim
    N_EXPERIMENTS_PER_FUNC_PER_DIM = 5  # TODO change to 51
    max_FES_coef = 10_000

    lcg_rng = LCG_PRNG(1234)
    mt_rng = MT_PRNG(1234)
    xoroshiro_rng = XOROSHIRO_PRNG(1234)
    halton_prng = HALTON_PRNG(1234)
    sobol_prng = SOBOL_PRNG(1234)
    urandom_rng = URANDOM_PRNG('prng/urandom.json')

    prngs = [lcg_rng, mt_rng, xoroshiro_rng, urandom_rng]  # halton_prng, sobol_prng

    result_directory = "results"
    create_directory_if_not_exists(result_directory)
    print(f"Experiments output are in {result_directory} direcotry")

    for prng in prngs:
        directory_path = f"results/{prng}"
        print(f"Running experiments for: {prng} generator")
        for func_i, func in enumerate(all_funcs_2017):
            run_experiment(directory_path, func_i, func, dim, algorithm_name, prng, N_EXPERIMENTS_PER_FUNC_PER_DIM, max_FES_coef)
