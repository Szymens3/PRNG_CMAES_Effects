import numpy as np
import os
import logging

import opfunu

from prng.lcg_prng import LCG_PRNG
from prng.mt_prng import MT_PRNG
from prng.xoroshio_prng import XOROSHIRO_PRNG
from prng.urandom_prng import URANDOM_PRNG
from prng.halton_prng import HALTON_PRNG
from prng.sobol_prng import SOBOL_PRNG
from cmaes.custom_cma import CustomCMA


def run_experiment(directory_path, algorithm_name, func_i, dim, func, rng, seeds, max_FES_coef = 10_000):
    max_FES = max_FES_coef * dim
    bounds = np.array([[-100, 100]] * dim)

    file_path = f"{directory_path}/{algorithm_name}_{func_i+1}_{dim}"
    problem = func(ndim=dim)

    x_global = problem.x_global
    y_global = problem.evaluate(x_global)

    run_checkpoints = max_FES * np.array([0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    experiments_results = np.zeros((len(run_checkpoints), len(seeds)))


    for experiment_i, seed in enumerate(seeds):
        rng_instance = rng(seed)
        initial_mean_vector = rng_instance.std_normal(dim)
        optimizer = CustomCMA(mean=initial_mean_vector, sigma=1.3, bounds=bounds, rng=rng_instance)

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
        logging.info(f"Experiment nr: {experiment_i+1} for function nr:{func_i+1}, dimension: {dim} executed")
    np.savetxt(file_path, experiments_results, fmt='%d', delimiter=' ')
    logging.info(f"Experiments for: function number {func_i+1}, dimension: {dim} have been completed and saved!")



    
    
def run_experiments_for_prngs(prngs, all_funcs_2017, algorithm_name, seeds, dim, max_FES_coef = 10_000):
    for prng in prngs:
        directory_path = f"results/{prng.name}"
        os.makedirs(directory_path, exist_ok=True)
        logging.info(f"Running experiments for: {prng.name} generator")
        for func_i, func in enumerate(all_funcs_2017):
            try:
                run_experiment(directory_path, algorithm_name, func_i, dim, func, prng, seeds, max_FES_coef)
            except SystemExit as e:
                logging.warning(f"Error for function {func_i+1} and dimension {dim}")
                logging.warning(e)
                continue
            except AssertionError as e:
                logging.warning(f"Error for function {func_i+1} and dimension {dim}")
                logging.warning(e)
                continue
            except FileNotFoundError as e:
                logging.warning(f"Error for function {func_i+1} and dimension {dim}")
                logging.warning(e)
                continue
            except Exception as e:
                logging.warning(f"Mysterious error for function {func_i+1} and dimension {dim}")
                logging.warning(e)
                continue

def main():
    logging.info(f"Starting experiments for Studying the Impact of Pseudo-Random Number Generator Selection on CMA-ES Algorithm.")
    result_directory = "results"
    logging.info(f"Experiments output are in {result_directory} directory")

    algorithm_name = "cmaes"
    seeds = list(range(1000,1051))
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
    prngs = [URANDOM_PRNG, SOBOL_PRNG, HALTON_PRNG,  XOROSHIRO_PRNG, MT_PRNG, LCG_PRNG]

    for dim in [10,30,50,100]:
        run_experiments_for_prngs(prngs, all_funcs_2017, algorithm_name, seeds, dim, max_FES_coef=10_000)

if __name__ == "__main__":
    log_name = 'experiments.log'
    logging.basicConfig(filename=log_name, encoding='utf-8', level=logging.DEBUG)
    main()
