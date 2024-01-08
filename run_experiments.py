"""
Module for Running Experiments to
Study the Impact of Pseudo-Random Number Generator Selection on CMA-ES Algorithm
"""
import os
import logging
import numpy as np

from cec2017.functions import all_functions

from prng.lcg_prng import LcgPrng
from prng.mt_prng import MtPrng
from prng.xoroshio_prng import XoroshiroPrng
from prng.urandom_prng import UrandomPrng
from prng.halton_prng import HaltonPrng
from prng.sobol_prng import SobolPrng
from cmaes.custom_cma import CustomCMA


def _run_experiment(
    directory_path, algorithm_name, func_i, dim, func, rng, seeds, max_fes_coef=10_000
):
    max_fes = max_fes_coef * dim
    bounds = np.array([[-100, 100]] * dim)
    run_checkpoints = max_fes * np.array(
        [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )

    file_path = f"{directory_path}/{algorithm_name}_{func_i+1}_{dim}"

    y_global = (
        func_i + 1
    ) * 100  # WORKS ONLY if EXPERIMENTS ARE RUN WITH ALL FUNCTIONS

    experiments_results = np.zeros((len(run_checkpoints), len(seeds)))

    for experiment_i, seed in enumerate(seeds):
        rng_instance = rng(seed, dim)
        initial_mean_vector = rng_instance.std_normal(dim)
        optimizer = CustomCMA(
            mean=initial_mean_vector, sigma=1.3, bounds=bounds, rng=rng_instance
        )

        run_results = np.zeros(len(run_checkpoints))
        checkpoint_pointer = 0
        # pylint: disable=W0212
        value = func([optimizer._mean])[0]
        fes = 1
        while abs(value - y_global) > 10 ** (-8) and fes < max_fes:
            xs = [optimizer.ask() for i in range(optimizer.population_size)]
            values = func(xs)
            solutions = [(xs[i], values[i]) for i in range(optimizer.population_size)]
            optimizer.tell(solutions)
            value = func([optimizer._mean])[0]
            fes += optimizer.population_size + 1  # + 1 for evaluating optimizer._mean
            if fes >= run_checkpoints[checkpoint_pointer]:
                run_results[checkpoint_pointer] = abs(value - y_global)
                checkpoint_pointer += 1

        experiments_results[:, experiment_i] = run_results
        logging.info(
            _get_msg_after_experiment_per_seed(experiment_i + 1, seed, func_i + 1, dim)
        )
    np.savetxt(file_path, experiments_results, fmt="%d", delimiter=" ")
    logging.info(_get_msg_after_all_seeds(func_i + 1, dim))


def _get_msg_after_all_seeds(func_nr, dim):
    return f"Experiments for: function number {func_nr}, \
        dimension: {dim} have been completed and saved!"


def _get_msg_after_experiment_per_seed(experiment_nr, seed, func_nr, dim):
    return f"Experiment nr: {experiment_nr} with seed: {seed} \
        for function nr:{func_nr}, dimension: {dim} executed"


def _run_experiments_for_prngs(
    prngs, all_funcs_2017, algorithm_name, seeds, dim, max_fes_coef=10_000
):
    # pylint: disable=W0640
    for prng in prngs:
        directory_path = f"results/{prng.name}"
        os.makedirs(directory_path, exist_ok=True)
        logging.info(f"Running experiments for: {prng.name} generator")
        for func_i, func in enumerate(all_funcs_2017):
            try:
                _run_experiment(
                    directory_path,
                    algorithm_name,
                    func_i,
                    dim,
                    func,
                    prng,
                    seeds,
                    max_fes_coef,
                )
            except SystemExit as e:
                logging.warning(
                    f"Error for function {func_i+1} and dimension {dim}"
                )
                logging.warning(e)
                continue
            except AssertionError as e:
                logging.warning(
                    f"Error for function {func_i+1} and dimension {dim}"
                )
                logging.warning(e)
                continue
            except FileNotFoundError as e:
                logging.warning(
                    f"Error for function {func_i+1} and dimension {dim}"
                )
                logging.warning(e)
                continue
            # pylint: disable=W0718
            except Exception as e:
                logging.warning(
                    f"Mysterious error for function {func_i+1} and dimension {dim}"
                )
                logging.warning(e)
                continue


def main():
    """Function to run all experiments"""
    logging.info(
        "Starting experiments for Studying the Impact of Pseudo-Random \
            Number Generator Selection on CMA-ES Algorithm."
    )
    result_directory = "results"
    logging.info(f"Experiments output are in {result_directory} directory")

    algorithm_name = "cmaes"
    seeds = list(range(1000, 1030))
    all_funcs_2017 = all_functions
    prngs = [UrandomPrng, SobolPrng, HaltonPrng, XoroshiroPrng, MtPrng, LcgPrng]
    for dim in [10, 30, 50, 100]:
        _run_experiments_for_prngs(
            prngs, all_funcs_2017, algorithm_name, seeds, dim, max_fes_coef=10_000
        )


# pylint: disable=W0613
def _handle_numpy_warnings(message, category, filename, lineno, file=None, line=None):
    logging.warning(f"NumPy warning: {category.__name__}: {message}")


if __name__ == "__main__":
    import warnings

    warnings.showwarning = _handle_numpy_warnings
    np.seterr(over="warn")
    # pylint: disable=C0103
    log_name = "experiments.log"
    logging.basicConfig(filename=log_name, encoding="utf-8", level=logging.DEBUG)
    main()
    warnings.resetwarnings()
