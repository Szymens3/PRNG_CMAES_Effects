"""
Module for Running Experiments to
Study the Impact of Pseudo-Random Number Generator Selection on CMA-ES Algorithm
"""
import os
import sys
import logging
import warnings
import argparse
from multiprocessing import Process

import numpy as np

from cec2017.functions import all_functions

from prng.lcg_prng import LcgPrng
from prng.mt_prng import MtPrng
from prng.xoroshio_prng import XoroshiroPrng
from prng.urandom_prng import UrandomPrng
from prng.halton_prng import HaltonPrng
from prng.sobol_prng import SobolPrng
from cmaes.custom_cma import CustomCMA

ALGORITHM_NAME = "cmaes"


def _run_experiment(
    logger,
    prng,
    i,
    func,
    seeds,
    dim,
    result_dir,
    max_fes_coef=10_000,
):
    max_fes = max_fes_coef * dim
    bounds = np.array([[-100, 100]] * dim)
    run_checkpoints = max_fes * np.array(
        [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )

    file_path = f"{result_dir}/{ALGORITHM_NAME}_{i}_{dim}"

    y_global = i * 100

    experiments_results = np.zeros((len(run_checkpoints), len(seeds)))

    for experiment_i, seed in enumerate(seeds):
        rng_instance = prng(seed, dim, logger=logger)
        initial_mean_vector = rng_instance.std_normal(dim)
        optimizer = CustomCMA(mean=initial_mean_vector, sigma=1.3, bounds=bounds, rng=rng_instance)

        run_results = np.zeros(len(run_checkpoints))
        checkpoint_pointer = 0
        # pylint: disable=W0212
        value = func([optimizer._mean])[0]
        fes = 1
        while abs(value - y_global) > 10 ** (-8) and fes < max_fes:
            xs = [optimizer.ask() for _ in range(optimizer.population_size)]
            values = func(xs)
            solutions = [(xs[j], values[j]) for j in range(optimizer.population_size)]
            optimizer.tell(solutions)
            value = func([optimizer._mean])[0]
            fes += optimizer.population_size + 1  # + 1 for evaluating optimizer._mean
            if fes >= run_checkpoints[checkpoint_pointer]:
                run_results[checkpoint_pointer] = abs(value - y_global)
                checkpoint_pointer += 1

        experiments_results[:, experiment_i] = run_results
        logger.info(_get_msg_after_experiment_per_seed((experiment_i + 1), seed, i, dim))
    np.savetxt(file_path, experiments_results, fmt="%d", delimiter=" ")
    logger.info(_get_msg_after_all_seeds(i, dim))


def _get_msg_after_all_seeds(func_nr, dim):
    return f"Experiments for: function number {func_nr}, dimension: {dim} have been completed and saved!"


def _get_msg_after_experiment_per_seed(experiment_nr, seed, func_nr, dim):
    return f"Experiment nr: {experiment_nr} with seed: {seed} for function nr:{func_nr}, dimension: {dim} executed"


def _run_experiments_for_prng_in_try_catch(
    logger,
    prng,
    i,
    func,
    seeds,
    dim,
    result_dir,
    max_fes_coef=10_000,
):
    try:
        _run_experiment(
            logger,
            prng,
            i,
            func,
            seeds,
            dim,
            result_dir,
            max_fes_coef=max_fes_coef,
        )
    except SystemExit as e:
        logger.warning(f"Error for function {i} and dimension {dim}")
        logger.warning(e)
    except AssertionError as e:
        logger.warning(f"Error for function {i} and dimension {dim}")
        logger.warning(e)
    except FileNotFoundError as e:
        logger.warning(f"Error for function {i} and dimension {dim}")
        logger.warning(e)
    # pylint: disable=W0718
    except Exception as e:
        logger.warning(f"Mysterious error for function {i} and dimension {dim}")
        logger.warning(e)


def _setup_logger_per_prng_per_seed(prng, seed, log_dir):
    # pylint: disable=C0103
    log_path = f"{log_dir}/file_generation_seed_{seed}.log"

    logger = logging.getLogger(f"logger_{prng.name}_seed_{seed}")
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    def _handle_numpy_warnings(message, category, filename, lineno, file=None, line=None):
        logger.warning(f"NumPy warning: {category.__name__}: {message}")

    np.seterr(over="warn")
    warnings.showwarning = _handle_numpy_warnings
    return logger


def _setup_main_logger(log_dir, main_log_name):
    # pylint: disable=C0103
    log_path = f"{log_dir}/{main_log_name}"

    logger = logging.getLogger(f"logger_main")
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    def _handle_numpy_warnings(message, category, filename, lineno, file=None, line=None):
        logger.warning(f"NumPy warning: {category.__name__}: {message}")

    np.seterr(over="warn")
    warnings.showwarning = _handle_numpy_warnings
    return logger


def _setup_logger_per_prgn_per_function(prng, i, log_dir):
    # pylint: disable=C0103
    log_path = f"{log_dir}/{prng.name}/experiments_for_func_{i}.log"

    logger = logging.getLogger(f"logger_for_function_{i}")
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    def _handle_numpy_warnings(message, category, filename, lineno, file=None, line=None):
        logger.warning(f"NumPy warning: {category.__name__}: {message}")

    np.seterr(over="warn")
    warnings.showwarning = _handle_numpy_warnings
    return logger


def _run_experiments_per_function(prng, i, func, seeds, dims, result_dir, log_dir):
    logger = _setup_logger_per_prgn_per_function(prng, i, log_dir)

    for dim in dims:
        _run_experiments_for_prng_in_try_catch(
            logger,
            prng,
            i,
            func,
            seeds,
            dim,
            result_dir,
            max_fes_coef=10_000,
        )


def _split_work_per_function(prng, seeds, dims, result_dir, log_dir):
    ps = []
    for i, func in enumerate(all_functions, start=1):
        p = Process(
            target=_run_experiments_per_function,
            args=(prng, i, func, seeds, dims, result_dir, log_dir),
        )
        p.start()
        ps.append(p)

    for i, func in enumerate(all_functions):
        ps[i].join()


def _split_work_per_generators(prngs, seeds, dims, result_dir, log_dir):
    ps = []
    for prng in prngs:
        results_per_prng_dir = f"{result_dir}/{prng.name}"
        os.makedirs(f"{log_dir}/{prng.name}", exist_ok=True)
        os.makedirs(results_per_prng_dir, exist_ok=True)
        p = Process(
            target=_split_work_per_function,
            args=(prng, seeds, dims, results_per_prng_dir, log_dir),
        )
        p.start()
        ps.append(p)

    for i in range(len(prngs)):
        ps[i].join()


def _initialize_prng_per_dims(prng, seed, dims, log_dir):
    logger = _setup_logger_per_prng_per_seed(prng, seed, log_dir)
    for dim in dims:
        _ = prng(seed, dim, logger=logger)


def _split_file_generation_per_seed(prng, seeds, dims, log_dir):
    ps = []
    for seed in seeds:
        p = Process(
            target=_initialize_prng_per_dims,
            args=(prng, seed, dims, log_dir),
        )
        p.start()
        ps.append(p)

    for i in range(len(seeds)):
        ps[i].join()


def _pregenerate_files(prngs, seeds, dims, log_dir):
    ps = []
    for prng in prngs:
        file_generation_subdirectory = f"{log_dir}/file_generation/{prng.name}"
        os.makedirs(f"{file_generation_subdirectory}", exist_ok=True)
        p = Process(
            target=_split_file_generation_per_seed,
            args=(prng, seeds, dims, file_generation_subdirectory),
        )
        p.start()
        ps.append(p)

    for i in range(len(prngs)):
        ps[i].join()


def _get_file_paths_for_sobol_or_halton(seeds, dims, sobol_or_halton, file_dir="prng"):
    paths = []
    for seed in seeds:
        for dim in dims:
            paths.append(f"{file_dir}/{sobol_or_halton.name}_prng_files/{seed}_{dim}")
    return paths


def _get_file_paths_for_urandom(seeds, urandom, file_dir="prng"):
    paths = []
    for seed in seeds:
        paths.append(f"{file_dir}/{urandom.name}_prng_files/{seed}")
    return paths


def _check_and_log_file_path_exists(file_path, logger: logging.Logger):
    if os.path.exists(file_path):
        logger.info(f"{file_path} exists")
        return True
    else:
        logger.critical(f"{file_path} does not exist!")
        return False


def _check_file_paths_exist(file_paths, logger):
    all_good = True
    for file_path in file_paths:
        if not _check_and_log_file_path_exists(file_path, logger):
            all_good = False
    return all_good


def _check_generators_file_paths_exist(prngs, seeds, dims, logger):
    all_good = True
    prng_good = False
    for prng in prngs:
        if prng.name == "halton" or prng.name == "sobol":
            sobol_or_halton_expected_file_paths = _get_file_paths_for_sobol_or_halton(
                seeds, dims, prng
            )
            prng_good = _check_file_paths_exist(sobol_or_halton_expected_file_paths, logger)
        elif prng.name == "urandom":
            urandom_expected_file_paths = _get_file_paths_for_urandom(seeds, prng)
            prng_good = _check_file_paths_exist(urandom_expected_file_paths, logger)
        else:
            logger.warning(f"Trying to check for file existance for generator: {prng}")
        if prng_good:
            logger.info(f"All files for {prng} exists")
        else:
            all_good = False
            logger.critical(f"Not all files for {prng} exist")
    return all_good


def _check_files_size_per_dim(file_paths, dim, logger):
    all_good = True
    CHUNK_SIZE = 2**20
    FLOAT_32_BYTE_SIZE = 4

    correct_file_size = CHUNK_SIZE * dim * FLOAT_32_BYTE_SIZE
    for file_path in file_paths:
        real_file_size = os.path.getsize(file_path)
        if real_file_size == correct_file_size:
            logger.info(f"File: {file_path} has correct size")
        else:
            logger.critical(
                f"File {file_path} does not have expected size: Real size: {real_file_size}, expected size: {correct_file_size}"
            )
            all_good = False
    return all_good


def _group_file_paths_based_on_dim(file_paths, expected_dims=[10, 30, 50, 100]):
    grouped_paths = {dim: [] for dim in expected_dims}

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        file_name_parts = file_name.split("_")

        if len(file_name_parts) == 2:
            second_part = file_name_parts[1]
            grouped_paths[second_part].append(file_path)
        else:
            Exception
    return grouped_paths


def _check_file_sizes_for_all_gens(prngs, logger, seeds, dims):
    all_expected_file_paths = []
    for prng in prngs:
        if prng.name == "halton" or prng.name == "sobol":
            sobol_or_halton_expected_file_paths = _get_file_paths_for_sobol_or_halton(
                seeds, dims, prng
            )
            all_expected_file_paths.extend(sobol_or_halton_expected_file_paths)
        elif prng.name == "urandom":
            urandom_expected_file_paths = _get_file_paths_for_urandom(seeds, prng)
            all_expected_file_paths.extend(urandom_expected_file_paths)
        else:
            logger.warning(f"Trying to get expected file paths for generator: {prng}")

    groupped_on_dim_file_paths = _group_file_paths_based_on_dim(all_expected_file_paths)
    are_dims_good = [
        _check_files_size_per_dim(grouped_paths, dim, logger)
        for dim, grouped_paths in groupped_on_dim_file_paths.items()
    ]
    for is_dim_good in are_dims_good:
        if not is_dim_good:
            return False
    return True


if __name__ == "__main__":
    LOG_DIR = "logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    main_logger = _setup_main_logger(LOG_DIR, "main.log")

    pid = os.getpid()
    main_logger.info("Python Script Started")
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--second_half",
        action="store_true",
        help="Run SOBOL HALTON and URANDOM instead of MT LCG and XOROSHIRO",
    )
    args = parser.parse_args()

    prngs_first_half = [MtPrng, LcgPrng, XoroshiroPrng]
    prngs_second_half = [SobolPrng, HaltonPrng, UrandomPrng]

    DIMS = [10, 30, 50, 100]
    SEEDS = list(range(1000, 1030))

    if args.second_half:
        curr_prngs = prngs_second_half
        main_logger.info("Starting file pre-generation")
        _pregenerate_files(curr_prngs, SEEDS, DIMS, LOG_DIR)
        all_expected_files_exist = _check_generators_file_paths_exist(
            curr_prngs, SEEDS, DIMS, main_logger
        )
        if not all_expected_files_exist:
            main_logger.critical("Not all files exist! Exited")
            print("ERROR: not all files exist")
            print("Please contact Projekt Specjalny Team ASAP")
            sys.exit()

        all_files_have_correct_sizes = _check_file_sizes_for_all_gens(
            curr_prngs, main_logger, SEEDS, DIMS
        )
        if not all_files_have_correct_sizes:
            main_logger.critical("Not all files have right size! Exited")
            print("ERROR: not all files have right size")
            print("Please contact Projekt Specjalny Team ASAP")
            sys.exit()

        main_logger.info("All expected files detected and checked sizes.")
    else:
        curr_prngs = prngs_first_half

    main_logger.info(f"Running experiments for: {curr_prngs}")

    RESULT_DIR = "results"
    os.makedirs(RESULT_DIR, exist_ok=True)

    _split_work_per_generators(curr_prngs, SEEDS, DIMS, RESULT_DIR, LOG_DIR)
