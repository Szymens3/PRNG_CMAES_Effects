"""Helper module for validating pregenerated files"""
import os
import logging


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


def check_generators_file_paths_exist(prngs, seeds, dims, logger):
    """
    Checks if all expected files do exist
    Works only for halton, sobol, urandom 
    If all files exist then returns True
    """
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
    # pylint: disable=C0103
    CHUNK_SIZE = 2**20
    FLOAT_32_BYTE_SIZE = 4

    if dim == 0:
        correct_file_size = CHUNK_SIZE * 100 * FLOAT_32_BYTE_SIZE  # URANDOM SPECIAL CASE
    else:
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


def _group_file_paths_based_on_dim(file_paths, expected_dims=(10, 30, 50, 100)):
    grouped_paths = {dim: [] for dim in expected_dims + (0,)}

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        file_name_parts = file_name.split("_")

        if len(file_name_parts) == 2:
            second_part = int(file_name_parts[1])
            grouped_paths[second_part].append(file_path)
        elif len(file_name_parts) == 1:
            grouped_paths[0].append(file_path)
    return grouped_paths


def check_file_sizes_for_all_gens(prngs, logger, seeds, dims):
    """
    Assumes that expected files do exist.
    Checks the sizes of all expected files.
    Works only for halton, sobol, urandom 
    It logs the result to specified logger.
    If all files have expected size then returns True
    """
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
