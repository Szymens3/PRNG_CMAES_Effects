"""Base Class for pseudo random numbers generators that read values from files"""

import os
import logging

from scipy.special import erfinv
import numpy as np

from .prng import Prng


class MockingPrng(Prng):
    """Base Class for pseudo random numbers generators that read values from files"""

    def __init__(self, seed, dim, max_fes_coef=10_000, chunk_size=2**20, logger=None):
        self._logger = logger
        self._seed = seed
        self.max_fes_coef = max_fes_coef
        self._dim = dim
        self.chunk_size = chunk_size
        self._file_path = None
        self.from_start_counter = 0
        self._current_idx = 0
        self.buffered_values = None

        self._setup()

    def _init_file_path(self):
        self._file_path = f"prng/{str(self)}_files/{self._seed}_{self._dim}"

    def _setup(self):
        self._init_file_path()
        directory_path = os.path.dirname(self._file_path)
        os.makedirs(directory_path, exist_ok=True)
        if not os.path.exists(self._file_path):
            self._logger.info(f"File: {self._file_path} not found. Generating...")
            self._generate_file()
            self._logger.info("File generated.")

        try:
            self.file = open(self._file_path, "rb")
        except FileNotFoundError as file_not_found_error:
            self._logger.error(f"File not found: {self._file_path}")
            raise file_not_found_error
        except PermissionError as permission_error:
            self._logger.error(
                f"Permission denied while opening file: {self._file_path}"
            )
            raise permission_error
        except Exception as exc:
            self._logger.error(f"Error encountered while opening file: {exc}")
            raise exc
        try:
            self._get_next_chunk()
        except Exception as exc:
            self._logger.error("Error encountered while getting next chunk from file")
            raise exc

    def _generate_file(self):
        target_nr_of_values_to_generate = self.max_fes_coef * self._dim
        with open(self._file_path, "ab") as file:
            self._write_chunks_to_file(file, target_nr_of_values_to_generate)

    def _gen_std_normal(self, dim: int, n=1):
        g_u = self._gen_uniform(dim, n)
        g_norm = self._uniform_to_std_normal(g_u)
        return g_norm

    def _gen_uniform(self, dim: int, n: int = 1):
        raise NotImplementedError

    def _uniform_to_std_normal(self, uniform_numbers):
        # prepare for function.

        left_threshold = np.float32(1.4901163e-08)
        uniform_numbers[uniform_numbers < left_threshold] = left_threshold
        right_threshold = np.float32(0.9999999)
        uniform_numbers[uniform_numbers > right_threshold] = right_threshold

        normal_numbers = np.sqrt(2) * erfinv(2 * uniform_numbers - 1)
        return normal_numbers

    def std_normal(self, dim: int, n: int = 1):
        if self._current_idx + dim <= len(self.buffered_values):
            v = self.buffered_values[self._current_idx : self._current_idx + dim]
            self._current_idx += dim
            return v
        init = self.buffered_values[self._current_idx :]
        self._current_idx = 0
        self._get_next_chunk()
        rest = self.std_normal(dim - len(init))
        return np.concatenate((init, rest), axis=0)

    def _get_file_reusing_msg(self):
        return (
            f"{str(self)} got back to the beginning of the file: "
            + f"{self._file_path} {self.from_start_counter} times already!"
        )

    def _get_next_chunk(self):
        chunk = self.file.read(
            self.chunk_size * 4
        )  # 4 is np.dtype(np.float32).itemsize
        if chunk:
            self.buffered_values = np.frombuffer(chunk, dtype=np.float32)
            return
        self.from_start_counter += 1
        self._logger.warning(self._get_file_reusing_msg())
        self.file.seek(0)
        chunk = self.file.read(self.chunk_size * 4)
        if chunk:
            self.buffered_values = np.frombuffer(chunk, dtype=np.float32)
            return
        self.file.close()
        raise MockingPrng.ReadingChunkException("File is probably empty")

    def _write_chunk_to_file(self, file, chunk_size):
        normal_numbers = self._gen_std_normal(self._dim ,chunk_size)
        normal_numbers.astype(np.float32).tofile(file)

    def _write_chunks_to_file(self, file, target_nr_of_values_to_generate):
        num_chunks = target_nr_of_values_to_generate // self.chunk_size + 1
        for _ in range(num_chunks):
            self._write_chunk_to_file(file, self.chunk_size)

    class ReadingChunkException(Exception):
        """Raised when trying to read content of a file but no content is returned"""
