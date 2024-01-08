"""Module with Urandom Generator  class"""
import os

import numpy as np

from .mocking_prng import MockingPrng


class UrandomPrng(MockingPrng):
    """Get numbers from OS - pregenerates file and reads it"""

    name = "urandom"

    def __init__(self, seed, dim, max_fes_coef=10_000, chunk_size=2**20):
        super().__init__(seed, dim, max_fes_coef=max_fes_coef, chunk_size=chunk_size)

    def __str__(self) -> str:
        return f"urandom_{super().__str__()}"

    def _gen_uniform(self, dim: int, n:int =1):
        random_bytes = os.urandom(dim * 4)
        random_array = np.frombuffer(random_bytes, dtype=np.uint32)
        random_array = random_array.reshape((dim,))
        return random_array / (2**32 - 1)

    def _init_file_path(self):
        self._file_path = f"prng/{str(self)}_files/{self._seed}"
