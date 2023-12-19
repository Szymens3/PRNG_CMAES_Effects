import json

import numpy as np
from functools import reduce

from .prng import PRNG

class URANDOM_PRNG(PRNG):
    def __init__(self, seed):
        # seed is path to JSON file
        with open(seed, 'r') as input_file:
            self._random_numbers = json.load(input_file)
        self._current_index = 0

    def __str__(self) -> str:
        return f"urandom_{super().__str__()}"

    def std_normal(self, dim: int):
        try:
            n = reduce(lambda x, y: x*y, dim)
        except TypeError:
            # if n not an array
            n = dim
        randoms = np.array([
            self._random_numbers[i%len(self._random_numbers)]
            for i in range(self._current_index, self._current_index + n)
        ])
        self._current_index = self._current_index + n
        return randoms.reshape(dim)
