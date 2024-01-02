import json, os

import numpy as np
from functools import reduce

from .mocking_prng import MOCKING_PRNG

class URANDOM_PRNG(MOCKING_PRNG):
    def __init__(self, seed, chunk_size=2**20):
        super().__init__(seed, chunk_size)
        
                
    def __str__(self) -> str:
        return f"urandom_{super().__str__()}"

    def _gen_uniform(self, dim: int):
            random_bytes = os.urandom(dim * 4)
            random_array = np.frombuffer(random_bytes, dtype=np.uint32)
            random_array = random_array.reshape((dim,))
            return random_array/(2**32-1)
    

