import json, os

import numpy as np
from functools import reduce

from .mocking_prng import MOCKING_PRNG

class URANDOM_PRNG(MOCKING_PRNG):
    name='urandom'
    def __init__(self, seed, dim, n_values_per_file=2**27, chunk_size=2**20):
        super().__init__(seed, dim, n_values_per_file, chunk_size)
        
                
    def __str__(self) -> str:
        return f"urandom_{super().__str__()}"

    def _gen_uniform(self, dim: int):
        random_bytes = os.urandom(dim * 4)
        random_array = np.frombuffer(random_bytes, dtype=np.uint32)
        random_array = random_array.reshape((dim,))
        return random_array/(2**32-1)
    
    def setup(self):
        self.file_path = f"prng/{self.__str__()}_files/{self._seed}"
        
    

