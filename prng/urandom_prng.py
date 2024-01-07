import os

import numpy as np

from .mocking_prng import MOCKING_PRNG

class URANDOM_PRNG(MOCKING_PRNG):
    name='urandom'
    def __init__(self, seed, dim, max_FES_coef=10_000, chunk_size=2**20):
        super().__init__(seed, dim, max_FES_coef=max_FES_coef, chunk_size=chunk_size)
        
                
    def __str__(self) -> str:
        return f"urandom_{super().__str__()}"

    def _gen_uniform(self, dim: int):
        random_bytes = os.urandom(dim * 4)
        random_array = np.frombuffer(random_bytes, dtype=np.uint32)
        random_array = random_array.reshape((dim,))
        return random_array/(2**32-1)
    
    def init_file_path(self):
        self.file_path = f"prng/{self.__str__()}_files/{self._seed}"
        
    

