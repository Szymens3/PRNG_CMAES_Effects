from scipy.special import erfinv
import numpy as np

from .prng import PRNG

class MOCKING_PRNG(PRNG):

    def __init__(self, seed, chunk_size=2**20):
        self._seed = (seed)
        self.chunk_size = chunk_size
        self.file_path = f"prng/{self.__str__()}_files/{seed}"
        try:
            self.c =0
            self.file = open(self.file_path, 'rb')
            self._current_idx=0
            self.buffered_values=None
            self._get_next_chunk()
        except:
            pass

    def gen_std_normal(self, dim: int):
        return self._uniform_to_std_normal(self._gen_uniform(dim))
    
    def _gen_uniform(self, dim: int):
        raise NotImplemented
    
    def _uniform_to_std_normal(self, uniform_numbers):
        normal_numbers = np.sqrt(2) * erfinv(2 * uniform_numbers - 1)
        return normal_numbers
    
    def std_normal(self, dim: int):
        if self._current_idx + dim <= len(self.buffered_values):
            v = self.buffered_values[self._current_idx : self._current_idx + dim]
            self._current_idx += dim
            return v
        init = self.buffered_values[self._current_idx : self._current_idx + dim]
        self._current_idx = 0
        self._get_next_chunk()
        rest = self.std_normal(dim-len(init))
        return np.concatenate((init, rest), axis=0)

    def _get_next_chunk(self):
        chunk = self.file.read(self.chunk_size * 4) # 4 is np.dtype(np.float32).itemsize
        if chunk:
            self.buffered_values = np.frombuffer(chunk, dtype=np.float32)
            return
        self.file.seek(0)
        chunk = self.file.read(self.chunk_size * 4)
        if chunk:
            self.buffered_values = np.frombuffer(chunk, dtype=np.float32)
            return
        self.file.close()
        raise Exception

    
