from scipy.special import erfinv
import numpy as np
import os
import logging

from .prng import PRNG

class MOCKING_PRNG(PRNG):

    def __init__(self, seed, n_values_per_file=2**27, chunk_size=2**20):
        self._seed = seed
        self.chunk_size = chunk_size
        self.n_values_per_file = n_values_per_file
        self.file_path = f"prng/{self.__str__()}_files/{seed}"
        directory_path = os.path.dirname(self.file_path)
        os.makedirs(directory_path, exist_ok=True)
        if not os.path.exists(self.file_path):
            logging.info(f"File: {self.file_path} not found. Generating...")
            self.generate_file()
            logging.info("File generated.")
        
        try:
            self.file = open(self.file_path, 'rb')
            self._current_idx=0
            self.buffered_values=None
            self._get_next_chunk()
        except:
            raise Exception
    
    def generate_file(self):
        with open(self.file_path, 'ab') as file:
            num_chunks = self.n_values_per_file // self.chunk_size
            self._write_chunks_to_file(file, num_chunks)

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
    
    def _write_chunk_to_file(self, file, chunk_size):
        normal_numbers = self.gen_std_normal(chunk_size)
        normal_numbers.astype(np.float32).tofile(file)

    def _write_chunks_to_file(self, file, num_chunks):
        for i in range(num_chunks):
            self._write_chunk_to_file(file, self.chunk_size)

        size_of_last_chunk = self.n_values_per_file % self.chunk_size
        if size_of_last_chunk != 0:
            self._write_chunk_to_file(file, size_of_last_chunk)

    
