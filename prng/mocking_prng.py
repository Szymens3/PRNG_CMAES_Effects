from scipy.special import erfinv
import numpy as np
import os
import logging

from .prng import PRNG

class MOCKING_PRNG(PRNG):

    def __init__(self, seed, dim, max_FES_coef=10_000, chunk_size=2**20):
        self._seed = seed
        self.max_FES_coef = max_FES_coef
        self._dim = dim
        self.chunk_size = chunk_size
        
        self.setup()


    def init_file_path(self):
        self.file_path = f"prng/{self.__str__()}_files/{self._seed}_{self._dim}"
        
    def setup(self):
        self.init_file_path()
        directory_path = os.path.dirname(self.file_path)
        os.makedirs(directory_path, exist_ok=True)
        if not os.path.exists(self.file_path):
            logging.info(f"File: {self.file_path} not found. Generating...")
            self.generate_file()
            logging.info("File generated.")
        try:
            self.from_start_counter = 0
            self.file = open(self.file_path, 'rb')
            self._current_idx=0
            self.buffered_values=None
            self._get_next_chunk()
        except:
            raise Exception

    
    def generate_file(self):
        target_nr_of_values_to_generate = self.max_FES_coef * self._dim
        with open(self.file_path, 'ab') as file:
            self._write_chunks_to_file(file, target_nr_of_values_to_generate)

    def gen_std_normal(self, dim: int, n=1):
        g_u = self._gen_uniform(dim, n)
        g_norm = self._uniform_to_std_normal(g_u)
        return g_norm
    
    def _gen_uniform(self, dim: int):
        raise NotImplemented
    
    def _uniform_to_std_normal(self, uniform_numbers):
        # prepare for function. 
        
        left_threshold = np.float32(1.4901163e-08)
        uniform_numbers[uniform_numbers < left_threshold] = left_threshold
        right_threshold = np.float32(0.9999999)
        uniform_numbers[uniform_numbers > right_threshold] = right_threshold


        normal_numbers = np.sqrt(2) * erfinv(2 * uniform_numbers - 1)
        return normal_numbers
    
    def std_normal(self, dim: int):
        if self._current_idx + dim <= len(self.buffered_values):
            v = self.buffered_values[self._current_idx : self._current_idx + dim]
            self._current_idx += dim
            return v
        init = self.buffered_values[self._current_idx:]
        self._current_idx = 0
        self._get_next_chunk()
        rest = self.std_normal(dim-len(init))
        return np.concatenate((init, rest), axis=0)

    def _get_next_chunk(self):
        chunk = self.file.read(self.chunk_size * 4) # 4 is np.dtype(np.float32).itemsize
        if chunk:
            self.buffered_values = np.frombuffer(chunk, dtype=np.float32)
            return
        self.from_start_counter+=1
        logging.warning(f"{self.__str__()} got back to the beggining of the file: {self.file_path} {self.from_start_counter} times already!")
        self.file.seek(0)
        chunk = self.file.read(self.chunk_size * 4)
        if chunk:
            self.buffered_values = np.frombuffer(chunk, dtype=np.float32)
            return
        self.file.close()
        raise Exception
    
    def _write_chunk_to_file(self, file, chunk_size):
        normal_numbers = self.gen_std_normal(None, chunk_size)
        normal_numbers.astype(np.float32).tofile(file)

    def _write_chunks_to_file(self, file, target_nr_of_values_to_generate):
        num_chunks = target_nr_of_values_to_generate // self.chunk_size + 1
        for i in range(num_chunks):
            self._write_chunk_to_file(file, self.chunk_size)

    
