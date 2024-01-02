import os
import numpy as np

from prng.file_generator import generate_file, generate_files, read_chunk_from_file
from prng.file_generator import SOBOL_PRNG, HALTON_PRNG, URANDOM_PRNG

def generate_urandom_array(n):
    random_bytes = os.urandom(n * 4)
    random_array = np.frombuffer(random_bytes, dtype=np.uint32)
    random_array = random_array.reshape((n,))
    return random_array/(2**32-1)


if __name__ == "__main__":
    generate_files()