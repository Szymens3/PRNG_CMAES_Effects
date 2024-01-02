import numpy as np

from .urandom_prng import URANDOM_PRNG
from .halton_prng import HALTON_PRNG
from .sobol_prng import SOBOL_PRNG


prngs = [URANDOM_PRNG, HALTON_PRNG, SOBOL_PRNG]
seeds = [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050 ]


def _write_chunk_to_file(file, prng_instance, chunk_size):
    normal_numbers = prng_instance.gen_std_normal(chunk_size)
    #np.savetxt(file, normal_numbers, delimiter='\n', fmt='%s')
    normal_numbers.astype(np.float32).tofile(file)

def _write_chunks_to_file(file, prng_instance, num_chunks, n_values_per_file, chunk_size):
    for i in range(num_chunks):
        _write_chunk_to_file(file, prng_instance, chunk_size)
        # print(f"\tChunk: {i+1} done")

    size_of_last_chunk = n_values_per_file % chunk_size
    if size_of_last_chunk != 0:
        _write_chunk_to_file(file, prng_instance, size_of_last_chunk)
        # print(f"Last chunk done! This was not a full chunk")

def generate_file(file_path, prng_instance, n_values_per_file=2**27, chunk_size=2**20):
    with open(file_path, 'ab') as file:
        num_chunks = n_values_per_file // chunk_size
        print("Starting:", file_path)
        _write_chunks_to_file(file, prng_instance, num_chunks, n_values_per_file, chunk_size)
        print("Ended:", file_path)
        

def generate_files(prngs=prngs, seeds=seeds, n_values_per_file=2**27, chunk_size=2**20):
    for prng in prngs:
        for seed in seeds:
            prng_instance = prng(seed=seed)
            file_path = f"./prng/{prng_instance}_files/{seed}"
            generate_file(file_path, prng_instance, n_values_per_file, chunk_size)

def read_chunk_from_file(file, chunk_size = 2**20):
    chunk = file.read(chunk_size * 4)
    if not chunk:
        return None
    return np.frombuffer(chunk, dtype=np.float32)
