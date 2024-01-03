import os
import numpy as np
import matplotlib.pyplot as plt

def generate_histograms_from_directory(directory_path, gen_name):
    files = os.listdir(directory_path)
    
    plt.figure(figsize=(10, 6), )  # Adjust figure size if needed
    plt.suptitle(gen_name, fontsize=16)
    for idx, file_name in enumerate(files):
        file_path = os.path.join(directory_path, file_name)
        values = []
        chunk_size = 1_000_000

        with open(file_path, 'rb') as file:
            while True:
                chunk = file.read(chunk_size * 4)
                if not chunk:
                    break
                values_from_chunk = np.frombuffer(chunk, dtype=np.float32)
                values.extend(values_from_chunk)

        plt.subplot(len(files)//2 + 1, 2, idx+1)
        plt.hist(values, bins=100)
        plt.title(file_name)
        plt.xlabel('Value')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

try:
    generate_histograms_from_directory('./prng/halton_prng_files', 'halton')
except:
    pass
try:
    generate_histograms_from_directory('./prng/sobol_prng_files', 'sobol')
except:
    pass
try:
    generate_histograms_from_directory('./prng/urandom_prng_files', 'urandom')
except:
    pass


