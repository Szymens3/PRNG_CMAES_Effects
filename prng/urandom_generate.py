'''
Script for generating normal distribution random numbers to a file
by reading /dev/urandom.
'''
import os
import json
import math
import struct

import numpy as np
from matplotlib import pyplot as plt


def read_urandom_range_0_1() -> float:
    '''
    Read a random number from /dev/urandom in range from 0 to 1 
    '''
    # Read 4 bytes from /dev/urandom and convert them to an integer
    random_bytes = os.urandom(4)
    random_int = struct.unpack('I', random_bytes)[0]

    # Normalize the integer to a float in the range [0, 1)
    random_float = random_int / (2**32 - 1)
    return random_float


def uniform_to_normal(u1, u2):
    '''
    Use Box-Muller transform to transform numbers from uniform [0, 1] to
    standard normal distribution
    '''
    z1 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    z2 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
    return [z1, z2]


if __name__ == "__main__":
    random_values = [read_urandom_range_0_1() for _ in range(1000000)]
    print("Average of random uniform numbers: ", np.average(random_values))
    normal_distribution = [
        uniform_to_normal(random_values[i*2], random_values[i*2 + 1])
        for i in range(len(random_values)//2)
    ]
    norm = []
    for el in normal_distribution:
        norm.extend(el)
    print("Average of standard normal numbers: ", np.average(norm))
    print("Standard deviation: ", np.std(norm))

    # dump to a json file:
    with open('urandom.json', 'w') as out_file:
        json.dump(norm, out_file, indent=4)
    
    # show a histogram of generated samples
    plt.hist(norm, bins=500)
    plt.show()
