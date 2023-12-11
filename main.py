import numpy as np
from cmaes.custom_cma import CustomCMA
from prng.lcg_prng import LCG_PRNG
from prng.mt_prng import MT_PRNG
from prng.xoroshio_prng import XOROSHIRO_PRNG
from prng.urandom_prng import URANDOM_PRNG
def quadratic(x1, x2):
    return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2

def qubic(x1, x2, x3):
    return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2 + (x3 - 5) ** 2


lcg_rng = LCG_PRNG(1234)
mt_rng = MT_PRNG(1234)
xoroshiro_rng = XOROSHIRO_PRNG(1234)
urandom_rng = URANDOM_PRNG('prng/urandom.json')
optimizer = CustomCMA(mean=np.zeros(3), sigma=1.3, rng=xoroshiro_rng)

for generation in range(50):
    solutions = []
    for _ in range(optimizer.population_size):
        # Ask a parameter
        x = optimizer.ask()
        value = qubic(x[0], x[1], x[2])
        solutions.append((x, value))
        print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]}, x3 = {x[2]})")

    # Tell evaluation values.
    optimizer.tell(solutions)
