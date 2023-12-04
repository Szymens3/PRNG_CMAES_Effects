import numpy as np
from cmaes.custom_cma import CustomCMA
from prng.lcg_prng import LCG_PRNG
def quadratic(x1, x2):
    return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2


lcg_rng = LCG_PRNG(1234)
optimizer = CustomCMA(mean=np.zeros(2), sigma=1.3, rng=lcg_rng)

for generation in range(50):
    solutions = []
    for _ in range(optimizer.population_size):
        # Ask a parameter
        x = optimizer.ask()
        value = quadratic(x[0], x[1])
        solutions.append((x, value))
        print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")

    # Tell evaluation values.
    optimizer.tell(solutions)
