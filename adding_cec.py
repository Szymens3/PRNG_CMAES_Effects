import opfunu
import numpy as np
import os

from prng.xoroshio_prng import XOROSHIRO_PRNG
from cmaes.custom_cma import CustomCMA

all_funcs_2017 = opfunu.get_functions_based_classname("2017")

print(len(all_funcs_2017))

#dims = [10, 30, 50 , 100]
dims = [10]
algorithm_name = "cmaes"

N_EXPERIMENTS_PER_FUNC_PER_DIM = 51


xoroshiro_rng = XOROSHIRO_PRNG(1234)
print(xoroshiro_rng)

if not os.path.exists("results"):
    os.makedirs("results")

for rng in [xoroshiro_rng]:
    directory_path = f"results/{rng}"
    for func_i, func in enumerate(all_funcs_2017):
        for dim in dims:
            max_FES = 10000*dim
            bounds = np.array([[-100, 100]] * dim)

            initial_mean_vector = np.zeros(dim) # starting point TODO change it to get values from dims from approprieate randomiser
            optimizer = CustomCMA(mean=initial_mean_vector, sigma=1.3, bounds=bounds, rng=xoroshiro_rng)

            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            file_path = f"{directory_path}/{algorithm_name}_{func_i}_{dim}"
            problem = func(ndim=dim)

            x_global = problem.x_global
            y_global = problem.evaluate(x_global)
            

            run_checkpoints = max_FES * [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            experiments_results = np.zeros((len(run_checkpoints), N_EXPERIMENTS_PER_FUNC_PER_DIM))

            for experiment_i in range(N_EXPERIMENTS_PER_FUNC_PER_DIM):
                run_results = np.zeros(len(run_checkpoints))
                checkpoint_pointer = 1
            
                x = initial_mean_vector
                value = problem.evaluate(x)
                FES = 1
                while abs(value - y_global) > 10**(-8) and FES < max_FES:
                    for _ in range(optimizer.population_size):
                        x = optimizer.ask()
                        value = problem.evaluate(x)
                        FES+=1
                    if FES >= run_checkpoints[checkpoint_pointer]:
                        run_results[checkpoint_pointer] = x
                        checkpoint_pointer+=1
                experiments_results[:,experiment_i] = run_results

            np.savetxt(file_path, experiments_results, fmt='%d', delimiter=' ')



