import numpy as np
from cmaes.custom_cma import CustomCMA
from prng.lcg_prng import LCG_PRNG
from prng.mt_prng import MT_PRNG
from prng.xoroshio_prng import XOROSHIRO_PRNG
import argparse
from tqdm import tqdm


def rng_generator(name: str, seed: int):
    return {
        "lcg": LCG_PRNG,
        "mt": MT_PRNG,
        "xoro": XOROSHIRO_PRNG
    }[name](seed)


def problem_from_dim(problem_dim):
    return {
        2: lambda x: (x[0] - 3) ** 2 + (10 * (x[1] + 2)) ** 2,
        3: lambda x: (x[0] - 3) ** 2 + (10 * (x[1] + 2)) ** 2 + (x[2] - 5) ** 2
    }[problem_dim]


def train(optimizer: CustomCMA, max_iters, eval_function):
    anchor_history = []
    for _ in tqdm(range(max_iters)):
        if optimizer.should_stop():
            break

        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = eval_function(x)
            solutions.append((x, value))

        optimizer.tell(solutions)
        anchor_history.append(optimizer.anchor())

    return np.array(anchor_history)


def main():
    parser = argparse.ArgumentParser(prog="CMA-ES trainer")
    parser.add_argument('rng_generator')
    parser.add_argument('problem_dim', type=int)
    parser.add_argument('sigma', type=float)
    parser.add_argument('max_iters', type=int)
    parser.add_argument('rng_seeds', type=int, nargs="+")

    args = parser.parse_args()

    optimizer = CustomCMA(
        mean=np.zeros(args.problem_dim),
        sigma=args.sigma,
        rng=rng_generator(args.rng_generator, args.rng_seeds)
    )

    optim_histories = []
    for seed in args.rng_seeds:
        eval_function = problem_from_dim(args.problem_dim)
        anchor_history = train(optimizer, args.max_iters, eval_function)
        optim_histories.append([seed, anchor_history])

    # handle optim histories
    print(optim_histories)

    # TO GET SCORE OF HISTORY POINTS:
    # np.apply_along_axis(eval_function, 1, optim_histories[0][1])


if __name__ == "__main__":
    main()
