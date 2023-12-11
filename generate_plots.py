import matplotlib.pyplot as plt
import numpy as np


def ecdf(scores: np.ndarray) -> None:
    fig, ax = plt.subplots(1, 1)
    ax.ecdf(scores)
    ax.grid(True)
    fig.suptitle("ECDF")
    ax.set_xlabel("k")
    ax.set_ylabel("Score")
    plt.show()
