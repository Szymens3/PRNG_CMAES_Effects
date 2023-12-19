from abc import ABC, abstractmethod


class PRNG(ABC):
    @abstractmethod
    def __init__(self, seed):
        pass

    def __str__(self) -> str:
        return "prng"

    @abstractmethod
    def std_normal(self, dim: int):
        """
        Return a vector of standard normal random variables.
        :param dim:
        :return:
        """
        pass