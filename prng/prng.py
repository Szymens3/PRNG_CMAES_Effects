from abc import ABC, abstractmethod


class PRNG(ABC):
    @abstractmethod
    def __init__(self, seed):
        pass

    @abstractmethod
    def std_normal(self, dim: int):
        """
        Return a vector of standard normal random variables.
        :param dim:
        :return:
        """
        pass