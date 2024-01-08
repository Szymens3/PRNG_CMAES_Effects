"""Module for base class for Generators"""
from abc import ABC, abstractmethod


class Prng(ABC):
    """Base Class for generators for CustomCMA class"""
    @abstractmethod
    def __init__(self, seed, dim, logger):
        pass

    def __str__(self) -> str:
        return "prng"

    @abstractmethod
    def std_normal(self, dim: int, n: int):
        """
        Return a vector of standard normal random variables.
        :param dim:
        :return:
        """
