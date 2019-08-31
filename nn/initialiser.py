from abc import abstractmethod
import numpy as np


class Initializer:

    @abstractmethod
    def __call__(self, shape: np.ndarray) -> np.ndarray:
        pass
