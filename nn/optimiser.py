from typing import List
from .layer import (
    Layer
)
from abc import abstractmethod
import numpy as np


class Optimiser:

    @abstractmethod
    def update(self, layers: List[Layer], gradients: List[np.ndarray]):
        pass

    @abstractmethod
    def propagate(self, gradients: np.ndarray) -> np.ndarray:
        pass
