from typing import List

import numpy as np

from nn.layer import Layer
from nn.optimiser import Optimiser


class SGDOptimiser(Optimiser):
    def __init__(self, learning_rate: float = 1e-1):
        self.learning_rate = learning_rate

    def update(self, layers: List[Layer], gradients: List[np.ndarray]):
        for l, g in zip(layers, gradients):
            l.W -= g * self.learning_rate

    def propagate(self, gradients: np.ndarray) -> np.ndarray:
        return gradients
