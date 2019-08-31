import numpy as np

from nn.initialiser import Initializer


class NormalAvg(Initializer):
    def __init__(self, loc: float = 0, std: float = 1):
        self.loc = loc
        self.std = std

    def __call__(self, shape: np.ndarray) -> np.ndarray:
        w = np.random.normal(self.loc, self.std, shape)
        return w / shape[-1]
