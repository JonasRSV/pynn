import numpy as np

from nn.loss import Loss


class Msq(Loss):
    def gradient(self, labels: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        return -2 * (labels - inputs) / len(inputs)

    def __call__(self, labels: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        return np.mean(np.square(labels - inputs))
