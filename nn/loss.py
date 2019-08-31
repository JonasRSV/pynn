from abc import abstractmethod
import numpy as np

class Loss:

    @abstractmethod
    def gradient(self, labels: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def __call__(self, labels: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        pass
