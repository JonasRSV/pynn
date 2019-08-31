import numpy as np
from abc import abstractmethod, abstractproperty


class Layer:

    def __init__(self):
        self.W: np.ndarray = None

    @abstractmethod
    def forward(self, tensor: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def __call__(self, inputs: 'Layer'):
        pass

    @abstractmethod
    def backward(self, gradients: np.ndarray) -> np.ndarray:
        pass

    @abstractproperty
    def shape(self) -> np.ndarray:
        pass
