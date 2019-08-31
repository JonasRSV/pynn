from typing import List
from nn.layer import (
    Layer
)

import numpy as np


class Input(Layer):

    def __init__(self, shape: List[int]):
        self.s = np.array(shape)

    def __call__(self, inputs: Layer):
        raise ValueError("Input layer should be the first layer")

    def forward(self, tensor: np.ndarray) -> np.ndarray:
        if tensor.shape[1:] != self.s:
            raise ValueError("Shape mismatch expected %s got %s" % (self.s, tensor.shape[1:]))

        return tensor

    def backward(self, gradients: np.ndarray) -> np.ndarray:
        raise ValueError("Input layer has no back-propagation")

    @property
    def shape(self):
        return self.s

