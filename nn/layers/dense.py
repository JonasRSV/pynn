from nn.layer import (
    Layer
)

from nn.initializers import NormalAvg

from typing import Callable, Tuple
import numpy as np
import nn.autograd as autograd


class Dense(Layer):

    def __init__(self, units: int, activation: Callable[[np.ndarray], np.ndarray],
                 initializer: Callable[[np.ndarray], np.ndarray] = NormalAvg()):
        self.units = units
        self.activation: Callable[[np.ndarray], np.ndarray] = activation
        self.d_activation: Callable[[np.ndarray], np.ndarray] = autograd.central(activation)
        self.initializer: Callable[[np.ndarray], np.ndarray] = initializer
        self.W = None

        # State for back-propagation
        self.intermediary = None
        self.input = None

    def __call__(self, inputs: Layer):
        shape = inputs.shape
        if len(shape) != 1:
            raise ValueError("Dense takes 1 dimensional input not %s dimensional" % len(shape))

        self.W = self.initializer(np.array([inputs.shape[0], self.units]))
        return self

    def forward(self, tensor: np.ndarray) -> np.ndarray:
        self.input = tensor
        self.intermediary = tensor @ self.W
        return self.activation(self.intermediary)

    def backward(self, gradients: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # The gradient variable is d_error__d_out
        # Which will have shape [None, N] where N is the number of neurons in this layer

        # Gradient of error with respect to intermediary
        # Which will have shape [None, N] where N is neurons of this layer
        d_err__d_intermediary = gradients * self.d_activation(self.intermediary)

        # Gradient of error with respect to weights
        # is the gradient of the intermediary times the input
        d_err__d_weights = (self.input.T @ d_err__d_intermediary)

        # Now need to calculate the error of the input with respect to the error
        # to continue propagating

        # The gradient of the intermediary with respect to the input is
        # the sum of the inputs effect on each neuron

        # A cell in the inputs gradient is going to be the sum of
        # the first weight in each neuron times the gradient of the intermediary

        # (None M)  (M, N)           (None, N)
        # a b c    e . . .       (ae + bf + cg)  (a. + b. + c.) (a. + b. + c.)
        # . . .    f . . .   ->  (.. ..       )        ..             ..
        # . . .    g . . .       (.. ..       )        ..             ..

        # Gradient of input with respect to error is going to be the gradient of
        # the input with respect to the cell value and the cell value with respect to the error

        d_err__d_in = d_err__d_intermediary @ self.W.T

        return d_err__d_weights, d_err__d_in

    @property
    def shape(self):
        return np.array([self.units])
