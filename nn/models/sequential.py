from typing import Union, List
import halo

import numpy as np
import random

from nn.layer import Layer
from nn.loss import Loss
from nn.model import (
    Model,
    opt_by_name,
    loss_by_name
)
from nn.optimiser import Optimiser


class Sequential(Model):

    def __init__(self):
        self.layers: List[Layer] = []

        self.loss = None
        self.optimiser = None

    def add(self, layer: Layer):
        if self.layers:
            layer(self.layers[-1])

        self.layers.append(layer)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        o = inputs
        for layer in self.layers:
            o = layer.forward(o)
        return o

    def compile(self, optimiser: Union[Optimiser, str], loss: Union[Loss, str]):
        if type(optimiser) == str:
            optimiser = opt_by_name[optimiser]()

        if type(loss) == str:
            loss = loss_by_name[loss]()

        self.loss = loss
        self.optimiser = optimiser

    def train(self, X: np.ndarray, Y: np.ndarray, batch_sz: int, epochs: int, shuffle=True):
        if len(X) != len(Y):
            raise ValueError("length of domain is not same as labels -- domain: %s, labels: %s" % (len(X), len(Y)))

        indexes = np.arange(len(X))
        spinner = halo.Halo(text="Starting Training", spinner="dots")

        progress_symbol = "#"
        for epoch in range(epochs):
            progress = []

            spinner.start(f"epoch: {epoch} [ " + "".join(progress) + (" " * (20 - len(progress))) + " ] " + "loss: ?")

            if shuffle:
                random.shuffle(indexes)

                X = X[indexes]
                Y = Y[indexes]

            p1 = 0
            p2 = batch_sz

            total_loss = 0
            while p1 < len(X):

                """First run a forward pass"""
                output = self.predict(X[p1:p2])

                gradients = []
                """Now run backward pass
                
                Step 1.
                    Get gradient with respect to the output
                """

                d_error__d_output = self.loss.gradient(Y[p1:p2], output)

                """Step 2.
                    Loop backwards through layers and control gradient with optimiser
                """

                d_error__d_input = d_error__d_output
                for layer in self.layers[:0:-1]:
                    d_err__d_weights, d_error__d_input = layer.backward(
                        self.optimiser.propagate(d_error__d_input)
                    )

                    gradients.append(d_err__d_weights)

                """Step 3.
                    Update all layers with optimiser
                """
                self.optimiser.update(self.layers[1:], gradients[::-1])

                """Step 4
                    Make pretty progress bar
                """
                while (p2 / len(X)) * 20 > len(progress):
                    progress.append(progress_symbol)

                total_loss += self.loss(Y[p1:p2], output)

                spinner.text = f"epoch: {epoch}  [ " + "".join(progress) + (" " * (20 - len(progress))) + " ] " \
                               + f"loss: {total_loss / (p2 / batch_sz):.2f}"

                """Step 5
                    Move pointers to next batch
                """

                p1 = p2
                p2 += batch_sz

            spinner.succeed(f"epoch: {epoch} -- loss: {total_loss / (p2 / batch_sz)}")
