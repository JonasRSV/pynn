import numpy as np
from typing import Union
from .layer import (
    Layer
)

from .optimiser import (
    Optimiser
)

import nn.optimisers as optimisers
import nn.losses as losses

from .loss import (
    Loss
)

from abc import abstractmethod

opt_by_name = {
    "sgd": optimisers.SGDOptimiser
}

loss_by_name = {
    "msq": losses.Msq
}


class Model:

    @abstractmethod
    def add(self, layer: Layer):
        pass

    @abstractmethod
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def compile(self, optimizer: Union[Optimiser, str], loss: Union[Loss, str]):
        pass

    @abstractmethod
    def train(self, X: np.ndarray, Y: np.ndarray, batch_sz: int, epochs: int):
        pass
