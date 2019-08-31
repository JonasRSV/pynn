from nn.models import Sequential
from nn.layers import Dense, Input
from nn.optimisers import SGDOptimiser
from nn.losses import Msq
import numpy as np


X = np.arange(10).reshape(-1, 1) / 10
Y = X


def test_forward():
    model = Sequential()

    model.add(
        Input(shape=[1])
    )

    model.add(
        Dense(40,
              np.tanh
          )
    )

    model.add(
        Dense(
            31,
            lambda x: x
        )
    )

    model.add(
        Dense(
            2,
            lambda x: x
        )
    )

    print(model.predict(X))

    assert True


def test_dense_integration():

    X = np.arange(10).reshape(-1, 1) / 10
    Y = X

    print("Linear")
    model = Sequential()

    model.add(
        Input(shape=[1])
    )

    model.add(
        Dense(1,
              lambda x: x
          )
    )

    model.add(
        Dense(
            1,
            lambda x: x
        )
    )

    model.compile(optimiser=SGDOptimiser(), loss=Msq())

    p = model.predict(X)

    print(np.mean(p))
    print(np.std(p))

    model.train(X, Y, 5, 10, True)

    print("Sinusoidal")
    model = Sequential()

    model.add(
        Input(shape=[1])
    )

    model.add(
        Dense(128,
              np.tanh
              )
    )

    model.add(
        Dense(
            1,
            lambda x: x
        )
    )

    model.compile(optimiser=SGDOptimiser(learning_rate=1e-3), loss=Msq())

    X = np.linspace(-5, 5, 10000).reshape(-1, 1)
    Y = np.sin(X)

    model.train(X, Y, 64, 10, True)


    print("Random Multidim")
    model = Sequential()

    model.add(
        Input(shape=[5])
    )

    model.add(
        Dense(256,
              np.tanh
              )
    )

    model.add(
        Dense(256,
              np.tanh
              )
    )

    model.add(
        Dense(
            5,
            lambda x: x
        )
    )

    model.compile(optimiser=SGDOptimiser(learning_rate=1e-3), loss=Msq())

    X = np.random.rand(10000, 5)
    Y = np.sin(X)

    model.train(X, Y, 64, 10, True)
