from typing import Callable
import numpy as np

delta = 1e-15


def forward(f: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    def f_diff(x: np.ndarray) -> np.ndarray:
        return (f(x + delta) - f(x)) / delta

    return f_diff


def backward(f: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    def b_diff(x: np.ndarray) -> np.ndarray:
        return (f(x) - f(x - delta)) / delta

    return b_diff


def central(f: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    f_diff = forward(f)
    b_diff = backward(f)

    def c_diff(x: np.ndarray) -> np.ndarray:
        return (f_diff(x) + b_diff(x)) / 2.0

    return c_diff
