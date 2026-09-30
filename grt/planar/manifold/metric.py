from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from common.typing import FloatArray, FloatValue
from common.utils import asarray_2d


class Metric(ABC):
    r"""Conformal Riemannian metric :math:``g(x,y) = e^{log_factor(x,y)} (dx^2 + dy^2)`` in xy-plane."""

    @abstractmethod
    def log(self, x: ArrayLike, y: ArrayLike) -> FloatArray | FloatValue:
        raise NotImplementedError

    @abstractmethod
    def grad_log(self, x: ArrayLike, y: ArrayLike) -> tuple[FloatArray | FloatValue, FloatArray | FloatValue]:
        raise NotImplementedError

class EuclideanMetric(Metric):
    def log(self, x: ArrayLike, y: ArrayLike) -> FloatArray | FloatValue:
        x = asarray_2d(x, y)[0]
        return np.zeros_like(x, dtype=FloatValue)

    def grad_log(self, x: ArrayLike, y: ArrayLike) -> tuple[FloatArray | FloatValue, FloatArray | FloatValue]:
        x, y = asarray_2d(x, y)
        return np.zeros_like(x, dtype=FloatValue), np.zeros_like(y, dtype=FloatValue)

class PoincareRMetric(Metric):
    def __init__(self, R: FloatValue):
        if R <= 0:
            raise ValueError("parameter R must be positive.")
    
        self.R = R

    def log(self, x: ArrayLike, y: ArrayLike) -> FloatArray | FloatValue:
        x, y = asarray_2d(x, y)
        return np.log(4 * self.R ** 4) - 2 * np.log(self.R ** 2 - x ** 2 - y ** 2)

    def grad_log(self, x: ArrayLike, y: ArrayLike) -> tuple[FloatArray | FloatValue, FloatArray | FloatValue]:
        x, y = asarray_2d(x, y)
        denom = self.R ** 2 - x ** 2 - y ** 2
        return 4. * x / denom, 4. * y / denom

class SphericalRMetric(Metric):
    def __init__(self, R: FloatValue):
        if R <= 0:
            raise ValueError("parameter R must be positive.")
    
        self.R = R

    def log(self, x: ArrayLike, y: ArrayLike) -> FloatArray | FloatValue:
        x, y = asarray_2d(x, y)
        return np.log(4 * self.R ** 4) - 2 * np.log(self.R ** 2 + x ** 2 + y ** 2)

    def grad_log(self, x: ArrayLike, y: ArrayLike) -> tuple[FloatArray | FloatValue, FloatArray | FloatValue]:
        x, y = asarray_2d(x, y)
        denom = self.R ** 2 + x ** 2 + y ** 2
        return - 4. * x / denom, - 4. * y / denom

class FocusingLens(Metric):
    def __init__(self, k: FloatValue, sigma: FloatValue, x0: FloatValue, y0: FloatValue):
        if k <= 0 or sigma <= 0:
            raise ValueError("parameters k and sigma must be positive.")
    
        self.k, self.sigma = k, sigma
        self.x0, self.y0 = x0, y0

    def log(self, x: ArrayLike, y: ArrayLike) -> FloatArray | FloatValue:
        x, y = asarray_2d(x, y)
        dist_sq = (x - self.x0) ** 2 + (y - self.y0) ** 2 
        return self.k * np.exp(- dist_sq / (2 * self.sigma ** 2))

    def grad_log(self, x: ArrayLike, y: ArrayLike) -> tuple[FloatArray | FloatValue, FloatArray | FloatValue]:
        x, y = asarray_2d(x, y)
        factor = self.log(x, y) / (self.sigma ** 2)
        return - (x - self.x0) * factor, - (y - self.y0) * factor
