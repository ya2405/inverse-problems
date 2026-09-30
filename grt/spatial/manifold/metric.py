"""Conformal metrics in three Cartesian dimensions."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from common.typing import FloatArray, FloatValue
from common.utils import asarray_3d


class Metric(ABC):
    """A metric conformal to the Euclidean metric in three dimensions."""

    @abstractmethod
    def log(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> FloatArray | FloatValue:
        raise NotImplementedError

    @abstractmethod
    def grad_log(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> tuple[FloatArray | FloatValue, FloatArray | FloatValue, FloatArray | FloatValue]:
        raise NotImplementedError


class EuclideanMetric(Metric):
    def log(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> FloatArray | FloatValue:
        x = asarray_3d(x, y, z)[0]
        return np.zeros_like(x, dtype=FloatValue)

    def grad_log(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> tuple[FloatArray | FloatValue, FloatArray | FloatValue, FloatArray | FloatValue]:
        x, y, z = asarray_3d(x, y, z)
        return np.zeros_like(x, dtype=FloatValue), np.zeros_like(y, dtype=FloatValue), np.zeros_like(z, dtype=FloatValue)


class PoincareRMetric(Metric):
    """Poincare ball metric with constant sectional curvature -1/R^2."""

    def __init__(self, R: FloatValue):
        if R <= 0:
            raise ValueError("parameter R must be positive.")
        
        self.R = R

    def log(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> FloatArray | FloatValue:
        x, y, z = asarray_3d(x, y, z)
        return np.log(4 * self.R ** 4) - 2 * np.log(self.R ** 2 - x ** 2 - y ** 2 - z ** 2)

    def grad_log(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> tuple[FloatArray | FloatValue, FloatArray | FloatValue, FloatArray | FloatValue]:
        x, y, z = asarray_3d(x, y, z)
        denom = self.R ** 2 - x ** 2 - y ** 2 - z ** 2
        return 4 * x / denom, 4 * y / denom, 4 * z / denom


class SphericalRMetric(Metric):
    """Stereographic spherical metric with sectional curvature 1/R^2."""

    def __init__(self, R: FloatValue):
        if R <= 0:
            raise ValueError("parameter R must be positive.")
        
        self.R = R

    def log(self, x: ArrayLike, y: ArrayLike, z: ArrayLike):
        x, y, z = asarray_3d(x, y, z)
        return np.log(4 * self.R ** 4) - 2 * np.log(self.R ** 2 + x ** 2 + y ** 2 + z ** 2)

    def grad_log(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> tuple[FloatArray | FloatValue, FloatArray | FloatValue, FloatArray | FloatValue]:
        x, y, z = asarray_3d(x, y, z)
        denom = self.R ** 2 + x ** 2 + y ** 2 + z ** 2
        return - 4 * x / denom,  - 4 * y / denom,  - 4 * z / denom


class FocusingLens(Metric):
    """Gaussian conformal perturbation centered at a point in 3D."""

    def __init__(self, k: FloatValue, sigma: FloatValue, x0: FloatValue, y0: FloatValue, z0: FloatValue):
        if k <= 0 or sigma <= 0:
            raise ValueError("parameters k and sigma must be positive.")
        
        self.k, self.sigma = k, sigma
        self.x0, self.y0, self.z0 = x0, y0, z0

    def log(self, x: ArrayLike, y: ArrayLike, z: ArrayLike):
        x, y, z = asarray_3d(x, y, z)
        dist_sq = (x - self.x0) ** 2 + (y - self.y0) ** 2 + (z - self.z0) ** 2
        return self.k * np.exp(- dist_sq / (2 * self.sigma ** 2))

    def grad_log(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> tuple[FloatArray | FloatValue, FloatArray | FloatValue, FloatArray | FloatValue]:
        x, y, z = asarray_3d(x, y, z)
        factor = self.log(x, y, z) / (self.sigma ** 2)
        return - (x - self.x0) * factor, - (y - self.y0) * factor, - (z - self.z0) * factor
