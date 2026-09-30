from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from common.typing import BoolArray, BoolValue, FloatArray, FloatValue
from common.utils import asarray_2d


class Disk(ABC):
    r"""A smooth domain in $xy$-plane, star-shaped about the origin, with polar boundary radius $r$."""

    @abstractmethod
    def r(self, th: ArrayLike) -> FloatArray | FloatValue:
        """Return boundary radius at polar angle ``th``."""
        raise NotImplementedError

    @abstractmethod
    def dr_dth(self, th: ArrayLike) -> FloatArray | FloatValue:
        """Return first derivative of boundary radius with respect to polar angle ``th``."""
        raise NotImplementedError

    @abstractmethod
    def boundary_point(self, th: ArrayLike) -> tuple[FloatArray | FloatValue, FloatArray | FloatValue]:
        raise NotImplementedError

    @abstractmethod
    def contains(self, x: ArrayLike, y: ArrayLike) -> BoolArray | BoolValue:
        raise NotImplementedError

    @abstractmethod
    def boundary_frame(self, th: ArrayLike) -> tuple[FloatArray, FloatArray]:
        raise NotImplementedError

    @abstractmethod
    def boundary_arc_length(self, th: ArrayLike) -> FloatArray | FloatValue:
        raise NotImplementedError


class CircularDisk(Disk):
    def __init__(self, radius: FloatValue):
        if radius <= 0:
            raise ValueError("radius must be positive.")
        
        self.radius = radius

    def r(self, th: ArrayLike) -> FloatArray | FloatValue:
        th = np.asarray(th, dtype=FloatValue)
        return np.full_like(th, self.radius, dtype=FloatValue)

    def bounds(self) -> tuple[FloatValue, FloatValue, FloatValue, FloatValue]:
        return - self.radius, self.radius, - self.radius, self.radius

    def dr_dth(self, th: ArrayLike) -> FloatArray | FloatValue:
        th = np.asarray(th, dtype=FloatValue)
        return np.zeros_like(th, dtype=FloatValue)
    
    def boundary_point(self, th: ArrayLike) -> tuple[FloatArray | FloatValue, FloatArray | FloatValue]:
        """Return Cartesian coordinates of the boundary point at polar angle ``th``."""
        th = np.asarray(th, dtype=FloatValue)
        radius = self.r(th)
        return radius * np.cos(th), radius * np.sin(th)

    def contains(self, x: ArrayLike, y: ArrayLike) -> BoolArray | BoolValue:
        """Return a boolean mask indicating which points are in the domain."""
        x, y = asarray_2d(x, y)
        th = np.arctan2(y, x)
        return x ** 2 + y ** 2 <= self.r(th) ** 2

    def boundary_frame(self, th: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Return inward normal and two orthonormal tangent vector."""
        th = np.asarray(th, dtype=FloatValue)
        e_r = np.stack([np.cos(th), np.sin(th)], axis=-1)
        e_th = np.stack([np.sin(th), - np.cos(th)], axis=-1)

        return - e_r, e_th

    def boundary_arc_length(self, th: ArrayLike) -> FloatArray | FloatValue:
        """Euclidean speed of the polar boundary parametrization."""
        th = np.asarray(th, dtype=FloatValue)
        return np.sqrt(self.r(th) ** 2 + self.dr_dth(th) ** 2)
