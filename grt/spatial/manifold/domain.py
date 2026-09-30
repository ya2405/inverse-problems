from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from common.typing import BoolArray, BoolValue, FloatArray, FloatValue
from common.utils import asarray_2d, asarray_3d


class Ball(ABC):
    """A smooth 3D domain star-shaped about the origin."""

    @abstractmethod
    def r(self, th: ArrayLike, phi: ArrayLike) -> FloatArray | FloatValue:
        """Return the boundary radius at polar angle ``th`` and azimuth angle ``phi``."""
        raise NotImplementedError

    @abstractmethod
    def dr_dth(self, th: ArrayLike, phi: ArrayLike) -> FloatArray | FloatValue:
        raise NotImplementedError

    @abstractmethod
    def dr_dphi(self, th: ArrayLike, phi: ArrayLike) -> FloatArray | FloatValue:
        raise NotImplementedError

    @abstractmethod
    def boundary_point(self, th: ArrayLike, phi: ArrayLike) -> tuple[FloatArray | FloatValue, FloatArray | FloatValue, FloatArray | FloatValue]:
        raise NotImplementedError

    @abstractmethod
    def contains(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> BoolArray | BoolValue:
        raise NotImplementedError

    @abstractmethod
    def boundary_frame(self, th: ArrayLike, phi: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
        raise NotImplementedError

    @abstractmethod
    def boundary_area_element(self, th: ArrayLike, azimuth: ArrayLike) -> FloatArray | FloatValue:
        raise NotImplementedError


class SphericalBall(Ball):
    """Euclidean ball centered at the origin."""

    def __init__(self, radius: FloatValue):
        if radius <= 0:
            raise ValueError("radius must be positive.")
        
        self.radius = radius

    def r(self, th: ArrayLike, phi: ArrayLike) -> FloatArray | FloatValue:
        th, phi = asarray_2d(th, phi)
        return np.full_like(th, self.radius, dtype=FloatValue)

    def bounds(self) -> tuple[FloatValue, FloatValue, FloatValue, FloatValue, FloatValue, FloatValue]:
        return - self.radius, self.radius, - self.radius, self.radius, - self.radius, self.radius

    def dr_dth(self, th: ArrayLike, phi: ArrayLike) -> FloatArray | FloatValue:
        th, phi = asarray_2d(th, phi)
        return np.zeros_like(th, dtype=FloatValue)

    def dr_dphi(self, th: ArrayLike, phi: ArrayLike) -> FloatArray | FloatValue:
        th, phi = asarray_2d(th, phi)
        return np.zeros_like(phi, dtype=FloatValue)

    def boundary_point(self, th: ArrayLike, phi: ArrayLike) -> tuple[FloatArray | FloatValue, FloatArray | FloatValue, FloatArray | FloatValue]:
        """Return Cartesian boundary points with final axis x, y, z at polar angle ``th`` and azimuth angle ``phi``."""
        th, phi = asarray_2d(th, phi)
        radius, sin_th = self.r(th, phi), np.sin(th)
        return radius * sin_th * np.cos(phi), radius * sin_th * np.sin(phi), radius * np.cos(th)

    def contains(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> BoolArray | BoolValue:
        """Return whether Cartesian points lie in domain."""
        x, y, z = asarray_3d(x, y, z)
        radius = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        nonzero = radius > 0
        th = np.zeros_like(radius, dtype=FloatValue)
        th[nonzero] = np.arccos(np.clip(z[nonzero] / radius[nonzero], -1, 1))
        phi = np.zeros_like(radius, dtype=FloatValue)
        phi[nonzero] = np.mod(np.arctan2(y[nonzero], x[nonzero]), 2 * np.pi)
        
        return radius <= self.r(th, phi)

    def boundary_frame(self, th: ArrayLike, phi: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return inward normal and two orthonormal tangent vectors."""
        th, phi = asarray_2d(th, phi)
        sin_th, cos_th = np.sin(th), np.cos(th)
        sin_phi, cos_phi = np.sin(phi), np.cos(phi)
        e_r = np.stack([sin_th * cos_phi, sin_th * sin_phi, cos_th], axis=-1)
        e_th = np.stack([cos_th * cos_phi, cos_th * sin_phi, - sin_th], axis=-1)
        e_phi = np.stack((- sin_phi, cos_phi, np.zeros_like(phi)), axis=-1)
        return - e_r, e_th, e_phi

    def boundary_area_element(self, th: ArrayLike, phi: ArrayLike) -> FloatArray | FloatValue:
        """Euclidean surface Jacobian of the spherical parametrization."""
        th, phi = asarray_2d(th, phi)
        return self.radius ** 2 * np.sin(th)
