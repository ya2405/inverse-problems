"""Spherical discretization of the incoming boundary bundle in 3D."""

import numpy as np

from common.typing import FloatArray, FloatValue, IntegerValue
from common.utils import midpoints
from grt.spatial.manifold.domain import SphericalBall
from grt.spatial.manifold.grid import UniformCartesianGrid
from grt.spatial.manifold.metric import Metric


class UniformSphericalBeamGrid:
    """Product grid on the four-dimensional incoming boundary bundle.

    Boundary points use spherical polar and azimuthal angles. Incoming
    directions use an inclination alpha from the inward normal and an
    azimuth gamma in the tangent plane.
    """

    def __init__(self, n_th: IntegerValue, n_phi: IntegerValue, n_alpha: IntegerValue, n_gamma: IntegerValue, metric: Metric, domain: SphericalBall, domain_grid: UniformCartesianGrid):
        self.th = midpoints(0, np.pi, n_th)
        self.phi = midpoints(0, 2 * np.pi, n_phi)
        self.alpha = midpoints(0, np.pi / 2, n_alpha)
        self.gamma = midpoints(0, 2 * np.pi, n_gamma)
        self.size = self.th.size * self.phi.size * self.alpha.size * self.gamma.size

        if self.size < domain_grid.size:
            raise ValueError(
                "spherical beam grid must contain at least as many measurements as object coefficients."
            )

        self.cell_ids = np.arange(self.size, dtype=IntegerValue)
        self.cell_areas = self.compute_cell_areas(domain, metric)

    def compute_cell_areas(self, domain: SphericalBall, metric: Metric) -> FloatArray:
        """Return product-quadrature weights for the natural influx measure."""
        shape = (self.th.size, self.phi.size, self.alpha.size, 1)

        th = np.broadcast_to(self.th[:, None, None, None], shape)
        phi = np.broadcast_to(self.phi[None, :, None, None], shape)
        alpha = np.broadcast_to(self.alpha[None, None, :, None], shape)
        x, y, z = domain.boundary_point(th, phi)

        dth, dphi = np.pi / self.th.size, 2 * np.pi / self.phi.size
        boundary_area = domain.boundary_area_element(th, phi) * np.exp(metric.log(x, y, z)) * dth * dphi

        dalpha, dgamma = (np.pi / 2) / self.alpha.size, 2 * np.pi / self.gamma.size
        influx_hemisphere_area = np.cos(alpha) * np.sin(alpha) * dalpha * dgamma

        shape = (shape[0], shape[1], shape[2], self.gamma.size)
        return np.broadcast_to(boundary_area * influx_hemisphere_area, shape).ravel()
