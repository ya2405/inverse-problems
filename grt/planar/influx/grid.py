import numpy as np

from common.typing import FloatArray, IntegerValue
from common.utils import midpoints
from grt.planar.manifold.domain import CircularDisk
from grt.planar.manifold.grid import UniformCartesianGrid
from grt.planar.manifold.metric import Metric


class UniformFanBeamGrid:
    r"""Uniform FanBeam discretization on :math:``\partial_+ SM``."""
    def __init__(self, n_alpha: IntegerValue, n_th: IntegerValue, metric: Metric, domain: CircularDisk, domain_grid: UniformCartesianGrid):
        self.alpha = midpoints(- np.pi / 2, np.pi / 2, n_alpha)
        self.th = midpoints(0, 2 * np.pi, n_th)
        self.size = self.alpha.size * self.th.size

        if self.size < domain_grid.size:
            raise ValueError(
                "fanbeam grid must contain at least as many measurements as object coefficients."
            )
        
        if np.any(np.abs(self.alpha) >= np.pi / 2):
            raise ValueError("alpha must lie strictly in (-pi/2, pi/2).")
        
        # Beta is the first axis and alpha is the second (fast) axis.
        self.cell_ids = np.arange(self.size, dtype=IntegerValue)
        self.cell_areas = self.compute_cell_areas(domain, metric)

    def compute_cell_areas(self, domain: CircularDisk, metric: Metric) -> FloatArray:
        cos_alpha = np.cos(self.alpha)[None, :]
        dalphadth = (np.pi / self.alpha.size) * (2 * np.pi / self.th.size)
        boundary_density = np.exp(0.5 * metric.log(*domain.boundary_point(self.th)))[:, None]
        ds_dth = domain.boundary_arc_length(self.th)[:, None]
        return (cos_alpha * dalphadth * ds_dth * boundary_density).ravel()
