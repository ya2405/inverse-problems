import numpy as np

from common.typing import FloatArray, FloatValue, IntegerValue, IntegerArray, FloatFunction2D
from grt.planar.manifold.domain import CircularDisk
from grt.planar.manifold.metric import Metric


class UniformCartesianGrid:
    """Uniform Cartesian discretization of a domain in xy-plane."""
    def __init__(self, domain: CircularDisk, metric: Metric, shape: tuple[IntegerValue, IntegerValue]):
        if shape[0] <= 0 or shape[1] <= 0:
            raise ValueError("shape entries must be positive.")
        
        self.shape = shape

        self.xx, self.yy = self.meshgrid(domain)
        self.domain_mask = domain.contains(self.xx, self.yy)
        self.size = self.domain_mask.sum()

        self.cell_ids = - np.ones(self.shape, dtype=IntegerValue)
        self.cell_ids[self.domain_mask] = np.arange(self.size)

        # 1d array containing cell areas ordered by cell ids
        self.cell_areas = self.compute_cell_areas(domain, metric)

    def meshgrid(self, domain: CircularDisk) -> tuple[FloatArray, FloatArray]:
        ny, nx = self.shape
        x_edges = np.linspace(- domain.radius, domain.radius, nx + 1, dtype=FloatValue)
        y_edges = np.linspace(- domain.radius, domain.radius, ny + 1, dtype=FloatValue)
        return np.meshgrid((x_edges[:-1] + x_edges[1:]) / 2, (y_edges[:-1] + y_edges[1:]) / 2)

    def compute_cell_areas(self, domain: CircularDisk, metric: Metric) -> FloatArray:
        ny, nx = self.shape
        dxdy = ((2 * domain.radius) / nx) * ((2 * domain.radius) / ny)
        logs = metric.log(self.xx[self.domain_mask], self.yy[self.domain_mask])
        return dxdy * np.exp(logs)
        
    def points_to_cell_ids(self, x: FloatArray, y: FloatArray, domain: CircularDisk) -> IntegerArray:
        if x.shape != y.shape:
            raise ValueError(f"x and y must of the same shape; got x.shape={x.shape} and y.shape={y.shape}.")
        
        ny, nx = self.shape
        ix = np.floor((x + domain.radius) / (2 * domain.radius) * nx).astype(IntegerValue)
        iy = np.floor((y + domain.radius) / (2 * domain.radius) * ny).astype(IntegerValue)
        valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        cell_ids = np.full(x.shape, -1, dtype=IntegerValue)
        cell_ids[valid] = self.cell_ids[iy[valid], ix[valid]]
        return cell_ids

    def coefficients_from_function(self, f: FloatFunction2D) -> FloatArray:
        values = f(self.xx[self.domain_mask], self.yy[self.domain_mask])
        return values * np.sqrt(self.cell_areas)

    def coefficients_from_image(self, image: FloatArray) -> FloatArray:
        if image.shape != self.shape:
            raise ValueError(
                f"image must have shape {self.shape}; got {image.shape}."
            )

        values = np.nan_to_num(image[self.domain_mask], nan=0.0, copy=True)
        return values * np.sqrt(self.cell_areas)

    def values_from_coefficients(self, coefficients: FloatArray, outer_vals: FloatValue = np.nan) -> FloatArray:
        values = np.full(self.shape, outer_vals, dtype=FloatValue)
        values[self.domain_mask] = coefficients / np.sqrt(self.cell_areas)
        return values
