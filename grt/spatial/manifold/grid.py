import numpy as np

from common.typing import (
    FloatArray,
    FloatFunction3D,
    FloatValue,
    IntegerArray,
    IntegerValue,
)
from grt.spatial.manifold.domain import SphericalBall
from grt.spatial.manifold.metric import Metric


class UniformCartesianGrid:
    """Uniform Cartesian voxel grid with array order nz, ny, nx."""

    def __init__(self, domain: SphericalBall, metric: Metric, shape: tuple[IntegerValue, IntegerValue, IntegerValue]):
        if shape[0] <= 0 or shape[1] <= 0 or shape[2] <= 0:
            raise ValueError("shape entries must be positive.")
        
        self.shape = shape

        self.xx, self.yy, self.zz = self.meshgrid(domain)
        self.domain_mask = domain.contains(self.xx, self.yy, self.zz)
        self.size = self.domain_mask.sum()

        self.cell_ids = - np.ones(self.shape, dtype=IntegerValue)
        self.cell_ids[self.domain_mask] = np.arange(self.size, dtype=IntegerValue)

        # 1d array containing cell volumes ordered by cell ids
        self.cell_volumes = self.compute_cell_volumes(domain, metric)

    def meshgrid(self, domain: SphericalBall) -> tuple[FloatArray, FloatArray, FloatArray]:
        nz, ny, nx = self.shape
        x_edges = np.linspace(- domain.radius, domain.radius, nx + 1, dtype=FloatValue)
        y_edges = np.linspace(- domain.radius, domain.radius, ny + 1, dtype=FloatValue)
        z_edges = np.linspace(- domain.radius, domain.radius, nz + 1, dtype=FloatValue)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        z_centers = (z_edges[:-1] + z_edges[1:]) / 2
        zz, yy, xx = np.meshgrid(
            z_centers, y_centers, x_centers, indexing="ij"
        )
        return xx, yy, zz

    def compute_cell_volumes(self, domain: SphericalBall, metric: Metric) -> FloatArray:
        nz, ny, nx = self.shape
        dxdydz = ((2 * domain.radius) / nx) * ((2 * domain.radius) / ny) * ((2 * domain.radius) / nz)
        logs = metric.log(self.xx[self.domain_mask], self.yy[self.domain_mask], self.zz[self.domain_mask])
        return dxdydz * np.exp(1.5 * logs)

    def points_to_cell_ids(self, x: FloatArray, y: FloatArray, z: FloatArray, domain: SphericalBall) -> IntegerArray:
        if x.shape != y.shape or x.shape != z.shape or y.shape != z.shape:
                raise ValueError(f"x, y and y must of the same shape; got x.shape={x.shape}, y.shape={y.shape} and z.shape={z.shape}.")
        
        nz, ny, nx = self.shape
        ix = np.floor((x + domain.radius) / (2 * domain.radius) * nx).astype(IntegerValue)
        iy = np.floor((y + domain.radius) / (2 * domain.radius) * ny).astype(IntegerValue)
        iz = np.floor((z + domain.radius) / (2 * domain.radius) * nz).astype(IntegerValue)
        valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
        cell_ids = np.full(x.shape, -1, dtype=IntegerValue)
        cell_ids[valid] = self.cell_ids[iz[valid], iy[valid], ix[valid]]
        return cell_ids

    def coefficients_from_function(self, f: FloatFunction3D) -> FloatArray:
        values = f(self.xx[self.domain_mask], self.yy[self.domain_mask], self.zz[self.domain_mask])
        return np.asarray(values) * np.sqrt(self.cell_volumes)

    def values_from_coefficients(self, coefficients: FloatArray, outer_vals: FloatValue = np.nan) -> FloatArray:
        values = np.full(self.shape, outer_vals, dtype=FloatValue)
        values[self.domain_mask] = coefficients / np.sqrt(self.cell_volumes)
        return values
