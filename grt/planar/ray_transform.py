import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix

from common.typing import FloatArray, FloatValue
from grt.planar.manifold.domain import Disk
from grt.planar.manifold.grid import UniformCartesianGrid
from grt.planar.influx.grid import UniformFanBeamGrid
from grt.planar.geodesics import Geodesics


class RayTransform:
    def __init__(self, geodesics: Geodesics, domain: Disk, domain_grid: UniformCartesianGrid, influx_grid: UniformFanBeamGrid):
        self.matrix = self.build_matrix(geodesics, domain, domain_grid, influx_grid)
        self.weights = 1 / np.sqrt(influx_grid.cell_areas)

    def apply(self, f_t: FloatArray) -> FloatArray:
        if (self.matrix.shape[1], ) != f_t.shape:
            raise ValueError(f"f_t must be of shape ({self.matrix.shape[1]},); got truth.shape={f_t.shape}.")

        return self.weights * (self.matrix @ f_t)

    def build_matrix(self, geodesics: Geodesics, domain: Disk, domain_grid: UniformCartesianGrid, influx_grid: UniformFanBeamGrid) -> csr_matrix:
        """Build the matrix of the ray transform in orthonormal cell coordinates."""
        row_indices, column_indices, entries = [], [], []

        for path_cell_id, path_points in enumerate(geodesics.influx_flow):

            if not len(path_points):
                continue

            path_point_cell_ids = domain_grid.points_to_cell_ids(path_points[:, 0], path_points[:, 1], domain)
            path_point_cell_ids = path_point_cell_ids[path_point_cell_ids >= 0]
            point_in_cell_counts = np.bincount(path_point_cell_ids, minlength=domain_grid.size)
            values = geodesics.step_size * point_in_cell_counts * np.sqrt(influx_grid.cell_areas[path_cell_id]) / np.sqrt(domain_grid.cell_areas)

            nonzero = np.flatnonzero(values)
            row_indices.extend(np.full(nonzero.size, path_cell_id))
            column_indices.extend(nonzero)
            entries.extend(values[nonzero])
            
        return csr_matrix((entries, (row_indices, column_indices)), shape=(influx_grid.size, domain_grid.size))
