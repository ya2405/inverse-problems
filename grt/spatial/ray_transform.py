"""Discrete geodesic ray transform in three dimensions."""

import numpy as np
from scipy.sparse import csr_matrix

from common.typing import FloatArray
from grt.spatial.geodesics import Geodesics
from grt.spatial.influx.grid import UniformSphericalBeamGrid
from grt.spatial.manifold.domain import Ball
from grt.spatial.manifold.grid import UniformCartesianGrid


class RayTransform:
    """Matrix representation in orthonormal voxel and measurement bases."""

    def __init__(
        self,
        geodesics: Geodesics,
        domain: Ball,
        domain_grid: UniformCartesianGrid,
        influx_grid: UniformSphericalBeamGrid,
    ):
        self.matrix = self.build_matrix(
            geodesics, domain, domain_grid, influx_grid
        )
        self.weights = 1 / np.sqrt(influx_grid.cell_areas)

    def apply(self, coefficients: FloatArray) -> FloatArray:
        coefficients = np.asarray(coefficients)
        if coefficients.shape != (self.matrix.shape[1],):
            raise ValueError(
                f"coefficients must have shape ({self.matrix.shape[1]},); "
                f"got {coefficients.shape}"
            )
        return self.weights * (self.matrix @ coefficients)

    @staticmethod
    def build_matrix(
        geodesics: Geodesics,
        domain: Ball,
        domain_grid: UniformCartesianGrid,
        influx_grid: UniformSphericalBeamGrid,
    ) -> csr_matrix:
        column_blocks, entry_blocks = [], []
        row_pointer = np.empty(influx_grid.size + 1, dtype=np.int64)
        row_pointer[0] = 0
        inverse_sqrt_volumes = 1 / np.sqrt(domain_grid.cell_volumes)
        ray_scales = geodesics.step_size * np.sqrt(influx_grid.cell_areas)

        for ray_id, path_points in enumerate(geodesics.influx_flow):
            if not len(path_points):
                row_pointer[ray_id + 1] = row_pointer[ray_id]
                continue
            cell_ids = domain_grid.points_to_cell_ids(
                path_points[:, 0],
                path_points[:, 1],
                path_points[:, 2],
                domain,
            )
            cell_ids = cell_ids[cell_ids >= 0]
            if not cell_ids.size:
                row_pointer[ray_id + 1] = row_pointer[ray_id]
                continue

            visited_cells, counts = np.unique(cell_ids, return_counts=True)
            column_blocks.append(visited_cells)
            entry_blocks.append(
                ray_scales[ray_id]
                * counts
                * inverse_sqrt_volumes[visited_cells]
            )
            row_pointer[ray_id + 1] = (
                row_pointer[ray_id] + visited_cells.size
            )

        if not entry_blocks:
            return csr_matrix((influx_grid.size, domain_grid.size))

        column_indices = np.concatenate(column_blocks)
        entries = np.concatenate(entry_blocks)

        return csr_matrix(
            (entries, column_indices, row_pointer),
            shape=(influx_grid.size, domain_grid.size),
        )
