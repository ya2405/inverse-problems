"""Numerical geodesics for conformal metrics in three dimensions."""

import numpy as np
from numpy.typing import ArrayLike

from common.typing import FloatArray, FloatValue, IntegerValue
from common.utils import normalize_3d
from grt.spatial.influx.grid import UniformSphericalBeamGrid
from grt.spatial.manifold.domain import SphericalBall
from grt.spatial.manifold.metric import Metric

class Geodesics:
    """Trace incoming unit-speed geodesics with a second-order Heun step."""

    def __init__(self, metric: Metric, domain: SphericalBall, influx_grid: UniformSphericalBeamGrid, step_size: FloatValue, max_length: FloatValue = 100.0, batch_size: IntegerValue = 4096):
        if step_size <= 0:
            raise ValueError("step_size must be positive.")

        self.step_size = step_size
        
        if max_length <= 0:
            raise ValueError("max_length must be positive.")

        self.max_length = max_length
        
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        
        self.batch_size = batch_size
        self.influx_flow = self.trace_influx_flow(metric, domain, influx_grid)

    def geodesic_step(
        self, x: FloatArray, y: FloatArray, z: FloatArray, v_x: FloatArray, v_y: FloatArray, v_z: FloatArray, metric: Metric
    ) -> tuple[tuple[FloatArray, FloatArray, FloatArray], tuple[FloatArray, FloatArray, FloatArray]]:
        v_x, v_y, v_z = normalize_3d(v_x, v_y, v_z)

        (k1_x, k1_y, k1_z), (k1_v_x, k1_v_y, k1_v_z) = geodesic_flow_derivative(x, y, z, v_x, v_y, v_z, metric)
        xm, ym, zm = x + self.step_size * k1_x, y + self.step_size * k1_y, z + self.step_size * k1_z
        v_xm, v_ym, v_zm = normalize_3d(v_x + self.step_size * k1_v_x, v_y + self.step_size * k1_v_y, v_z + self.step_size * k1_v_z)

        (k2_x, k2_y, k2_z), (k2_v_x, k2_v_y, k2_v_z) = geodesic_flow_derivative(xm, ym, zm, v_xm, v_ym, v_zm, metric)
        xf, yf, zf = x + self.step_size * (k1_x + k2_x) / 2, y + self.step_size * (k1_y + k2_y) / 2, z + self.step_size * (k1_z + k2_z) / 2
        v_xf, v_yf, v_zf = normalize_3d(
            v_x + self.step_size * (k1_v_x + k2_v_x) / 2,
            v_y + self.step_size * (k1_v_y + k2_v_y) / 2,
            v_z + self.step_size * (k1_v_z + k2_v_z) / 2,
        )
        
        return (xf, yf, zf), (v_xf, v_yf, v_zf)

    def trace_influx_flow(
        self,
        metric: Metric,
        domain: SphericalBall,
        influx_grid: UniformSphericalBeamGrid,
    ) -> list[FloatArray]:
        """Trace rays in batches and return midpoint arrays of shape ``(n, 3)``.

        The returned list has the same flattened order as ``influx_grid.cell_ids``:
        boundary polar, boundary azimuth, alpha, then gamma (fastest).
        """
        paths = []
        max_steps = IntegerValue(np.ceil(self.max_length / self.step_size))

        for start in range(0, influx_grid.size, self.batch_size):
            stop = min(start + self.batch_size, influx_grid.size)
            indices = np.arange(start, stop, dtype=np.intp)
            th_ids, phi_ids, alpha_ids, gamma_ids = np.unravel_index(
                indices, (influx_grid.th.size, influx_grid.phi.size, influx_grid.alpha.size, influx_grid.gamma.size)
            )
            th = influx_grid.th[th_ids]
            phi = influx_grid.phi[phi_ids]
            alpha = influx_grid.alpha[alpha_ids]
            gamma = influx_grid.gamma[gamma_ids]

            x, y, z = domain.boundary_point(th, phi)
            v_x, v_y, v_z = influx_direction(th, phi, alpha, gamma, domain)
            
            batch_size = stop - start
            active = np.ones(batch_size, dtype=bool)
            lengths = np.zeros(batch_size, dtype=np.intp)
            point_steps: list[FloatArray] = []

            for _ in range(max_steps):
                if not np.any(active):
                    break

                active_ids = np.flatnonzero(active)
                (xn, yn, zn), (v_xn, v_yn, v_zn) = self.geodesic_step(
                    x[active_ids],
                    y[active_ids],
                    z[active_ids],
                    v_x[active_ids],
                    v_y[active_ids],
                    v_z[active_ids],
                    metric,
                )
                next_inside = domain.contains(xn, yn, zn)
                accepted_ids = active_ids[next_inside]

                step_points = np.full(
                    (batch_size, 3), np.nan, dtype=FloatValue
                )
                step_points[accepted_ids, 0] = xn[next_inside]
                step_points[accepted_ids, 1] = yn[next_inside]
                step_points[accepted_ids, 2] = zn[next_inside]
                point_steps.append(step_points)
                lengths[accepted_ids] += 1

                x[accepted_ids] = xn[next_inside]
                y[accepted_ids] = yn[next_inside]
                z[accepted_ids] = zn[next_inside]
                v_x[accepted_ids] = v_xn[next_inside]
                v_y[accepted_ids] = v_yn[next_inside]
                v_z[accepted_ids] = v_zn[next_inside]
                active[:] = False
                active[accepted_ids] = True
            else:
                if np.any(active):
                    raise RuntimeError(
                        f"{np.count_nonzero(active)} rays did not exit; "
                        "increase max_length or check the nontrapping condition"
                    )

            if point_steps:
                stored_points = np.stack(point_steps)
                paths.extend(
                    stored_points[:length, ray_id].copy()
                    for ray_id, length in enumerate(lengths)
                )
            else:
                paths.extend(
                    np.empty((0, 3), dtype=FloatValue)
                    for _ in range(batch_size)
                )

        return paths


def geodesic_flow_derivative(
    x: FloatArray, y: FloatArray, z: FloatArray, v_x: FloatArray, v_y: FloatArray, v_z: FloatArray, metric: Metric
) -> tuple[tuple[FloatArray, FloatArray, FloatArray], tuple[FloatArray, FloatArray, FloatArray]]:
    scale = np.exp(- 0.5 * metric.log(x, y, z))
    grad_x, grad_y, grad_z = metric.grad_log(x, y, z)
    proj = grad_x * v_x + grad_y * v_y + grad_z * v_z
    
    return (scale * v_x, scale * v_y, scale * v_z), (0.5 * scale * (grad_x - proj * v_x), 0.5 * scale * (grad_y - proj * v_y), 0.5 * scale * (grad_z - proj * v_z))


def influx_direction(th: FloatArray, phi: FloatArray, alpha: FloatArray, gamma: FloatArray, domain: SphericalBall) -> tuple[FloatArray, FloatArray, FloatArray]:
    inward, tangent_th, tangent_phi = domain.boundary_frame(th, phi)
    v = np.cos(alpha)[:, None] * inward
    v += np.sin(alpha)[:, None] * (np.cos(gamma)[:, None] * tangent_th + np.sin(gamma)[:, None] * tangent_phi)
    return normalize_3d(v[:, 0], v[:, 1], v[:, 2])
