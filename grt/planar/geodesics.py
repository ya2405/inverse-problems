import numpy as np
from numpy.typing import ArrayLike

from common.typing import FloatArray, FloatValue, IntegerValue
from common.utils import normalize_2d
from grt.planar.manifold.domain import CircularDisk
from grt.planar.manifold.metric import Metric
from grt.planar.influx.grid import UniformFanBeamGrid


class Geodesics:
    def __init__(self, metric: Metric, domain: CircularDisk, influx_grid: UniformFanBeamGrid, step_size: FloatValue, max_length: FloatValue = 100.0, batch_size: IntegerValue = 4096):
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
        self, x: FloatArray, y: FloatArray, v_x: FloatArray, v_y: FloatArray, metric: Metric
    ) -> tuple[tuple[FloatArray, FloatArray, FloatArray], tuple[FloatArray, FloatArray, FloatArray]]:
        v_x, v_y = normalize_2d(v_x, v_y)

        (k1_x, k1_y), (k1_v_x, k1_v_y) = geodesic_flow_derivative(x, y, v_x, v_y, metric)
        xm, ym = x + self.step_size * k1_x, y + self.step_size * k1_y
        v_xm, v_ym = normalize_2d(v_x + self.step_size * k1_v_x, v_y + self.step_size * k1_v_y)

        (k2_x, k2_y), (k2_v_x, k2_v_y) = geodesic_flow_derivative(xm, ym, v_xm, v_ym, metric)
        xf, yf = x + self.step_size * (k1_x + k2_x) / 2, y + self.step_size * (k1_y + k2_y) / 2
        v_xf, v_yf = normalize_2d(
            v_x + self.step_size * (k1_v_x + k2_v_x) / 2, v_y + self.step_size * (k1_v_y + k2_v_y) / 2
        )
        
        return (xf, yf), (v_xf, v_yf)


    def trace_influx_flow(
        self, metric: Metric, domain: CircularDisk, influx_grid: UniformFanBeamGrid
    ) -> list[FloatArray]:
        """Trace rays in batches and return sample arrays of shape ``(n, 2)``.

        Rays retain beta-major order, with alpha varying fastest, and the
        returned ragged-list format is unchanged from the scalar implementation.
        """
        paths: list[FloatArray] = []
        max_steps = int(np.ceil(self.max_length / self.step_size))

        for start in range(0, influx_grid.size, self.batch_size):
            stop = min(start + self.batch_size, influx_grid.size)
            indices = np.arange(start, stop, dtype=np.intp)
            th_ids, alpha_ids = np.unravel_index(
                indices, (influx_grid.th.size, influx_grid.alpha.size)
            )
            th, alpha = influx_grid.th[th_ids], influx_grid.alpha[alpha_ids]

            x, y = domain.boundary_point(th)
            v_x, v_y = influx_direction(th, alpha, domain)
            
            batch_size = stop - start
            active = np.ones(batch_size, dtype=bool)
            lengths = np.zeros(batch_size, dtype=np.intp)
            point_steps: list[FloatArray] = []

            for _ in range(max_steps):
                if not np.any(active):
                    break

                active_ids = np.flatnonzero(active)
                (xn, yn), (v_xn, v_yn) = self.geodesic_step(x[active_ids], y[active_ids], v_x[active_ids], v_y[active_ids], metric)
                next_inside = domain.contains(xn, yn)
                accepted_ids = active_ids[next_inside]

                step_points = np.full(
                    (batch_size, 2), np.nan, dtype=FloatValue
                )
                step_points[accepted_ids, 0] = xn[next_inside]
                step_points[accepted_ids, 1] = yn[next_inside]
                point_steps.append(step_points)
                lengths[accepted_ids] += 1

                x[accepted_ids] = xn[next_inside]
                y[accepted_ids] = yn[next_inside]
                v_x[accepted_ids] = v_xn[next_inside]
                v_y[accepted_ids] = v_yn[next_inside]
                active[:] = False
                active[accepted_ids] = True
            else:
                if np.any(active):
                    raise RuntimeError(
                        f"{np.count_nonzero(active)} rays did not exit; "
                        "increase max_length or check the nontrapping condition."
                    )

            if point_steps:
                stored_points = np.stack(point_steps)
                paths.extend(
                    stored_points[:length, ray_id].copy()
                    for ray_id, length in enumerate(lengths)
                )
            else:
                paths.extend(
                    np.empty((0, 2), dtype=FloatValue)
                    for _ in range(batch_size)
                )

        return paths


def geodesic_flow_derivative(
    x: FloatArray, y: FloatArray, v_x: FloatArray, v_y: FloatArray, metric: Metric
) -> tuple[tuple[FloatArray, FloatArray], tuple[FloatArray, FloatArray]]:
    scale = np.exp(- 0.5 * metric.log(x, y))
    grad_x, grad_y = metric.grad_log(x, y)
    proj = grad_x * v_x + grad_y * v_y
    
    return (scale * v_x, scale * v_y), (0.5 * scale * (grad_x - proj * v_x), 0.5 * scale * (grad_y - proj * v_y))


def influx_direction(th: FloatArray, alpha: FloatArray, domain: CircularDisk) -> tuple[FloatArray, FloatArray, FloatArray]:
    inward, tangent_th = domain.boundary_frame(th)
    v = np.cos(alpha)[:, None] * inward + np.sin(alpha)[:, None] * tangent_th
    return normalize_2d(v[:, 0], v[:, 1])
