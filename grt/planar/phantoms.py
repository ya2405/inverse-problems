import numpy as np


def shepp_logan(x, y):
    """Evaluate the modified Shepp-Logan/Toft phantom at (x, y)."""
    x, y = np.broadcast_arrays(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
    )

    ellipses = np.array([
        [ 1.0, 0.6900, 0.9200,  0.00,  0.0000,   0.0],
        [-0.8, 0.6624, 0.8740,  0.00, -0.0184,   0.0],
        [-0.2, 0.1100, 0.3100,  0.22,  0.0000, -18.0],
        [-0.2, 0.1600, 0.4100, -0.22,  0.0000,  18.0],
        [ 0.1, 0.2100, 0.2500,  0.00,  0.3500,   0.0],
        [ 0.1, 0.0460, 0.0460,  0.00,  0.1000,   0.0],
        [ 0.1, 0.0460, 0.0460,  0.00, -0.1000,   0.0],
        [ 0.1, 0.0460, 0.0230, -0.08, -0.6050,   0.0],
        [ 0.1, 0.0230, 0.0230,  0.00, -0.6060,   0.0],
        [ 0.1, 0.0230, 0.0460,  0.06, -0.6050,   0.0],
    ])

    values = np.zeros(x.shape, dtype=float)

    for amplitude, a, b, x0, y0, angle_degrees in ellipses:
        angle = np.deg2rad(angle_degrees)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        dx = x - x0
        dy = y - y0

        rotated_x = dx * cos_angle + dy * sin_angle
        rotated_y = dy * cos_angle - dx * sin_angle

        inside = (
            rotated_x**2 / a**2
            + rotated_y**2 / b**2
            <= 1.0
        )

        values += amplitude * inside

    return values
