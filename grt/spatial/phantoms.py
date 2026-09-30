import numpy as np


def shepp_logan(x, y, z):
    """Evaluate the 3D modified Shepp-Logan/Toft phantom."""
    x, y, z = np.broadcast_arrays(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(z, dtype=float),
    )

    # Columns:
    # amplitude, a, b, c, x0, y0, z0, rotation_about_z_degrees
    ellipsoids = np.array([
        [ 1.0, 0.6900, 0.9200, 0.9000,  0.00,  0.0000,  0.00,   0.0],
        [-0.8, 0.6624, 0.8740, 0.8800,  0.00, -0.0184,  0.00,   0.0],
        [-0.2, 0.1100, 0.3100, 0.2200,  0.22,  0.0000,  0.00, -18.0],
        [-0.2, 0.1600, 0.4100, 0.2800, -0.22,  0.0000,  0.00,  18.0],
        [ 0.1, 0.2100, 0.2500, 0.4100,  0.00,  0.3500, -0.15,   0.0],
        [ 0.1, 0.0460, 0.0460, 0.0500,  0.00,  0.1000,  0.25,   0.0],
        [ 0.1, 0.0460, 0.0230, 0.0500, -0.08, -0.6050,  0.00,   0.0],
        [ 0.1, 0.0230, 0.0230, 0.0200,  0.00, -0.6060,  0.00,   0.0],
        [ 0.1, 0.0230, 0.0460, 0.0200,  0.06, -0.6050,  0.00,   0.0],
    ], dtype=float)

    values = np.zeros(x.shape, dtype=float)

    for amplitude, a, b, c, x0, y0, z0, angle_degrees in ellipsoids:
        angle = np.deg2rad(angle_degrees)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        dx = x - x0
        dy = y - y0
        dz = z - z0

        # Rotation about the z-axis.
        rotated_x = dx * cos_angle + dy * sin_angle
        rotated_y = dy * cos_angle - dx * sin_angle

        inside = (
            rotated_x**2 / a**2
            + rotated_y**2 / b**2
            + dz**2 / c**2
            <= 1.0
        )

        values += amplitude * inside

    return values
