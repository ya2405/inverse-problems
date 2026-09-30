import numpy as np
from numpy.typing import ArrayLike

from common.typing import FloatArray, FloatValue, IntegerValue


def asarray_2d(x: ArrayLike, y: ArrayLike) -> tuple[FloatArray | FloatValue, FloatArray | FloatValue]:
    x, y = np.asarray(x, dtype=FloatValue), np.asarray(y, dtype=FloatValue)

    if x.shape != y.shape:
        raise ValueError(f"x and y must of the same shape; got x.shape={x.shape} and y.shape={y.shape}.")

    return x, y


def asarray_3d(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> tuple[FloatArray | FloatValue, FloatArray | FloatValue, FloatArray | FloatValue]:
    x, y, z = np.asarray(x, dtype=FloatValue), np.asarray(y, dtype=FloatValue), np.asarray(z, dtype=FloatValue)

    if x.shape != y.shape or x.shape != z.shape or y.shape != z.shape:
        raise ValueError(f"x, y and y must of the same shape; got x.shape={x.shape}, y.shape={y.shape} and z.shape={z.shape}.")

    return x, y, z


def midpoints(start: FloatValue, stop: FloatValue, count: IntegerValue) -> FloatArray:
    if count <= 0:
        raise ValueError("count must be positive integer.")
    
    edges = np.linspace(start, stop, count + 1, dtype=FloatValue)
    return (edges[:-1] + edges[1:]) / 2


def normalize_2d(x: ArrayLike, y: ArrayLike) -> tuple[FloatArray | FloatValue, FloatArray | FloatValue]:
    x, y = np.asarray(x, dtype=FloatValue), np.asarray(y, dtype=FloatValue)

    if x.shape != y.shape:
        raise ValueError(f"x and y must of the same shape; got x.shape={x.shape} and y.shape={y.shape}.")
    
    norm = np.sqrt(x ** 2 + y ** 2)

    if np.any(norm <= np.finfo(FloatValue).eps):
        raise ValueError("cannot normalize a zero vector.")
    
    return x / norm, y / norm


def normalize_3d(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> tuple[FloatArray | FloatValue, FloatArray | FloatValue, FloatArray | FloatValue]:
    x, y, z = np.asarray(x, dtype=FloatValue), np.asarray(y, dtype=FloatValue), np.asarray(z, dtype=FloatValue)

    if x.shape != y.shape or x.shape != z.shape or y.shape != z.shape:
        raise ValueError(f"x, y and y must of the same shape; got x.shape={x.shape}, y.shape={y.shape} and z.shape={z.shape}.")
    
    norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    if np.any(norm <= np.finfo(FloatValue).eps):
        raise ValueError("cannot normalize a zero vector.")
    
    return x / norm, y / norm, z / norm
