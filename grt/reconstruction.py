from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from common.typing import FloatArray, FloatValue

@dataclass
class Result:
    coefficients: np.ndarray
    stopping_iteration: int
    algorithm_name: str


def steepest_gradient_descent(matrix: csr_matrix, y: FloatArray, f_0: FloatArray, abs_error_threshold: FloatValue, n_iterations: int | None = None):
    if n_iterations is not None:
        if isinstance(n_iterations, (bool, np.bool_)) or not isinstance(n_iterations, (int, np.integer)) or n_iterations < 0:
            raise ValueError("iterations must be a nonnegative integer")
    
    if (matrix.shape[0],) != y.shape:
        raise ValueError(f"y must be of shape ({matrix.shape[0]},); got y.shape={y.shape}.")
    if (matrix.shape[1],) != f_0.shape:
        raise ValueError(f"f_0 must be of shape ({matrix.shape[1]},); got f_0.shape={f_0.shape}.")
    if not np.isfinite(abs_error_threshold) or abs_error_threshold < 0:
        raise ValueError("abs_error_threshold must be finite and nonnegative.")
    
    f = f_0.copy()
    matrix_T = matrix.T

    print("iterating...")
    k = 0
    while n_iterations is None or k < n_iterations:

        r = matrix @ f - y

        if np.linalg.norm(r) <= abs_error_threshold:
            print("error threshold is satisfied.")
            break

        q = matrix_T @ r
        if not np.any(q):
            print("least-squares gradient is zero.")
            break
        eta = np.linalg.norm(q) ** 2 / np.linalg.norm(matrix @ q) ** 2
        f -= eta * q
        k += 1

    return Result(
        coefficients=f,
        stopping_iteration=k,
        algorithm_name="Steepest Gradient Descent",
    )


def conjugate_gradient_method(matrix: csr_matrix, y: FloatArray, f_0: FloatArray, abs_error_threshold: FloatValue):    
    if (matrix.shape[0],) != y.shape:
        raise ValueError(f"y must be of shape ({matrix.shape[0]},); got y.shape={y.shape}.")
    if (matrix.shape[1],) != f_0.shape:
        raise ValueError(f"f_0 must be of shape ({matrix.shape[1]},); got f_0.shape={f_0.shape}.")
    if not np.isfinite(abs_error_threshold) or abs_error_threshold < 0:
        raise ValueError("abs_error_threshold must be finite and nonnegative.")

    n_iterations = matrix.shape[1]
    
    f = f_0.copy()
    matrix_T = matrix.T
    b = matrix_T @ y
    ds = []

    print("iterating...")
    k = 0
    for _ in range(n_iterations):
        matrix_f = matrix @ f
        
        if np.linalg.norm(y - matrix_f) <= abs_error_threshold:
            print("error threshold is satisfied.")
            break

        r = b - matrix_T @ matrix_f

        if np.linalg.norm(r) <= np.finfo(float).eps :
            print("residual is very small.")
            break

        matrix_r = matrix @ r
        d = r.copy()
        for (dp, matrix_dp, matrix_dp_norm_sq) in ds:
            d -= dp * (matrix_dp).dot(matrix_r) / matrix_dp_norm_sq

        if np.linalg.norm(d) <= np.finfo(float).eps :
            print("conjugate gradient direction is very small.")
            break

        matrix_d = matrix @ d
        matrix_d_norm_sq = matrix_d.dot(matrix_d)
        if matrix_d_norm_sq == 0:
            print("conjugate gradient direction is in the numerical nullspace.")
            break
        ds.append((d, matrix_d, matrix_d_norm_sq))

        alpha = d.dot(r) / matrix_d_norm_sq
        f += alpha * d
        k += 1

    return Result(
        coefficients=f,
        stopping_iteration=k,
        algorithm_name="Conjugate Gradient Method",
    )
