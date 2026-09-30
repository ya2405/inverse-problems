import numpy as np
import matplotlib.pyplot as plt

from common.typing import FloatArray, IntegerValue
from grt.planar.geodesics import Geodesics
from grt.planar.influx.grid import UniformFanBeamGrid
from grt.planar.manifold.domain import CircularDisk
from grt.planar.manifold.grid import UniformCartesianGrid
from grt.reconstruction import Result


def plot_phantom(truth: FloatArray, domain: CircularDisk, domain_grid: UniformCartesianGrid):
    fig = plt.figure(figsize=(4, 4), constrained_layout=True)
    grid = fig.add_gridspec(1, 1)
    image_options = dict(origin='lower', cmap='RdBu_r')

    # Phantom
    ax_phantom = fig.add_subplot(grid[0, 0])
    f_phantom = ax_phantom.imshow(domain_grid.values_from_coefficients(truth), extent=domain.bounds(), **image_options)
    ax_phantom.axis("off")
    phantom_colorbar = fig.colorbar(f_phantom, ax=ax_phantom, location="bottom", orientation="horizontal")
    phantom_colorbar.ax.tick_params(labelsize=16)


def plot_reconstruction(truth: FloatArray, output: Result, domain: CircularDisk, domain_grid: UniformCartesianGrid):
    fig = plt.figure(figsize=(8, 5), constrained_layout=True)
    fig.suptitle(t=output.algorithm_name, fontsize=20)
    grid = fig.add_gridspec(1, 2)
    image_options = dict(origin='lower', cmap='RdBu_r')

    # Reconstruction
    ax_recovery = fig.add_subplot(grid[0, 0])
    f_recovery = ax_recovery.imshow(domain_grid.values_from_coefficients(output.coefficients), extent=domain.bounds(), **image_options)
    ax_recovery.set_xticks([]), ax_recovery.set_yticks([])
    ax_recovery.set(title='Reconstruction', xlabel=f'{output.stopping_iteration} iterations');
    recovery_colorbar = fig.colorbar(f_recovery, ax=ax_recovery, location="bottom", orientation="horizontal")
    recovery_colorbar.ax.tick_params(labelsize=16)

    # Absolute Pointwise Error
    ax_error = fig.add_subplot(grid[0, 1])
    f_error = ax_error.imshow(np.abs(domain_grid.values_from_coefficients(truth) - domain_grid.values_from_coefficients(output.coefficients)), extent=domain.bounds(), **image_options)
    ax_error.set_xticks([]), ax_error.set_yticks([])
    ax_error.set(title='Absolute Pointwise Error', xlabel=f'relative error {round(100 * np.linalg.norm(truth - output.coefficients) / np.linalg.norm(truth), 2):.2f}%');
    error_colorbar = fig.colorbar(f_error, ax=ax_error, location="bottom", orientation="horizontal")
    error_colorbar.ax.tick_params(labelsize=16)

    for ax in fig.axes:
        ax.title.set_size(16)
        ax.xaxis.label.set_size(16)


def plot_sinogram(sinogram: FloatArray, n_geodesics: IntegerValue, domain: CircularDisk, influx_grid: UniformFanBeamGrid, geodesics: Geodesics):
    if influx_grid.alpha.size // n_geodesics == 0:
        raise ValueError(f"n_geodesics must not be greater than influx_grid.alpha.size; got n_geodesics={n_geodesics} and influx_grid.alpha.size={influx_grid.alpha.size}.")

    fig = plt.figure(figsize=(12, 4), constrained_layout=True)
    grid = fig.add_gridspec(1, 3)
    image_options = dict(origin='lower', cmap='RdBu_r')

    # Geodesics
    ax_geodesics = fig.add_subplot(grid[0, 0])
    ax_geodesics.set_facecolor("white")
    xmin, xmax, ymin, ymax = domain.bounds()
    x_padding, y_padding = 0.03 * (xmax - xmin), 0.03 * (ymax - ymin)
    ax_geodesics.set_xlim(xmin - x_padding, xmax + x_padding)
    ax_geodesics.set_ylim(ymin - y_padding, ymax + y_padding)

    # plot boundary
    theta = np.linspace(0.0, 2 * np.pi, 512, endpoint=True)
    x, y = domain.boundary_point(theta)
    ax_geodesics.plot(x, y, color="black", linewidth=1.5)

    # plot geodesics
    h = influx_grid.alpha.size // n_geodesics
    for idx in [i * h for i in range(n_geodesics)]:
        path = geodesics.influx_flow[idx]
        if len(path):
            ax_geodesics.plot(path[:, 0], path[:, 1], color='red', linewidth=1)
            
    ax_geodesics.set_aspect("equal")
    ax_geodesics.axis("off")

    # Sinogram
    ax_sinogram = fig.add_subplot(grid[0, 1:])
    f_sinogram = ax_sinogram.imshow(sinogram.reshape(len(influx_grid.th), len(influx_grid.alpha)).T, aspect='auto', **image_options)
    ax_sinogram.yaxis.label.set_rotation(0)
    ax_sinogram.set_xticks([]), ax_sinogram.set_yticks([])
    sinogram_colorbar = fig.colorbar(f_sinogram, ax=ax_sinogram)
    sinogram_colorbar.ax.tick_params(labelsize=16)
