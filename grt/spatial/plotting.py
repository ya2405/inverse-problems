import numpy as np
import matplotlib.pyplot as plt

from common.typing import FloatArray
from grt.spatial.influx.grid import UniformSphericalBeamGrid
from grt.spatial.manifold.domain import SphericalBall
from grt.spatial.manifold.grid import UniformCartesianGrid
from grt.reconstruction import Result


def plot_phantom(truth: FloatArray, domain: SphericalBall, domain_grid: UniformCartesianGrid):
    fig = plt.figure(figsize=(12, 4), constrained_layout=True)
    grid = fig.add_gridspec(1, 3)
    image_options = dict(origin='lower', cmap='RdBu_r')

    ax_phantom_slice_1 = fig.add_subplot(grid[0, 0])
    ax_phantom_slice_2 = fig.add_subplot(grid[0, 1])
    ax_phantom_slice_3 = fig.add_subplot(grid[0, 2])

    ### Phantom slice
    phantom = domain_grid.values_from_coefficients(truth)

    index_shift = phantom.shape[0] // 16
    z_index_2 = phantom.shape[0] // 2
    z_index_1 = z_index_2 - index_shift
    z_index_3 = z_index_2 + index_shift

    # Slice 1
    phantom_slice_1_image = ax_phantom_slice_1.imshow(phantom[z_index_1], extent=domain.bounds(), **image_options)
    ax_phantom_slice_1.set_aspect("equal")
    ax_phantom_slice_1.axis("off")
    phantom_slice_1_colorbar = fig.colorbar(phantom_slice_1_image, ax=ax_phantom_slice_1, location="bottom", orientation="horizontal")
    phantom_slice_1_colorbar.ax.tick_params(labelsize=16)

    # Slice 2
    phantom_slice_2_image = ax_phantom_slice_2.imshow(phantom[z_index_2], extent=domain.bounds(), **image_options)
    ax_phantom_slice_2.set_aspect("equal")
    ax_phantom_slice_2.axis("off")
    phantom_slice_2_colorbar = fig.colorbar(phantom_slice_2_image, ax=ax_phantom_slice_2, location="bottom", orientation="horizontal")
    phantom_slice_2_colorbar.ax.tick_params(labelsize=16)

    # Slice 3
    phantom_slice_3_image = ax_phantom_slice_3.imshow(phantom[z_index_3], extent=domain.bounds(), **image_options)
    ax_phantom_slice_3.set_aspect("equal")
    ax_phantom_slice_3.axis("off")
    phantom_slice_3_colorbar = fig.colorbar(phantom_slice_3_image, ax=ax_phantom_slice_3, location="bottom", orientation="horizontal")
    phantom_slice_3_colorbar.ax.tick_params(labelsize=16)


def plot_reconstruction(truth: FloatArray, output: Result, domain: SphericalBall, domain_grid: UniformCartesianGrid):
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    fig.suptitle(t=output.algorithm_name, fontsize=20)
    grid = fig.add_gridspec(2, 3)
    image_options = dict(origin='lower', cmap='RdBu_r')

    ax_recovery_slice_1 = fig.add_subplot(grid[0, 0])
    ax_recovery_slice_2 = fig.add_subplot(grid[0, 1])
    ax_recovery_slice_3 = fig.add_subplot(grid[0, 2])

    ax_error_slice_1 = fig.add_subplot(grid[1, 0])
    ax_error_slice_2 = fig.add_subplot(grid[1, 1])
    ax_error_slice_3 = fig.add_subplot(grid[1, 2])

    phantom = domain_grid.values_from_coefficients(truth)
    recovery = domain_grid.values_from_coefficients(output.coefficients)
    error = np.abs(phantom - recovery)

    index_shift = phantom.shape[0] // 16
    z_index_2 = phantom.shape[0] // 2
    z_index_1 = z_index_2 - index_shift
    z_index_3 = z_index_2 + index_shift

    # Reconstruction
    # Slice 1
    recovery_slice_1_image = ax_recovery_slice_1.imshow(recovery[z_index_1], extent=domain.bounds(), **image_options)
    ax_recovery_slice_1.set_aspect("equal")
    ax_recovery_slice_1.axis("off")
    recovery_slice_1_colorbar = fig.colorbar(recovery_slice_1_image, ax=ax_recovery_slice_1, location="bottom", orientation="horizontal")
    recovery_slice_1_colorbar.ax.tick_params(labelsize=16)

    # Slice 2
    recovery_slice_2_image = ax_recovery_slice_2.imshow(recovery[z_index_2], extent=domain.bounds(), **image_options)
    ax_recovery_slice_2.set_aspect("equal")
    ax_recovery_slice_2.axis("off")
    recovery_slice_2_colorbar = fig.colorbar(recovery_slice_2_image, ax=ax_recovery_slice_2, location="bottom", orientation="horizontal")
    recovery_slice_2_colorbar.ax.tick_params(labelsize=16)

    # Slice 3
    recovery_slice_3_image = ax_recovery_slice_3.imshow(recovery[z_index_3], extent=domain.bounds(), **image_options)
    ax_recovery_slice_3.set_aspect("equal")
    ax_recovery_slice_3.axis("off")
    recovery_slice_3_colorbar = fig.colorbar(recovery_slice_3_image, ax=ax_recovery_slice_3, location="bottom", orientation="horizontal")
    recovery_slice_3_colorbar.ax.tick_params(labelsize=16)

    # Absolute Pointwise Error
    # Slice 1
    error_slice_1_image = ax_error_slice_1.imshow(error[z_index_1], extent=domain.bounds(), **image_options)
    ax_error_slice_1.set_aspect("equal")
    ax_error_slice_1.axis("off")
    error_slice_1_colorbar = fig.colorbar(error_slice_1_image, ax=ax_error_slice_1, location="bottom", orientation="horizontal")
    error_slice_1_colorbar.ax.tick_params(labelsize=16)

    # Slice 2
    error_slice_2_image = ax_error_slice_2.imshow(error[z_index_2], extent=domain.bounds(), **image_options)
    ax_error_slice_2.set_aspect("equal")
    ax_error_slice_2.axis("off")
    error_slice_2_colorbar = fig.colorbar(error_slice_2_image, ax=ax_error_slice_2, location="bottom", orientation="horizontal")
    error_slice_2_colorbar.ax.tick_params(labelsize=16)

    # Slice 3
    error_slice_3_image = ax_error_slice_3.imshow(error[z_index_3], extent=domain.bounds(), **image_options)
    ax_error_slice_3.set_aspect("equal")
    ax_error_slice_3.axis("off")
    error_slice_3_colorbar = fig.colorbar(error_slice_3_image, ax=ax_error_slice_3, location="bottom", orientation="horizontal")
    error_slice_3_colorbar.ax.tick_params(labelsize=16)

    caption_text = f'{output.stopping_iteration} iterations, relative error {round(100 * np.linalg.norm(truth - output.coefficients) / np.linalg.norm(truth), 2):.2f}%'
    fig.text(0.5, - 0.03, caption_text, ha='center', fontsize=16, wrap=True)


def plot_sinogram(sinogram: FloatArray, domain: SphericalBall, influx_grid: UniformSphericalBeamGrid):
    fig = plt.figure(figsize=(15, 4), constrained_layout=True)
    grid = fig.add_gridspec(1, 3)
    image_options = dict(origin='lower', cmap='RdBu_r')

    ax_sinogram_slice_1 = fig.add_subplot(grid[0, 0])
    ax_sinogram_slice_2 = fig.add_subplot(grid[0, 1])
    ax_sinogram_slice_3 = fig.add_subplot(grid[0, 2])

    ### Sinogram slice
    sinogram_4D = sinogram.reshape((influx_grid.th.size, influx_grid.phi.size, influx_grid.alpha.size, influx_grid.gamma.size))

    th_index_shift, phi_index_shift = influx_grid.th.shape[0] // 4, influx_grid.phi.shape[0] // 4
    th_index_2, phi_index_2 = influx_grid.th.shape[0] // 2, influx_grid.phi.shape[0] // 2
    th_index_1, phi_index_1 = th_index_2 - th_index_shift, phi_index_2 - phi_index_shift
    th_index_3, phi_index_3 = th_index_2 + th_index_shift, phi_index_2 + phi_index_shift

    # Slice 1
    sinogram_slice_1_image = ax_sinogram_slice_1.imshow(sinogram_4D[th_index_1, phi_index_1], extent=(0, 2 * np.pi, 0, np.pi / 2), **image_options)
    ax_sinogram_slice_1.set_aspect("equal")
    ax_sinogram_slice_1.axis("off")
    phantom_slice_1_colorbar = fig.colorbar(sinogram_slice_1_image, ax=ax_sinogram_slice_1, location="bottom", orientation="horizontal")
    phantom_slice_1_colorbar.ax.tick_params(labelsize=16)

    # Slice 2
    sinogram_slice_2_image = ax_sinogram_slice_2.imshow(sinogram_4D[th_index_2, phi_index_2], extent=(0, 2 * np.pi, 0, np.pi / 2), **image_options)
    ax_sinogram_slice_2.set_aspect("equal")
    ax_sinogram_slice_2.axis("off")
    phantom_slice_2_colorbar = fig.colorbar(sinogram_slice_2_image, ax=ax_sinogram_slice_2, location="bottom", orientation="horizontal")
    phantom_slice_2_colorbar.ax.tick_params(labelsize=16)

    # Slice 3
    sinogram_slice_3_image = ax_sinogram_slice_3.imshow(sinogram_4D[th_index_3, phi_index_3], extent=(0, 2 * np.pi, 0, np.pi / 2), **image_options)
    ax_sinogram_slice_3.set_aspect("equal")
    ax_sinogram_slice_3.axis("off")
    phantom_slice_3_colorbar = fig.colorbar(sinogram_slice_3_image, ax=ax_sinogram_slice_3, location="bottom", orientation="horizontal")
    phantom_slice_3_colorbar.ax.tick_params(labelsize=16)
