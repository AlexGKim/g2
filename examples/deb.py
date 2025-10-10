"""
Detached Eclipsing Binary Analysis using TotallyOccultedDisk
===========================================================

This module implements a comprehensive analysis function for detached eclipsing binaries
using the TotallyOccultedDisk model. It demonstrates various plotting and analysis
capabilities for the KIC 8410637 system.

The function performs:
1. Intensity image plotting over angular extent
2. V_squared plotting for a grid of lambda_0 u, lambda_0 v coordinates
3. Difference plotting compared to a UniformDisk model
4. Jacobian calculation and printing
5. Inverse noise calculation for observations
6. Fisher Matrix calculation with uncertainty estimates

Example usage for KIC 8410637:
>>> analyze_detached_eclipsing_binary()
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.special import j1, jv
import jax.numpy as jnp

from g2.models.sources.simple import TotallyOccultedDisk, UniformDiskFixR
from g2.core import Observation, inverse_noise, fisher_matrix


def calculate_spectral_exitance_from_kepler_magnitude(kp_mag, radius_m, distance_pc):
    """
    Calculate spectral exitance from Kepler magnitude.
    
    Parameters
    ----------
    kp_mag : float
        Kepler magnitude
    radius_m : float
        Radius in meters
    distance_pc : float
        Distance in parsecs
        
    Returns
    -------
    spectral_exitance : float
        Spectral exitance in W m⁻² Hz⁻¹
    """
    # Convert distance to meters
    distance_m = distance_pc * 3.0857e16  # parsecs to meters
    
    # Kepler band effective wavelength ~640 nm, frequency ~4.7e14 Hz
    nu_kepler = 4.7e14  # Hz
    
    # Convert Kepler magnitude to flux density (approximate)
    # Kp = 0 corresponds to ~3631 Jy in AB system
    flux_density_jy = 3631 * 10**(-kp_mag/2.5)  # Jansky
    flux_density = flux_density_jy * 1e-26  # W m⁻² Hz⁻¹
    
    # Calculate spectral exitance from flux density
    # F = L / (4π d²) where L is luminosity
    # L = spectral_exitance * 4π R²
    # So: spectral_exitance = F * d² / R²
    spectral_exitance = flux_density * distance_m**2 / radius_m**2
    
    return spectral_exitance


def analyze_detached_eclipsing_binary():
    """
    Comprehensive analysis function for detached eclipsing binary KIC 8410637.
    
    This function implements all the requested analysis steps:
    1. Plots intensity image over angular extent
    2. Plots V_squared for grid of lambda_0 u, lambda_0 v
    3. Plots difference with UniformDisk model
    4. Prints Jacobian
    5. Prints inverse_noise for observations
    6. Prints Fisher Matrix and uncertainties
    """
    
    # Physical constants
    solar_radius = 6.96e8  # meters
    parsec = 3.0857e16  # meters
    c = 2.99792458e8  # m/s
    
    # KIC 8410637 parameters
    # Star 0 (primary)
    radius_m_0 = 10.74 * solar_radius  # meters
    distance = 998 * parsec  # meters
    kp_magnitude = 10.77
    
    # Star 1 (secondary) 
    radius_m_1 = 1.571 * solar_radius  # meters
    flux_ratio = 7.86  # star 0 to star 1 flux ratio
    
    # Calculate spectral exitances
    spectral_exitance_0 = calculate_spectral_exitance_from_kepler_magnitude(
        kp_magnitude, radius_m_0, 998)
    
    # Confirm the flux ratio calculation
    # spectral_exitance_1 should be lower by factor: flux_ratio * (radius_m_1/radius_m_0)**2
    radius_ratio_squared = (radius_m_1 / radius_m_0)**2
    expected_factor = flux_ratio * radius_ratio_squared
    spectral_exitance_1 = spectral_exitance_0 / expected_factor
    
    print(f"Flux ratio: {flux_ratio}")
    print(f"Radius ratio squared: {radius_ratio_squared:.6f}")
    print(f"Expected spectral exitance factor: {expected_factor:.6f}")
    print(f"Spectral exitance 0: {spectral_exitance_0:.3e} W m⁻² Hz⁻¹")
    print(f"Spectral exitance 1: {spectral_exitance_1:.3e} W m⁻² Hz⁻¹")
    print(f"Confirmed: spectral_exitance_1 is lower by factor {spectral_exitance_0/spectral_exitance_1:.3f}")
    
    # Choose dx, dy such that star 1 profile is wholly within star 0
    # Make sure |dx|, |dy| + radius_1 < radius_0
    angular_radius_0 = radius_m_0 / distance
    angular_radius_1 = radius_m_1 / distance
    
    # Choose dx, dy to be about 60% of the way from center to edge
    dx = 0.6 * (angular_radius_0 - angular_radius_1) * distance
    dy = 0.4 * (angular_radius_0 - angular_radius_1) * distance
    
    print(f"\nChosen offsets:")
    print(f"dx: {dx/solar_radius:.3f} solar radii = {dx/distance*206265:.1f} arcsec")
    print(f"dy: {dy/solar_radius:.3f} solar radii = {dy/distance*206265:.1f} arcsec")
    print(f"Check: offset + radius_1 = {(np.sqrt(dx**2 + dy**2) + radius_m_1)/solar_radius:.3f} < radius_0 = {radius_m_0/solar_radius:.1f} solar radii")
    
    # Create TotallyOccultedDisk source
    source = TotallyOccultedDisk(
        spectral_exitance_0=spectral_exitance_0,
        radius_m_0=radius_m_0,
        spectral_exitance_1=spectral_exitance_1, 
        radius_m_1=radius_m_1,
        dx=dx,
        dy=dy,
        distance=distance
    )
    
    # Create comparison UniformDisk for star 0
    uniform_disk = UniformDiskFixR(
        spectral_exitance=spectral_exitance_0,
        radius_m=radius_m_0,
        distance=distance
    )
    
    # Observation parameters
    nu_0 = 5e14  # 600 nm
    wavelength = c / nu_0
    
    # Create observation configuration
    observations = [
        Observation(integration_time=3600, telescope_area=1.0, throughput=1.0, detector_jitter=1e-11),
        Observation(integration_time=7200, telescope_area=2.0, throughput=0.8, detector_jitter=5e-12),
        Observation(integration_time=1800, telescope_area=4.0, throughput=0.9, detector_jitter=2e-11)
    ]
    
    # Create PDF file for all plots
    pdf_filename = 'kic8410637_analysis.pdf'
    print(f"\nCreating plots and saving to {pdf_filename}...")
    
    with PdfPages(pdf_filename) as pdf:
        # 1. Plot intensity image over angular extent
        print(f"1. Plotting intensity image...")
        
        # Create angular grid
        angular_extent = 3 * angular_radius_0  # 3 times the primary radius
        n_pixels = 64  # Reduced for faster computation
        angles = np.linspace(-angular_extent, angular_extent, n_pixels)
        angle_x, angle_y = np.meshgrid(angles, angles)
        
        # Calculate intensity on grid
        intensity_grid = np.zeros((n_pixels, n_pixels))
        for i in range(n_pixels):
            for j in range(n_pixels):
                n_hat = np.array([angle_x[i, j], angle_y[i, j]])
                intensity_grid[i, j] = source.intensity(nu_0, n_hat)
        
        # Plot intensity
        fig, ax = plt.subplots(figsize=(10, 8))
        extent = [-angular_extent*206265, angular_extent*206265,
                  -angular_extent*206265, angular_extent*206265]  # Convert to arcsec
        im = ax.imshow(intensity_grid, extent=extent, origin='lower', cmap='hot')
        plt.colorbar(im, label='Intensity (W m⁻² Hz⁻¹ sr⁻¹)')
        ax.set_xlabel('Angular offset (arcsec)')
        ax.set_ylabel('Angular offset (arcsec)')
        ax.set_title('TotallyOccultedDisk Intensity Distribution\nKIC 8410637')
        
        # Add circles showing stellar radii
        circle_0 = plt.Circle((0, 0), angular_radius_0*206265, fill=False, color='white', linestyle='--', linewidth=2)
        circle_1 = plt.Circle((dx/distance*206265, dy/distance*206265), angular_radius_1*206265,
                             fill=False, color='cyan', linestyle='-', linewidth=2)
        ax.add_patch(circle_0)
        ax.add_patch(circle_1)
        ax.legend(['Primary boundary', 'Secondary boundary'])
        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
        
        # 2. Plot V_squared for grid of lambda_0 u, lambda_0 v
        print(f"2. Plotting V_squared for grid of lambda_0 u, lambda_0 v...")
        
        # Create baseline grid in units of lambda_0
        max_baseline_lambda = 2.0  # Maximum baseline in units of wavelength
        n_baseline = 32  # Reduced for faster computation
        u_lambda = np.linspace(-max_baseline_lambda, max_baseline_lambda, n_baseline)
        v_lambda = np.linspace(-max_baseline_lambda, max_baseline_lambda, n_baseline)
        u_grid, v_grid = np.meshgrid(u_lambda, v_lambda)
        
        # Calculate V_squared on grid
        v_squared_grid = np.zeros((n_baseline, n_baseline))
        for i in range(n_baseline):
            for j in range(n_baseline):
                baseline = np.array([u_grid[i, j] * wavelength, v_grid[i, j] * wavelength, 0.0])
                v_squared_grid[i, j] = source.V_squared(nu_0, baseline)
        
        # Plot V_squared
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(v_squared_grid, extent=[-max_baseline_lambda, max_baseline_lambda,
                                             -max_baseline_lambda, max_baseline_lambda],
                       origin='lower', cmap='viridis')
        plt.colorbar(im, label='|V|²')
        ax.set_xlabel('λ₀ u')
        ax.set_ylabel('λ₀ v')
        ax.set_title('Squared Visibility |V|² for TotallyOccultedDisk\nKIC 8410637')
        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
        
        # 3. Plot difference with UniformDisk
        print(f"3. Plotting difference with UniformDisk...")
        
        # Calculate V_squared for uniform disk on same grid
        v_squared_uniform = np.zeros((n_baseline, n_baseline))
        for i in range(n_baseline):
            for j in range(n_baseline):
                baseline = np.array([u_grid[i, j] * wavelength, v_grid[i, j] * wavelength, 0.0])
                v_squared_uniform[i, j] = uniform_disk.V_squared(nu_0, baseline)
        
        # Calculate difference
        v_squared_diff = v_squared_grid - v_squared_uniform
        
        # Plot difference
        fig, ax = plt.subplots(figsize=(10, 8))
        vmax = np.max(np.abs(v_squared_diff))
        im = ax.imshow(v_squared_diff, extent=[-max_baseline_lambda, max_baseline_lambda,
                                             -max_baseline_lambda, max_baseline_lambda],
                       origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.colorbar(im, label='Δ|V|² (TotallyOccultedDisk - UniformDisk)')
        ax.set_xlabel('λ₀ u')
        ax.set_ylabel('λ₀ v')
        ax.set_title('Difference in |V|²: TotallyOccultedDisk - UniformDisk\nKIC 8410637')
        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
    
    print(f"   All plots saved to {pdf_filename}")
    
    # 4. Print Jacobian
    print(f"\n4. Jacobian calculation...")
    
    # Choose a representative baseline
    baseline_test = np.array([0.5 * wavelength, 0.3 * wavelength, 0.0])
    jacobian = source.V_squared_jacobian(nu_0, baseline_test)
    
    print("Jacobian of |V|² with respect to source parameters:")
    for param_name, grad_value in jacobian.items():
        if np.isscalar(grad_value):
            print(f"  ∂|V|²/∂{param_name}: {grad_value:.6e}")
        else:
            print(f"  ∂|V|²/∂{param_name}: {grad_value}")
    
    # 5. Print inverse_noise for observations
    print(f"\n5. Inverse noise calculations...")
    
    for i, obs in enumerate(observations):
        inv_noise = inverse_noise(source, nu_0, obs)
        print(f"Observation {i+1}:")
        print(f"  Integration time: {obs.integration_time} s")
        print(f"  Telescope area: {obs.telescope_area} m²")
        print(f"  Throughput: {obs.throughput}")
        print(f"  Detector jitter: {obs.detector_jitter:.1e} s")
        print(f"  Inverse noise: {inv_noise:.3e}")
    
    # 6. Fisher Matrix calculation
    print(f"\n6. Fisher Matrix calculation...")
    
    # Use the first observation for Fisher matrix calculation
    obs_fisher = observations[0]
    
    try:
        fisher_mat = fisher_matrix(source, nu_0, baseline_test, obs_fisher)
        
        print("Fisher Matrix:")
        print(fisher_mat)
        
        # Calculate uncertainties (square root of diagonal)
        if fisher_mat.size > 0:
            diagonal_elements = np.diag(fisher_mat)
            uncertainties = np.sqrt(1.0 / np.maximum(diagonal_elements, 1e-20))  # Avoid division by zero
            
            print("\nParameter uncertainties (√(F⁻¹ᵢᵢ)):")
            param_names = list(source.get_params().keys())
            for i, (param_name, uncertainty) in enumerate(zip(param_names, uncertainties)):
                print(f"  σ({param_name}): {uncertainty:.6e}")
        else:
            print("Fisher matrix is empty or singular")
            
    except Exception as e:
        print(f"Error calculating Fisher matrix: {e}")
        print("This may be due to the complexity of the TotallyOccultedDisk model")
    
    # Summary
    print(f"\n" + "="*60)
    print("SUMMARY FOR KIC 8410637")
    print("="*60)
    print(f"Primary star (star 0):")
    print(f"  Radius: {radius_m_0/solar_radius:.2f} R☉")
    print(f"  Spectral exitance: {spectral_exitance_0:.3e} W m⁻² Hz⁻¹")
    print(f"Secondary star (star 1):")
    print(f"  Radius: {radius_m_1/solar_radius:.3f} R☉") 
    print(f"  Spectral exitance: {spectral_exitance_1:.3e} W m⁻² Hz⁻¹")
    print(f"  Flux ratio (0/1): {flux_ratio:.2f}")
    print(f"System:")
    print(f"  Distance: {distance/parsec:.0f} pc")
    print(f"  Angular separation: {np.sqrt(dx**2 + dy**2)/distance*206265:.1f} arcsec")
    print(f"  Primary angular radius: {angular_radius_0*206265:.3f} arcsec")
    print(f"  Secondary angular radius: {angular_radius_1*206265:.3f} arcsec")
    
    return source, uniform_disk, observations


if __name__ == "__main__":
    # Run the analysis
    source, uniform_disk, observations = analyze_detached_eclipsing_binary()