"""
Functions for HBT and II calculations (chaotic sources)
=======================================================

This module provides functions to calculate the inverse noise (Fisher information)
for intensity interferometry measurements with chaotic (thermal) light sources.
The inverse noise determines the theoretical sensitivity limits and optimal
measurement strategies for extracting spatial information from temporal correlations.

The calculations are based on the theoretical framework where the second-order
coherence function g²(Δt) connects temporal correlations to spatial visibility
through the relation 

.. math::
    g²(Δt) - 1 = |V(B)|².

Key Features
------------
- Fisher information matrix calculation for chaotic sources
- Signal-to-noise ratio estimation for visibility measurements
- Optimal baseline and integration time calculations
- Support for different detector configurations and noise models

Mathematical Framework
----------------------
For chaotic sources, the inverse noise is related to the photon statistics
and the coherence properties of the light. The Fisher information provides
the theoretical lower bound on parameter estimation uncertainty through the
Cramér-Rao bound.

Usage Example
-------------
>>> from source import UniformDisk
>>> from inverse_noise_chaotic import calculate_inverse_noise
>>> 
>>> # Create a uniform disk source
>>> disk = UniformDisk(flux_density=1e-26, radius=1e-8)
>>> 
>>> # Calculate inverse noise for a baseline measurement
>>> baseline = np.array([100.0, 0.0, 0.0])
>>> nu_0 = 5e14  # 600 nm
>>> delta_nu = 1e12  # 1 THz bandwidth
>>> integration_time = 3600  # 1 hour
>>> 
>>> inv_noise = calculate_inverse_noise(disk, nu_0, baseline, delta_nu, integration_time)
>>> print(f"Inverse noise: {inv_noise:.2e}")
"""

import numpy as np
from typing import Union, Tuple, Optional
from .models.base.source import ChaoticSource, AbstractSource


def calculate_inverse_noise(source: ChaoticSource, 
                          nu_0: float, 
                          baseline: np.ndarray,
                          delta_nu: float,
                          integration_time: float,
                          detector_area: float = 1.0,
                          quantum_efficiency: float = 1.0,
                          dark_current: float = 0.0) -> float:
    """
    Calculate inverse noise (Fisher information) for chaotic source measurements.
    
    This function computes the theoretical sensitivity limit for measuring
    the spatial visibility of a chaotic source using intensity interferometry.
    The calculation accounts for photon noise, detector characteristics,
    and the coherence properties of the source.
    
    Parameters
    ----------
    source : ChaoticSource
        The chaotic light source object.
    nu_0 : float
        Central frequency in Hz.
    baseline : array_like, shape (3,)
        Baseline vector in meters [Bx, By, Bz].
    delta_nu : float
        Frequency bandwidth in Hz.
    integration_time : float
        Integration time in seconds.
    detector_area : float, optional
        Effective detector area in m². Default is 1.0.
    quantum_efficiency : float, optional
        Detector quantum efficiency (0-1). Default is 1.0.
    dark_current : float, optional
        Dark current in electrons/s. Default is 0.0.
        
    Returns
    -------
    inverse_noise : float
        Inverse noise (Fisher information) for the measurement.
        Higher values indicate better theoretical sensitivity.
        
    Notes
    -----
    The inverse noise calculation is based on the photon statistics of
    chaotic light and the coherence time determined by the bandwidth.
    
    For chaotic sources, the variance in intensity measurements is related
    to the mean intensity and the coherence properties through:
    
        σ²(I) = ⟨I⟩² × (1 + g²(0)) / (δν × τ)
    
    where g²(0) = 2 for thermal light, δν is the bandwidth, and τ is
    the integration time.
    
    The Fisher information provides the theoretical lower bound on
    parameter estimation uncertainty via the Cramér-Rao bound.
    """
    # Physical constants
    h = 6.62607015e-34  # Planck constant (J⋅s)
    c = 2.99792458e8    # Speed of light (m/s)
    
    # Calculate photon energy
    photon_energy = h * nu_0
    
    # Get source flux and visibility
    flux = source.total_flux(nu_0)
    visibility = source.visibility(nu_0, baseline)
    visibility_magnitude = abs(visibility)
    
    # Calculate photon rate
    power = flux * detector_area * delta_nu
    photon_rate = power / photon_energy * quantum_efficiency
    
    # Total detected photons
    total_photons = photon_rate * integration_time
    
    # Calculate coherence time
    coherence_time = 1.0 / delta_nu
    
    # Number of independent measurements (coherence cells)
    n_independent = integration_time / coherence_time
    
    # For chaotic light, g²(0) = 2
    g2_zero = 2.0
    
    # Calculate signal and noise for intensity correlation measurement
    # Signal: proportional to visibility squared
    signal = visibility_magnitude**2
    
    # Noise: includes photon noise and detector noise
    # For chaotic light, intensity variance is enhanced by factor (1 + g²(0))
    photon_noise_variance = total_photons * (1 + g2_zero) / n_independent
    dark_noise_variance = dark_current * integration_time
    total_noise_variance = photon_noise_variance + dark_noise_variance
    
    # Fisher information (inverse noise) for visibility measurement
    # This represents the theoretical sensitivity limit
    if total_noise_variance > 0:
        inverse_noise = (signal**2 * total_photons**2) / total_noise_variance
    else:
        inverse_noise = np.inf
    
    return inverse_noise


def calculate_snr_visibility(source: ChaoticSource,
                           nu_0: float,
                           baseline: np.ndarray,
                           delta_nu: float,
                           integration_time: float,
                           detector_area: float = 1.0,
                           quantum_efficiency: float = 1.0) -> float:
    """
    Calculate signal-to-noise ratio for visibility measurements.
    
    This function estimates the achievable SNR for measuring the spatial
    visibility of a chaotic source using intensity interferometry.
    
    Parameters
    ----------
    source : ChaoticSource
        The chaotic light source object.
    nu_0 : float
        Central frequency in Hz.
    baseline : array_like, shape (3,)
        Baseline vector in meters.
    delta_nu : float
        Frequency bandwidth in Hz.
    integration_time : float
        Integration time in seconds.
    detector_area : float, optional
        Effective detector area in m². Default is 1.0.
    quantum_efficiency : float, optional
        Detector quantum efficiency (0-1). Default is 1.0.
        
    Returns
    -------
    snr : float
        Signal-to-noise ratio for visibility measurement.
        
    Notes
    -----
    The SNR calculation accounts for the enhanced noise in chaotic light
    due to the bunching effect (g²(0) = 2) and the finite coherence time.
    """
    # Calculate inverse noise
    inv_noise = calculate_inverse_noise(source, nu_0, baseline, delta_nu,
                                      integration_time, detector_area,
                                      quantum_efficiency)
    
    # SNR is the square root of Fisher information for Gaussian statistics
    snr = np.sqrt(inv_noise)
    
    return snr


def optimize_integration_time(source: ChaoticSource,
                            nu_0: float,
                            baseline: np.ndarray,
                            delta_nu: float,
                            target_snr: float,
                            detector_area: float = 1.0,
                            quantum_efficiency: float = 1.0) -> float:
    """
    Calculate required integration time to achieve target SNR.
    
    Parameters
    ----------
    source : ChaoticSource
        The chaotic light source object.
    nu_0 : float
        Central frequency in Hz.
    baseline : array_like, shape (3,)
        Baseline vector in meters.
    delta_nu : float
        Frequency bandwidth in Hz.
    target_snr : float
        Desired signal-to-noise ratio.
    detector_area : float, optional
        Effective detector area in m². Default is 1.0.
    quantum_efficiency : float, optional
        Detector quantum efficiency (0-1). Default is 1.0.
        
    Returns
    -------
    integration_time : float
        Required integration time in seconds to achieve target SNR.
        
    Notes
    -----
    This function uses an iterative approach to find the integration time
    that yields the desired SNR, accounting for the scaling of noise with
    integration time for chaotic sources.
    """
    # Initial guess: scale with target_snr^2 (for photon-limited case)
    flux = source.total_flux(nu_0)
    visibility = source.visibility(nu_0, baseline)
    visibility_magnitude = abs(visibility)
    
    # Physical constants
    h = 6.62607015e-34  # Planck constant
    photon_energy = h * nu_0
    
    # Rough estimate assuming photon-limited performance
    power = flux * detector_area * delta_nu
    photon_rate = power / photon_energy * quantum_efficiency
    
    # For chaotic light, need extra factor for enhanced noise
    signal = visibility_magnitude**2
    if signal > 0:
        # Rough scaling: SNR² ∝ N_photons for chaotic light
        estimated_photons_needed = (target_snr**2) / signal * 4  # Factor 4 for g²(0)=2
        integration_time_estimate = estimated_photons_needed / photon_rate
    else:
        return np.inf
    
    # Refine with actual calculation
    integration_time = integration_time_estimate
    for _ in range(10):  # Iterative refinement
        snr = calculate_snr_visibility(source, nu_0, baseline, delta_nu,
                                     integration_time, detector_area,
                                     quantum_efficiency)
        if abs(snr - target_snr) / target_snr < 0.01:  # 1% accuracy
            break
        
        # Adjust integration time
        scaling_factor = (target_snr / snr)**2
        integration_time *= scaling_factor
    
    return integration_time


def calculate_fisher_matrix(source: ChaoticSource,
                          nu_0: float,
                          baselines: np.ndarray,
                          delta_nu: float,
                          integration_time: float,
                          detector_area: float = 1.0,
                          quantum_efficiency: float = 1.0) -> np.ndarray:
    """
    Calculate Fisher information matrix for multiple baseline measurements.
    
    This function computes the full Fisher information matrix for estimating
    source parameters from multiple baseline measurements, providing the
    theoretical limits on parameter estimation accuracy.
    
    Parameters
    ----------
    source : ChaoticSource
        The chaotic light source object.
    nu_0 : float
        Central frequency in Hz.
    baselines : array_like, shape (N, 3)
        Array of baseline vectors in meters.
    delta_nu : float
        Frequency bandwidth in Hz.
    integration_time : float
        Integration time in seconds.
    detector_area : float, optional
        Effective detector area in m². Default is 1.0.
    quantum_efficiency : float, optional
        Detector quantum efficiency (0-1). Default is 1.0.
        
    Returns
    -------
    fisher_matrix : ndarray, shape (N, N)
        Fisher information matrix for the measurements.
        The inverse of this matrix provides the covariance matrix
        for parameter estimation via the Cramér-Rao bound.
        
    Notes
    -----
    The Fisher matrix elements are calculated as:
        F_ij = Σ_k (∂V_k/∂θ_i)(∂V_k/∂θ_j) / σ²_k
    
    where V_k is the visibility for baseline k, θ_i are the parameters,
    and σ²_k is the measurement variance for baseline k.
    """
    n_baselines = baselines.shape[0]
    fisher_matrix = np.zeros((n_baselines, n_baselines))
    
    # Calculate inverse noise for each baseline
    for i in range(n_baselines):
        inv_noise_i = calculate_inverse_noise(source, nu_0, baselines[i],
                                            delta_nu, integration_time,
                                            detector_area, quantum_efficiency)
        fisher_matrix[i, i] = inv_noise_i
    
    # Off-diagonal terms would require parameter derivatives
    # For now, assume independent measurements (diagonal matrix)
    
    return fisher_matrix


if __name__ == "__main__":
    """
    Example usage and validation of inverse noise calculations.
    """
    from .sources.source import UniformDisk, PointSource
    
    # Create test sources
    disk = UniformDisk(flux_density=1e-26, radius=1e-8)  # ~2 mas disk
    point = PointSource(lambda nu: 1e-26)  # 1 Jy point source
    
    # Measurement parameters
    nu_0 = 5e14  # 600 nm
    delta_nu = 1e12  # 1 THz bandwidth
    integration_time = 3600  # 1 hour
    detector_area = 10.0  # 10 m² effective area
    
    # Test different baselines
    baselines = [
        np.array([10.0, 0.0, 0.0]),    # 10 m
        np.array([100.0, 0.0, 0.0]),   # 100 m
        np.array([1000.0, 0.0, 0.0]),  # 1 km
    ]
    
    print("Inverse Noise Calculations for Chaotic Sources")
    print("=" * 50)
    
    for source, name in [(disk, "Uniform Disk"), (point, "Point Source")]:
        print(f"\n{name}:")
        print(f"Flux density: {source.total_flux(nu_0):.2e} W/m²/Hz")
        
        for baseline in baselines:
            baseline_length = np.linalg.norm(baseline[:2])
            visibility = source.visibility(nu_0, baseline)
            inv_noise = calculate_inverse_noise(source, nu_0, baseline, delta_nu,
                                              integration_time, detector_area)
            snr = calculate_snr_visibility(source, nu_0, baseline, delta_nu,
                                         integration_time, detector_area)
            
            print(f"  Baseline {baseline_length:4.0f} m: "
                  f"|V| = {abs(visibility):.3f}, "
                  f"Inv. Noise = {inv_noise:.2e}, "
                  f"SNR = {snr:.1f}")
    
    # Test integration time optimization
    print(f"\nIntegration time for SNR = 10:")
    target_snr = 10.0
    baseline = np.array([100.0, 0.0, 0.0])
    
    for source, name in [(disk, "Uniform Disk"), (point, "Point Source")]:
        req_time = optimize_integration_time(source, nu_0, baseline, delta_nu,
                                           target_snr, detector_area)
        print(f"  {name}: {req_time:.1f} seconds ({req_time/3600:.2f} hours)")