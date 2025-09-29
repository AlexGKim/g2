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

    g²(Δt) - 1 = \\|V(B)\\|².

Key Features
------------
* Fisher information matrix calculation for chaotic sources
* Signal-to-noise ratio estimation for visibility measurements  
* Optimal baseline and integration time calculations
* Support for different detector configurations and noise models

Mathematical Framework
----------------------
For chaotic sources, the inverse noise is related to the photon statistics
and the coherence properties of the light. The Fisher information provides
the theoretical lower bound on parameter estimation uncertainty through the
Cramér-Rao bound.

Usage Example
-------------
.. code-block:: python

    from source import UniformDisk
    from inverse_noise_chaotic import calculate_inverse_noise, Observation
    
    # Create a uniform disk source
    disk = UniformDisk(flux_density=1e-26, radius=1e-8)
    
    # Define observation parameters
    observation = Observation(
        nu_0=5e14,
        baseline=np.array([100.0, 0.0, 0.0]),
        integration_time=3600,
        telescope_area=1.0,
        throughput=1.0,
        detector_jitter=0.0
    )
    
    # Calculate inverse noise
    inv_noise = calculate_inverse_noise(disk, observation)
    print(f"Inverse noise: {inv_noise:.2e}")
"""

import numpy as np
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from .models.base.source import ChaoticSource, AbstractSource


@dataclass
class Observation:
    """
    Represents an observational configuration for intensity interferometry.
    
    Attributes
    ----------
    nu_0 : float
        Central frequency in Hz.
    baseline : np.ndarray
        Baseline vector in meters [Bx, By, Bz].
    integration_time : float
        Integration time in seconds.
    telescope_area : float, optional
        Effective telescope area in m². Default is 1.0.
    throughput : float, optional
        System throughput efficiency (0-1). Default is 1.0.
    detector_jitter : float, optional
        Detector timing jitter in seconds. Default is 0.0.
    """
    nu_0: float
    baseline: np.ndarray
    integration_time: float
    telescope_area: float = 1.0
    throughput: float = 1.0
    detector_jitter: float = 0.0


def inverse_noise(source: ChaoticSource, observation: Observation) -> float:
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
    observation : Observation
        Observational configuration containing parameters for the measurement.
        
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
    
    .. math::
        σ²(I) = ⟨I⟩² × (1 + g²(0)) / (δν × τ)
    
    where g²(0) = 2 for thermal light, δν is the bandwidth, and τ is
    the integration time.
    
    The Fisher information provides the theoretical lower bound on
    parameter estimation uncertainty via the Cramér-Rao bound.
    """
    # Physical constants
    h = 6.62607015e-34  # Planck constant (J⋅s)
    
    # Calculate photon energy
    photon_energy = h * observation.nu_0
    
    # Get source flux and visibility
    flux = source.total_flux(observation.nu_0)
    
    # Calculate photon rate per frequency
    photon_rate_per_nu = (
        observation.throughput * 
        observation.telescope_area * 
        flux / 
        photon_energy
    )
    
    inverse_noise = photon_rate_per_nu * np.sqrt(
        observation.integration_time / observation.detector_jitter
    ) * (128 * np.pi)**(-0.25)
    
    return inverse_noise


def fisher_matrix(
    source: ChaoticSource,
    observations: List[Observation]
) -> np.ndarray:
    """
    Calculate the Fisher matrix for the parameters of a chaotic source model
    based on a set of intensity interferometry measurements.
    
    The Fisher matrix is computed using the Jacobian of the squared visibility
    (V_squared) with respect to the source parameters and the inverse noise
    (variance) of each measurement.
    
    Parameters
    ----------
    source : ChaoticSource
        The chaotic source model with parameters to estimate.
    observations : list of Observation
        List of observational configurations.
            
    Returns
    -------
    fisher_matrix : numpy.ndarray
        The Fisher matrix with shape (n_params, n_params), where n_params is
        the number of parameters in the source model. The order of parameters
        matches the keys from source.get_params().
        
    Notes
    -----
    The Fisher matrix provides the Cramér-Rao lower bound on the covariance
    matrix of parameter estimates. The inverse of the Fisher matrix gives
    the minimum achievable covariance matrix.
    
    Example
    -------
    >>> disk = UniformDisk(flux_density=1e-26, radius=1e-8)
    >>> observations = [
    ...     Observation(nu_0=5e14, baseline=np.array([100, 0, 0]), integration_time=3600),
    ...     Observation(nu_0=5e14, baseline=np.array([0, 100, 0]), integration_time=3600)
    ... ]
    >>> fisher = calculate_fisher_matrix(disk, observations)
    """
    # Get parameter names in consistent order
    params = list(source.get_params().keys())
    n_params = len(params)
    fisher = np.zeros((n_params, n_params))
    
    for obs in observations:
        # Compute Jacobian of V_squared with respect to parameters
        jacobian_dict = source.V_squared_jacobian(nu_0=obs.nu_0, baseline=obs.baseline)
        jacobian = np.array([jacobian_dict[p] for p in params])
        
        # Calculate inverse noise (standard deviation) for this observation
        sigma = inverse_noise(source, obs)
        
        # Avoid division by zero (if sigma is zero, skip this measurement)
        if sigma <= 0:
            continue
        
        # Compute contribution to Fisher matrix
        inv_variance = 1.0 / (sigma ** 2)
        fisher += np.outer(jacobian, jacobian) * inv_variance
    
    return fisher