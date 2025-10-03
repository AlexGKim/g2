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
    
    # Define observation parameters (telescope/detector configuration)
    observation = Observation(
        integration_time=3600,
        telescope_area=1.0,
        throughput=1.0,
        detector_jitter=0.0
    )
    
    # Calculate inverse noise for specific frequency and baseline
    nu_0 = 5e14  # Hz
    baseline = np.array([100.0, 0.0, 0.0])  # meters
    inv_noise = calculate_inverse_noise(disk, nu_0, baseline, observation)
    print(f"Inverse noise: {inv_noise:.2e}")
"""

import numpy as np
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from g2.models.base.source import ChaoticSource, AbstractSource


@dataclass
class Observation:
    """
    Represents telescope and detector configuration for intensity interferometry.
    
    This dataclass contains the instrumental parameters that define the 
    observational setup, independent of the specific measurement frequency
    and baseline configuration.
    
    Attributes
    ----------
    integration_time : float
        Integration time in seconds.
    telescope_area : float, optional
        Effective telescope area in m². Default is 1.0.
    throughput : float, optional
        System throughput efficiency (0-1). Default is 1.0.
    detector_jitter : float, optional
        Detector timing jitter in seconds. Default is 0.0.
    """
    integration_time: float
    telescope_area: float = 1.0
    throughput: float = 1.0
    detector_jitter: float = 0.0


def inverse_noise(
    source: ChaoticSource,
    nu_0: float,
    observation: Observation
) -> float:
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
    observation : Observation
        Telescope and detector configuration parameters.
        
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
    photon_energy = h * nu_0
    
    # Get source flux and visibility
    flux = source.specific_flux(nu_0)
    
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
    nu_0: float,
    baseline: np.ndarray,
    observation: Observation,
    params = None
) -> np.ndarray:
    """
    Calculate the Fisher matrix for the parameters of a chaotic source model
    based on a set of intensity interferometry measurements.
    
    The Fisher matrix is computed using the Jacobian of the squared visibility
    (V_squared) with respect to the source parameters and the inverse noise
    (variance) of each measurement. All measurements use the same telescope
    and detector configuration but different frequencies and baselines.
    
    Parameters
    ----------
    source : ChaoticSource
        The chaotic source model with parameters to estimate.
    nu_0 : float
        central frequencies in Hz for each measurement.
    baseline : np.ndarray
        baseline vectors in meters [Bx, By, Bz] for each measurement.
    observation : Observation
        Telescope and detector configuration used for all measurements.
            
    Returns
    -------
    fisher_matrix : numpy.ndarray
        The Fisher matrix with shape (n_params, n_params), where n_params is
        the number of parameters in the source model. The order of parameters
        matches the keys from source.get_params().
        
    Raises
    ------
    ValueError
        If nu_0_list and baseline_list have different lengths.
        
    Notes
    -----
    The Fisher matrix provides the Cramér-Rao lower bound on the covariance
    matrix of parameter estimates. The inverse of the Fisher matrix gives
    the minimum achievable covariance matrix.
    
    Example
    -------
    >>> disk = UniformDisk(flux_density=1e-26, radius=1e-8)
    >>> nu_0_list = [5e14, 5e14, 6e14]
    >>> baseline_list = [
    ...     np.array([100, 0, 0]),
    ...     np.array([0, 100, 0]),
    ...     np.array([50, 50, 0])
    ... ]
    >>> observation = Observation(integration_time=3600, telescope_area=1.0)
    >>> fisher = fisher_matrix(disk, nu_0_list, baseline_list, observation)
    """

    if params is None:
        params = source.get_params()

    # Get parameter names in consistent order
    params_keys = list(params.keys())
    
    # # Flatten all parameter values to get total number of scalar parameters
    # flattened_params = []
    # param_indices = {}  # Track which indices correspond to which parameters
    
    # start_idx = 0
    # for param_name in params_keys:
    #     value = params[param_name]
    #     if np.isscalar(value):
    #         # Scalar parameter
    #         flattened_params.append(value)
    #         param_indices[param_name] = slice(start_idx, start_idx + 1)
    #         start_idx += 1
    #     else:
    #         # Array parameter - flatten it
    #         value_array = np.asarray(value)
    #         flat_value = value_array.flatten()
    #         flattened_params.extend(flat_value)
    #         param_indices[param_name] = slice(start_idx, start_idx + len(flat_value))
    #         start_idx += len(flat_value)
    

    # Compute Jacobian of V_squared with respect to parameters
    jacobian_dict = source.V_squared_jacobian(nu_0=nu_0, baseline=baseline, params=params)
    
    # Flatten the jacobian values in the same order as parameters
    jacobian = []
    for param_name in params_keys:
        grad_value = jacobian_dict[param_name]
        if np.isscalar(grad_value):
            jacobian.append(grad_value)
        else:
            # Flatten array gradients
            grad_array = np.asarray(grad_value)
            jacobian.extend(grad_array.flatten())
    
    jacobian = np.array(jacobian)
    
    # Calculate inverse noise (standard deviation) for this measurement
    inverse_sigma = inverse_noise(source, nu_0, observation)
    
    # Compute contribution to Fisher matrix
    inv_variance = inverse_sigma**2
    fisher = np.outer(jacobian, jacobian) * inv_variance
    
    return fisher.real