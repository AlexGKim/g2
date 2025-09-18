"""
Intensity Interferometry Source Models
======================================

This module provides a comprehensive framework for intensity interferometry calculations
based on the theoretical foundations from "Probing H0 and resolving AGN disks with 
ultrafast photon counters" (arXiv:2403.15903v1).

The module implements equations 1-14 from the paper for abstract intensity I_nu(nu, n̂)
using a modular, object-oriented design that supports various astronomical source models
and observational configurations.

Key Features
------------
- Abstract base class for defining custom intensity sources
- FFT-based visibility calculations for efficient computation
- Built-in implementations for common source types (point sources, uniform disks)
- Support for both analytical and numerical visibility calculations
- Proper normalization and units throughout

Mathematical Framework
----------------------
The core calculation is the simplified fringe visibility:

    V_simple(ν₀,B) = ∫ d²n̂ I(ν₀,n̂) exp(2πiB_⊥⋅n̂/λ₀) / ∫ d²n̂ I(ν₀,n̂)

This represents the normalized spatial Fourier transform of the intensity distribution,
which is fundamental to intensity interferometry measurements.

Usage Example
-------------
>>> # Create a uniform disk source
>>> disk = UniformDisk(flux_density=1e-26, radius=1e-8)  # 1e-8 rad = ~2 mas
>>> 
>>> # Calculate visibility for a 100m baseline
>>> baseline = np.array([100.0, 0.0, 0.0])  # 100m E-W baseline
>>> nu_0 = 5e14  # 600 nm
>>> visibility = disk.visibility(nu_0, baseline)
>>> print(f"Visibility magnitude: {abs(visibility):.3f}")

Classes
-------
AbstractSource : Abstract base class for all intensity sources
ChaoticSource : Abstract base class for chaotic (thermal) light sources
PointSource : Point source implementation
UniformDisk : Uniform circular disk implementation

Notes
-----
- All angular quantities are in radians
- Frequencies are in Hz
- Flux densities are in W m⁻² Hz⁻¹
- Intensities are in W m⁻² Hz⁻¹ sr⁻¹
- Baseline vectors are in meters

The FFT-based visibility calculation uses proper grid setup and normalization
to ensure accurate results for extended sources.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Callable, Union
import scipy.special as sp
from scipy.integrate import quad, dblquad
from scipy.fft import fft2, ifft2, fftfreq, fftshift, ifftshift
from scipy.interpolate import RegularGridInterpolator
from dataclasses import dataclass
try:
    import jax
    import jax.numpy as jnp
    from jax.typing import ArrayLike

except ImportError:
    # Mock jax for documentation generation
    class MockJax:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    jax = MockJax()
    jnp = MockJax()

from typing import Dict, Any


class AbstractSource(ABC):
    """
    Abstract base class for intensity sources described by I_nu(nu, n̂).
    
    Spatial coherence is described by visibility V, which depends on the specific
    intensity I_nu(nu, n̂) as a function of frequency nu and sky direction n̂. A
    concrete method is implemented for the visibility calculations that works with
    any intensity profile.

    Temporal coherence is described by the second order coherence g²(nu, delta_t)
    which must be specified by subclasses. g²(Δt) - 1 is directly related to the
    HBT observable, the correlation in light intensity as a function of time lag.
    Its integral over all time lags gives the signal given in Eq. 3 in Dalal et al.
    The g2_minus_one method is implemented as an abstract method to be overridden 
    by subclasses. For example, for chaotic (thermal) sources, a concrete 
    implementation is provided in the ChaoticSource subclass where
    g²(Δt) - 1 equals the square of the magnitude of g¹(Δt).
    
    Methods to Implement
    --------------------
    intensity : Calculate specific intensity at given frequency and direction
    total_flux : Calculate total integrated flux at given frequency
    g2_minus_one : Calculate second-order temporal coherence function minus one
    get_params : Return source model parameters
    
    Provided Methods
    ----------------
    V : Calculate fringe visibility using FFT (works for any intensity distribution)
    V_squared : Calculate squared visibility magnitude \\|V\\|²
    V_squared_jacobian : Calculate Jacobian of \\|V\\|² with respect to parameters
    V_fft : FFT-based visibility calculation implementation
    """
    
    @abstractmethod
    def intensity(self, nu: Union[float, np.ndarray], n_hat: np.ndarray) -> Union[float, np.ndarray]:
        """
        Calculate specific intensity I_nu(nu, n̂).
        
        This is the fundamental quantity in intensity interferometry - the specific
        intensity as a function of frequency and sky direction. The implementation
        should return the intensity in proper units and handle both scalar and
        array inputs appropriately.
        
        Parameters
        ----------
        nu : float or array_like
            Frequency in Hz. Can be a single value or array of frequencies.
        n_hat : array_like, shape (2,) or (N, 2)
            Direction vector(s) on sky in radians. For a single direction,
            should be [theta_x, theta_y]. For multiple directions, should be
            an array where each row is a direction vector.
            
        Returns
        -------
        intensity : float or array_like
            Specific intensity in W m⁻² Hz⁻¹ sr⁻¹. Shape matches input:
            - If nu is scalar and n_hat is (2,): returns scalar
            - If nu is array and n_hat is (2,): returns array matching nu
            - If nu is scalar and n_hat is (N,2): returns array of length N
            
        Notes
        -----
        The intensity should be properly normalized such that integrating over
        all solid angles gives the total flux density at frequency nu.
        """
        pass

    def V(self, nu_0: float, baseline: "ArrayLike", params: dict = None) -> complex:
        """
        Calculate the fringe visibility that corresponds to the intensity distribution.

        This method provides a general interface for visibility calculation.
        The default implementation uses FFT-based computation which works for
        any intensity distribution.
        
        Parameters
        ----------
        nu_0 : float
            Central frequency in Hz.
        baseline : array_like, shape (3,)
            Baseline vector in meters [Bx, By, Bz].
        params : dict, optional
            Source parameters. If None, uses current source parameters.
            
        Returns
        -------
        V : complex
            Complex fringe visibility.
            
        Notes
        -----
        Subclasses may override this method to provide analytical solutions
        for specific source geometries when available.
        """
        return self.V_fft(nu_0, baseline)

    @abstractmethod
    def total_flux(self, nu: float) -> float:
        """
        Calculate total flux F_nu = ∫ I_nu d²n̂.
        
        This method should return the total flux density by integrating the
        specific intensity over all solid angles. This is used for normalization
        in visibility calculations and for checking conservation of flux.
        
        Parameters
        ----------
        nu : float
            Frequency in Hz.
            
        Returns
        -------
        flux : float
            Total flux density in W m⁻² Hz⁻¹.
            
        Notes
        -----
        For most implementations, this can be calculated analytically from the
        source parameters. The result should be consistent with numerical
        integration of the intensity() method over the sky.

        Think about implementing a general implementation here.
        """
        pass
    
    @abstractmethod
    def g2_minus_one(self, delta_t: float, nu_0: float, delta_nu: float) -> float:
        """
        Calculate second-order temporal coherence function minus one: g²(Δt) - 1.
        
        This method computes g²(Δt) - 1, which directly represents the excess
        correlation above the uncorrelated baseline. This quantity is fundamental
        to intensity interferometry and equals \\|V(B)\\|² for chaotic sources.

        Usually we are interested in the coherence function through an observational
        setup with a finite bandwidth. The bandwidth is described by a tophat function
        centered at frequency nu_0 with width delta_nu.
        
        Parameters
        ----------
        delta_t : float
            Time lag in seconds between the two intensity measurements.
        nu_0 : float
            Central frequency in Hz.
        delta_nu : float
            Frequency window (bandwidth) in Hz over which the coherence
            function is calculated. This represents the spectral resolution
            or filter bandwidth of the measurement.
            
        Returns
        -------
        g2_minus_one : float
            Second-order temporal coherence function minus one at time lag Δt.
            For thermal (chaotic) light: g²(0) - 1 = 1, g²(∞) - 1 = 0
            For coherent light: g²(Δt) - 1 = 0 for all Δt
            
        Notes
        -----
        The second-order coherence function is defined as:
        
        .. math::
            g²(Δt) = ⟨I(t)I(t+Δt)⟩ / I₀²
        
        where I₀ = ⟨I(t)⟩ is the mean intensity.
        
        For intensity interferometry, the key relation is:

        .. math::
            g²(Δt) - 1 = \\|g¹(Δt)\\|²
        
        where g¹(Δt) relates to the spatial visibility function V(B) through the
        van Cittert-Zernike theorem. This connects the temporal correlations 
        measured by g² to the spatial structure of the source through the visibility.
        
        The frequency window δν determines the coherence time τ_c ≈ 1/δν,
        which affects the temporal correlations. The g²-1 function typically
        decays from its peak value at Δt = 0 to zero at large time lags.
        """
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """
        Return the parameters that define the source model.
        
        These parameters are particularly important for those that may be
        varied in fitting or optimization procedures.
        
        Returns
        -------
        params : dict
            Dictionary containing source parameters with parameter names
            as keys and their current values as values.
            
        Notes
        -----
        The returned parameters should be those that can be optimized
        or fitted to observational data. Internal derived quantities
        should not typically be included.
        """
        pass

    def V_squared(self, nu_0: float, baseline: np.ndarray, params: dict = None) -> float:
        """
        Calculate squared visibility magnitude \\|V\\|².
        
        The spatial dependence of the intensity interferometry signal (function 
        of baseline) is directly proportional to \\|V\\|².
        
        Parameters
        ----------
        nu_0 : float
            Central frequency in Hz.
        baseline : array_like, shape (3,)
            Baseline vector in meters [Bx, By, Bz].
        params : dict, optional
            Source parameters. If None, uses current source parameters.
            
        Returns
        -------
        V_squared : float
            Squared visibility magnitude, always real and non-negative.
        """
        ans = self.V(nu_0, baseline, params) 
        return jnp.abs(ans)**2
    
    def V_squared_jacobian(self, nu_0: float, baseline: np.ndarray, params: dict = None):
        """
        Calculate the Jacobian of \\|V\\|² with respect to the source parameters.
        
        This is useful for gradient-based optimization and uncertainty
        estimation in parameter fitting.
        
        Parameters
        ----------
        nu_0 : float
            Central frequency in Hz.
        baseline : array_like, shape (3,)
            Baseline vector in meters [Bx, By, Bz].
        params : dict, optional
            Source parameters. If None, uses current source parameters.
            
        Returns
        -------
        jacobian : dict
            Dictionary with same keys as params, containing the partial
            derivatives of \\|V\\|² with respect to each parameter.
        """
        def pure_V_squared(params):
            return self.V_squared(nu_0, baseline, params)
        
        if params is None:
            params = self.get_params()

        return jax.jacrev(pure_V_squared)(params)

    def V_fft(self, nu_0: float, baseline: np.ndarray,
              grid_size: int = 512, sky_extent: float = 2e-7) -> complex:
        """
        Calculate the spatial visibility function V using FFT.
        
        Computes V(ν₀,B) = ∫ d²n̂ I(ν₀,n̂) exp(2πiB_⊥⋅n̂/λ₀) / ∫ d²n̂ I(ν₀,n̂)
        
        This is Equation 8 in Dalal et al. 2024 (arXiv:2403.15903v1).
        The implementation uses FFT to accurately approximate the continuous
        Fourier transform.
        
        Parameters
        ----------
        nu_0 : float
            Central frequency in Hz. Determines the wavelength λ₀ = c/ν₀.
        baseline : array_like, shape (3,)
            Baseline vector in meters [Bx, By, Bz]. Only the perpendicular
            components (Bx, By) are used in the calculation.
        grid_size : int, optional
            Size of the FFT grid. Larger values give better accuracy but
            slower computation. Default is 512.
        sky_extent : float, optional
            Angular extent of the sky grid in radians. Should be large enough
            to contain the source but small enough for good frequency resolution.
            Default is 2e-7 rad (~0.04 arcsec).
            
        Returns
        -------
        V : complex
            Normalized fringe visibility. The magnitude gives the visibility
            amplitude, and the phase gives the visibility phase.
            
        Notes
        -----
        The FFT method works well for extended sources but may have numerical
        artifacts for very compact sources. For simple geometries like point
        sources or uniform disks, analytical methods may be more accurate.
        
        The grid parameters (grid_size, sky_extent) should be chosen based on
        the source size and desired accuracy:
        - sky_extent should be several times the source angular size
        - grid_size should be large enough for good sampling of the source
        
        Examples
        --------
        >>> source = UniformDisk(flux_density=1e-26, radius=1e-8)
        >>> baseline = np.array([100.0, 0.0, 0.0])  # 100m E-W
        >>> nu_0 = 5e14  # 600 nm
        >>> vis = source.V(nu_0, baseline)
        >>> print(f"Visibility: {vis:.3f}")
        """
        # Physical constants
        c = 2.99792458e8  # Speed of light in m/s
        wavelength = c / nu_0
        
        # Extract perpendicular baseline components (ignore Bz)
        baseline_perp = baseline[:2]
        
        # Set up coordinate grids
        pixel_scale = sky_extent / grid_size
        coords_1d = np.linspace(-sky_extent/2, sky_extent/2, grid_size, endpoint=False)
        sky_x, sky_y = np.meshgrid(coords_1d, coords_1d)
        
        # Calculate intensity on grid
        intensity_grid = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                n_hat = np.array([sky_x[i, j], sky_y[i, j]])
                intensity_grid[i, j] = self.intensity(nu_0, n_hat)

        # Compute 2D FFT with proper shifting
        intensity_fft = fft2(intensity_grid)
        intensity_fft = fftshift(intensity_fft)
        
        # Normalize by pixel area and total flux
        pixel_area = pixel_scale**2
        total_flux = np.sum(intensity_grid) * pixel_area
        
        if total_flux > 0:
            # Proper normalization for discrete FFT to approximate continuous transform
            intensity_fft *= pixel_area / total_flux
        else:
            return 0.0 + 0.0j
        
        # Convert baseline to spatial frequency coordinates
        u_freq = baseline_perp[0] / wavelength if len(baseline_perp) > 0 else 0.0
        v_freq = baseline_perp[1] / wavelength if len(baseline_perp) > 1 else 0.0
        
        # Convert to grid indices
        freq_resolution = 1.0 / sky_extent
        u_idx = u_freq / freq_resolution + grid_size // 2
        v_idx = v_freq / freq_resolution + grid_size // 2
        
        # Interpolate FFT result at the desired spatial frequency
        return self._bilinear_interpolate(intensity_fft, u_idx, v_idx)
    
    def _bilinear_interpolate(self, grid: np.ndarray, x: float, y: float) -> complex:
        """
        Perform bilinear interpolation on a 2D complex grid.
        
        This is a helper method for the visibility calculation that interpolates
        the FFT result at non-integer grid positions corresponding to the
        baseline spatial frequency.
        
        Parameters
        ----------
        grid : ndarray, shape (M, N)
            2D complex array to interpolate from.
        x, y : float
            Interpolation coordinates. Can be fractional values.
            
        Returns
        -------
        value : complex
            Interpolated complex value.
            
        Notes
        -----
        Currently uses nearest-neighbor interpolation for simplicity and
        numerical stability. Could be upgraded to true bilinear interpolation
        if higher accuracy is needed.
        """
        grid_size = grid.shape[0]
        
        # Handle boundary conditions - return zero outside grid
        if (x < 0 or x >= grid_size - 1 or
            y < 0 or y >= grid_size - 1):
            return 0.0 + 0.0j
        
        # Use nearest neighbor interpolation for numerical stability
        x_int = int(round(x))
        y_int = int(round(y))
        
        # Ensure indices are within bounds
        x_int = max(0, min(x_int, grid_size - 1))
        y_int = max(0, min(y_int, grid_size - 1))
        
        return grid[y_int, x_int]


class ChaoticSource(AbstractSource):
    """
    Abstract base class for chaotic (thermal) light sources.
    
    This class extends AbstractSource by providing a concrete implementation
    of the g² function for chaotic light sources. For thermal sources, the
    second-order coherence function is related to the Fourier transform of
    the spectral power distribution within the measurement bandwidth.
    
    The g² function is implemented as:

        g²(ν₀, Δν) = FT[tophat(ν₀, Δν)]/Δν
    
    where FT[tophat] is the
    Fourier transform of the rectangular (tophat) function defined by the
    central frequency and bandwidth.
    
    Methods to Implement
    --------------------
    intensity : Calculate specific intensity at given frequency and direction
    total_flux : Calculate total integrated flux at given frequency
    
    Provided Methods
    ----------------
    g1 : Calculate first-order temporal coherence function (implemented)
    g2_minus_one : Calculate second-order temporal coherence function minus one (implemented)
    V : Calculate fringe visibility using FFT (inherited)
    """
    
    def g1(self, delta_t: float, nu_0: float, delta_nu: float) -> complex:
        """
        Calculate first-order temporal coherence function for chaotic light.
        
        For chaotic (thermal) sources, the first-order coherence function g¹(Δt)
        is the Fourier Transform of the spectrum by the Wiener-Khinchin theorem.
        Practically we are interested in the coherence function through an observational
        pass band.  In cases where the spectrum can be approximated as constant within
        a rectangular spectral window. the Fourier transform of the tophat function which
        gives the sinc function.

        g1(Δt) = sinc(π × Δν × Δt)

        g¹(Δt) is fundamental to the second-order correlations measured in
        intensity interferometry through the relation g²(Δt) - 1 = \\|g¹(Δt)\\|²
        for chaotic light.
        
        Parameters
        ----------
        delta_t : float
            Time lag in seconds.
        nu_0 : float
            Central frequency in Hz.
        delta_nu : float
            Frequency window (bandwidth) in Hz.
            
        Returns
        -------
        g1 : complex
            First-order temporal coherence function at time lag Δt.
            For chaotic light, this exhibits the characteristic sinc
            behavior with correlations decaying as a function of time lag.
            
        Notes
        -----
        The implementation uses
        
        .. math::

            g¹(Δt) = sinc(π × Δν × Δt)
        
        where the sinc function arises from the Fourier transform of the
        rectangular spectral window. This gives:
        
        - At Δt = 0: g¹(0) = 1 (maximum coherence)
        - At large Δt: g¹(∞) → 0 (no coherence)
        
        The first-order coherence function is fundamental to the second-order
        correlations measured in intensity interferometry through the relation:
        
            g²(Δt) - 1 = \\|g¹(Δt)\\|²
        """
        # Calculate sinc function: sinc(π × Δν × Δt)
        if delta_t == 0:
            sinc_value = 1.0
        else:
            x = np.pi * delta_nu * delta_t
            sinc_value = np.sin(x) / x
        
        # For chaotic light: g¹(Δt) = sinc(π × Δν × Δt)
        # Return as complex number (phase is zero for this simple case)
        return sinc_value + 0.0j
    
    def g2_minus_one(self, delta_t: float, nu_0: float, delta_nu: float) -> float:
        """
        Calculate second-order temporal coherence function minus one for chaotic light.
        
        For chaotic (thermal) sources, g²(Δt) - 1 is implemented as the square
        magnitude of the first-order coherence function g¹(Δt). This relationship
        is fundamental to chaotic light statistics and intensity interferometry.
        
        Parameters
        ----------
        delta_t : float
            Time lag in seconds.
        nu_0 : float
            Central frequency in Hz.
        delta_nu : float
            Frequency window (bandwidth) in Hz.
            
        Returns
        -------
        g2_minus_one : float
            Second-order temporal coherence function minus one at time lag Δt.
            For chaotic light, this exhibits the characteristic sinc²
            behavior with correlations decaying as a function of time lag.
            
        Notes
        -----
        The implementation uses:

        .. math::
            g²(Δt) - 1 = \\|g¹(Δt)\\|²
        
        where g¹(Δt) is the first-order temporal coherence function.
        This gives:
        - At Δt = 0: g²(0) - 1 = \\|g¹(0)\\|² = 1 (maximum excess correlation)
        - At large Δt: g²(∞) - 1 → \\|g¹(∞)\\|² = 0 (no excess correlation)
        
        The key relation for intensity interferometry is:

        .. math::
            g²(Δt) - 1 = \\|V(B)\\|² = \\|g¹(Δt)\\|²
        
        where V(B) is the spatial visibility function (Equation 8). This connects
        the temporal correlations measured by g²-1 to the spatial structure through
        the first-order coherence function.
        """
        # Calculate g²(Δt) - 1 = \\|g¹(Δt)\\|² - 1
        # For chaotic light: g²(Δt) = 1 + \\|g¹(Δt)\\|², so g²(Δt) - 1 = \\|g¹(Δt)\\|²
        g1_value = self.g1(delta_t, nu_0, delta_nu)
        return abs(g1_value)**2


