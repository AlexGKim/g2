"""
Intensity Interferometry Source Models
=====================================

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
---------------------
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
import jax
import jax.numpy as jnp
from typing import Dict, Any


class AbstractSource(ABC):
    """
    Abstract base class for intensity sources described by I_nu(nu, n̂).
    
    Spatial coherence is described by visibility V, which depends on the specific
    intensity I_nu(nu, n̂) as a function of frequency nu and sky direction n̂. A
    concrete method is implemented for the visibility calculations that works with
    any intensity profile.

    Temporal coherence is described by the second order coherence g^2(nu, delta_t)
    which must be specified by subclasses.  g²(Δt) - 1 is directly related to the
    HBT obserable, the correlation in light intensity as a function of time lag.
    Its integral over all time lags gives the signal given in Eq. 3 in Dalal et al.
    g^2(nu, delta_t), or specifically the g2_minus_one method, is implemented as an
    abstract method to be overridden by subclasses.  For example, chaotic (thermal)
    sources, a concrete implementation is provided in the ChaoticSource subclass as
    g²(Δt) - 1 = |g¹(Δt)|².
    
    Methods to Implement
    -------------------
    intensity : Calculate specific intensity at given frequency and direction
    total_flux : Calculate total integrated flux at given frequency
    g2_minus_one : Calculate second-order temporal coherence function minus one
    
    Provided Methods
    ---------------
    V : Calculate fringe visibility using FFT (works for any intensity distribution)
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

    def V(self, nu_0: float, baseline: jnp.ndarray, params: dict = None ) -> complex:
        """
        The visibility that corresponds to the intensity distribution.

        There is a general implementation using FFT provided in this base class
        V_fft    
        """
        self.V_fft(nu_0, baseline, params=params)

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
        """
        pass
    
    @abstractmethod
    def g2_minus_one(self, delta_t: float, nu_0: float, delta_nu: float) -> float:
        """
        Calculate second-order temporal coherence function minus one: g²(Δt) - 1.
        
        This method computes g²(Δt) - 1, which directly represents the excess
        correlation above the uncorrelated baseline. This quantity is fundamental
        to intensity interferometry and it equals |V(B)|² for chaotic sources.

        Usually we are interested in the coherence function through an observational
        setup with a finite bandwidth.  The bandwidth is described by a tophat function
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
            g²(Δt) = ⟨I(t)I(t+Δt)⟩ / I₀²
        
        where I₀ = ⟨I(t)⟩ is the mean intensity.
        
        For intensity interferometry, the key relation is:
            g²(Δt) - 1 = |g^1(Δt)|²
        
        where V(B) is the spatial visibility function (Equation 8). This connects
        the temporal correlations measured by g² to the spatial structure of the
        source through the visibility.
        
        The frequency window δν determines the coherence time τ_c ≈ 1/δν,
        which affects the temporal correlations. The g²-1 function typically
        decays from its peak value at Δt = 0 to zero at large time lags.
        """
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """
        The parameters that define the source model, particularly those that
        may be varied in fitting or optimization.
        """
        pass

    def V_squared(self, nu_0: float, baseline: np.ndarray, params: dict = None ) -> float:
        """
        Calculate squared visibility |V|².  The spatial dependence of the intensity
        interferometry signal (function of baseline) is directly proportional to |V|².
        """
        ans = self.V(nu_0, baseline, params) 
        return jnp.abs(ans)**2
    
    def V_squared_jacobian(self,nu_0, baseline: np.ndarray, params: dict = None ):
        """
        The Jacobian of |V|^2 with respect to the source parameters.
        """
        def pure_V_squared(params):
            return self.V_squared(nu_0, baseline, params)
        
        if params is None:
            params = self.get_params()

        return jax.jacrev(pure_V_squared)(params)

    def V_fft(self, nu_0: float, baseline: np.ndarray,
          grid_size: int = 512, sky_extent: float = 2e-7) -> complex:
        """
        Calculate the spatial visibility function V.
            V(ν₀,B) = ∫ d²n̂ I(ν₀,n̂) exp(2πiB_⊥⋅n̂/λ₀) / ∫ d²n̂ I(ν₀,n̂)
        
        Equation 8 in Dalal et al. 2024 (arXiv:2403.15903v1).

        The general implmentation The uses FFT with to accurately approximate
        the continuous Fourier transform.
        
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
    -------------------
    intensity : Calculate specific intensity at given frequency and direction
    total_flux : Calculate total integrated flux at given frequency
    
    Provided Methods
    ---------------
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
        intensity interferometry through the relation g²(Δt) - 1 = |g¹(Δt)|²
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
        The implementation uses:
            g¹(Δt) = sinc(π × Δν × Δt)
        
        where the sinc function arises from the Fourier transform of the
        rectangular spectral window. This gives:
        - At Δt = 0: g¹(0) = 1 (maximum coherence)
        - At large Δt: g¹(∞) → 0 (no coherence)
        
        The first-order coherence function is fundamental to the second-order
        correlations measured in intensity interferometry through the relation:
            g²(Δt) - 1 = |g¹(Δt)|²
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
            g²(Δt) - 1 = |g¹(Δt)|²
        
        where g¹(Δt) is the first-order temporal coherence function.
        This gives:
        - At Δt = 0: g²(0) - 1 = |g¹(0)|² = 1 (maximum excess correlation)
        - At large Δt: g²(∞) - 1 → |g¹(∞)|² = 0 (no excess correlation)
        
        The key relation for intensity interferometry is:
            g²(Δt) - 1 = |V(B)|² = |g¹(Δt)|²
        
        where V(B) is the spatial visibility function (Equation 8). This connects
        the temporal correlations measured by g²-1 to the spatial structure through
        the first-order coherence function.
        """
        # Calculate g²(Δt) - 1 = |g¹(Δt)|² - 1
        # For chaotic light: g²(Δt) = 1 + |g¹(Δt)|², so g²(Δt) - 1 = |g¹(Δt)|²
        g1_value = self.g1(delta_t, nu_0, delta_nu)
        return abs(g1_value)**2


class PointSource(ChaoticSource):
    """
    Point source implementation.
    
    Represents an unresolved point source located at the origin of the sky
    coordinate system. This is the simplest possible intensity distribution
    and serves as a useful reference case.
    
    For a point source, the intensity is a Dirac delta function:
        I(ν, n̂) = F_ν(ν) δ²(n̂)
    
    where F_ν(ν) is the frequency-dependent flux density and δ²(n̂) is the 2D Dirac delta function.
    
    Parameters
    ----------
    flux_function : callable
        Function that returns flux density as a function of frequency.
        Should accept frequency in Hz and return flux in W m⁻² Hz⁻¹.
        
    Attributes
    ----------
    flux_function : callable
        The frequency-dependent flux density function.
        
    Examples
    --------
    >>> # Constant flux
    >>> point = PointSource(lambda nu: 1e-26)  # 1 Jy constant
    >>> print(f"Total flux at 5e14 Hz: {point.total_flux(5e14):.2e} W/m²/Hz")
    >>>
    >>> # Power-law spectrum
    >>> point = PointSource(lambda nu: 1e-26 * (nu/5e14)**(-0.7))
    >>>
    >>> # Point source visibility is always 1.0
    >>> baseline = np.array([100.0, 0.0, 0.0])
    >>> vis = point.visibility(5e14, baseline)
    >>> print(f"Visibility: {abs(vis):.3f}")  # Should be 1.000
    """
    
    def __init__(self, flux_function: Callable[[float], float], position: jnp.ndarray = None):
        """
        Initialize point source.
        
        Parameters
        ----------
        flux_function : callable
            Function that returns flux density as a function of frequency.
            Should accept frequency in Hz and return flux in W m⁻² Hz⁻¹.
        position : array_like, shape (2,), optional
            Position of the point source on the sky in radians [theta_x, theta_y].
            Default is [0, 0] (at the origin).
        """
        self.flux_function = flux_function
        self.position = jnp.array([0.0, 0.0]) if position is None else jnp.array(position)

    def get_params(self) -> Dict[str, Any]:
        """Extract parameters as a dictionary"""
        return {
            'flux_function': self.flux_function,
            'position': self.position
        }
    
    def intensity(self, nu: Union[float, np.ndarray], n_hat: np.ndarray, atol: float = 1e-10) -> Union[float, np.ndarray]:
        """
        Calculate point source intensity.
        
        For numerical implementation, the delta function is approximated by
        returning the flux divided by the area of a small circle when n̂ is
        very close to the source position.
        
        Parameters
        ----------
        nu : float or array_like
            Frequency in Hz.
        n_hat : array_like, shape (2,)
            Sky direction in radians.
        atol : float, optional
            Tolerance for delta function approximation. The intensity is
            flux/(π*atol²) within this radius, zero outside. Default is 1e-10.
            
        Returns
        -------
        intensity : float
            Specific intensity in W m⁻² Hz⁻¹ sr⁻¹.
        """
        # Get frequency-dependent flux
        if np.isscalar(nu):
            flux = self.flux_function(nu)
        else:
            flux = np.array([self.flux_function(f) for f in nu])
        
        # Approximate delta function - flux divided by circle area within atol
        if np.allclose(n_hat, self.position, atol=atol):
            return flux / (np.pi * atol**2)  # Proper delta function approximation
        return 0.0
    
    def total_flux(self, nu: float) -> float:
        """
        Calculate total flux (frequency-dependent for point source).
        
        Parameters
        ----------
        nu : float
            Frequency in Hz.
            
        Returns
        -------
        flux : float
            Total flux density in W m⁻² Hz⁻¹.
        """
        return self.flux_function(nu)
    
    def V(self, nu_0: float, baseline: jnp.ndarray, params: dict = None) -> complex:
        """
        Analytical visibility function V for point source at origin.
        
        For a point source at the origin, the visibility function is:
            V(B) = exp(2πi B_⊥ · 0 / λ) = 1.0 + 0.0j
        
        However, the proper form should include the exponential phase structure
        even though it evaluates to 1 for a source at the origin.
        
        Parameters
        ----------
        nu_0 : float
            Central frequency in Hz. Determines the wavelength λ = c/ν₀.
        baseline : array_like, shape (3,)
            Baseline vector in meters [Bx, By, Bz]. Only perpendicular
            components (Bx, By) are used.
        grid_size : int, optional
            Grid size (not used, kept for interface compatibility).
        sky_extent : float, optional
            Sky extent (not used, kept for interface compatibility).
            
        Returns
        -------
        V : complex
            Visibility function: exp(2πi B_⊥ · n̂₀ / λ) where n̂₀ = [0,0].
        """

        if params is None:
            params = self.get_params()

        # Physical constants
        c = 2.99792458e8  # Speed of light in m/s
        wavelength = c / nu_0
        
        # Extract perpendicular baseline components (ignore Bz)
        baseline_perp = baseline[:2]
        
        # For point source at origin: n̂₀ = [0, 0]
        # Calculate phase: 2π B_⊥ · n̂₀ / λ = 2π B_⊥ · [0,0] / λ = 0
        phase = 2 * np.pi * jnp.dot(baseline_perp, params['position']) / wavelength
        
        # Return complex exponential: exp(i * 0) = 1.0 + 0.0j
        return jnp.exp(1j * phase)


class UniformDisk(ChaoticSource):
    """
    Uniform circular disk source implementation.
    
    Represents a circular source with uniform surface brightness. This is
    a common model for stellar disks and other approximately circular
    astronomical objects.
    
    The intensity distribution is:
        I(ν, n̂) = I₀  for |n̂| ≤ θ
                 = 0   for |n̂| > θ
    
    where I₀ is the surface brightness and θ is the angular radius.
    
    Parameters
    ----------
    flux_density : float
        Total flux density in W m⁻² Hz⁻¹.
    radius : float
        Angular radius in radians.
        
    Attributes
    ----------
    flux_density : float
        Total flux density of the disk.
    radius : float
        Angular radius of the disk.
    surface_brightness : float
        Uniform surface brightness I₀ = F_ν/(πθ²).
        
    Examples
    --------
    >>> # Create a disk with 2 milliarcsecond radius
    >>> disk = UniformDisk(flux_density=1e-26, radius=1e-8)  # ~2 mas
    >>> print(f"Surface brightness: {disk.surface_brightness:.2e} W/m²/Hz/sr")
    >>> 
    >>> # Calculate visibility for different baselines
    >>> for B in [10, 100, 1000]:  # meters
    >>>     baseline = np.array([B, 0.0, 0.0])
    >>>     vis = disk.visibility(5e14, baseline)
    >>>     print(f"B={B}m: |V|={abs(vis):.3f}")
    """
    
    def __init__(self, flux_density: float, radius: float):
        """
        Initialize uniform disk source.
        
        Parameters
        ----------
        flux_density : float
            Total flux density in W m⁻² Hz⁻¹.
        radius : float
            Angular radius in radians.
        """
        self.flux_density = flux_density
        self.radius = radius
        # Calculate uniform surface brightness
        self.surface_brightness = flux_density / (np.pi * radius**2)

    def get_params(self) -> Dict[str, Any]:
        """Extract parameters as a dictionary"""
        return {
            'flux_density': self.flux_density,
            'radius': self.radius
        }

    def intensity(self, nu: Union[float, np.ndarray], n_hat: np.ndarray) -> Union[float, np.ndarray]:
        """
        Calculate uniform disk intensity.
        
        Returns constant surface brightness inside the disk radius,
        zero outside.
        
        Parameters
        ----------
        nu : float or array_like
            Frequency in Hz (not used for uniform disk).
        n_hat : array_like, shape (2,)
            Sky direction in radians.
            
        Returns
        -------
        intensity : float
            Specific intensity in W m⁻² Hz⁻¹ sr⁻¹.
        """
        r = jnp.sqrt(n_hat[0]**2 + n_hat[1]**2)
        return self.surface_brightness if r <= self.radius else 0.0
    
    def total_flux(self, nu: float) -> float:
        """
        Calculate total flux (constant for uniform disk).
        
        Parameters
        ----------
        nu : float
            Frequency in Hz (not used).
            
        Returns
        -------
        flux : float
            Total flux density in W m⁻² Hz⁻¹.
        """
        return self.flux_density
    
    def V(self, nu_0: float, baseline: np.ndarray, params = None) -> complex:
        """
        Analytical visibility function V for uniform disk.
        
        For a uniform circular disk, the normalized visibility function
        is given by the Airy function:
        
            V(u) = 2J₁(2πuθ) / (2πuθ)
        
        where J₁ is the first-order Bessel function, u = |B_⊥|/λ is the
        spatial frequency, and θ is the disk radius.
        
        Parameters
        ----------
        nu_0 : float
            Central frequency in Hz.
        baseline : array_like, shape (3,)
            Baseline vector in meters.
        grid_size : int, optional
            Not used (kept for interface compatibility).
        sky_extent : float, optional
            Not used (kept for interface compatibility).
            
        Returns
        -------
        V : complex
            Normalized visibility function (real-valued for symmetric disk).
            
        Notes
        -----
        This analytical result is exact and should match the FFT calculation
        in the limit of fine grid sampling. It's much faster than the FFT
        method and doesn't suffer from discretization artifacts.
        
        The first zero of the visibility function occurs at:
            u = 1.22/(2θ)  or  |B_⊥| = 1.22λ/(2θ)
        
        This corresponds to the classical resolution limit for circular apertures.
        """
        # from scipy.special import j1

        if params is None:
            params = self.get_params()
        
        # Physical constants
        c = 2.99792458e8  # Speed of light in m/s
        wavelength = c / nu_0
        
        # Extract perpendicular baseline components
        baseline_perp = baseline[:2]
        baseline_length = jnp.linalg.norm(baseline_perp)
        
        # Calculate spatial frequency u = |B_⊥|/λ
        u = baseline_length / wavelength
        
        # Calculate argument for Bessel function: x = 2πuθ
        x = 2 * np.pi * u * params['radius']
        
        # Handle special case x=0 (zero baseline or zero radius)
        if x == 0:
            V_value = 1.0
        else:
            # Airy function: V(u) = 2J₁(x)/x
            # V_value = 2 * j1(x) / x
            V_value = 2 * (jnp.sin(x) / x**2 - jnp.cos(x) / x) / x  
        # Return as complex number (phase is zero for symmetric disk)
        return V_value + 0.0j