"""
Intensity Interferometry Core Module

Implementation of equations 1-14 from "Probing H0 and resolving AGN disks with ultrafast photon counters"
(arXiv:2403.15903v1) for abstract intensity I_nu(nu,\hat{n}) using a modular design.

This module provides a general-purpose framework for intensity interferometry calculations
with support for various source models and observational configurations.

Uses FFT-based methods for efficient integral calculations.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Callable, Union
import scipy.special as sp
from scipy.integrate import quad, dblquad
from scipy.fft import fft2, ifft2, fftfreq, fftshift, ifftshift
from scipy.interpolate import RegularGridInterpolator
from dataclasses import dataclass


@dataclass
class ObservationalParameters:
    """Container for observational parameters"""
    nu_0: float  # Central frequency [Hz]
    delta_nu: float  # Bandwidth [Hz]
    baseline: np.ndarray  # Baseline vector [m]
    delta_t: float  # Time lag [s]
    sigma_t: float  # Timing jitter RMS [s]
    T_obs: float  # Total observing time [s]
    A: float  # Telescope area [m^2]
    n_t: int  # Number of telescopes


class AbstractIntensitySource(ABC):
    """Abstract base class for intensity sources I_nu(nu, n_hat)"""
    
    @abstractmethod
    def intensity(self, nu: Union[float, np.ndarray], n_hat: np.ndarray) -> Union[float, np.ndarray]:
        """
        Calculate specific intensity I_nu(nu, n_hat)
        
        Parameters:
        -----------
        nu : float or array
            Frequency [Hz]
        n_hat : array_like, shape (2,) or (N, 2)
            Direction vector(s) on sky [dimensionless]
            
        Returns:
        --------
        intensity : float or array
            Specific intensity [W m^-2 Hz^-1 sr^-1]
        """
        pass
    
    @abstractmethod
    def total_flux(self, nu: float) -> float:
        """
        Calculate total flux F_nu = ∫ I_nu d²n_hat
        
        Parameters:
        -----------
        nu : float
            Frequency [Hz]
            
        Returns:
        --------
        flux : float
            Total flux [W m^-2 Hz^-1]
        """
        pass
    
    def visibility_fft(self, nu: float, baseline: np.ndarray, grid_size: int = 128,
                      sky_extent: float = 1e-4) -> complex:
        """
        Default visibility calculation using FFT-based integration
        
        V(ν,B) = ∫ I_ν(ν,n̂) exp(2πi B⊥⋅n̂/λ) d²n̂ / ∫ I_ν(ν,n̂) d²n̂
        
        Parameters:
        -----------
        nu : float
            Frequency [Hz]
        baseline : array_like, shape (3,)
            Baseline vector [m]
        grid_size : int, optional
            Size of FFT grid (default: 128)
        sky_extent : float, optional
            Angular extent of sky grid [radians] (default: 1e-4)
            
        Returns:
        --------
        visibility : complex
            Complex visibility
        """
        # Calculate wavelength
        c = 2.99792458e8
        wavelength = c / nu
        
        # Use only perpendicular baseline components
        baseline_perp = baseline[:2]
        
        # Set up coordinate grids
        pixel_scale = sky_extent / grid_size
        coords_1d = np.linspace(-sky_extent/2, sky_extent/2, grid_size)
        sky_x, sky_y = np.meshgrid(coords_1d, coords_1d)
        
        # Calculate intensity on grid
        intensity_grid = np.zeros((grid_size, grid_size))
        n_hat_array = np.stack([sky_x.ravel(), sky_y.ravel()], axis=1)
        
        for i, n_hat in enumerate(n_hat_array):
            intensity_grid.ravel()[i] = self.intensity(nu, n_hat)
        
        # Compute 2D FFT of intensity distribution
        intensity_fft = fft2(intensity_grid)
        intensity_fft = fftshift(intensity_fft)
        
        # Normalize by pixel area and total flux
        pixel_area = pixel_scale**2
        total_flux = np.sum(intensity_grid) * pixel_area
        
        if total_flux > 0:
            intensity_fft *= pixel_area / total_flux
        
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
        Perform bilinear interpolation using SciPy's RegularGridInterpolator
        
        Parameters:
        -----------
        grid : ndarray
            2D complex array
        x, y : float
            Interpolation coordinates (can be fractional)
            
        Returns:
        --------
        value : complex
            Interpolated value
        """
        grid_size = grid.shape[0]
        
        # Handle boundary conditions
        if (x < 0 or x >= grid_size - 1 or
            y < 0 or y >= grid_size - 1):
            return 0.0 + 0.0j
        
        # Create coordinate arrays
        coords = (np.arange(grid_size), np.arange(grid_size))
        
        # Handle complex interpolation by interpolating real and imaginary parts separately
        if np.iscomplexobj(grid):
            interp_real = RegularGridInterpolator(coords, grid.real, method='linear',
                                                bounds_error=False, fill_value=0.0)
            interp_imag = RegularGridInterpolator(coords, grid.imag, method='linear',
                                                bounds_error=False, fill_value=0.0)
            
            real_val = interp_real([y, x])[0]  # Note: y, x order for RegularGridInterpolator
            imag_val = interp_imag([y, x])[0]
            return complex(real_val, imag_val)
        else:
            interp = RegularGridInterpolator(coords, grid, method='linear',
                                           bounds_error=False, fill_value=0.0)
            return complex(interp([y, x])[0], 0.0)


class IntensityInterferometry:
    """
    Core intensity interferometry calculations implementing equations 1-14
    from arXiv:2403.15903v1
    
    Uses FFT-based methods for efficient visibility calculations.
    """
    
    def __init__(self, source: AbstractIntensitySource, grid_size: int = 128,
                 sky_extent: float = 1e-4):
        """
        Initialize with an intensity source
        
        Parameters:
        -----------
        source : AbstractIntensitySource
            Source model providing I_nu(nu, n_hat)
        grid_size : int, optional
            Size of FFT grid for calculations (default: 512)
        sky_extent : float, optional
            Angular extent of sky grid [radians] (default: 1e-4 = ~20 arcsec)
        """
        self.source = source
        self.c = 2.99792458e8  # Speed of light [m/s]
        self.h = 6.62607015e-34  # Planck constant [J⋅s]
        
        # FFT grid parameters (passed to source visibility methods)
        self.grid_size = grid_size
        self.sky_extent = sky_extent
    
    def visibility(self, nu: float, baseline: np.ndarray, delta_t: float = 0.0) -> complex:
        """
        Calculate complex visibility V(ν,B,Δt) using source's visibility method - Equation (1)
        
        V(ν,B,Δt) = ∫ I_ν(ν,n̂) exp(i[k⋅B - ωΔt]) d²n̂
        
        Uses the source's visibility calculation method (analytical if available, FFT otherwise).
        
        Parameters:
        -----------
        nu : float
            Frequency [Hz]
        baseline : array_like, shape (3,)
            Baseline vector [m]
        delta_t : float, optional
            Time lag [s]
            
        Returns:
        --------
        visibility : complex
            Complex visibility
        """
        # Check if source has analytical visibility method
        if hasattr(self.source, 'visibility_analytical'):
            visibility = self.source.visibility_analytical(nu, baseline)
        else:
            # Use default FFT-based visibility calculation
            visibility = self.source.visibility_fft(nu, baseline, self.grid_size, self.sky_extent)
        
        # Apply time delay phase factor exp(-iωΔt)
        if delta_t != 0.0:
            omega = 2 * np.pi * nu
            time_phase = np.exp(-1j * omega * delta_t)
            visibility *= time_phase
        
        return visibility
    
    def normalized_fringe_visibility(self, nu_0: float, delta_nu: float, 
                                   baseline: np.ndarray, delta_t: float = 0.0) -> complex:
        """
        Calculate normalized fringe visibility V̄(ν₀,Δν,B,Δt) - Equation (2)
        
        V̄ = ∫[ν₀-Δν/2 to ν₀+Δν/2] dν V(ν,B,Δt) / ∫[ν₀-Δν/2 to ν₀+Δν/2] dν ∫ d²n̂ I_ν(ν,n̂)
        
        Parameters:
        -----------
        nu_0 : float
            Central frequency [Hz]
        delta_nu : float
            Bandwidth [Hz]
        baseline : array_like, shape (3,)
            Baseline vector [m]
        delta_t : float, optional
            Time lag [s]
            
        Returns:
        --------
        normalized_visibility : complex
            Normalized fringe visibility
        """
        nu_min = nu_0 - delta_nu / 2
        nu_max = nu_0 + delta_nu / 2
        
        # Numerator: ∫ dν V(ν,B,Δt)
        def visibility_integrand(nu):
            return self.visibility(nu, baseline, delta_t)
        
        # For complex integration, integrate real and imaginary parts separately
        real_num, _ = quad(lambda nu: np.real(visibility_integrand(nu)), nu_min, nu_max)
        imag_num, _ = quad(lambda nu: np.imag(visibility_integrand(nu)), nu_min, nu_max)
        numerator = complex(real_num, imag_num)
        
        # Denominator: ∫ dν ∫ d²n̂ I_ν(ν,n̂)
        def flux_integrand(nu):
            return self.source.total_flux(nu)
        
        denominator, _ = quad(flux_integrand, nu_min, nu_max)
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def photon_count_covariance(self, params: ObservationalParameters) -> float:
        """
        Calculate photon count covariance ⟨δNᵢδNⱼ⟩ - Equation (3)
        
        ⟨δNᵢδNⱼ⟩ = (Γ²/2) ∫ dtᵢ dtⱼ |V(ν₀,Δν,B,tᵢ-tⱼ)|²
        
        Parameters:
        -----------
        params : ObservationalParameters
            Observational configuration
            
        Returns:
        --------
        covariance : float
            Photon count covariance
        """
        # Calculate photon detection rate
        F_nu = self.source.total_flux(params.nu_0)
        Gamma = params.A * F_nu * params.delta_nu / (self.h * params.nu_0)
        
        # Calculate normalized visibility
        V = self.normalized_fringe_visibility(params.nu_0, params.delta_nu, 
                                            params.baseline, params.delta_t)
        
        # Time integration factor (assuming observation time T)
        T = params.T_obs
        
        return (Gamma**2 / 2) * T**2 * np.abs(V)**2
    
    def timing_jitter_correlation(self, params: ObservationalParameters) -> float:
        """
        Calculate correlation function with timing jitter C(tᵢ-tⱼ) - Equation (5)
        
        C(tᵢ-tⱼ) = ∫ dτᵢ dτⱼ |V(tᵢ-tⱼ+τⱼ-τᵢ)|² W(τᵢ)W(τⱼ)
        
        Parameters:
        -----------
        params : ObservationalParameters
            Observational configuration
            
        Returns:
        --------
        correlation : float
            Timing jitter correlation
        """
        # For Gaussian timing jitter with RMS σₜ
        sigma_t = params.sigma_t
        
        # Calculate visibility
        V = self.normalized_fringe_visibility(params.nu_0, params.delta_nu, 
                                            params.baseline, params.delta_t)
        
        # Apply timing jitter convolution (Gaussian approximation)
        # For two Gaussian jitters: effective σ = √2 σₜ
        effective_sigma = np.sqrt(2) * sigma_t
        
        # Timing jitter factor for narrow bandwidth
        delta_omega = 2 * np.pi * params.delta_nu
        if sigma_t * delta_omega > 1:
            # Equation (11): f ≈ exp(-Δt²/4σₜ²) / (√4πσₜ Δν)
            jitter_factor = (np.exp(-params.delta_t**2 / (4 * sigma_t**2)) / 
                           (np.sqrt(4 * np.pi) * sigma_t * params.delta_nu))
        else:
            # Use full expression from Equation (10)
            jitter_factor = self._calculate_jitter_factor(params)
        
        return np.abs(V)**2 * jitter_factor
    
    def _calculate_jitter_factor(self, params: ObservationalParameters) -> float:
        """
        Calculate timing jitter factor f from Equation (10)
        
        f = ∫ dτ sinc²((Δω(Δt-τ))/2) exp(-τ²/4σₜ²) / √4πσₜ
        """
        sigma_t = params.sigma_t
        delta_omega = 2 * np.pi * params.delta_nu
        delta_t = params.delta_t
        
        def integrand(tau):
            sinc_arg = delta_omega * (delta_t - tau) / 2
            sinc_val = np.sinc(sinc_arg / np.pi)**2  # numpy sinc is sin(πx)/(πx)
            gaussian = np.exp(-tau**2 / (4 * sigma_t**2))
            return sinc_val * gaussian
        
        integral, _ = quad(integrand, -10*sigma_t, 10*sigma_t)
        return integral / np.sqrt(4 * np.pi * sigma_t**2)
    
    def signal_to_noise_ratio(self, params: ObservationalParameters) -> float:
        """
        Calculate signal-to-noise ratio - Equations (6) and (12)
        
        SNR² = (Γ²T_obs/4) ∫ dΔt C²(ν₀,Δν,B,Δt)
        
        For narrow bandwidth and Gaussian jitter:
        SNR² = |V|⁴/(128π)^(1/2) * (Γ/Δν)² * T_obs/σₜ
        
        Parameters:
        -----------
        params : ObservationalParameters
            Observational configuration
            
        Returns:
        --------
        snr : float
            Signal-to-noise ratio
        """
        # Calculate photon detection rate per unit frequency
        F_nu = self.source.total_flux(params.nu_0)
        dGamma_dnu = params.A * F_nu / (self.h * params.nu_0)
        
        # Calculate visibility
        V = self.normalized_fringe_visibility(params.nu_0, params.delta_nu, 
                                            params.baseline, params.delta_t)
        
        # Use simplified expression for narrow bandwidth (Equation 12)
        if params.sigma_t * 2 * np.pi * params.delta_nu > 1:
            snr_squared = (np.abs(V)**4 / np.sqrt(128 * np.pi) * 
                          (dGamma_dnu / params.delta_nu)**2 * 
                          params.T_obs / params.sigma_t)
        else:
            # Use full expression (Equation 6)
            C = self.timing_jitter_correlation(params)
            Gamma = dGamma_dnu * params.delta_nu
            snr_squared = (Gamma**2 * params.T_obs / 4) * C**2
        
        return np.sqrt(snr_squared)
    
    def visibility_error(self, params: ObservationalParameters) -> float:
        """
        Calculate visibility measurement error σ_{|V|²} - Equation (14)
        
        σ_{|V|²}^(-1) = (dΓ/dν) * √(T_obs/σₜ) * (128π)^(-1/4)
        
        Parameters:
        -----------
        params : ObservationalParameters
            Observational configuration
            
        Returns:
        --------
        sigma_v_squared : float
            Visibility measurement error
        """
        F_nu = self.source.total_flux(params.nu_0)
        dGamma_dnu = params.A * F_nu / (self.h * params.nu_0)
        
        sigma_inv = (dGamma_dnu * 
                    np.sqrt(params.T_obs / params.sigma_t) * 
                    (128 * np.pi)**(-0.25))
        
        return 1.0 / sigma_inv if sigma_inv > 0 else np.inf


class FactorizedVisibility:
    """
    Handle factorized visibility for narrow bandwidth sources - Equations (7-8)
    Uses FFT-based methods for efficient calculation.
    """
    
    def __init__(self, source: AbstractIntensitySource, grid_size: int = 128,
                 sky_extent: float = 1e-4):
        """
        Parameters:
        -----------
        source : AbstractIntensitySource
            Source model
        grid_size : int, optional
            Size of FFT grid (default: 512)
        sky_extent : float, optional
            Angular extent of sky grid [radians] (default: 1e-4)
        """
        self.source = source
        self.c = 2.99792458e8
        self.grid_size = grid_size
        self.sky_extent = sky_extent
        self.pixel_scale = sky_extent / grid_size
        
        # Set up coordinate grids
        coords_1d = np.linspace(-sky_extent/2, sky_extent/2, grid_size)
        self.sky_x, self.sky_y = np.meshgrid(coords_1d, coords_1d)
    
    def _get_intensity_grid(self, nu: float) -> np.ndarray:
        """Get intensity on sky grid"""
        intensity_grid = np.zeros((self.grid_size, self.grid_size))
        n_hat_array = np.stack([self.sky_x.ravel(), self.sky_y.ravel()], axis=1)
        
        for i, n_hat in enumerate(n_hat_array):
            intensity_grid.ravel()[i] = self.source.intensity(nu, n_hat)
        
        return intensity_grid
    
    def spatial_visibility(self, nu_0: float, baseline_perp: np.ndarray) -> complex:
        """
        Calculate spatial visibility V̄(ν₀,B) using FFT - Equation (8)
        
        V̄(ν₀,B) = ∫ d²n̂ I(ν₀,n̂) exp(2πiB_⊥⋅n̂/λ₀) / ∫ d²n̂ I(ν₀,n̂)
        
        Parameters:
        -----------
        nu_0 : float
            Central frequency [Hz]
        baseline_perp : array_like, shape (2,)
            Perpendicular baseline component [m]
            
        Returns:
        --------
        visibility : complex
            Spatial visibility
        """
        # Get intensity distribution
        intensity_grid = self._get_intensity_grid(nu_0)
        
        # Calculate wavelength
        lambda_0 = self.c / nu_0
        
        # Compute FFT
        intensity_fft = fft2(intensity_grid)
        intensity_fft = fftshift(intensity_fft)
        
        # Normalize by total flux
        pixel_area = self.pixel_scale**2
        total_flux = np.sum(intensity_grid) * pixel_area
        
        if total_flux > 0:
            intensity_fft *= pixel_area / total_flux
        
        # Convert baseline to spatial frequency
        u_freq = baseline_perp[0] / lambda_0 if len(baseline_perp) > 0 else 0.0
        v_freq = baseline_perp[1] / lambda_0 if len(baseline_perp) > 1 else 0.0
        
        # Convert to grid indices
        freq_resolution = 1.0 / self.sky_extent
        u_idx = u_freq / freq_resolution + self.grid_size // 2
        v_idx = v_freq / freq_resolution + self.grid_size // 2
        
        # Interpolate at desired frequency
        return self._bilinear_interpolate(intensity_fft, u_idx, v_idx)
    
    def _bilinear_interpolate(self, grid: np.ndarray, x: float, y: float) -> complex:
        """Bilinear interpolation on complex grid"""
        if (x < 0 or x >= self.grid_size - 1 or
            y < 0 or y >= self.grid_size - 1):
            return 0.0 + 0.0j
        
        x0, x1 = int(x), min(int(x) + 1, self.grid_size - 1)
        y0, y1 = int(y), min(int(y) + 1, self.grid_size - 1)
        
        fx = x - int(x)
        fy = y - int(y)
        
        value = (grid[y0, x0] * (1 - fx) * (1 - fy) +
                grid[y0, x1] * fx * (1 - fy) +
                grid[y1, x0] * (1 - fx) * fy +
                grid[y1, x1] * fx * fy)
        
        return value
    
    def temporal_factor(self, delta_nu: float, delta_t: float) -> complex:
        """
        Calculate temporal factor from Equation (7)
        
        sinc(ΔωΔt/2) * exp(-iω₀Δt)
        
        Parameters:
        -----------
        delta_nu : float
            Bandwidth [Hz]
        delta_t : float
            Time lag [s]
            
        Returns:
        --------
        factor : complex
            Temporal factor
        """
        delta_omega = 2 * np.pi * delta_nu
        sinc_factor = np.sinc(delta_omega * delta_t / (2 * np.pi))
        # Note: assuming ω₀Δt term is absorbed in redefinition as mentioned in paper
        return sinc_factor


# Example source implementations
class PointSource(AbstractIntensitySource):
    """Point source at origin"""
    
    def __init__(self, flux_density: float):
        """
        Parameters:
        -----------
        flux_density : float
            Flux density [W m^-2 Hz^-1]
        """
        self.flux_density = flux_density
    
    def intensity(self, nu: Union[float, np.ndarray], n_hat: np.ndarray) -> Union[float, np.ndarray]:
        # Delta function at origin - return large value only at n_hat = [0, 0]
        if np.allclose(n_hat, [0, 0], atol=1e-6):
            return self.flux_density / (4 * np.pi * 1e-12)  # Approximate delta function
        return 0.0
    
    def total_flux(self, nu: float) -> float:
        return self.flux_density
    
    def visibility_analytical(self, nu: float, baseline: np.ndarray) -> complex:
        """
        Analytical visibility for point source
        
        A point source has unit visibility at all baselines
        """
        return 1.0 + 0.0j


class UniformDisk(AbstractIntensitySource):
    """Uniform circular disk source"""
    
    def __init__(self, flux_density: float, radius: float):
        """
        Parameters:
        -----------
        flux_density : float
            Total flux density [W m^-2 Hz^-1]
        radius : float
            Angular radius [radians]
        """
        self.flux_density = flux_density
        self.radius = radius
        self.surface_brightness = flux_density / (np.pi * radius**2)
    
    def intensity(self, nu: Union[float, np.ndarray], n_hat: np.ndarray) -> Union[float, np.ndarray]:
        r = np.sqrt(n_hat[0]**2 + n_hat[1]**2)
        return self.surface_brightness if r <= self.radius else 0.0
    
    def total_flux(self, nu: float) -> float:
        return self.flux_density