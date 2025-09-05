"""
Intensity Interferometry Core Module

Implementation of equations 1-14 from "Probing H0 and resolving AGN disks with ultrafast photon counters"
(arXiv:2403.15903v1) for abstract intensity I_nu(nu,\hat{n}) using a modular design.

This module provides a general-purpose framework for intensity interferometry calculations
with support for various source models and observational configurations.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Callable, Union
import scipy.special as sp
from scipy.integrate import quad, dblquad
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


class IntensityInterferometry:
    """
    Core intensity interferometry calculations implementing equations 1-14
    from arXiv:2403.15903v1
    """
    
    def __init__(self, source: AbstractIntensitySource):
        """
        Initialize with an intensity source
        
        Parameters:
        -----------
        source : AbstractIntensitySource
            Source model providing I_nu(nu, n_hat)
        """
        self.source = source
        self.c = 2.99792458e8  # Speed of light [m/s]
        self.h = 6.62607015e-34  # Planck constant [J⋅s]
    
    def visibility(self, nu: float, baseline: np.ndarray, delta_t: float = 0.0) -> complex:
        """
        Calculate complex visibility V(ν,B,Δt) - Equation (1)
        
        V(ν,B,Δt) = ∫ I_ν(ν,n̂) exp(i[k⋅B - ωΔt]) d²n̂
        
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
        omega = 2 * np.pi * nu
        k_mag = omega / self.c
        
        def integrand(theta, phi):
            # Convert spherical to Cartesian direction
            n_hat = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
            
            # Calculate k⋅B - ωΔt
            k_dot_B = k_mag * np.dot(n_hat, baseline)
            phase = k_dot_B - omega * delta_t
            
            # Get intensity and apply phase factor
            intensity = self.source.intensity(nu, n_hat[:2])  # Use only sky coordinates
            return intensity * np.exp(1j * phase) * np.sin(theta)
        
        # Integrate over solid angle
        real_part, _ = dblquad(lambda phi, theta: np.real(integrand(theta, phi)), 
                              0, np.pi, 0, 2*np.pi)
        imag_part, _ = dblquad(lambda phi, theta: np.imag(integrand(theta, phi)), 
                              0, np.pi, 0, 2*np.pi)
        
        return complex(real_part, imag_part)
    
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
        if sigma_t * delta_omega >> 1:
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
        if params.sigma_t * 2 * np.pi * params.delta_nu >> 1:
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
    """
    
    def __init__(self, source: AbstractIntensitySource):
        self.source = source
        self.c = 2.99792458e8
    
    def spatial_visibility(self, nu_0: float, baseline_perp: np.ndarray) -> complex:
        """
        Calculate spatial visibility V̄(ν₀,B) - Equation (8)
        
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
        lambda_0 = self.c / nu_0
        
        def integrand(n_hat):
            # n_hat is 2D sky coordinate
            phase = 2 * np.pi * np.dot(baseline_perp, n_hat) / lambda_0
            intensity = self.source.intensity(nu_0, n_hat)
            return intensity * np.exp(1j * phase)
        
        # Integrate over sky coordinates
        # This is a simplified 2D integration - in practice, need proper solid angle
        def real_integrand(nx, ny):
            n_hat = np.array([nx, ny])
            return np.real(integrand(n_hat))
        
        def imag_integrand(nx, ny):
            n_hat = np.array([nx, ny])
            return np.imag(integrand(n_hat))
        
        # Integration limits depend on source extent
        limit = 0.1  # Adjust based on source size
        real_part, _ = dblquad(real_integrand, -limit, limit, -limit, limit)
        imag_part, _ = dblquad(imag_integrand, -limit, limit, -limit, limit)
        
        numerator = complex(real_part, imag_part)
        denominator = self.source.total_flux(nu_0)
        
        return numerator / denominator if denominator != 0 else 0.0
    
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