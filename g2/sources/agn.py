"""
Simplified AGN Source Models for Intensity Interferometry

Streamlined implementation focusing on computational efficiency
while maintaining the simplified integration method.

All AGN source models now inherit from ChaoticSource to provide
proper temporal coherence functions for intensity interferometry.
"""

import numpy as np
import scipy.special as sp
from scipy.integrate import quad, dblquad
from typing import Union, Tuple, Optional
import sys
import os

from .source import ChaoticSource


class ShakuraSunyaevDisk(ChaoticSource):
    """
    Simplified Shakura-Sunyaev accretion disk model - Equations (21-22)
    
    I(R) = I₀ [e^f(R) - 1]^(-1)
    f(R) = (ν/ν₀(R)) = [(R₀/R)^n (1 - √(R_in/R))]^(-1/4)
    
    Uses simplified integration for efficient visibility calculations.
    Inherits temporal coherence functions from ChaoticSource.
    """
    
    def __init__(self, I_0: float, R_0: float, R_in: float, n: float = 3.0, 
                 inclination: float = 0.0, phi_B: float = 0.0, distance: float = 1.0, 
                 GM_over_c2: float = 1.0):
        """
        Parameters:
        -----------
        I_0 : float
            Normalization intensity [W m^-2 Hz^-1 sr^-1]
        R_0 : float
            Characteristic radius in units of GM/c² [dimensionless]
        R_in : float
            Inner disk radius in units of GM/c² [dimensionless]
        n : float, optional
            Power law index (default: 3.0 for standard SS disk)
        inclination : float, optional
            Disk inclination angle [radians]
        phi_B : float, optional
            Position angle of disk [radians]
        distance : float, optional
            Distance to source [m]
        GM_over_c2 : float, optional
            Gravitational radius GM/c² [m]
        """
        self.I_0 = I_0
        self.R_0 = R_0
        self.R_in = R_in
        self.n = n
        self.inclination = inclination
        self.phi_B = phi_B
        self.distance = distance
        self.GM_over_c2 = GM_over_c2
        
        # Precompute cos(i) for efficiency
        self.cos_i = np.cos(inclination)
        self.sin_i = np.sin(inclination)

        self.cos_phi_B = np.cos(phi_B)
        self.sin_phi_B = np.sin(phi_B)  

    def get_params(self) -> dict:
        """
        The parameters that define the source model, particularly those that
        may be varied in fitting or optimization.
        """
        pass

    def _f_function(self, R: float) -> float:
        """Calculate f(R) from Equation (22)"""
        if R <= self.R_in:
            return np.inf  # No emission inside R_in
        
        ratio = self.R_0 / R
        sqrt_term = np.sqrt(self.R_in / R)
        return (ratio**self.n * (1 - sqrt_term))**(-0.25)
    
    def _disk_intensity(self, R: float) -> float:
        """Calculate disk intensity I(R) from Equation (21)"""
        if R <= self.R_in:
            return 0.0
        
        f_val = self._f_function(R)
    
        if f_val == np.inf:
            return 0.0
        
        try:    
            exp_f = np.exp(f_val)
        except OverflowError:
            return 0.0
        
        return self.I_0 / (exp_f - 1)
    
    def intensity(self, nu: Union[float, np.ndarray], n_hat: np.ndarray) -> Union[float, np.ndarray]:
        """
        Calculate intensity at sky position n_hat
        
        Parameters:
        -----------
        nu : float or array
            Frequency [Hz]
        n_hat : array_like, shape (2,)
            Sky coordinates [x, y] in angular units
            
        Returns:
        --------
        intensity : float or array
            Specific intensity [W m^-2 Hz^-1 sr^-1]
        """
        # Convert sky coordinates to disk coordinates
        x, y = n_hat[0], n_hat[1]

        # # Calculate coordinates in disk plane
        x_prime = x * self.cos_phi_B + y * self.sin_phi_B 
        y_prime = (- x * self.sin_phi_B + y * self.cos_phi_B) / self.cos_i

        y_prime = self.distance * y_prime
        x_prime = self.distance * x_prime

        # q = np.sqrt(self.cos_i**2 * self.cos_phi_B**2 + self.sin_phi_B**2)
        
        # Convert angular radius to physical radius in proper unts
        R_physical = np.sqrt(x_prime**2 + y_prime**2) / self.GM_over_c2
        
        return self._disk_intensity(R_physical)
    
    def total_flux(self, nu: float) -> float:
        """Calculate total flux by integrating over disk"""
        def integrand(R):
            return R * self._disk_intensity(R)
        
        try:
            flux, _ = quad(integrand, self.R_in, 100 * self.R_0, epsrel=1e-6)
            angular_area_factor = (self.GM_over_c2 / self.distance)**2
            return flux * angular_area_factor * abs(self.cos_i)
        except:
            return 1e-12  # Fallback value
    
    def V(self, nu_0: float, baseline: np.ndarray,
          grid_size: int = 128, sky_extent: float = 1e-4) -> complex:
        """
        Analytical simplified fringe visibility for Shakura-Sunyaev disk - Equation (8)
        
        V_simple(ν₀,B) = ∫ dR R I(R) J₀(2πqνBR/cD) / ∫ dR R I(R)
        where q = √(cos²i cos²φ_B + sin²φ_B)
        """
        # Calculate baseline parameters
        B_mag = np.linalg.norm(baseline[:2])  # Use perpendicular component
        
        # Calculate q factor
        q = np.sqrt(self.cos_i**2 * np.cos(self.phi_B)**2 + np.sin(self.phi_B)**2)
        
        # Wave number
        c = 2.99792458e8
        k = 2 * np.pi * nu_0 / c
        
        # Calculate the oscillatory parameter the integral is done over R in GM/c^2 units
        alpha = k * B_mag *  q * self.GM_over_c2 / self.distance

        if alpha == 0:
            return 1.0 + 0.0j
        
        # Use simplified integration method
        return self._visibility_integration(alpha)
    
    def _visibility_integration(self, alpha: float) -> complex:
        """
        Proper integration using Bessel function J₀ for visibility calculation
        
        V = ∫ r * I(r) * J₀(αr) dr / ∫ r * I(r) dr
        """
        if alpha == 0:
            return 1.0 + 0.0j
        
        # For very small alpha, J₀(αr) ≈ 1, so visibility ≈ 1
        if alpha < 1e-6:
            return 1.0 + 0.0j
        
        # Set up integration range - need to go well beyond R_0 for proper oscillations
        R_min = self.R_in
        R_max = max(100 * self.R_0, 50 / alpha if alpha > 0 else 100 * self.R_0)
        
        # Use sufficient point density for accurate Bessel function sampling
        n_points = max(50000, int(30 * alpha * R_max / (2 * np.pi)))
        n_points = min(n_points, 20000)  # Keep computational load reasonable
        
        r_array = np.linspace(R_min, R_max, n_points)
        
        # Calculate intensity values
        I_values = np.array([self._disk_intensity(r) for r in r_array])
        
        # Use proper Bessel function J₀(αr) for the visibility integral
        from scipy.special import j0
        bessel_values = j0(alpha * r_array)
        
        # Calculate numerator and denominator integrals
        numerator_integrand = r_array * I_values * bessel_values
        denominator_integrand = r_array * I_values
        
        # Use trapezoidal integration
        numerator = np.trapz(numerator_integrand, r_array)
        denominator = np.trapz(denominator_integrand, r_array)
        
        if denominator == 0:
            return 0.0 + 0.0j
        
        visibility = numerator / denominator
        
        # Ensure physical bounds
        visibility_mag = abs(visibility)
        if visibility_mag > 1.0:
            visibility = visibility / visibility_mag
        
        return complex(visibility.real, 0.0)


class BroadLineRegion(ChaoticSource):
    """
    Simplified Broad Line Region model for AGN - Section IV
    
    Models BLR as Keplerian disk with velocity-dependent emission.
    Inherits temporal coherence functions from ChaoticSource.
    """
    
    def __init__(self, beta_function: callable, R_in: float, R_out: float,
                 GM: float, inclination: float, distance: float,
                 line_center_freq: float):
        """
        Parameters:
        -----------
        beta_function : callable
            Response function β(R) [dimensionless]
        R_in : float
            Inner BLR radius [m]
        R_out : float
            Outer BLR radius [m]
        GM : float
            Gravitational parameter [m³/s²]
        inclination : float
            Inclination angle [radians]
        distance : float
            Distance to source [m]
        line_center_freq : float
            Rest frequency of emission line [Hz]
        """
        self.beta_function = beta_function
        self.R_in = R_in
        self.R_out = R_out
        self.GM = GM
        self.inclination = inclination
        self.distance = distance
        self.nu_c = line_center_freq
        
        self.cos_i = np.cos(inclination)
        self.sin_i = np.sin(inclination)
        self.c = 2.99792458e8

    def get_params(self) -> dict:
        """
        The parameters that define the source model, particularly those that
        may be varied in fitting or optimization.
        """
        pass

    def _keplerian_velocity(self, R: float, phi: float) -> float:
        """
        Calculate line-of-sight Keplerian velocity - Equation (37)
        
        v_LOS(R,φ) = √(GM/R) sin(i) sin(φ)
        """
        v_circular = np.sqrt(self.GM / R)
        return v_circular * self.sin_i * np.sin(phi)
    
    def _doppler_shift(self, v_los: float) -> float:
        """Calculate Doppler-shifted frequency"""
        return self.nu_c * (1 - v_los / self.c)
    
    def intensity(self, nu: Union[float, np.ndarray], n_hat: np.ndarray) -> Union[float, np.ndarray]:
        """
        Calculate BLR intensity including Doppler shifts
        """
        # Convert sky coordinates to disk coordinates
        x, y = n_hat[0], n_hat[1]
        
        # Convert to polar coordinates in disk
        R_angular = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        
        # Convert to physical radius
        R_physical = R_angular * self.distance
        
        # Check if within BLR bounds
        if R_physical < self.R_in or R_physical > self.R_out:
            return 0.0
        
        # Calculate line-of-sight velocity
        v_los = self._keplerian_velocity(R_physical, phi)
        
        # Calculate Doppler-shifted frequency at this location
        nu_local = self._doppler_shift(v_los)
        
        # Check if observed frequency matches local emission
        freq_tolerance = 0.01 * self.nu_c  # 1% tolerance
        if abs(nu - nu_local) > freq_tolerance:
            return 0.0
        
        # Return intensity proportional to β(R)
        return self.beta_function(R_physical)
    
    def total_flux(self, nu: float) -> float:
        """Calculate total flux for given observed frequency"""
        def integrand(R, phi):
            # Check velocity matching
            v_los = self._keplerian_velocity(R, phi)
            nu_local = self._doppler_shift(v_los)
            
            freq_tolerance = 0.01 * self.nu_c
            if abs(nu - nu_local) > freq_tolerance:
                return 0.0
            
            return R * self.beta_function(R) * abs(self.cos_i)
        
        try:
            flux, _ = dblquad(integrand, 0, 2*np.pi, self.R_in, self.R_out)
            
            # Convert to observed flux
            angular_area_factor = 1.0 / self.distance**2
            return flux * angular_area_factor
        except:
            return 1e-12  # Fallback value


class RelativisticDisk(ShakuraSunyaevDisk):
    """
    Simplified relativistic accretion disk including basic relativistic effects
    
    Extension of Shakura-Sunyaev disk with simplified relativistic corrections.
    Inherits temporal coherence functions from ChaoticSource via ShakuraSunyaevDisk.
    """
    
    def __init__(self, *args, spin_parameter: float = 0.0, **kwargs):
        """
        Parameters:
        -----------
        spin_parameter : float, optional
            Black hole spin parameter a/M (default: 0.0)
        *args, **kwargs : 
            Arguments for ShakuraSunyaevDisk
        """
        super().__init__(*args, **kwargs)
        self.spin_parameter = spin_parameter
        
        # Calculate ISCO radius for given spin
        self.R_isco = self._calculate_isco_radius()
        
        # Update inner radius to ISCO if not specified
        if self.R_in < self.R_isco:
            self.R_in = self.R_isco
    
    def _calculate_isco_radius(self) -> float:
        """Calculate ISCO radius for given spin parameter"""
        a = self.spin_parameter
        
        # Bardeen, Press & Teukolsky (1972) formula
        Z1 = 1 + (1 - a**2)**(1/3) * ((1 + a)**(1/3) + (1 - a)**(1/3))
        Z2 = np.sqrt(3 * a**2 + Z1**2)
        
        if a >= 0:  # Prograde
            R_isco = 3 + Z2 - np.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))
        else:  # Retrograde
            R_isco = 3 + Z2 + np.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))
        
        return R_isco
    
    def _doppler_factor(self, R: float, phi: float) -> float:
        """
        Calculate simplified relativistic Doppler factor
        """
        # Keplerian velocity
        v_phi = np.sqrt(self.GM_over_c2 / R)  # In units of c
        
        # Line-of-sight component
        v_los = v_phi * self.sin_i * np.sin(phi)
        
        # Simplified relativistic Doppler factor
        gamma = 1.0 / np.sqrt(1 - self.GM_over_c2 / R)
        return gamma * (1 - v_los)
    
    def intensity(self, nu: Union[float, np.ndarray], n_hat: np.ndarray) -> Union[float, np.ndarray]:
        """
        Calculate intensity including simplified relativistic effects
        """
        # Get base intensity from SS disk
        base_intensity = super().intensity(nu, n_hat)
        
        if base_intensity == 0.0:
            return 0.0
        
        # Convert to disk coordinates
        x, y = n_hat[0], n_hat[1]
        y_disk = y / self.cos_i if self.cos_i != 0 else y
        
        R_angular = np.sqrt(x**2 + y_disk**2)
        phi = np.arctan2(y_disk, x)
        
        R_physical = R_angular * self.distance / self.GM_over_c2
        
        # Apply simplified Doppler beaming
        # doppler_factor = self._doppler_factor(R_physical, phi)
        # turn this off
        doppler_factor = 1.0
        
        # Intensity transforms as I' = D³ I in observer frame
        return base_intensity * doppler_factor**3


# Utility functions for AGN models
def power_law_beta(R: float, R_0: float, n: float, normalization: float = 1.0) -> float:
    """
    Power law response function β(R) ∝ R^n
    
    Parameters:
    -----------
    R : float
        Radius [m]
    R_0 : float
        Normalization radius [m]
    n : float
        Power law index
    normalization : float, optional
        Normalization constant
        
    Returns:
    --------
    beta : float
        Response function value
    """
    return normalization * (R / R_0)**n


def lognormal_beta(R: float, R_0: float, sigma: float, normalization: float = 1.0) -> float:
    """
    Lognormal response function for BLR
    
    Parameters:
    -----------
    R : float
        Radius [m]
    R_0 : float
        Characteristic radius [m]
    sigma : float
        Width parameter
    normalization : float, optional
        Normalization constant
        
    Returns:
    --------
    beta : float
        Response function value
    """
    if R <= 0:
        return 0.0
    
    log_term = np.log(R / R_0)
    exponent = -log_term**2 / (2 * sigma**2) - sigma**2 / 2
    return normalization / (R * np.sqrt(2 * np.pi) * sigma) * np.exp(exponent)
