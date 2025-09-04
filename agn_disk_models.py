"""
AGN Accretion Disk Models for Intensity Interferometry
Based on "Probing H0 and resolving AGN disks with ultrafast photon counters" (Dalal et al. 2024)

This module implements the Shakura-Sunyaev disk profile and visibility calculations.
"""

import numpy as np
import scipy.special as special
import scipy.integrate as integrate
from typing import Tuple
import astropy.constants as const
from dataclasses import dataclass


@dataclass
class DiskParameters:
    """Parameters for AGN accretion disk models"""
    black_hole_mass: float  # Solar masses
    distance: float  # Angular diameter distance in Mpc
    inclination: float  # Disk inclination angle in radians
    inner_radius: float  # Inner disk radius in GM/c²
    power_law_index: float = 3.0  # Temperature profile power law index n
    normalization_radius: float = 43.0  # R₀ in GM/c² at reference wavelength
    
    @property
    def gravitational_radius(self) -> float:
        """Gravitational radius GM/c² in meters"""
        return const.G.value * self.black_hole_mass * const.M_sun.value / const.c.value**2
    
    @property
    def angular_scale(self) -> float:
        """Angular scale: GM/c² in radians at given distance"""
        distance_m = self.distance * 3.086e22  # Mpc to meters
        return self.gravitational_radius / distance_m


class ShakuraSunyaevDisk:
    """
    Shakura-Sunyaev accretion disk model
    
    Implements Equations (21-22) from the paper:
    I(R) = I₀[e^f(R) - 1]⁻¹
    f(R) = (R₀/R)ⁿ * (1 - √(R_in/R))^(-1/4)
    """
    
    def __init__(self, params: DiskParameters):
        self.params = params
    
    def temperature_profile_function(self, radius: float, wavelength: float = 550e-9) -> float:
        """Calculate f(R) from Equation (22)"""
        if radius <= self.params.inner_radius:
            return np.inf
        
        # Scale normalization radius with wavelength
        r0_scaled = self.params.normalization_radius * (wavelength / 550e-9)**(4.0/3.0)
        
        ratio_term = (r0_scaled / radius)**self.params.power_law_index
        sqrt_term = (1 - np.sqrt(self.params.inner_radius / radius))**(-0.25)
        
        return ratio_term * sqrt_term
    
    def intensity_profile(self, radius: float, wavelength: float = 550e-9) -> float:
        """Calculate disk intensity I(R) from Equation (21)"""
        if radius <= self.params.inner_radius:
            return 0.0
        
        f_r = self.temperature_profile_function(radius, wavelength)
        
        if f_r > 50:  # Avoid overflow
            return 0.0
        
        return 1.0 / (np.exp(f_r) - 1.0)
    
    def visibility_amplitude(self, baseline_length: float, baseline_angle: float = 0.0,
                           wavelength: float = 550e-9) -> float:
        """
        Calculate visibility amplitude for Shakura-Sunyaev disk
        Based on Equation (18) from the paper
        """
        # Calculate q factor for inclination
        cos_i = np.cos(self.params.inclination)
        cos_phi = np.cos(baseline_angle)
        sin_phi = np.sin(baseline_angle)
        q = np.sqrt(cos_i**2 * cos_phi**2 + sin_phi**2)
        
        def integrand_numerator(r):
            if r <= self.params.inner_radius:
                return 0.0
            intensity = self.intensity_profile(r, wavelength)
            r_angular = r * self.params.angular_scale
            arg = 2 * np.pi * q * baseline_length * r_angular / wavelength
            bessel = special.j0(arg) if arg < 100 else 0.0
            return r * intensity * bessel
        
        def integrand_denominator(r):
            if r <= self.params.inner_radius:
                return 0.0
            return r * self.intensity_profile(r, wavelength)
        
        r_min = self.params.inner_radius
        r_max = 1000.0  # Limit integration range
        
        try:
            numerator, _ = integrate.quad(integrand_numerator, r_min, r_max)
            denominator, _ = integrate.quad(integrand_denominator, r_min, r_max)
            return abs(numerator / denominator) if denominator > 0 else 0.0
        except:
            return 0.0


def create_fiducial_agn_disk() -> ShakuraSunyaevDisk:
    """Create the fiducial AGN disk model from the paper"""
    params = DiskParameters(
        black_hole_mass=1e8,  # Solar masses
        distance=20.0,  # Mpc
        inclination=np.pi/3,  # 60 degrees
        inner_radius=6.0,  # GM/c² (Schwarzschild ISCO)
        power_law_index=3.0,  # n = 3 for Shakura-Sunyaev
        normalization_radius=43.0  # R₀ = 43 GM/c² at 550 nm
    )
    return ShakuraSunyaevDisk(params)


if __name__ == "__main__":
    # Example usage
    disk = create_fiducial_agn_disk()
    print(f"Angular scale: {disk.params.angular_scale * 1e6:.2f} μas per GM/c²")
    
    # Test visibility calculation
    baseline = 1000  # 1 km
    vis = disk.visibility_amplitude(baseline)
    print(f"Visibility at 1 km baseline: |V| = {vis:.3f}")