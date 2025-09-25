"""
JAX-Compatible AGN Source Models for Intensity Interferometry

Streamlined implementation focusing on computational efficiency
and JAX compatibility for automatic differentiation.

All AGN source models now inherit from ChaoticSource to provide
proper temporal coherence functions for intensity interferometry.
"""

import numpy as np
from scipy.integrate import quad, dblquad
from typing import Union, Optional
import jax
import jax.numpy as jnp
from jax import custom_jvp, pure_callback
from functools import partial
from scipy.special import j0, j1

from ..base import source


# JAX-compatible Bessel function j0
@partial(jax.custom_jvp, nondiff_argnums=())
def _j0(x):
    """Zeroth-order Bessel function J0 using scipy.special.j0"""
    result_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return pure_callback(
        lambda x: j0(np.asarray(x)).astype(x.dtype),
        result_shape,
        x,
        vmap_method='sequential'
    )

@_j0.defjvp
def _j0_jvp(primals, tangents):
    """Custom JVP rule for J0: d/dx J0(x) = -J1(x)"""
    x, = primals
    dx, = tangents
    y = _j0(x)
    
    # Derivative of J0 is -J1
    result_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    j1_val = pure_callback(
        lambda x: j1(np.asarray(x)).astype(x.dtype),
        result_shape,
        x,
        vmap_method='sequential'
    )
    
    dy = -j1_val
    return y, dy * dx


class ShakuraSunyaevDisk(source.ChaoticSource):
    """
    JAX-compatible Shakura-Sunyaev accretion disk model - Equations (21-22)
    
    I(R) = I₀ [e^f(R) - 1]^(-1)
    f(R) = (ν/ν₀(R)) = [(R₀/R)^n (1 - √(R_in/R))]^(-1/4)
    
    Uses JAX-compatible operations for automatic differentiation.
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
        self.I_0 = float(I_0)
        self.R_0 = float(R_0)
        self.R_in = float(R_in)
        self.n = float(n)
        self.inclination = float(inclination)
        self.phi_B = float(phi_B)
        self.distance = float(distance)
        self.GM_over_c2 = float(GM_over_c2)
        
        # Precompute trigonometric values for efficiency
        self.cos_i = jnp.cos(inclination)
        self.sin_i = jnp.sin(inclination)
        self.cos_phi_B = jnp.cos(phi_B)
        self.sin_phi_B = jnp.sin(phi_B)

    def get_params(self) -> dict:
        """
        Get parameters that define the source model.
        
        Returns
        -------
        dict
            Dictionary containing source parameters that may be varied in fitting
        """
        return {
            'I_0': self.I_0,
            'R_0': self.R_0,
            'R_in': self.R_in,
            'n': self.n,
            'inclination': self.inclination,
            'phi_B': self.phi_B,
            'distance': self.distance,
            'GM_over_c2': self.GM_over_c2
        }

    def _f_function(self, R: float) -> float:
        """Calculate f(R) from Equation (22) - JAX compatible"""
        # Use JAX operations to avoid if-then statements
        # For R <= R_in, we want f -> infinity, so use large value
        ratio = self.R_0 / jnp.maximum(R, 1e-10)  # Avoid division by zero
        sqrt_term = jnp.sqrt(self.R_in / jnp.maximum(R, 1e-10))
        
        # Use jnp.where to handle the R <= R_in case
        f_val = jnp.where(
            R <= self.R_in,
            1e10,  # Large value instead of infinity
            (ratio**self.n * jnp.maximum(1 - sqrt_term, 1e-10))**(-0.25)
        )
        return f_val
    
    def _disk_intensity(self, R: float) -> float:
        """Calculate disk intensity I(R) from Equation (21) - JAX compatible"""
        f_val = self._f_function(R)
        
        # Use JAX operations to avoid if-then statements
        exp_f = jnp.exp(jnp.minimum(f_val, 100))  # Clip to avoid overflow
        intensity = self.I_0 / jnp.maximum(exp_f - 1, 1e-10)  # Avoid division by zero
        
        # Zero out intensity for R <= R_in
        intensity = jnp.where(R <= self.R_in, 0.0, intensity)
        
        return intensity
    
    def intensity(self, nu: Union[float, np.ndarray], n_hat: np.ndarray) -> Union[float, np.ndarray]:
        """
        Calculate intensity at sky position n_hat - JAX compatible
        
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

        # Calculate coordinates in disk plane
        x_prime = x * self.cos_phi_B + y * self.sin_phi_B 
        y_prime = (-x * self.sin_phi_B + y * self.cos_phi_B) / self.cos_i

        y_prime = self.distance * y_prime
        x_prime = self.distance * x_prime
        
        # Convert angular radius to physical radius
        R_physical = jnp.sqrt(x_prime**2 + y_prime**2) / self.GM_over_c2
        
        return self._disk_intensity(R_physical)
    
    def total_flux(self, nu: float) -> float:
        """Calculate total flux by integrating over disk"""
        def integrand(R):
            return R * self._disk_intensity(R)
        
        try:
            flux, _ = quad(integrand, self.R_in, 100 * self.R_0, epsrel=1e-6)
            angular_area_factor = (self.GM_over_c2 / self.distance)**2
            return flux * angular_area_factor * jnp.abs(self.cos_i)
        except:
            return 1e-12  # Fallback value
    
    def V(self, nu_0: float, baseline: np.ndarray,
          grid_size: int = 128, sky_extent: float = 1e-4) -> complex:
        """
        JAX-compatible analytical fringe visibility for Shakura-Sunyaev disk
        
        V_simple(ν₀,B) = ∫ dR R I(R) J₀(2πqνBR/cD) / ∫ dR R I(R)
        where q = √(cos²i cos²φ_B + sin²φ_B)
        """
        # Calculate baseline parameters
        B_mag = jnp.linalg.norm(jnp.array(baseline[:2]))
        
        # Calculate q factor
        q = jnp.sqrt(self.cos_i**2 * self.cos_phi_B**2 + self.sin_phi_B**2)
        
        # Wave number
        c = 2.99792458e8
        k = 2 * jnp.pi * nu_0 / c
        
        # Calculate the oscillatory parameter
        alpha = k * B_mag * q * self.GM_over_c2 / self.distance

        # Use JAX-compatible operations instead of if-then
        return jnp.where(
            alpha == 0,
            1.0 + 0.0j,
            self._visibility_integration(alpha)
        )
    
    def _visibility_integration(self, alpha: float) -> complex:
        """
        JAX-compatible visibility integration using Bessel function J₀
        
        V = ∫ r * I(r) * J₀(αr) dr / ∫ r * I(r) dr
        """
        # Use JAX-compatible operations
        alpha_safe = jnp.maximum(alpha, 1e-10)  # Avoid division by zero
        
        # For very small alpha, return 1
        small_alpha_result = 1.0 + 0.0j
        
        # Set up integration range
        R_min = self.R_in
        R_max = jnp.maximum(100 * self.R_0, 50 / alpha_safe)
        
        # Use fixed number of points for JAX compatibility
        n_points = 10000
        r_array = jnp.linspace(R_min, R_max, n_points)
        
        # Calculate intensity values using vectorized operations
        I_values = jax.vmap(self._disk_intensity)(r_array)
        
        # Use JAX-compatible Bessel function
        bessel_values = _j0(alpha_safe * r_array)
        
        # Calculate integrals using trapezoidal rule
        numerator_integrand = r_array * I_values * bessel_values
        denominator_integrand = r_array * I_values
        
        numerator = jnp.trapezoid(numerator_integrand, r_array)
        denominator = jnp.trapezoid(denominator_integrand, r_array)
        
        # Avoid division by zero
        denominator_safe = jnp.maximum(denominator, 1e-10)
        visibility = numerator / denominator_safe
        
        # Ensure physical bounds using JAX operations
        visibility_mag = jnp.abs(visibility)
        visibility = jnp.where(
            visibility_mag > 1.0,
            visibility / visibility_mag,
            visibility
        )
        
        # Return based on alpha value
        return jnp.where(
            alpha < 1e-6,
            small_alpha_result,
            visibility.real + 0.0j
        )


class BroadLineRegion(source.ChaoticSource):
    """
    JAX-compatible Broad Line Region model for AGN - Section IV
    
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
        
        self.cos_i = jnp.cos(inclination)
        self.sin_i = jnp.sin(inclination)
        self.c = 2.99792458e8

    def get_params(self) -> dict:
        """
        Get parameters that define the source model.
        
        Returns
        -------
        dict
            Dictionary containing source parameters that may be varied in fitting
        """
        return {
            'R_in': self.R_in,
            'R_out': self.R_out,
            'GM': self.GM,
            'inclination': self.inclination,
            'distance': self.distance,
            'line_center_freq': self.nu_c
        }

    def _keplerian_velocity(self, R: float, phi: float) -> float:
        """
        Calculate line-of-sight Keplerian velocity - JAX compatible
        
        v_LOS(R,φ) = √(GM/R) sin(i) sin(φ)
        """
        R_safe = jnp.maximum(R, 1e-10)  # Avoid division by zero
        v_circular = jnp.sqrt(self.GM / R_safe)
        return v_circular * self.sin_i * jnp.sin(phi)
    
    def _doppler_shift(self, v_los: float) -> float:
        """Calculate Doppler-shifted frequency - JAX compatible"""
        return self.nu_c * (1 - v_los / self.c)
    
    def intensity(self, nu: Union[float, np.ndarray], n_hat: np.ndarray) -> Union[float, np.ndarray]:
        """
        Calculate BLR intensity including Doppler shifts - JAX compatible
        """
        # Convert sky coordinates to disk coordinates
        x, y = n_hat[0], n_hat[1]
        
        # Convert to polar coordinates in disk
        R_angular = jnp.sqrt(x**2 + y**2)
        phi = jnp.arctan2(y, x)
        
        # Convert to physical radius
        R_physical = R_angular * self.distance
        
        # Check if within BLR bounds using JAX operations
        in_bounds = (R_physical >= self.R_in) & (R_physical <= self.R_out)
        
        # Calculate line-of-sight velocity
        v_los = self._keplerian_velocity(R_physical, phi)
        
        # Calculate Doppler-shifted frequency at this location
        nu_local = self._doppler_shift(v_los)
        
        # Check if observed frequency matches local emission
        freq_tolerance = 0.01 * self.nu_c
        freq_match = jnp.abs(nu - nu_local) <= freq_tolerance
        
        # Return intensity using JAX where operations
        beta_val = self.beta_function(R_physical)
        return jnp.where(in_bounds & freq_match, beta_val, 0.0)
    
    def total_flux(self, nu: float) -> float:
        """Calculate total flux for given observed frequency"""
        def integrand(R, phi):
            # Check velocity matching
            v_los = self._keplerian_velocity(R, phi)
            nu_local = self._doppler_shift(v_los)
            
            freq_tolerance = 0.01 * self.nu_c
            freq_match = jnp.abs(nu - nu_local) <= freq_tolerance
            
            return jnp.where(
                freq_match,
                R * self.beta_function(R) * jnp.abs(self.cos_i),
                0.0
            )
        
        try:
            flux, _ = dblquad(integrand, 0, 2*jnp.pi, self.R_in, self.R_out)
            
            # Convert to observed flux
            angular_area_factor = 1.0 / self.distance**2
            return flux * angular_area_factor
        except:
            return 1e-12  # Fallback value
    
    def V(self, nu_0: float, baseline: np.ndarray, params: dict = None) -> complex:
        """
        JAX-compatible fringe visibility for BLR using Equation 45
        """
        # Calculate baseline parameters
        B_mag = jnp.linalg.norm(jnp.array(baseline[:2]))
        
        # Use JAX where instead of if-then
        zero_baseline_result = 1.0 + 0.0j
        
        # Physical constants
        c = 2.99792458e8
        wavelength = c / nu_0
        
        # Calculate the oscillatory parameter
        alpha = 2 * jnp.pi * B_mag / (wavelength * self.distance)
        
        return jnp.where(
            B_mag == 0,
            zero_baseline_result,
            jnp.where(
                alpha == 0,
                zero_baseline_result,
                self._blr_visibility_integration(alpha, nu_0)
            )
        )
    
    def _blr_visibility_integration(self, alpha: float, nu_0: float) -> complex:
        """
        JAX-compatible BLR visibility integration
        """
        # Set up integration range
        R_min = self.R_in
        R_max = jnp.minimum(self.R_out, 100 * R_min)  # Limit range for stability
        
        # Use fixed number of points for JAX compatibility
        n_points = 5000
        r_array = jnp.linspace(R_min, R_max, n_points)
        
        # Calculate beta values
        beta_values = jax.vmap(self.beta_function)(r_array)
        
        # Bessel function term
        bessel_values = _j0(alpha * r_array)
        
        # Simplified Doppler factor (could be enhanced)
        doppler_factor = 1.0
        
        # Calculate integrals
        numerator_integrand = r_array * beta_values * bessel_values * doppler_factor
        denominator_integrand = r_array * beta_values
        
        numerator = jnp.trapz(numerator_integrand, r_array)
        denominator = jnp.trapz(denominator_integrand, r_array)
        
        # Avoid division by zero
        denominator_safe = jnp.maximum(denominator, 1e-10)
        visibility_real = numerator / denominator_safe
        
        # Ensure physical bounds
        visibility_mag = jnp.abs(visibility_real)
        visibility_real = jnp.where(
            visibility_mag > 1.0,
            visibility_real / visibility_mag,
            visibility_real
        )
        
        return visibility_real + 0.0j


class RelativisticDisk(ShakuraSunyaevDisk):
    """
    JAX-compatible relativistic accretion disk with basic relativistic effects
    
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
        
        # Update inner radius to ISCO using JAX operations
        self.R_in = jnp.maximum(self.R_in, self.R_isco)
    
    def get_params(self) -> dict:
        """
        Get parameters that define the relativistic disk model.
        
        Returns
        -------
        dict
            Dictionary containing source parameters including spin parameter
        """
        # Get base parameters from ShakuraSunyaevDisk
        params = super().get_params()
        # Add relativistic-specific parameter
        params['spin_parameter'] = self.spin_parameter
        return params
    
    def _calculate_isco_radius(self) -> float:
        """Calculate ISCO radius for given spin parameter - JAX compatible"""
        a = self.spin_parameter
        
        # Bardeen, Press & Teukolsky (1972) formula
        Z1 = 1 + (1 - a**2)**(1/3) * ((1 + a)**(1/3) + (1 - a)**(1/3))
        Z2 = jnp.sqrt(3 * a**2 + Z1**2)
        
        # Use JAX where instead of if-then
        R_isco_prograde = 3 + Z2 - jnp.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))
        R_isco_retrograde = 3 + Z2 + jnp.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))
        
        return jnp.where(a >= 0, R_isco_prograde, R_isco_retrograde)
    
    def _doppler_factor(self, R: float, phi: float) -> float:
        """
        Calculate simplified relativistic Doppler factor - JAX compatible
        """
        R_safe = jnp.maximum(R, 1e-10)  # Avoid division by zero
        
        # Keplerian velocity
        v_phi = jnp.sqrt(self.GM_over_c2 / R_safe)  # In units of c
        
        # Line-of-sight component
        v_los = v_phi * self.sin_i * jnp.sin(phi)
        
        # Simplified relativistic Doppler factor
        gamma = 1.0 / jnp.sqrt(jnp.maximum(1 - self.GM_over_c2 / R_safe, 1e-10))
        return gamma * (1 - v_los)
    
    def intensity(self, nu: Union[float, np.ndarray], n_hat: np.ndarray) -> Union[float, np.ndarray]:
        """
        Calculate intensity including simplified relativistic effects - JAX compatible
        """
        # Get base intensity from SS disk
        base_intensity = super().intensity(nu, n_hat)
        
        # Convert to disk coordinates
        x, y = n_hat[0], n_hat[1]
        y_disk = jnp.where(self.cos_i != 0, y / self.cos_i, y)
        
        R_angular = jnp.sqrt(x**2 + y_disk**2)
        phi = jnp.arctan2(y_disk, x)
        
        R_physical = R_angular * self.distance / self.GM_over_c2
        
        # Apply simplified Doppler beaming (turned off for now)
        doppler_factor = 1.0  # Could use self._doppler_factor(R_physical, phi)
        
        # Intensity transforms as I' = D³ I in observer frame
        return base_intensity * doppler_factor**3


# JAX-compatible utility functions for AGN models
def power_law_beta(R: float, R_0: float, n: float, normalization: float = 1.0) -> float:
    """
    JAX-compatible power law response function β(R) ∝ R^n
    
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
    R_safe = jnp.maximum(R, 1e-10)  # Avoid division by zero
    R_0_safe = jnp.maximum(R_0, 1e-10)
    return normalization * (R_safe / R_0_safe)**n


def lognormal_beta(R: float, R_0: float, sigma: float, normalization: float = 1.0) -> float:
    """
    JAX-compatible lognormal response function for BLR
    
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
    R_safe = jnp.maximum(R, 1e-10)  # Avoid log(0)
    R_0_safe = jnp.maximum(R_0, 1e-10)
    
    log_term = jnp.log(R_safe / R_0_safe)
    exponent = -log_term**2 / (2 * sigma**2) - sigma**2 / 2
    prefactor = normalization / (R_safe * jnp.sqrt(2 * jnp.pi) * sigma)
    
    return prefactor * jnp.exp(exponent)
