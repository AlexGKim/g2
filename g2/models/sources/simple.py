import numpy as np
from typing import Callable, Union, Any, Dict
import jax.numpy as jnp # Use JAX for array operations
from jax import custom_jvp, pure_callback
import jax
import jax.lax
from functools import partial

from scipy.special import j1, jv

from ..base.source import ChaoticSource


# @custom_jvp
# def _j1(x):
#     """First-order Bessel function J1 using scipy.special.j1"""
#     return j1(x)

# @_j1.defjvp
# def _j1_jvp(primals, tangents):
#     """Custom JVP rule for J1 using scipy.special.jv"""
#     x, = primals
#     dx, = tangents
#     y = j1(x)
#     dy = y/x - jv(2, x) 
#     return y, dy * dx 

from functools import partial

@partial(jax.custom_jvp, nondiff_argnums=())
def _j1(x):
    """First-order Bessel function J1 using scipy.special.j1"""
    result_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return pure_callback(
        lambda x: j1(np.asarray(x)).astype(x.dtype),
        result_shape,
        x,
        vmap_method='sequential'
    )

@_j1.defjvp
def _j1_jvp(primals, tangents):
    """Custom JVP rule for J1"""
    x, = primals
    dx, = tangents
    y = _j1(x)
    
    # Also need to wrap jv for the derivative
    result_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    jv2 = pure_callback(
        lambda x: jv(2, np.asarray(x)).astype(x.dtype),
        result_shape,
        x,
        vmap_method='sequential'
    )
    
    dy = y/x - jv2
    return y, dy * dx

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
        I(ν, n̂) = I₀
    
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
    >>>     print(f"B={B}m: \\V\\|={abs(vis):.3f}")
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
        
        where J₁ is the first-order Bessel function, u = \\|B_⊥\\|/λ is the
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
            u = 1.22/(2θ)  or  \\|B_⊥\\| = 1.22λ/(2θ)
        
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
        
        # Calculate spatial frequency u = \\|B_⊥\\|/λ
        u = baseline_length / wavelength
        
        # Calculate argument for Bessel function: zeta = 2πuθ
        zeta = np.pi * u * (2 * params['radius'])
        # Handle special case x=0 (zero baseline or zero radius)
        V_value = jnp.where(zeta == 0, 1.0, 2 * _j1(zeta) / zeta)
        # if zeta == 0:
        #     V_value = 1.0
        # else:
        #     # Airy function: V(u) = 2J₁(x)/x
        #     # V_value = 2 * j1(x) / x
        #     V_value = 2 * _j1(zeta) / zeta   
        # Return as complex number (phase is zero for symmetric disk)
        return V_value + 0.0j


class MultiPoint(ChaoticSource):
    """
    Multiple point sources implementation.
    
    Represents multiple unresolved point sources at specified positions on the sky.
    This extends the single PointSource to handle collections of point sources,
    which is useful for modeling binary stars, multiple AGN components, or
    other multi-component sources.
    
    The total intensity is the sum of individual point source intensities:
        I(ν, n̂) = Σᵢ Fᵢ(ν) δ²(n̂ - n̂ᵢ)
    
    where Fᵢ(ν) is the flux density of the i-th source and n̂ᵢ is its position.
    
    Parameters
    ----------
    flux_densities : array_like
        Array of flux densities for each source in W m⁻² Hz⁻¹.
    positions : array_like, shape (N, 2)
        Array of source positions in radians [[theta_x1, theta_y1], ...].
    spectral_indices : array_like, optional
        Array of spectral indices for each source (F_ν ∝ ν^α). Default is 0 (flat spectrum).
    reference_frequency : float, optional
        Reference frequency in Hz for spectral index scaling. Default is 5e14 Hz.
        
    Attributes
    ----------
    flux_densities : jnp.ndarray
        Flux densities of individual sources.
    positions : jnp.ndarray
        Positions of individual sources.
    spectral_indices : jnp.ndarray
        Spectral indices of individual sources.
    reference_frequency : float
        Reference frequency for spectral scaling.
        
    Examples
    --------
    >>> # Binary star system with flat spectra
    >>> flux_densities = [1e-26, 5e-27]  # W/m²/Hz
    >>> positions = [[1e-8, 0], [-1e-8, 0]]  # radians
    >>> binary = MultiPoint(flux_densities, positions)
    >>> print(f"Total flux at 5e14 Hz: {binary.total_flux(5e14):.2e} W/m²/Hz")
    >>>
    >>> # Triple system with different spectral indices
    >>> flux_densities = [1e-26, 8e-27, 3e-27]
    >>> positions = [[0, 0], [2e-8, 1e-8], [-1e-8, -2e-8]]
    >>> spectral_indices = [-0.5, -1.0, 0.0]  # Different spectral slopes
    >>> triple = MultiPoint(flux_densities, positions, spectral_indices)
    >>>
    >>> # Visibility is coherent sum of individual source visibilities
    >>> baseline = np.array([100.0, 0.0, 0.0])
    >>> vis = triple.V(5e14, baseline)
    >>> print(f"Visibility: {abs(vis):.3f}")
    """
    
    def __init__(self, flux_densities: Union[list, np.ndarray],
                 positions: Union[list, np.ndarray],
                 spectral_indices: Union[list, np.ndarray, None] = None,
                 reference_frequency: float = 5e14):
        """
        Initialize multiple point sources.
        
        Parameters
        ----------
        flux_densities : array_like
            Array of flux densities for each source in W m⁻² Hz⁻¹.
        positions : array_like, shape (N, 2)
            Array of source positions in radians [[theta_x1, theta_y1], ...].
        spectral_indices : array_like, optional
            Array of spectral indices for each source (F_ν ∝ ν^α). Default is 0.
        reference_frequency : float, optional
            Reference frequency in Hz for spectral index scaling. Default is 5e14 Hz.
        """
        self.flux_densities = jnp.array(flux_densities)
        self.positions = jnp.array(positions)
        self.reference_frequency = reference_frequency
        
        # Set default spectral indices to 0 (flat spectrum) if not provided
        if spectral_indices is None:
            self.spectral_indices = jnp.zeros(len(flux_densities))
        else:
            self.spectral_indices = jnp.array(spectral_indices)
        
        # Validate input dimensions
        if len(self.flux_densities) != len(self.positions):
            raise ValueError("Number of flux densities must match number of positions")
        if len(self.spectral_indices) != len(self.flux_densities):
            raise ValueError("Number of spectral indices must match number of sources")

    def get_params(self) -> Dict[str, Any]:
        """Extract parameters as a dictionary"""
        return {
            'flux_densities': self.flux_densities,
            'positions': self.positions,
            'spectral_indices': self.spectral_indices,
            'reference_frequency': self.reference_frequency
        }
    
    def intensity(self, nu: Union[float, np.ndarray], n_hat: np.ndarray, atol: float = 1e-10) -> Union[float, np.ndarray]:
        """
        Calculate multiple point sources intensity.
        
        The total intensity is the sum of individual point source intensities.
        Each point source contributes a delta function at its position.
        
        Parameters
        ----------
        nu : float or array_like
            Frequency in Hz.
        n_hat : array_like, shape (2,)
            Sky direction in radians.
        atol : float, optional
            Tolerance for delta function approximation. Default is 1e-10.
            
        Returns
        -------
        intensity : float
            Total specific intensity in W m⁻² Hz⁻¹ sr⁻¹.
        """
        total_intensity = 0.0
        
        # Calculate flux for each source at the given frequency
        if np.isscalar(nu):
            source_fluxes = self.flux_densities * (nu / self.reference_frequency) ** self.spectral_indices
        else:
            nu_array = jnp.array(nu)
            source_fluxes = self.flux_densities[:, None] * (nu_array[None, :] / self.reference_frequency) ** self.spectral_indices[:, None]
        
        # Check if n_hat is close to any source position
        for i in range(len(self.positions)):
            if jnp.allclose(n_hat, self.positions[i], atol=atol):
                if np.isscalar(nu):
                    total_intensity += source_fluxes[i] / (np.pi * atol**2)
                else:
                    total_intensity += source_fluxes[i, :] / (np.pi * atol**2)
        
        return total_intensity
    
    def total_flux(self, nu: float) -> float:
        """
        Calculate total flux from all point sources.
        
        Parameters
        ----------
        nu : float
            Frequency in Hz.
            
        Returns
        -------
        flux : float
            Total flux density in W m⁻² Hz⁻¹.
        """
        # Calculate flux for each source and sum
        source_fluxes = self.flux_densities * (nu / self.reference_frequency) ** self.spectral_indices
        return jnp.sum(source_fluxes)
    
    def V_squared_jacobian(self, nu_0: float, baseline: np.ndarray, params: dict = None):
        """
        Calculate the Jacobian of \\V\\² with respect to the source parameters.
        
        For multiple point sources, provides correct derivatives:
        - flux_densities: 0 (\\V\\² is invariant to uniform flux scaling)
        - positions: computed analytically
        - spectral_indices: 0 (for flat spectra at reference frequency)
        - reference_frequency: 0 (for evaluation at reference frequency)
        
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
            derivatives of \\V\\² with respect to each parameter.
        """
        if params is None:
            params = self.get_params()
        
        # Physical constants
        c = 2.99792458e8  # Speed of light in m/s
        wavelength = c / nu_0
        baseline_perp = baseline[:2]
        
        # Calculate current visibility for position derivatives
        V_current = self.V(nu_0, baseline, params)
        
        # Flux densities: derivative is 0 (invariant to uniform scaling)
        flux_grad = jnp.zeros_like(params['flux_densities'])
        
        # Positions: compute analytical derivative
        source_fluxes = params['flux_densities'] * (nu_0 / params['reference_frequency']) ** params['spectral_indices']
        total_flux = jnp.sum(source_fluxes)
        
        # Calculate derivative w.r.t. each position component
        pos_grad = jnp.zeros_like(params['positions'])
        
        for i in range(len(params['positions'])):
            # Derivative of phase w.r.t. position: ∂φᵢ/∂nᵢ = 2π B_⊥ / λ
            dphase_dpos = 2 * jnp.pi * baseline_perp / wavelength
            
            # Weight of this source
            weight_i = source_fluxes[i] / total_flux
            
            # Phase of this source
            phase_i = 2 * jnp.pi * jnp.dot(params['positions'][i], baseline_perp) / wavelength
            
            # Derivative of V w.r.t. position of source i
            dV_dpos_i = weight_i * 1j * jnp.exp(1j * phase_i) * dphase_dpos
            
            # Derivative of \\V\\² = 2 * Re(V* * dV/dpos)
            d_abs_V_squared_dpos_i = 2 * jnp.real(jnp.conj(V_current) * dV_dpos_i)
            
            pos_grad = pos_grad.at[i].set(d_abs_V_squared_dpos_i)
        
        # Spectral indices: 0 at reference frequency
        spectral_grad = jnp.zeros_like(params['spectral_indices'])
        
        # Reference frequency: 0 when evaluated at reference frequency
        ref_freq_grad = 0.0
        
        return {
            'flux_densities': flux_grad,
            'positions': pos_grad,
            'spectral_indices': spectral_grad,
            'reference_frequency': ref_freq_grad
        }

    def V(self, nu_0: float, baseline: jnp.ndarray, params: dict = None) -> complex:
        """
        Analytical visibility function V for multiple point sources.
        
        For multiple point sources, the visibility is the coherent sum of
        individual source visibilities weighted by their flux densities.
        
        Uses log-sum-exp trick for numerical stability:
        Σᵢ Fᵢ * exp(iφᵢ) = max(Fᵢ) * Σᵢ exp(ln(Fᵢ/max(Fᵢ)) + iφᵢ)
        
        Parameters
        ----------
        nu_0 : float
            Central frequency in Hz. Determines the wavelength λ = c/ν₀.
        baseline : array_like, shape (3,)
            Baseline vector in meters [Bx, By, Bz]. Only perpendicular
            components (Bx, By) are used.
        params : dict, optional
            Source parameters. If None, uses current source parameters.
            
        Returns
        -------
        V : complex
            Complex visibility function as coherent sum of individual sources.
        """
        if params is None:
            params = self.get_params()

        # Physical constants
        c = 2.99792458e8  # Speed of light in m/s
        wavelength = c / nu_0
        
        # Extract perpendicular baseline components (ignore Bz)
        baseline_perp = baseline[:2]
        
        # Calculate flux for each source at the given frequency
        source_fluxes = params['flux_densities'] * (nu_0 / params['reference_frequency']) ** params['spectral_indices']
        
        # Calculate phases for each source position
        phases = 2 * jnp.pi * jnp.dot(params['positions'], baseline_perp) / wavelength
        
        # Find maximum flux for numerical stability
        max_flux = jnp.max(source_fluxes)
        
        # Use log-sum-exp trick for numerically stable computation
        # Σᵢ Fᵢ * exp(iφᵢ) = max(Fᵢ) * Σᵢ exp(ln(Fᵢ/max(Fᵢ)) + iφᵢ)
        def compute_visibility():
            # Calculate log ratios: ln(Fᵢ/max(Fᵢ))
            log_ratios = jnp.log(source_fluxes / max_flux)
            
            # Calculate stable exponentials: exp(ln(Fᵢ/max(Fᵢ)) + iφᵢ)
            stable_exponentials = jnp.exp(log_ratios + 1j * phases)
            
            # Sum the stable exponentials
            stable_sum = jnp.sum(stable_exponentials)
            
            # Calculate total flux for normalization
            total_flux = jnp.sum(source_fluxes)
            
            # Return normalized visibility: (max_flux * stable_sum) / total_flux
            return (max_flux * stable_sum) / total_flux
        
        def zero_visibility():
            return 0.0 + 0.0j
        
        # Use JAX conditional to handle zero flux case
        visibility = jax.lax.cond(max_flux == 0, zero_visibility, compute_visibility)
        
        return visibility
    
