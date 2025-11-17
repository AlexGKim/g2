"""This class implements a source model using radial coordinates (impact parameter)
and applies the polar DFT algorithms from II.ipynb for efficient visibility
calculations.

The intensity data I_nu_p(lambda, p) has arbitrary normalization - it represents
the relative intensity profile as a function of wavelength and radial coordinate,
but does not need to be in physical units. The actual flux normalization is
provided separately via the specific_flux parameter.

Parameters:
- lambda: wavelength grid
- p: impact parameter (radial coordinate)
- I_nu_p: intensity profile with arbitrary normalization
- specific_flux: total flux density at each wavelength (physical units)
"""
class RadialSource(source.ChaoticSource):
    def __init__(self, specific_flux: np.ndarray, lambdas: np.ndarray,
                I_nu_p: np.ndarray, p_rays: np.ndarray, s: float = 1.0):
        """
        Initialize RadialGrid2 with flux, wavelength, intensity, and radial coordinate data.
        
        Parameters
        ----------
        specific_flux : np.ndarray
            Total flux density at each wavelength in W m⁻² Hz⁻¹, shape (n_wavelengths,)
        lambdas : np.ndarray
            Wavelength grid in Angstrom, shape (n_wavelengths,)
        I_nu_p : np.ndarray
            Intensity data with arbitrary normalization, shape (n_wavelengths, n_radial_points)
        p_rays : np.ndarray
            Impact parameter coordinates in radians, shape (n_radial_points,)
        s : float
            Size parameter (default: 1.0)
        """
        c = 2.99792458e8  # m/s
        
        # Store input data
        self.specific_flux_array = jnp.array(specific_flux)  # [W m⁻² Hz⁻¹]
        self.lambdas = jnp.array(lambdas)  # [Angstrom]
        self.I_nu_p = jnp.array(I_nu_p)    # [arbitrary normalization]
        self.p_rays = jnp.array(p_rays)    # [radians]
        
        # Calculate frequency grid
        self.frequency_grid = c / (self.lambdas * 1e-10)  # [Hz]
        
        # Store parameters
        self.s = s
        
        # Get dimensions
        self.n_wavelengths, self.n_radial_points = self.I_nu_p.shape
        
        # Validate input dimensions
        if len(self.specific_flux_array) != self.n_wavelengths:
            raise ValueError(f"Specific flux array length {len(self.specific_flux_array)} doesn't match number of wavelengths {self.n_wavelengths}")
        if len(self.lambdas) != self.n_wavelengths:
            raise ValueError(f"Wavelength grid length {len(self.lambdas)} doesn't match intensity wavelength dimension {self.n_wavelengths}")
        if len(self.p_rays) != self.n_radial_points:
            raise ValueError(f"Radial coordinate length {len(self.p_rays)} doesn't match intensity radial dimension {self.n_radial_points}")
        
        # Store frequency range for reference
        self.freq_min = jnp.min(self.frequency_grid)
        self.freq_max = jnp.max(self.frequency_grid)


    def intensity(self, nu: Union[float, np.ndarray], n_hat: np.ndarray, params=None) -> Union[float, np.ndarray]:
        """
        Calculate specific intensity I_nu(nu, n_hat) using radial interpolation.
        
        Parameters
        ----------
        nu : float or array_like
            Frequency in Hz
        n_hat : array_like, shape (2,)
            Direction vector on sky in radians
        params : dict, optional
            Source parameters
            
        Returns
        -------
        intensity : float or array_like
            Specific intensity in appropriate units
        """
        if params is None:
            params = self.get_params()
        
        # Convert direction to radial distance
        r = np.sqrt(n_hat[0]**2 + n_hat[1]**2)
        
        # Apply scaling parameter to radial coordinate
        # r is already in radians, scale by 's' parameter
        scaled_radial_coord = r * params.get('s', 1.0)
        
        # Handle scalar vs array frequency input
        if np.isscalar(nu):
            # Find closest frequency
            freq_idx = np.argmin(np.abs(self.frequency_grid - nu))
            intensity_profile = self.I_nu_p[freq_idx, :]
            
            # Interpolate at the requested radial coordinate
            return np.interp(scaled_radial_coord, self.p_rays, intensity_profile, right=0.)
        else:
            # Array of frequencies
            nu_array = np.asarray(nu)
            results = np.zeros_like(nu_array)
            for i, freq in enumerate(nu_array):
                results[i] = self.intensity(freq, n_hat, params=params)
            return results

    def specific_flux(self, nu: float) -> float:
        """
        Return the pre-computed specific flux at the given frequency.
        
        The specific flux values are provided during initialization and represent
        the total flux density F_nu = ∫ I_nu d²n̂ at each wavelength.
        
        Parameters
        ----------
        nu : float
            Frequency in Hz
            
        Returns
        -------
        flux : float
            Total flux density in W m⁻² Hz⁻¹
        """
        # Find closest frequency
        freq_idx = np.argmin(np.abs(self.frequency_grid - nu))
        return float(self.specific_flux_array[freq_idx])

    def V(self, nu_0: float, baseline: np.ndarray, params: dict = None) -> complex:
        """
        Calculate the spatial visibility function V using polar Fourier transform.
        
        This implements the Fourier transform of a 2D function in polar coordinates:
        V(u) = 2π ∫ I_nu_p(r) J_0(2π u r) r dr
        
        where the integral is computed using the trapezoidal rule.
        
        Parameters
        ----------
        nu_0 : float
            Central frequency in Hz
        baseline : array_like, shape (3,)
            Baseline vector in meters [Bx, By, Bz]
        params : dict, optional
            Source parameters
            
        Returns
        -------
        V : complex
            Complex fringe visibility
        """
        if params is None:
            params = self.get_params()
        
        # Find the closest frequency index
        freq_idx = jnp.argmin(jnp.abs(self.frequency_grid - nu_0))
        
        # Get intensity profile for this frequency
        intensity_profile = self.I_nu_p[freq_idx, :]
        
        # Convert baseline to spatial frequency
        c = 2.99792458e8
        wavelength = c / nu_0
        baseline_perp = baseline[:2]
        baseline_length = jnp.linalg.norm(baseline_perp)
        
        # Calculate spatial frequency u = |B|/λ
        u = baseline_length / wavelength
        
        # Apply size parameter scaling to the radial coordinates
        scaled_p_rays = self.p_rays / params.get('s', 1.0)
        
        # Calculate the polar Fourier transform:
        # V(u) = 2π ∫ I_nu_p(r) J_0(2π u r) r dr
        integrand = intensity_profile * _j0(2 * jnp.pi * u * scaled_p_rays) * scaled_p_rays
        integrand_0 = intensity_profile * scaled_p_rays
        
        # Use trapezoidal rule for integration
        visibility = jnp.trapezoid(integrand, scaled_p_rays) / jnp.trapezoid(integrand_0, scaled_p_rays)
        
        return visibility + 0.0j

    def get_params(self) -> Dict[str, Any]:
        """
        Get parameters that define the source model.
        
        Returns
        -------
        dict
            Dictionary containing source parameters
        """
        return {
            's': self.s  # Size parameter
        }

    def get_spectrum_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded spectrum.
        
        Returns
        -------
        dict
            Dictionary with spectrum information
        """
        return {
            'wavelength_range_angstrom': (np.min(self.lambdas), np.max(self.lambdas)),
            'frequency_range_hz': (self.freq_min, self.freq_max),
            'radial_range_rad': (np.min(self.p_rays), np.max(self.p_rays)),
            'wavelength_points': self.n_wavelengths,
            'radial_points': self.n_radial_points
        }