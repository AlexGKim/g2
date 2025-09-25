"""
Source Model for a spatial grid of the intensity profile
=========================================================
"""

import numpy as np
from typing import Union
import sys
import os
from pathlib import Path

from ..base import source
from scipy.interpolate import interp1d
from jax.numpy.fft import fftshift, fftfreq

from functools import partial
from jax import jit
from jax import numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

# @partial(jax.custom_jvp,  nondiff_argnums=())
def _compute_map_fft(intensity_map, wavelength_grid) -> np.ndarray:
        """
        Functional computation of FFT for a specific frequency using spatial gridding.
        Returns only the FFT intensity result for caching.
        
        Parameters
        ----------
        intensity_map : jnp.ndarray
            2D intensity map in [W m⁻² Hz⁻¹] (units don't matter for visibility)
        wavelength_grid : float
            Wavelength value (kept for compatibility)
            
        Returns
        -------
        np.ndarray
            2D FFT of intensity map, properly normalized
        """
        from jax.numpy.fft import fft2, fftshift
        
        # For visibility calculation, units don't matter since it's a normalized quantity
        # Just use the intensity_map directly and apply the same logic as before
        # pixel_solid_angle = pixel_scale**2  # steradians per pixel

        total_flux = jnp.sum(intensity_map)

        # Compute 2D FFT with proper shifting
        intensity_fft = fft2(intensity_map/total_flux)
        intensity_fft = fftshift(intensity_fft)
        
        # Calculate total flux for normalization
        # total_flux = jnp.sum(intensity_map)
        
        # Proper normalization: visibility should be normalized by total flux
        # so that V(0) = 1 (zero baseline gives unity visibility)
        # intensity_fft /= total_flux
        
        return intensity_fft

# @_compute_map_fft.defjvp
# def _compute_map_fft_jvp(primals, tangents):
#     """Custom JVP rule for J1"""
#     x, = primals
#     dx, = tangents
#     y = _compute_map_fft(x)
    
#     # Also need to wrap jv for the derivative
#     result_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
#     jv2 = pure_callback(
#         lambda x: jv(2, np.asarray(x)).astype(x.dtype),
#         result_shape,
#         x,
#         vmap_method='sequential'
#     )
    
#     dy = y/x - jv2
#     return y, dy * dx

# Removed standalone _interpolate_fft_result function - now a class method



class GridSource(source.ChaoticSource):
    """
    Sedona model source for SN2011fe using numpy data files with FFT-based visibility calculation
    
    This class implements a spatially extended source model using Sedona simulation data
    with efficient FFT-based visibility calculations and caching.
    """
    
    def __init__(self, wavelength_grid: np.ndarray, flux_grid: np.ndarray, pixel_scale_m: float,
                 B: float = 9.98, distance: float = 204379200000000.0, phi_B: float = 0.0):
        """
        Initialize Sedona SN2011fe source with wavelength and flux grids as parameters
        
        Parameters:
        -----------
        wavelength_grid : np.ndarray
            Wavelength grid in Angstrom, shape (n_wavelengths,)
        flux_grid : np.ndarray
            3D flux data in [erg/s/Å], shape (n_wavelengths, nx, ny)
        pixel_scale_m : float
            Pixel scale in m per pixel
        B : float
            Magnitude for flux normalization
        distance : float
            Distance to source in meters
        """
        c = 2.99792458e8  # m/s

        # Store input grids directly as class parameters
        self.wavelength_grid = jnp.array(wavelength_grid)  # [Angstrom]
        self.frequency_grid = c / (wavelength_grid * 1e-10)  # [Hz]
        self.pixel_scale_m = pixel_scale_m  # radians per pixel

        self.flux_grid = flux_grid  # [erg/s/Å] - 3D array
        
        # Convert flux_grid from [erg/s/Å] to [W Hz⁻¹] during initialization
        self.flux_grid_mks = flux_grid * 1e-7 * wavelength_grid[:,None,None]**2 / (c * 1e10) #Convert to [W/m²/Å] for internal use if needed
        
        # Convert from [W Hz^-1] to [W Hz^-1 m^-2 sr^-1] assuming isotropic emission
        self.intensity_data = jnp.array(self.flux_grid_mks / (4 * np.pi * pixel_scale_m**2))  # [W/m²/Hz] assuming isotropic emission

        # Calculate total flux spectrum by integrating over spatial dimensions
        # intensity_space is already in [W m⁻² Hz⁻¹], so sum gives total flux
        self.specific_flux = np.sum(self.intensity_data, axis=(1, 2))  # [W m⁻² Hz⁻¹]

        # Physical constants for unit conversion
        wavelength_m = wavelength_grid * 1e-10  # Convert Å to m
        
        # Convert units: [erg/s/cm²/Å] → [W m⁻² Hz⁻¹]
        # 1e-7: erg → J, 1e4: cm² → m², 1e-10: Å → m, (λ²/c): λ → ν
        conversion_factor = 1e-7 * 1e4 * 1e-10 * (wavelength_m[:, None, None]**2) / c
        self.intensity_space = jnp.array(self.flux_grid * conversion_factor)  # [W m⁻² Hz⁻¹] - 3D array
        

        # Store parameters
        self.B = B
        self.distance = distance
        self.phi_B = phi_B  # Position angle for baseline orientation (not used here

        # Precompute trigonometric values for baseline rotation
        self.cos_phi_B = jnp.cos(phi_B)
        self.sin_phi_B = jnp.sin(phi_B)
        
        # Get spatial dimensions
        self.n_wavelengths, self.nx, self.ny = self.intensity_space.shape
        
        # Validate input dimensions
        if len(self.wavelength_grid) != self.n_wavelengths:
            raise ValueError(f"Wavelength grid length {len(self.wavelength_grid)} doesn't match flux grid wavelength dimension {self.n_wavelengths}")
        
        # normalize flux scale
        # flux_int = self.flux_data_3d.sum(axis=(1,2))
        # spectrum = sncosmo.Spectrum(self.wavelength_grid, flux_int)
        # spectrum_mag = spectrum.bandmag('bessellb', magsys='vega')
        # self.flux_data_3d = self.flux_data_3d * 10**((spectrum_mag-B)/2.5) # now in units of  (erg / s / cm^2 / A) for B=12 mag
        
        # Convert wavelength to frequency (reuse wavelength_m from above)
        # self.frequency_grid = c / wavelength_m  # [Hz]
                
        # Keep old total_flux_spectrum for backward compatibility in plotting
        # Convert back to [erg/s/cm²/Å] for plotting method
        self.total_flux_spectrum = self.specific_flux * c / (wavelength_m**2) * 1e-10 * 1e4 / 1e-7  # [erg/s/cm²/Å]
        
        # Calculate total_photon_spectrum for backward compatibility
        # Convert from flux density [W m⁻² Hz⁻¹] to photon flux [photons/s/m²/Hz]
        h = 6.62607015e-34  # Planck constant
        self.specific_photon_flux = self.specific_flux / (h * self.frequency_grid)  # [photons/s/m²/Hz]
        
        # Create interpolation function for flux density
        sort_indices = np.argsort(self.frequency_grid)
        freq_sorted = self.frequency_grid[sort_indices]
        flux_sorted = self.specific_flux[sort_indices]
        
        # Remove any duplicate frequencies
        unique_mask = np.diff(freq_sorted, prepend=freq_sorted[0]-1) > 0
        freq_unique = freq_sorted[unique_mask]
        flux_unique = flux_sorted[unique_mask]
        
        # Store frequency range for reference
        self.freq_min = np.min(freq_unique)
        self.freq_max = np.max(freq_unique)

        # Initialize functional FFT cache for intensity results only
        self._intensity_fft_cache = {}  # Cache only FFT intensity results by frequency
        
        # Note: Frequency coordinates will be computed dynamically based on distance
        # since they depend on the angular pixel scale (radians per pixel)
        # which varies with distance parameter
        
        # print(f"Loaded Sedona SN2011fe model:")
        # print(f"  Wavelength range: {np.min(self.wavelength_grid):.1f} - {np.max(self.wavelength_grid):.1f} Å")
        # print(f"  Frequency range: {self.freq_min:.2e} - {self.freq_max:.2e} Hz")
        # print(f"  Peak flux density: {np.max(self.flux_density_grid):.2e} W/m²/Hz")
        # print(f"  Spatial grid: {self.nx} × {self.ny}")
        # print(f"  Wavelength points: {self.n_wavelengths}")
    
    def pixel_scale(self) -> float:

        """
        Get pixel scale in radians per pixel
        
        Returns
        -------
        float
            Pixel scale in radians per pixel
        """
        return self.pixel_scale_m / self.distance  # radians per pixel
    
    def intensity(self, nu: Union[float, np.ndarray], n_hat: np.ndarray, params=None) -> Union[float, np.ndarray]:
        """
        Calculate specific intensity I_nu(nu, n_hat)
        
        For SN2011fe, we use the 3D Sedona data to get spatially resolved intensity.
        This method is compatible with the updated AbstractSource interface.
        
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
        """

        if params is None:
            params = self.get_params()
        
        # Handle scalar vs array frequency input
        if jnp.isscalar(nu):
            # Single frequency
            freq_idx = np.argmin(np.abs(self.frequency_grid - nu))

            # Convert to intensity per steradian using pixel solid angle
            pixel_scale = self.pixel_scale_m/params['distance']  # radians per pixel
            # pixel_solid_angle = pixel_scale**2  # steradians per pixel

            # # Get the 2D intensity map at this frequency
            # intensity_map = (self.flux_grid_mks[freq_idx, :, :] 
            #                  / (4 * np.pi * params['distance']**2) / pixel_solid_angle ) # [W m⁻² Hz⁻¹]
            

            # intensity_map_si = intensity_map / pixel_solid_angle  # [W m⁻² Hz⁻¹ sr⁻¹]
            
            return self._interpolate_intensity(self.intensity_data[freq_idx,:,:], n_hat, pixel_scale)
        else:
            # Array of frequencies
            nu_array = np.asarray(nu)
            if np.ndim(n_hat) == 1:
                # Single direction, multiple frequencies
                results = np.zeros_like(nu_array)
                for i, freq in enumerate(nu_array):
                    results[i] = self.intensity(freq, n_hat, params=params)
                return results
            else:
                # Multiple directions and frequencies - not typically used
                raise NotImplementedError("Multiple frequencies and directions not implemented")
    
    def _transform_coordinates(self, coords: np.ndarray, params: dict, coord_type: str = 'direction') -> tuple:
        """
        Common coordinate transformation for both intensity and visibility calculations.
        
        Parameters
        ----------
        coords : ndarray
            Input coordinates (direction vectors for intensity, baseline for visibility)
        params : dict
            Source parameters including distance and phi_B
        coord_type : str
            Either 'direction' (for intensity) or 'baseline' (for visibility)
            
        Returns
        -------
        tuple
            (u_target, v_target) coordinates in the appropriate space
        """
        if coord_type == 'direction':
            # For intensity: convert direction to pixel coordinates
            pixel_scale = self.pixel_scale_m / params['distance']
            cos_phi_B = jnp.cos(params['phi_B'])
            sin_phi_B = jnp.sin(params['phi_B'])
            
            # Apply rotation and convert to pixel coordinates
            x_pixel = (coords[0] * cos_phi_B + coords[1] * sin_phi_B) / pixel_scale + self.nx // 2
            y_pixel = (-coords[0] * sin_phi_B + coords[1] * cos_phi_B) / pixel_scale + self.ny // 2
            
            return x_pixel, y_pixel
            
        elif coord_type == 'baseline':
            # For visibility: convert baseline to spatial frequency coordinates
            c = 2.99792458e8
            nu_0 = 5e14  # This should be passed as parameter, but for now use default
            wavelength = c / nu_0
            
            # Apply phi_B rotation to baseline
            cos_phi_B = jnp.cos(params['phi_B'])
            sin_phi_B = jnp.sin(params['phi_B'])
            baseline_rotated = jnp.array([
                coords[0] * cos_phi_B + coords[1] * sin_phi_B,
                -coords[0] * sin_phi_B + coords[1] * cos_phi_B
            ])
            
            # Convert to spatial frequency coordinates in cycles per meter
            u_freq_meters = baseline_rotated[0] / wavelength / params['distance']
            v_freq_meters = baseline_rotated[1] / wavelength / params['distance']
            
            return u_freq_meters, v_freq_meters
        else:
            raise ValueError(f"Unknown coord_type: {coord_type}")

    def _interpolate_grid(self, grid_data: jnp.ndarray, u_target: float, v_target: float,
                         coord_type: str = 'pixel') -> Union[float, complex]:
        """
        Common grid interpolation for both intensity and FFT data.
        
        Parameters
        ----------
        grid_data : jnp.ndarray
            2D grid data to interpolate from
        u_target, v_target : float
            Target coordinates
        coord_type : str
            Either 'pixel' (for intensity) or 'fft' (for visibility FFT)
            
        Returns
        -------
        Union[float, complex]
            Interpolated value
        """
        if coord_type == 'pixel':
            # For intensity: use pixel coordinates directly
            if (0 <= u_target < self.nx and 0 <= v_target < self.ny):
                x0, x1 = int(u_target), min(int(u_target) + 1, self.nx - 1)
                y0, y1 = int(v_target), min(int(v_target) + 1, self.ny - 1)
                return grid_data[y0, x0]
            else:
                return 0.0
                
        elif coord_type == 'fft':
            # For FFT: find closest grid point in spatial frequency space
            u_coords = fftshift(fftfreq(self.nx, d=self.pixel_scale_m))
            v_coords = fftshift(fftfreq(self.ny, d=self.pixel_scale_m))
            
            u_idx = jnp.argmin(jnp.abs(u_coords - u_target))
            v_idx = jnp.argmin(jnp.abs(v_coords - v_target))
            
            return grid_data[v_idx, u_idx]
        else:
            raise ValueError(f"Unknown coord_type: {coord_type}")

    def _interpolate_intensity(self, intensity_map_si: np.ndarray, n_hat: np.ndarray, pixel_scale: float) -> Union[float, np.ndarray]:
        """
        Helper method to interpolate intensity from the 2D map using common coordinate transformation.
        
        Parameters
        ----------
        intensity_map_si : ndarray
            2D intensity map in SI units
        n_hat : ndarray
            Direction vector(s)
        pixel_scale : float
            Scale factor for pixel coordinates
            
        Returns
        -------
        intensity : float or ndarray
            Interpolated intensity value(s)
        """
        # Use common transformation with pixel_scale embedded in params
        params = {'distance': self.pixel_scale_m / pixel_scale, 'phi_B': self.phi_B}
        
        if np.ndim(n_hat) == 1:
            # Single direction vector
            x_pixel, y_pixel = self._transform_coordinates(n_hat, params, 'direction')
            return self._interpolate_grid(intensity_map_si, x_pixel, y_pixel, 'pixel')
        else:
            # Multiple direction vectors
            intensities = np.zeros(n_hat.shape[0])
            for i, direction in enumerate(n_hat):
                x_pixel, y_pixel = self._transform_coordinates(direction, params, 'direction')
                intensities[i] = self._interpolate_grid(intensity_map_si, x_pixel, y_pixel, 'pixel')
            return intensities

    def _transform_coordinates_for_visibility(self, baseline_perp: np.ndarray, params: dict, nu_0: float) -> tuple:
        """
        Transform baseline coordinates to spatial frequency coordinates for visibility calculation.
        
        Parameters
        ----------
        baseline_perp : ndarray
            Perpendicular baseline components [Bx, By]
        params : dict
            Source parameters including distance and phi_B
        nu_0 : float
            Central frequency in Hz
            
        Returns
        -------
        tuple
            (u_freq_meters, v_freq_meters) in cycles per meter
        """
        # Physical constants
        c = 2.99792458e8
        wavelength = c / nu_0
        
        # Apply phi_B rotation to baseline
        cos_phi_B = jnp.cos(params['phi_B'])
        sin_phi_B = jnp.sin(params['phi_B'])
        baseline_rotated = jnp.array([
            baseline_perp[0] * cos_phi_B + baseline_perp[1] * sin_phi_B,
            -baseline_perp[0] * sin_phi_B + baseline_perp[1] * cos_phi_B
        ]) if len(baseline_perp) >= 2 else jnp.array([baseline_perp[0], 0.0])
        
        # Convert to spatial frequency coordinates in cycles per meter
        u_freq_meters = baseline_rotated[0] / wavelength / params['distance'] if len(baseline_rotated) > 0 else 0.0
        v_freq_meters = baseline_rotated[1] / wavelength / params['distance'] if len(baseline_rotated) > 1 else 0.0
        
        return u_freq_meters, v_freq_meters

    def V(self, nu_0: float, baseline: np.ndarray, params: dict = None) -> complex:
        """
        Calculate the spatial visibility function V using FFT with caching and interpolation.
        
        Uses the native spatial gridding of the flux_grid for FFT calculation,
        caches the result, and interpolates for specific baselines.
        
        Parameters
        ----------
        nu_0 : float
            Central frequency in Hz. Determines the wavelength λ₀ = c/ν₀.
        baseline : array_like, shape (3,)
            Baseline vector in meters [Bx, By, Bz]. Only the perpendicular
            components (Bx, By) are used in the calculation.
        params : dict, optional
            Additional parameters. Should contain 'distance' key for pixel scale calculation.
            If not provided, uses self.distance.
            
        Returns
        -------
        V : complex
            Normalized fringe visibility. The magnitude gives the visibility
            amplitude, and the phase gives the visibility phase.
        """

        if params is None:
            params = self.get_params()
        
        # Find the closest frequency index
        freq_idx = jnp.argmin(jnp.abs(self.frequency_grid - nu_0))
        
        # # Check if FFT is already cached for this frequency
        # if freq_idx not in self._intensity_fft_cache:
        #     self._intensity_fft_cache[freq_idx] = self._compute_intensity_fft(freq_idx)
        
        # # Get cached FFT data
        # intensity_fft = self._intensity_fft_cache[freq_idx]

        # Calculate pixel scale from distance (angular scale)
        pixel_scale = self.pixel_scale_m / params['distance']  # radians per pixel
        
        # Always compute FFT functionally using spatial gridding
        intensity_fft = _compute_map_fft(self.intensity_data[freq_idx, :, :],
                                                self.wavelength_grid[freq_idx])

        # Physical constants
        c = 2.99792458e8  # Speed of light in m/s
        wavelength = c / nu_0
        
        # Extract perpendicular baseline components (ignore Bz)
        baseline_perp = baseline[:2]
        
        # Use common coordinate transformation for visibility
        u_freq_meters, v_freq_meters = self._transform_coordinates_for_visibility(
            baseline_perp, params, nu_0)
        
        # Use common interpolation method
        return self._interpolate_grid(intensity_fft, u_freq_meters, v_freq_meters, 'fft')
    

    
    def V_squared_jacobian(self, nu_0: float, baseline: np.ndarray, params: dict = None):
        """
        Calculate the Jacobian of |V|² with respect to source parameters.
        
        This method uses JAX's automatic differentiation on the FFT computation,
        making the intensity data depend on the parameters through scaling.
        
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
            derivatives of |V|² with respect to each parameter.
        """
        if params is None:
            params = self.get_params()
        
        # Find the frequency index (use the same non-differentiable operation as V)
        freq_idx = jnp.argmin(jnp.abs(self.frequency_grid - nu_0))
        
        # Physical constants
        c = 2.99792458e8  # Speed of light in m/s
        wavelength = c / nu_0
        
        # Extract perpendicular baseline components
        baseline_perp = baseline[:2]
        
        # Define a function that computes |V|² with parameter dependence
        def V_squared_with_params(params_dict):
            # Make intensity data depend on parameters
            # Distance affects the angular scale, which we can model as a scaling of the intensity
            distance_scale = params_dict['distance'] / self.distance  # Relative distance change
            
            # Scale the intensity data based on distance (inverse square law for flux)
            scaled_intensity = self.intensity_data[freq_idx, :, :] / (distance_scale**2)
            
            # Apply magnitude scaling based on B parameter
            B_scale = 10**((self.B - params_dict['B']) / 2.5)  # Magnitude scaling
            scaled_intensity = scaled_intensity * B_scale
            
            # Compute FFT of the scaled intensity
            intensity_fft = _compute_map_fft(scaled_intensity, self.wavelength_grid[freq_idx])
            
            # Use common coordinate transformation (same as V method)
            u_freq_meters, v_freq_meters = self._transform_coordinates_for_visibility(
                baseline_perp, params_dict, nu_0)
            
            # Use common interpolation method (same as V method)
            fft_value = self._interpolate_grid(intensity_fft, u_freq_meters, v_freq_meters, 'fft')
            
            return jnp.abs(fft_value)**2
        
        # Use JAX to compute the gradient
        return jax.grad(V_squared_with_params)(params)

    def get_params(self) -> dict:
        """
        Get parameters that define the source model.
        
        Returns
        -------
        dict
            Dictionary containing source parameters
        """
        return {
            'B': self.B,
            'distance': self.distance,
            'phi_B': self.phi_B
        }
    
    def total_flux(self, nu: float) -> float:
        """
        Calculate total flux F_nu = ∫ I_nu d²n̂.
        
        Uses the pre-calculated total flux spectrum from intensity_space,
        compatible with the updated AbstractSource interface.
        
        Parameters
        ----------
        nu : float
            Frequency in Hz.
            
        Returns
        -------
        flux : float
            Total flux density in W m⁻² Hz⁻¹.
        """
        return jnp.interp(nu, self.frequency_grid, self.specific_flux)
    
    def get_spectrum_info(self):
        """
        Get information about the loaded spectrum
        
        Returns:
        --------
        dict : Dictionary with spectrum information
        """
        return {
            'wavelength_range_angstrom': (np.min(self.wavelength_grid), np.max(self.wavelength_grid)),
            'frequency_range_hz': (self.freq_min, self.freq_max),
            'peak_flux_density_w_m2_hz': np.max(self.specific_flux),
            'total_luminosity_estimate': np.trapezoid(self.specific_flux, self.frequency_grid),
            'spatial_grid': (self.nx, self.ny),
            'wavelength_points': self.n_wavelengths
        }
    
    def plot_spectrum(self, wavelength_units='angstrom'):
        """
        Plot the spectrum (requires matplotlib)
        
        Parameters:
        -----------
        wavelength_units : str
            Units for wavelength axis ('angstrom', 'nm', 'micron')
        """
        try:
            import matplotlib.pyplot as plt
            
            if wavelength_units == 'angstrom':
                wave_plot = self.wavelength_grid
                xlabel = 'Wavelength [Å]'
            elif wavelength_units == 'nm':
                wave_plot = self.wavelength_grid / 10
                xlabel = 'Wavelength [nm]'
            elif wavelength_units == 'micron':
                wave_plot = self.wavelength_grid / 10000
                xlabel = 'Wavelength [μm]'
            else:
                raise ValueError("wavelength_units must be 'angstrom', 'nm', or 'micron'")
            
            plt.figure(figsize=(10, 6))
            plt.plot(wave_plot, self.total_flux_spectrum, 'b-', linewidth=1)
            plt.xlabel(xlabel)
            plt.ylabel('Total Flux Density [erg/s/cm²/Å]')
            plt.title('Sedona SN2011fe Spectrum (Phase 0) - Spatially Integrated')
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            plt.show()
            
        except ImportError:
            print("matplotlib not available for plotting")

    @staticmethod
    def create_grid_source_from_files(wave_grid_file: str = "../data/WaveGrid.npy",
                                   flux_file: str = "../data/Phase0Flux.npy",
                                   B: float = 9.98, distance: float = 204379200000000.0) -> "GridSource":
        """
        Convenience factory function to create SedonaSN2011feSource from data files.
        
        This maintains backward compatibility with the old constructor interface.
        
        Parameters
        ----------
        wave_grid_file : str
            Path to WaveGrid.npy file containing wavelength grid [Angstrom]
        flux_file : str
            Path to Phase0Flux.npy file containing 3D flux data [erg/s/cm²/Å]
        B : float
            Magnitude for flux normalization
        distance : float
            Distance to source in meters
            
        Returns
        -------
        SedonaSN2011feSource
            Configured source instance
        """
        # Load the data files
        wavelength_grid = np.flip(np.load(wave_grid_file))  # [Angstrom]
        flux_grid = np.flip(np.load(flux_file), axis=0)  # [erg/s/cm²/Å] - 3D array
        # Check for duplicate values
        if len(wavelength_grid) != len(np.unique(wavelength_grid)):
            raise ValueError("Wavelength grid contains duplicate values")
        
        # Check for monotonically increasing values
        if not np.all(np.diff(wavelength_grid) > 0):
            raise ValueError("Wavelength grid is not monotonically increasing")
    
        return GridSource(wavelength_grid, flux_grid, B, distance)
    
    @staticmethod
    def getSN2011feSource(B: float = 9.98, distance: float = 204379200000000.0):
            # Get the current file's directory
            current_dir = Path(__file__).parent

            # Try to use real Sedona data first
            real_wave_file = os.path.join(current_dir, '../../data/WaveGrid.npy')
            real_flux_file = os.path.join(current_dir, '../../data/Phase0Flux.npy')

            wavelength_grid = np.flip(np.load(real_wave_file))  # [Angstrom]
            flux_grid = np.flip(np.load(real_flux_file), axis=0)  # [erg/s/cm²/Å] - 3D array

            pixel_scale_m = 3200. * 20 * 24 * 3600  # Spatial scale in km/s per pixel * time since explosion (20 days)

            return GridSource(wavelength_grid, flux_grid, pixel_scale_m, B=B,  distance=distance)