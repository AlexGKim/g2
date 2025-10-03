"""
Source Model for a spatial grid of the intensity profile
=========================================================
"""

import numpy as np
from typing import Union
import os
from pathlib import Path

from ..base import source
from jax.numpy.fft import fftshift, fftfreq
from jax import numpy as jnp
import jax

def _compute_map_fft(intensity_map) -> jnp.ndarray:
    """
    Compute 2D FFT of intensity map for visibility calculation.
    
    Parameters
    ----------
    intensity_map : jnp.ndarray
        2D intensity map in [W m⁻² Hz⁻¹]
    wavelength_grid : float
        Wavelength value (unused, kept for compatibility)
        
    Returns
    -------
    jnp.ndarray
        2D FFT of intensity map, normalized by total flux
    """
    from jax.numpy.fft import fft2, fftshift
    
    total_flux = jnp.sum(intensity_map)
    intensity_fft = fft2(intensity_map / total_flux)
    intensity_fft = fftshift(intensity_fft)
    
    return intensity_fft


class GridSource(source.ChaoticSource):
    """
    Spatially extended source model using Sedona simulation data.
    
    This class implements a spatially extended source model using Sedona simulation data
    with efficient FFT-based visibility calculations.
    """
    
    def __init__(self, wavelength_grid: np.ndarray, flux_grid: np.ndarray, pixel_scale_m: float,
                 distance: float = 204379200000000.0, phi_B: float = 0.0):
        """
        Initialize GridSource with wavelength and flux grids.
        
        Parameters
        ----------
        wavelength_grid : np.ndarray
            Wavelength grid in Angstrom, shape (n_wavelengths,)
        flux_grid : np.ndarray
            3D flux data in [erg/s/Å], shape (n_wavelengths, nx, ny)
        pixel_scale_m : float
            Pixel scale in meters per pixel
        distance : float
            Distance to source in meters
        phi_B : float
            Baseline orientation angle in radians
        """
        c = 2.99792458e8  # m/s

        # Store input grids
        self.wavelength_grid = wavelength_grid  # [Angstrom]
        self.frequency_grid = c / (wavelength_grid * 1e-10)  # [Hz]
        self.pixel_scale_m = pixel_scale_m
        self.flux_grid = flux_grid  # [erg/s/Å]
        
        # Convert flux_grid from [erg/s/Å] to [W Hz⁻¹]
        flux_grid_mks = flux_grid * 1e-7 * wavelength_grid[:,None,None]**2 / (c * 1e10)
        
        # Convert to intensity [W m⁻² Hz⁻¹ sr⁻¹]
        self.intensity_grid = jnp.array(flux_grid_mks / (4 * np.pi * pixel_scale_m**2))

        # Store intensity data for dynamic flux calculation
        # specific_flux will be calculated dynamically based on distance

        # Physical constants for unit conversion
        # wavelength_m = wavelength_grid * 1e-10  # Convert Å to m

        # Store parameters
        # self.B = B
        self.distance = distance
        self.phi_B = phi_B
        
        # Get spatial dimensions
        self.n_wavelengths, self.nx, self.ny = self.intensity_grid.shape
        
        # Validate input dimensions
        if len(self.wavelength_grid) != self.n_wavelengths:
            raise ValueError(f"Wavelength grid length {len(self.wavelength_grid)} doesn't match flux grid wavelength dimension {self.n_wavelengths}")
        
        # Calculate specific flux once for efficiency
        # specific_flux_values = self.specific_flux()
        
        # Keep old total_flux_spectrum for backward compatibility in plotting
        # self.total_flux_spectrum = specific_flux_values * c / (wavelength_m**2) * 1e-10 * 1e4 / 1e-7  # [erg/s/cm²/Å]
        
        # Calculate total_photon_spectrum for backward compatibility
        # h = 6.62607015e-34  # Planck constant
        # self.specific_photon_flux = specific_flux_values / (h * self.frequency_grid)  # [photons/s/m²/Hz]
        
        # Store frequency range for reference (using frequency grid directly)
        self.freq_min = np.min(self.frequency_grid)
        self.freq_max = np.max(self.frequency_grid)
    
    def pixel_scale(self) -> float:
        """
        Get pixel scale in radians per pixel.
        
        Returns
        -------
        float
            Pixel scale in radians per pixel
        """
        return self.pixel_scale_m / self.distance

    # def specific_photon_flux(self):
    #          # Calculate total_photon_spectrum for backward compatibility
    #     h = 6.62607015e-34  # Planck constant
    #     return   self.specific_flux() / (h * self.frequency_grid)  # [photons/s/m²/Hz]
    
    def intensity(self, nu: Union[float, np.ndarray], n_hat: np.ndarray, params=None) -> Union[float, np.ndarray]:
        """
        Calculate specific intensity I_nu(nu, n_hat).
        
        Parameters
        ----------
        nu : float or array_like
            Frequency in Hz
        n_hat : array_like, shape (2,) or (N, 2)
            Direction vector(s) on sky in radians
            
        Returns
        -------
        intensity : float or array_like
            Specific intensity in W m⁻² Hz⁻¹ sr⁻¹
        """
        if params is None:
            params = self.get_params()
        
        # Handle scalar vs array frequency input
        if jnp.isscalar(nu):
            # Single frequency
            freq_idx = np.argmin(np.abs(self.frequency_grid - nu))
            intensity_map = self.intensity_grid[freq_idx, :, :]
            
            # Transform coordinates and interpolate directly
            if np.ndim(n_hat) == 1:
                # Single direction vector
                x_pixel, y_pixel = self._transform_coordinates(n_hat, params, 'direction')
                return self._interpolate_grid(intensity_map, x_pixel, y_pixel, 'pixel')
            else:
                # Multiple direction vectors
                intensities = np.zeros(n_hat.shape[0])
                for i, direction in enumerate(n_hat):
                    x_pixel, y_pixel = self._transform_coordinates(direction, params, 'direction')
                    intensities[i] = self._interpolate_grid(intensity_map, x_pixel, y_pixel, 'pixel')
                return intensities
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
    
    def _transform_coordinates(self, coords: np.ndarray, params: dict, coord_type: str = 'direction', nu_0: float = None) -> tuple:
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
        nu_0 : float, optional
            Central frequency in Hz (required for baseline transformations)
            
        Returns
        -------
        tuple
            (u_target, v_target) coordinates in the appropriate space
        """
        if coord_type == 'direction':
            # For intensity: convert direction to pixel coordinates
            pixel_scale = self.pixel_scale_m/ (params['s']*self.distance) # params['pixel_scale_rad']
            cos_phi_B = jnp.cos(params['phi_B'])
            sin_phi_B = jnp.sin(params['phi_B'])
            
            # Apply rotation and convert to pixel coordinates
            x_pixel = (coords[0] * cos_phi_B + coords[1] * sin_phi_B) / pixel_scale + self.nx // 2
            y_pixel = (-coords[0] * sin_phi_B + coords[1] * cos_phi_B) / pixel_scale + self.ny // 2
            
            return x_pixel, y_pixel
            
        elif coord_type == 'baseline':
            # For visibility: convert baseline to spatial frequency coordinates
            if nu_0 is None:
                raise ValueError("nu_0 frequency parameter is required for baseline transformations")
                
            c = 2.99792458e8
            wavelength = c / nu_0
            
            # Apply phi_B rotation to baseline
            cos_phi_B = jnp.cos(params['phi_B'])
            sin_phi_B = jnp.sin(params['phi_B'])
            baseline_rotated = jnp.array([
                coords[0] * cos_phi_B + coords[1] * sin_phi_B,
                -coords[0] * sin_phi_B + coords[1] * cos_phi_B
            ]) / params['s']
            
            # Convert to spatial frequency coordinates in cycles per meter
            # distance = self.pixel_scale_m / params['pixel_scale_rad']
            distance = self.distance * params['s']

            u_freq_meters = baseline_rotated[0] / wavelength / distance
            v_freq_meters = baseline_rotated[1] / wavelength / distance
            
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
            # For intensity: use pixel coordinates directly (nearest neighbor)
            u_idx = jnp.round(u_target).astype(jnp.int32)
            v_idx = jnp.round(v_target).astype(jnp.int32)
            
            # Check bounds
            in_bounds = (u_idx >= 0) & (u_idx < self.nx) & (v_idx >= 0) & (v_idx < self.ny)
            
            # Clamp indices to valid range for array access
            u_idx_safe = jnp.clip(u_idx, 0, self.nx - 1)
            v_idx_safe = jnp.clip(v_idx, 0, self.ny - 1)
            
            value = grid_data[v_idx_safe, u_idx_safe]
            return jnp.where(in_bounds, value, 0.0)
            
        elif coord_type == 'fft':
            # For FFT: find closest grid point in spatial frequency space
            u_coords = fftshift(fftfreq(self.nx, d=self.pixel_scale_m))
            v_coords = fftshift(fftfreq(self.ny, d=self.pixel_scale_m))
            
            u_idx = jnp.argmin(jnp.abs(u_coords - u_target))
            v_idx = jnp.argmin(jnp.abs(v_coords - v_target))
            
            return grid_data[v_idx, u_idx]
        else:
            raise ValueError(f"Unknown coord_type: {coord_type}")



    def V(self, nu_0: float, baseline: np.ndarray, params: dict = None) -> complex:
        """
        Calculate the spatial visibility function V using FFT.
        
        Parameters
        ----------
        nu_0 : float
            Central frequency in Hz
        baseline : array_like, shape (3,)
            Baseline vector in meters [Bx, By, Bz]
        params : dict, optional
            Additional parameters. If not provided, uses self.get_params()
            
        Returns
        -------
        V : complex
            Normalized fringe visibility
        """
        if params is None:
            params = self.get_params()
        
        # Find the closest frequency index
        freq_idx = jnp.argmin(jnp.abs(self.frequency_grid - nu_0))
        
        # Always compute FFT functionally using spatial gridding
        intensity_fft = _compute_map_fft(self.intensity_grid[freq_idx, :, :])

        # Extract perpendicular baseline components (ignore Bz)
        baseline_perp = baseline[:2]
        
        # Use common coordinate transformation for visibility
        u_freq_meters, v_freq_meters = self._transform_coordinates(
            baseline_perp, params, 'baseline', nu_0)
        
        # Use common interpolation method
        return self._interpolate_grid(intensity_fft, u_freq_meters, v_freq_meters, 'fft')

    def _V_squared_jacobian_grid(self, nu_0: float, params: dict = None):
        """
        Calculate the Jacobian of |V|² with respect to source parameters using the sedona algorithm.
        
        This implementation follows the algorithm from sedona.ipynb that calculates Fisher matrix
        elements F00, F01, F02 using gamma2 (which corresponds to V_squared).
        
        Parameters
        ----------
        nu_0 : float
            Central frequency in Hz
        baseline : array_like, shape (3,)
            Baseline vector in meters [Bx, By, Bz]
        params : dict, optional
            Source parameters. If None, uses current source parameters
            
        Returns
        -------
        jacobian : dict
            Dictionary with same keys as params, containing the partial
            derivatives of |V|² with respect to each parameter
        """
        if params is None:
            params = self.get_params()
        
        # Find the frequency index
        freq_idx = jnp.argmin(jnp.abs(self.frequency_grid - nu_0))
        
        # Get normalized flux data (following sedona algorithm)
        flux_norm = self.intensity_grid[freq_idx, :, :]
        flux_norm = flux_norm / jnp.sum(flux_norm)
        
        # Create padded array (without padding as requested)
        paddedarray = flux_norm
        
        # Calculate gamma (FFT of normalized intensity)
        gamma = jax.numpy.fft.fft2(paddedarray)
        
        # Calculate coordinate grids for derivatives (following sedona algorithm)
        # theta represents pixel coordinates relative to center
        theta_x = jnp.arange(paddedarray.shape[1]) - paddedarray.shape[1] // 2 + 0.5
        theta_y = jnp.arange(paddedarray.shape[0]) - paddedarray.shape[0] // 2 + 0.5
        
        # Calculate derivatives of gamma with respect to u and v coordinates
        # Following sedona: dgammau = -2j*pi * fft2(paddedarray * theta[:,None])
        dgammau = -2j * jnp.pi * jax.numpy.fft.fft2(paddedarray * theta_y[:, None])
        dgammav = -2j * jnp.pi * jax.numpy.fft.fft2(paddedarray * theta_x[None, :])
        
        # Get frequency coordinates for spatial frequency space
        u = jax.numpy.fft.fftfreq(paddedarray.shape[0])
        v = jax.numpy.fft.fftfreq(paddedarray.shape[1])
        
        # Calculate derivatives in parameter space (following sedona algorithm)
        # dgammas corresponds to derivative with respect to size parameter
        # dgammaphi corresponds to derivative with respect to rotation parameter
        dgammas = -(u[:, None] * dgammau + v[None, :] * dgammav) / params['s']
        dgammaphi = -v[None, :] * dgammau + u[:, None] * dgammav

        jacobian_grid = {}

        if 's' in params:
            # Use F00 for size parameter derivative (dgammas corresponds to size changes)
            jacobian_grid['s'] = jnp.conjugate(gamma) * dgammas + gamma * jnp.conjugate(dgammas)
        
        # For phi_B parameter (corresponds to rotation parameter "phi" in sedona)
        if 'phi_B' in params:
            # Use F11 for rotation parameter derivative (dgammaphi corresponds to rotation changes)
            jacobian_grid['phi_B'] = jnp.conjugate(gamma) * dgammaphi + gamma * jnp.conjugate(dgammaphi)
        
        return jacobian_grid
        
    def V_squared_jacobian(self, nu_0: float, baseline: np.ndarray, params: dict = None):
        """
        Calculate the Jacobian of |V|² with respect to source parameters using the sedona algorithm.
        
        This implementation follows the algorithm from sedona.ipynb that calculates Fisher matrix
        elements F00, F01, F02 using gamma2 (which corresponds to V_squared).
        
        Parameters
        ----------
        nu_0 : float
            Central frequency in Hz
        baseline : array_like, shape (3,)
            Baseline vector in meters [Bx, By, Bz]
        params : dict, optional
            Source parameters. If None, uses current source parameters
            
        Returns
        -------
        jacobian : dict
            Dictionary with same keys as params, containing the partial
            derivatives of |V|² with respect to each parameter
        """
        if params is None:
            params = self.get_params()
        
        # Find the frequency index
        # freq_idx = jnp.argmin(jnp.abs(self.frequency_grid - nu_0))
        
        # Extract perpendicular baseline components
    
        
        # # Get normalized flux data (following sedona algorithm)
        # flux_norm = self.intensity_grid[freq_idx, :, :]
        # flux_norm = flux_norm / jnp.sum(flux_norm)
        
        # # Create padded array (without padding as requested)
        # paddedarray = flux_norm
        
        # # Calculate gamma (FFT of normalized intensity)
        # gamma = jax.numpy.fft.fft2(paddedarray)
        
        # # Calculate coordinate grids for derivatives (following sedona algorithm)
        # # theta represents pixel coordinates relative to center
        # theta_x = jnp.arange(paddedarray.shape[1]) - paddedarray.shape[1] // 2 + 0.5
        # theta_y = jnp.arange(paddedarray.shape[0]) - paddedarray.shape[0] // 2 + 0.5
        
        # # Calculate derivatives of gamma with respect to u and v coordinates
        # # Following sedona: dgammau = -2j*pi * fft2(paddedarray * theta[:,None])
        # dgammau = -2j * jnp.pi * jax.numpy.fft.fft2(paddedarray * theta_y[:, None])
        # dgammav = -2j * jnp.pi * jax.numpy.fft.fft2(paddedarray * theta_x[None, :])
        
        # # Get frequency coordinates for spatial frequency space
        # u = jax.numpy.fft.fftfreq(paddedarray.shape[0])
        # v = jax.numpy.fft.fftfreq(paddedarray.shape[1])
        
        # # Calculate derivatives in parameter space (following sedona algorithm)
        # # dgammas corresponds to derivative with respect to size parameter
        # # dgammaphi corresponds to derivative with respect to rotation parameter
        # dgammas = -(u[:, None] * dgammau + v[None, :] * dgammav) / params['s']
        # dgammaphi = -v[None, :] * dgammau + u[:, None] * dgammav

        # jacobian_grid = {}

        # if 's' in params:
        #     # Use F00 for size parameter derivative (dgammas corresponds to size changes)
        #     jacobian_grid['s'] = jnp.conjugate(gamma) * dgammas + gamma * jnp.conjugate(dgammas)
        
        # # For phi_B parameter (corresponds to rotation parameter "phi" in sedona)
        # if 'phi_B' in params:
        #     # Use F11 for rotation parameter derivative (dgammaphi corresponds to rotation changes)
        #     jacobian_grid['phi_B'] = jnp.conjugate(gamma) * dgammaphi + gamma * jnp.conjugate(dgammaphi)

        jacobian_grid = self._V_squared_jacobian_grid(nu_0, params)
        
        # # Calculate Fisher matrix elements (following sedona algorithm)
        # F00 = jnp.abs(2 * gamma.conjugate() * dgammas) ** 2
        # F01 = jnp.abs(2 * gamma.conjugate() * dgammas) * jnp.abs(2 * gamma.conjugate() * dgammaphi)
        # F11 = jnp.abs(2 * gamma.conjugate() * dgammaphi) ** 2
        
        # # Apply fftshift to center the arrays
        # F00 = jnp.fft.fftshift(F00)
        # F01 = jnp.fft.fftshift(F01)
        # F11 = jnp.fft.fftshift(F11)
        baseline_perp = baseline[:2]
        # Use common coordinate transformation and interpolation methods
        u_target, v_target = self._transform_coordinates(baseline_perp, params, 'baseline', nu_0)
        
        # # Get Fisher matrix values at this point
        # F00_at_point = F00[v_idx, u_idx]
        # F01_at_point = F01[v_idx, u_idx]
        # F11_at_point = F11[v_idx, u_idx]
        
        # Calculate Jacobian using Fisher matrix elements
        # Based on sedona algorithm: F00 corresponds to size parameter, F11 to rotation parameter
        jacobian = {}
        
        # For pixel_scale_rad parameter (corresponds to size parameter "s" in sedona)
        if 's' in params:
            # Use F00 for size parameter derivative (dgammas corresponds to size changes)
            jacobian['s'] = self._interpolate_grid(jacobian_grid['s'], u_target, v_target, 'fft')
        
        # For phi_B parameter (corresponds to rotation parameter "phi" in sedona)
        if 'phi_B' in params:
            # Use F11 for rotation parameter derivative (dgammaphi corresponds to rotation changes)
            jacobian['phi_B'] = self._interpolate_grid(jacobian_grid['phi_B'], u_target, v_target, 'fft')
        
        return jacobian
    


    def get_params(self) -> dict:
        """
        Get parameters that define the source model.
        
        Returns
        -------
        dict
            Dictionary containing source parameters
        """
        return {
            # 'pixel_scale_rad':  self.pixel_scale_m / self.distance,
            's': 1.,
            'phi_B': self.phi_B
        }
    
    def _specific_flux_grid(self, params: dict = None) -> np.ndarray:
        """
        Calculate specific flux spectrum accounting for distance-dependent pixel scale.
        
        Parameters
        ----------
        params : dict, optional
            Source parameters. If None, uses current source parameters
            
        Returns
        -------
        flux : np.ndarray
            Specific flux spectrum in W m⁻² Hz⁻¹
        """
        if params is None:
            params = self.get_params()
        
        # Calculate pixel scale in steradians based on current parameters
        pixel_scale_rad = self.pixel_scale_m / (params['s'] * self.distance)
        # params['pixel_scale_rad']  # radians per pixel
        pixel_area_sr = pixel_scale_rad**2  # steradians per pixel
        
        # Sum intensity over spatial dimensions and multiply by pixel area
        # intensity_data is in [W m⁻² Hz⁻¹ sr⁻¹], so multiplying by sr gives [W m⁻² Hz⁻¹]
        flux_spectrum = np.sum(self.intensity_grid, axis=(1, 2)) * pixel_area_sr
        
        return flux_spectrum

    def specific_flux(self, nu: float) -> float:
        """
        Calculate total flux F_nu = ∫ I_nu d²n̂.
        
        Parameters
        ----------
        nu : float
            Frequency in Hz
            
        Returns
        -------
        flux : float
            Total flux density in W m⁻² Hz⁻¹
        """
        flux_spectrum = self._specific_flux_grid()
        return jnp.interp(nu, self.frequency_grid, flux_spectrum)
    
    def get_spectrum_info(self):
        """
        Get information about the loaded spectrum.
        
        Returns
        -------
        dict
            Dictionary with spectrum information
        """
        return {
            'wavelength_range_angstrom': (np.min(self.wavelength_grid), np.max(self.wavelength_grid)),
            'frequency_range_hz': (self.freq_min, self.freq_max),
            'peak_flux_density_w_m2_hz': np.max(self._specific_flux_grid()),
            'total_luminosity_estimate': np.trapezoid(self._specific_flux_grid(), self.frequency_grid),
            'spatial_grid': (self.nx, self.ny),
            'wavelength_points': self.n_wavelengths
        }
    
    def plot_spectrum(self, wavelength_units='angstrom'):
        """
        Plot the spectrum (requires matplotlib).
        
        Parameters
        ----------
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
            plt.plot(wave_plot, self.flux_grid.sum(index(1,2)), 'b-', linewidth=1)
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
                                pixel_scale_m: float = None,
                                wavelength_scale: float = None,
                                distance: float = 204379200000000.0,
                                phi_B: float = 0.0, padfactor=1) -> "GridSource":
        """
        Convenience factory function to create GridSource from data files.
        
        Parameters
        ----------
        wave_grid_file : str
            Path to WaveGrid.npy file containing wavelength grid [Angstrom]
        flux_file : str
            Path to Phase0Flux.npy file containing 3D flux data [erg/s/cm²/Å]
        pixel_scale_m : float, optional
            Pixel scale in meters per pixel. If None, defaults to SN2011fe scale
        B : float
            Magnitude for flux normalization
        distance : float
            Distance to source in meters
        phi_B : float
            Baseline orientation angle in radians
            
        Returns
        -------
        GridSource
            Configured source instance
        """
        # Load the data files
        wavelength_grid = np.flip(np.load(wave_grid_file))  # [Angstrom]
        flux_grid = np.flip(np.load(flux_file), axis=0)  # [erg/s/cm²] - 3D array
        flux_grid = flux_grid/wavelength_scale # [erg/s/cm²/Å] - 3D array
        
        # Check for duplicate values
        if len(wavelength_grid) != len(np.unique(wavelength_grid)):
            raise ValueError("Wavelength grid contains duplicate values")
        
        # Check for monotonically increasing values
        if not np.all(np.diff(wavelength_grid) > 0):
            raise ValueError("Wavelength grid is not monotonically increasing")
        
        # Default pixel scale for SN2011fe if not provided
        if pixel_scale_m is None:
            pixel_scale_m = 3200. * 20 * 24 * 3600  # Spatial scale in km/s per pixel * time since explosion (20 days)


        # Padded grid
        if padfactor > 1:
            pad_x = (padfactor - 1) * flux_grid.shape[1] // 2
            pad_y = (padfactor - 1) * flux_grid.shape[2] // 2
            flux_grid = np.pad(flux_grid, ((0,0),(pad_x,pad_x),(pad_y,pad_y)), mode='constant', constant_values=0)
            # pixel_scale_m = pixel_scale_m / padfactor  # Adjust pixel scale accordingly

        return GridSource(wavelength_grid, flux_grid, pixel_scale_m, distance=distance, phi_B=phi_B)

    @staticmethod
    def getSN2011feSource(B: float = 9.98, distance: float = 204379200000000.0,
                                phi_B: float = 0.0):
        """
        Create a GridSource instance for SN2011fe using default data files.
        
        Parameters
        ----------
        distance : float
            Distance to source in meters
            
        Returns
        -------
        GridSource
            Configured SN2011fe source instance
        """
        # Get the current file's directory
        current_dir = Path(__file__).parent
        
        # Use real Sedona data
        wave_grid_file = os.path.join(current_dir, '../../data/WaveGrid.npy')
        flux_file = os.path.join(current_dir, '../../data/Phase0Flux.npy')
        
        # SN2011fe specific pixel scale
        pixel_scale_m = 3200 * 3600 * 24 * 20 * 1000 # patial scale in m/s per pixel * time since explosion (20 days)
        wavelength_scale = 200

        # Call the general factory method
        return GridSource.create_grid_source_from_files(
            wave_grid_file=wave_grid_file,
            flux_file=flux_file,
            pixel_scale_m=pixel_scale_m,
            wavelength_scale = 200,
            distance=distance,
            phi_B=phi_B,
            padfactor=4
        )