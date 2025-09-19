"""
Source Model for a spatial grid of the intensity profile
=========================================================
"""

import numpy as np
from typing import Union
import sys
import os
import sncosmo
from pathlib import Path

from ..base import source
from scipy.interpolate import interp1d
from scipy.fft import fftshift, fftfreq


class GridSource(source.ChaoticSource):
    """
    Sedona model source for SN2011fe using numpy data files with FFT-based visibility calculation
    
    This class implements a spatially extended source model using Sedona simulation data
    with efficient FFT-based visibility calculations and caching.
    """
    
    def __init__(self, wavelength_grid: np.ndarray, flux_grid: np.ndarray,
                 B: float = 9.98, distance: float = 204379200000000.0, phi_B: float = 0.0):
        """
        Initialize Sedona SN2011fe source with wavelength and flux grids as parameters
        
        Parameters:
        -----------
        wavelength_grid : np.ndarray
            Wavelength grid in Angstrom, shape (n_wavelengths,)
        flux_grid : np.ndarray
            3D flux data in [erg/s/cm²/Å], shape (n_wavelengths, nx, ny)
        B : float
            Magnitude for flux normalization
        distance : float
            Distance to source in meters
        """

        # Store input grids directly as class parameters
        self.wavelength_grid = wavelength_grid  # [Angstrom]
        self.flux_data_3d = flux_grid  # [erg/s/cm²/Å] - 3D array
        self.B = B
        self.distance = distance
        self.phi_B = phi_B  # Position angle for baseline orientation (not used here
        self.cos_phi_B = np.cos(phi_B)
        self.sin_phi_B = np.sin(phi_B)  
        
        # Get spatial dimensions
        self.n_wavelengths, self.nx, self.ny = self.flux_data_3d.shape
        
        # Validate input dimensions
        if len(self.wavelength_grid) != self.n_wavelengths:
            raise ValueError(f"Wavelength grid length {len(self.wavelength_grid)} doesn't match flux grid wavelength dimension {self.n_wavelengths}")
        
        # normalize angular scale
        self.length_scale = 3200. * 20 * 24 * 3600  # Spatial scale in km/s per pixel * time since explosion (20 days)
        self.pixel_scale = self.length_scale / distance # radians per pixel

        # normalize flux scale
        flux_int = self.flux_data_3d.sum(axis=(1,2))
        spectrum = sncosmo.Spectrum(self.wavelength_grid, flux_int)
        spectrum_mag = spectrum.bandmag('bessellb', magsys='vega')
        self.flux_data_3d = self.flux_data_3d * 10**((spectrum_mag-B)/2.5) # now in units of  (erg / s / cm^2 / A) for B=12 mag
        
        # Convert wavelength to frequency
        c = 2.99792458e8  # m/s
        wavelength_m = self.wavelength_grid * 1e-10  # Convert Å to m
        self.frequency_grid = c / wavelength_m  # [Hz]
        
        # Calculate total flux spectrum by integrating over spatial dimensions
        self.total_flux_spectrum = np.sum(self.flux_data_3d, axis=(1, 2))  # [erg/s/cm²/Å]
        self.total_photon_spectrum = self.total_flux_spectrum * wavelength_m / (6.62607015e-34 * c)  # [photons/s/m²/Å]
        
        # Convert to SI units for flux density
        flux_si_per_wavelength = self.total_flux_spectrum * 1e-7 / 1e-4  # [W/m²/Å]
        flux_si_per_wavelength *= 1e-10  # [W/m²/m]
        
        # Convert to per frequency: F_ν = F_λ * λ²/c
        self.flux_density_grid = flux_si_per_wavelength * (wavelength_m**2) / c  # [W/m²/Hz]
        
        # Create interpolation function for flux density
        sort_indices = np.argsort(self.frequency_grid)
        freq_sorted = self.frequency_grid[sort_indices]
        flux_sorted = self.flux_density_grid[sort_indices]
        
        # Remove any duplicate frequencies
        unique_mask = np.diff(freq_sorted, prepend=freq_sorted[0]-1) > 0
        freq_unique = freq_sorted[unique_mask]
        flux_unique = flux_sorted[unique_mask]
        
        self.flux_interpolator = interp1d(
            freq_unique, flux_unique,
            kind='linear', bounds_error=False, fill_value=0.0
        )
        
        # Store frequency range for reference
        self.freq_min = np.min(freq_unique)
        self.freq_max = np.max(freq_unique)

        # Initialize functional FFT cache for intensity results only
        self._intensity_fft_cache = {}  # Cache only FFT intensity results by frequency
        
        # print(f"Loaded Sedona SN2011fe model:")
        # print(f"  Wavelength range: {np.min(self.wavelength_grid):.1f} - {np.max(self.wavelength_grid):.1f} Å")
        # print(f"  Frequency range: {self.freq_min:.2e} - {self.freq_max:.2e} Hz")
        # print(f"  Peak flux density: {np.max(self.flux_density_grid):.2e} W/m²/Hz")
        # print(f"  Spatial grid: {self.nx} × {self.ny}")
        # print(f"  Wavelength points: {self.n_wavelengths}")
    
    def intensity(self, nu: Union[float, np.ndarray], n_hat: np.ndarray) -> Union[float, np.ndarray]:
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
        # Handle scalar vs array frequency input
        if np.isscalar(nu):
            # Single frequency
            freq_idx = np.argmin(np.abs(self.frequency_grid - nu))
            
            # Get the 2D intensity map at this frequency
            intensity_map = self.flux_data_3d[freq_idx, :, :]  # [erg/s/cm²/Å]
            
            # Convert units to [W/m²/Hz/sr]
            wavelength_m = self.wavelength_grid[freq_idx] * 1e-10
            c = 2.99792458e8
            intensity_map_si = intensity_map * 1e-7 / 1e-4 * 1e-10 * (wavelength_m**2) / c
            
            # Convert to intensity per steradian
            pixel_scale = self.pixel_scale  # radians per pixel (adjustable)
            pixel_solid_angle = pixel_scale**2  # steradians per pixel
            intensity_map_si /= pixel_solid_angle
            
            return self._interpolate_intensity(intensity_map_si, n_hat, pixel_scale)
        else:
            # Array of frequencies
            nu_array = np.asarray(nu)
            if np.ndim(n_hat) == 1:
                # Single direction, multiple frequencies
                results = np.zeros_like(nu_array)
                for i, freq in enumerate(nu_array):
                    results[i] = self.intensity(freq, n_hat)
                return results
            else:
                # Multiple directions and frequencies - not typically used
                raise NotImplementedError("Multiple frequencies and directions not implemented")
    
    def _interpolate_intensity(self, intensity_map_si: np.ndarray, n_hat: np.ndarray, pixel_scale: float) -> Union[float, np.ndarray]:
        """
        Helper method to interpolate intensity from the 2D map
        
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
        if np.ndim(n_hat) == 1:

            # Single direction vector
            x_pixel = (n_hat[0]* self.cos_phi_B + n_hat[1] * self.sin_phi_B) / pixel_scale + self.nx // 2
            y_pixel = (-n_hat[0]* self.sin_phi_B + n_hat[1] * self.cos_phi_B) / pixel_scale + self.ny // 2
            
            # Check if within bounds
            if (0 <= x_pixel < self.nx and 0 <= y_pixel < self.ny):
                # Bilinear interpolation
                x0, x1 = int(x_pixel), min(int(x_pixel) + 1, self.nx - 1)
                y0, y1 = int(y_pixel), min(int(y_pixel) + 1, self.ny - 1)
                
                intensity = intensity_map_si[y0, x0] 

                # if interpolated
                # fx = x_pixel - int(x_pixel)
                # fy = y_pixel - int(y_pixel)
                       
                # intensity = (intensity_map_si[y0, x0] * (1 - fx) * (1 - fy) +
                #            intensity_map_si[y0, x1] * fx * (1 - fy) +
                #            intensity_map_si[y1, x0] * (1 - fx) * fy +
                #            intensity_map_si[y1, x1] * fx * fy)
                return intensity
            else:
                return 0.0
        else:
            # Multiple direction vectors
            intensities = np.zeros(n_hat.shape[0])
            for i, direction in enumerate(n_hat):
                x_pixel = direction[0] / pixel_scale + self.nx // 2
                y_pixel = direction[1] / pixel_scale + self.ny // 2
                
                if (0 <= x_pixel < self.nx and 0 <= y_pixel < self.ny):
                    x0, x1 = int(x_pixel), min(int(x_pixel) + 1, self.nx - 1)
                    y0, y1 = int(y_pixel), min(int(y_pixel) + 1, self.ny - 1)

                    intensities[i] = intensity_map_si[y0, x0]
                    
                    # fx = x_pixel - int(x_pixel)
                    # fy = y_pixel - int(y_pixel)
                    
                    # intensities[i] = (intensity_map_si[y0, x0] * (1 - fx) * (1 - fy) +
                    #                 intensity_map_si[y0, x1] * fx * (1 - fy) +
                    #                 intensity_map_si[y1, x0] * (1 - fx) * fy +
                    #                 intensity_map_si[y1, x1] * fx * fy)
                else:
                    intensities[i] = 0.0
            return intensities

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
        freq_idx = np.argmin(np.abs(self.frequency_grid - nu_0))
        
        # Check if FFT is already cached for this frequency
        if freq_idx not in self._intensity_fft_cache:
            self._intensity_fft_cache[freq_idx] = self._compute_intensity_fft(freq_idx)
        
        # Get cached FFT data
        intensity_fft = self._intensity_fft_cache[freq_idx]
        
        # Physical constants
        c = 2.99792458e8  # Speed of light in m/s
        wavelength = c / nu_0
        
        # Extract perpendicular baseline components (ignore Bz)
        baseline_perp = baseline[:2]
        
        # Convert baseline to spatial frequency coordinates
        u_freq = baseline_perp[0] / wavelength if len(baseline_perp) > 0 else 0.0
        v_freq = baseline_perp[1] / wavelength if len(baseline_perp) > 1 else 0.0
        
        # Calculate pixel scale from distance
        pixel_scale = self.length_scale / params['distance']  # radians per pixel
        
        # Compute spatial frequency coordinate grids dynamically
        u_coords_ = fftshift(fftfreq(self.nx, d=pixel_scale))  # cycles per radian
        v_coords_ = fftshift(fftfreq(self.ny, d=pixel_scale))  # cycles per radian
        u_coords = u_coords_*np.cos(params['phi_B']) + v_coords_*np.sin(params['phi_B'])
        v_coords = -u_coords_*np.sin(params['phi_B']) + v_coords_*np.cos(params['phi_B'])
        
        # Get FFT result at the closest spatial frequency coordinates
        return self._interpolate_fft_result(intensity_fft, u_freq, v_freq, u_coords, v_coords)
    
    def _compute_intensity_fft(self, freq_idx: int) -> np.ndarray:
        """
        Functional computation of FFT for a specific frequency using native spatial gridding.
        Returns only the FFT intensity result for caching.
        
        Parameters
        ----------
        freq_idx : int
            Index in the frequency grid
            
        Returns
        -------
        np.ndarray
            2D FFT of intensity map, properly normalized
        """
        from scipy.fft import fft2, fftshift
        
        # Get the 2D intensity map at this frequency
        intensity_map = self.flux_data_3d[freq_idx, :, :]  # [erg/s/cm²/Å]
        
        # Convert units to [W/m²/Hz/sr] for proper intensity
        wavelength_m = self.wavelength_grid[freq_idx] * 1e-10
        c = 2.99792458e8
        intensity_map_si = intensity_map * 1e-7 / 1e-4 * 1e-10 * (wavelength_m**2) / c
        
        # Convert to intensity per steradian using pixel solid angle
        pixel_solid_angle = self.pixel_scale**2  # steradians per pixel
        intensity_map_si /= pixel_solid_angle
        
        # Compute 2D FFT with proper shifting
        intensity_fft = fft2(intensity_map_si)
        intensity_fft = fftshift(intensity_fft)
        
        # Calculate total flux for normalization
        total_flux = np.sum(intensity_map_si) * pixel_solid_angle
        
        if total_flux > 0:
            # Proper normalization for discrete FFT to approximate continuous transform
            intensity_fft *= pixel_solid_angle / total_flux
        
        return intensity_fft
    
    def _interpolate_fft_result(self, intensity_fft: np.ndarray, u_target: float, v_target: float, u_coords, v_coords) -> complex:
        """
        Get FFT result at the closest spatial frequency coordinates by computing coordinates dynamically.
        
        Parameters
        ----------
        intensity_fft : np.ndarray
            2D FFT of intensity map
        u_target, v_target : float
            Target spatial frequency coordinates
        distance : float
            Distance to source in meters for pixel scale calculation
            
        Returns
        -------
        complex
            FFT value at closest coordinates
        """

        
        # Find the closest indices for u and v coordinates
        u_idx = np.argmin(np.abs(u_coords - u_target))
        v_idx = np.argmin(np.abs(v_coords - v_target))
        
        # Return the FFT value at the closest grid point
        return intensity_fft[v_idx, u_idx]

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
        
        Uses the integrated flux from the 3D data, compatible with
        the updated AbstractSource interface.
        
        Parameters
        ----------
        nu : float
            Frequency in Hz.
            
        Returns
        -------
        flux : float
            Total flux density in W m⁻² Hz⁻¹.
        """
        return self.flux_interpolator(nu)
    
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
            'peak_flux_density_w_m2_hz': np.max(self.flux_density_grid),
            'total_luminosity_estimate': np.trapezoid(self.flux_density_grid, self.frequency_grid),
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
        flux_data_3d = np.flip(np.load(flux_file), axis=0)  # [erg/s/cm²/Å] - 3D array
        
        return GridSource(wavelength_grid, flux_data_3d, B, distance)
    
    @staticmethod
    def getSN2011feSource(B: float = 9.98, distance: float = 204379200000000.0):
            # Get the current file's directory
            current_dir = Path(__file__).parent

            # Try to use real Sedona data first
            real_wave_file = os.path.join(current_dir, '../../data/WaveGrid.npy')
            real_flux_file = os.path.join(current_dir, '../../data/Phase0Flux.npy')

            return GridSource.create_grid_source_from_files(wave_grid_file=real_wave_file,
                                                    flux_file =real_flux_file,
                                                    B=B,  distance=distance)