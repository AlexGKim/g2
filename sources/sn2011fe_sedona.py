"""
Sedona SN2011fe Source Model for Intensity Interferometry

Implementation of ChaoticSource using Sedona model data
for SN2011fe with WaveGrid.npy and Phase0Flux.npy files.
"""

import numpy as np
from typing import Union
import sys
import os
import sncosmo

# Add parent directory to path to import source module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from source import ChaoticSource
from scipy.interpolate import interp1d

class SedonaSN2011feSource(ChaoticSource):
    """
    Sedona model source for SN2011fe using numpy data files
    
    This class implements a point source model where the intensity
    is concentrated at the origin with frequency-dependent flux
    from the Sedona simulation data.


    """
    
    def __init__(self, wave_grid_file: str = "../data/WaveGrid.npy",
                 flux_file: str = "../data/Phase0Flux.npy", B: float = 9.98, distance: float = 204379200000000.0 ):
        """
        Initialize Sedona SN2011fe source
        
        The underlying model is in physical units.  The flux and angular size are normalized to observed values.

        Parameters:
        -----------
        wave_grid_file : str
            Path to WaveGrid.npy file containing wavelength grid [Angstrom]
        flux_file : str
            Path to Phase0Flux.npy file containing 3D flux data [erg/s/cm²/Å]
            Shape: (n_wavelengths, nx, ny)
        """
        # Load the data files
        try:
            self.wavelength_grid = np.flip(np.load(wave_grid_file))  # [Angstrom]
            self.flux_data_3d = np.flip(np.load(flux_file),axis=0)  # [erg/s/cm²/Å] - 3D array

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not load Sedona data files: {e}")
        
        # Get spatial dimensions
        self.n_wavelengths, self.nx, self.ny = self.flux_data_3d.shape
        
        # normalize angular scale
        self.length_scale = 3200. * 20 * 24 * 3600  # Spatial scale in km/s per pixel * time since explosion (20 days) as per Nov 25 email from XingZhuo
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
        # The flux_data_3d is in [erg/s/cm²/Å] per pixel
        # Sum over spatial dimensions to get total flux density as in Figure 3 of II_Telescopes.pdf
        self.total_flux_spectrum = np.sum(self.flux_data_3d, axis=(1, 2))  # [erg/s/cm²/Å]
        
        # Convert to photon flux density per frequency to match Figure 3
        # First convert energy flux to photon flux: divide by photon energy E = hν = hc/λ
        h = 6.626e-34  # Planck constant [J⋅s]
        photon_energy = h * c / wavelength_m  # [J] per photon
        photon_energy_erg = photon_energy / 1e-7  # [erg] per photon
        
        # Convert from energy flux per wavelength to photon flux per wavelength
        photon_flux_per_wavelength = self.total_flux_spectrum / photon_energy_erg  # [photons/s/cm²/Å]
        
        # Convert from per wavelength to per frequency: n_ν = n_λ * |dλ/dν| = n_λ * λ²/c
        self.photon_flux_density_grid = photon_flux_per_wavelength * (wavelength_m**2) / c * 1e10  # [photons/s/cm²/Hz]
        
        # Also keep energy flux density for compatibility
        # Convert flux units from [erg/s/cm²/Å] to [W/m²/Hz]
        # 1 erg = 1e-7 J, 1 cm² = 1e-4 m²
        # F_ν = F_λ * λ²/c (to convert per wavelength to per frequency)
        flux_si_per_wavelength = self.total_flux_spectrum * 1e-7 / 1e-4  # [W/m²/Å]
        flux_si_per_wavelength *= 1e-10  # [W/m²/m]
        
        # Convert to per frequency: F_ν = F_λ * λ²/c
        self.flux_density_grid = flux_si_per_wavelength * (wavelength_m**2) / c  # [W/m²/Hz]
        
        # Create interpolation function for flux density
        # Sort by frequency (ascending order)
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
        
        print(f"Loaded Sedona SN2011fe model:")
        print(f"  Wavelength range: {np.min(self.wavelength_grid):.1f} - {np.max(self.wavelength_grid):.1f} Å")
        print(f"  Frequency range: {self.freq_min:.2e} - {self.freq_max:.2e} Hz")
        print(f"  Peak flux density: {np.max(self.flux_density_grid):.2e} W/m²/Hz")
        print(f"  Spatial grid: {self.nx} × {self.ny}")
        print(f"  Wavelength points: {self.n_wavelengths}")
    
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
            x_pixel = n_hat[0] / pixel_scale + self.nx // 2
            y_pixel = n_hat[1] / pixel_scale + self.ny // 2
            
            # Check if within bounds
            if (0 <= x_pixel < self.nx and 0 <= y_pixel < self.ny):
                # Bilinear interpolation
                x0, x1 = int(x_pixel), min(int(x_pixel) + 1, self.nx - 1)
                y0, y1 = int(y_pixel), min(int(y_pixel) + 1, self.ny - 1)
                
                fx = x_pixel - int(x_pixel)
                fy = y_pixel - int(y_pixel)
                
                intensity = (intensity_map_si[y0, x0] * (1 - fx) * (1 - fy) +
                           intensity_map_si[y0, x1] * fx * (1 - fy) +
                           intensity_map_si[y1, x0] * (1 - fx) * fy +
                           intensity_map_si[y1, x1] * fx * fy)
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
                    
                    fx = x_pixel - int(x_pixel)
                    fy = y_pixel - int(y_pixel)
                    
                    intensities[i] = (intensity_map_si[y0, x0] * (1 - fx) * (1 - fy) +
                                    intensity_map_si[y0, x1] * fx * (1 - fy) +
                                    intensity_map_si[y1, x0] * (1 - fx) * fy +
                                    intensity_map_si[y1, x1] * fx * fy)
                else:
                    intensities[i] = 0.0
            return intensities
    
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
            
        Notes
        -----
        For most implementations, this can be calculated analytically from the
        source parameters. The result should be consistent with numerical
        integration of the intensity() method over the sky.
        """
        return float(self.flux_interpolator(nu))
    
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

def test_sedona_source():
    """Test the Sedona SN2011fe source implementation"""
    
    print("Testing Sedona SN2011fe Source")
    print("=" * 40)
    
    try:
        # Create the source
        source = SedonaSN2011feSource()
        
        # Get spectrum info
        info = source.get_spectrum_info()
        print(f"\nSpectrum Information:")
        for key, value in info.items():
            if isinstance(value, tuple):
                print(f"  {key}: {value[0]:.2e} - {value[1]:.2e}")
            else:
                print(f"  {key}: {value:.2e}")
        
        # Test at a few frequencies
        test_frequencies = [5e14, 6e14, 7e14]  # Around optical
        print(f"\nFlux density at test frequencies:")
        for nu in test_frequencies:
            flux = source.total_flux(nu)
            wavelength_nm = 3e8 / nu * 1e9
            print(f"  ν = {nu:.1e} Hz ({wavelength_nm:.0f} nm): F_ν = {flux:.2e} W/m²/Hz")
        
        # Test intensity at origin
        nu_test = 5e14
        n_hat_origin = np.array([0.0, 0.0])
        n_hat_offset = np.array([1e-6, 1e-6])
        
        intensity_origin = source.intensity(nu_test, n_hat_origin)
        intensity_offset = source.intensity(nu_test, n_hat_offset)
        
        print(f"\nIntensity test at ν = {nu_test:.1e} Hz:")
        print(f"  At origin [0,0]: I = {intensity_origin:.2e} W/m²/Hz/sr")
        print(f"  At offset [1μas,1μas]: I = {intensity_offset:.2e} W/m²/Hz/sr")
        
        # Test g2_minus_one function (inherited from ChaoticSource)
        nu_0 = 5e14  # 600 nm
        delta_nu = 1e12  # 1 THz bandwidth
        delta_t = 1e-9  # 1 ns time lag
        
        g2_minus_one_value = source.g2_minus_one(delta_t, nu_0, delta_nu)
        print(f"\nSecond-order coherence function test:")
        print(f"  Parameters: Δt={delta_t*1e9} ns, ν₀={nu_0:.1e} Hz, Δν={delta_nu:.1e} Hz")
        print(f"  g²(Δt) - 1 = {g2_minus_one_value:.3f}")
        
        print(f"\n✅ Sedona SN2011fe source test completed successfully!")
        
        return source
        
    except Exception as e:
        print(f"❌ Error testing Sedona source: {e}")
        return None

if __name__ == "__main__":
    # Test the implementation
    source = test_sedona_source()
    
    if source is not None:
        print(f"\n📊 Source ready for intensity interferometry calculations!")
        print(f"Use this source with IntensityInterferometry class for visibility calculations.")