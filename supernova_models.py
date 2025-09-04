"""
Supernova Models for Intensity Interferometry
Based on intensity interferometry techniques and existing supernova analysis

This module implements supernova ejecta models for intensity interferometry
observations, building on the existing supernova analysis framework.
"""

import numpy as np
import scipy.special as special
import scipy.integrate as integrate
from scipy.fft import fft2, ifft2, fftshift
from typing import Tuple, Optional, Callable, List, Dict
import astropy.units as u
import astropy.constants as const
import astropy.cosmology
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class SupernovaParameters:
    """Parameters for supernova models"""
    sn_type: str  # "Ia", "II", "Ib", "Ic", etc.
    explosion_time: float  # Days since explosion
    expansion_velocity: float  # km/s
    distance: float  # Mpc
    absolute_magnitude: float  # Absolute magnitude
    ejecta_mass: float = 1.4  # Solar masses
    explosion_energy: float = 1e51  # erg
    
    @property
    def angular_radius(self) -> float:
        """Angular radius of photosphere in radians"""
        # Physical radius = v * t
        radius_km = self.expansion_velocity * self.explosion_time * 24 * 3600
        radius_m = radius_km * 1000
        
        # Angular radius
        distance_m = self.distance * 3.086e22  # Mpc to meters
        return radius_m / distance_m
    
    @property
    def angular_radius_microarcsec(self) -> float:
        """Angular radius in microarcseconds"""
        return self.angular_radius * (180 / np.pi) * 3600 * 1e6


class SupernovaEjecta:
    """
    Supernova ejecta model for intensity interferometry
    
    Models the expanding photosphere and ejecta structure based on
    the existing supernova analysis and intensity interferometry theory.
    """
    
    def __init__(self, params: SupernovaParameters):
        self.params = params
        
        # Load supernova type-specific parameters from existing code
        self.type_parameters = self._get_type_parameters()
    
    def _get_type_parameters(self) -> Dict:
        """Get type-specific parameters from existing supernova analysis"""
        # Based on the existing main.py supernova parameters
        type_params = {
            "Ia": {
                "rate": 2.43e-5,  # SNe/yr/Mpc^3/h^3_70
                "abs_mag": -19.46 + 5 * np.log10(70/60),
                "velocity": 1e4,  # km/s
                "timescale": 18,  # days
            },
            "II": {
                "rate": 9.1e-5 * 0.649,
                "abs_mag": -15.97,
                "velocity": 1e4,  # km/s
                "timescale": 25,  # days
            },
            "Ib": {
                "rate": 9.1e-5 * 0.108,
                "abs_mag": -18.26,
                "velocity": 1e4,  # km/s
                "timescale": 25,  # days
            },
            "Ic": {
                "rate": 9.1e-5 * 0.075,
                "abs_mag": -17.44,
                "velocity": 1e4,  # km/s
                "timescale": 25,  # days
            }
        }
        
        return type_params.get(self.params.sn_type, type_params["Ia"])
    
    def photosphere_intensity_profile(self, radius_norm: float) -> float:
        """
        Intensity profile of supernova photosphere
        
        Parameters:
        -----------
        radius_norm : float
            Normalized radius (r/R_photosphere)
            
        Returns:
        --------
        intensity : float
            Normalized intensity
        """
        if radius_norm > 1.0:
            return 0.0
        
        # Simple limb-darkening model
        # I(μ) = I₀(1 - u + u*μ) where μ = cos(θ) = √(1 - r²)
        limb_darkening = 0.6  # Typical value for supernovae
        
        if radius_norm < 1.0:
            mu = np.sqrt(1 - radius_norm**2)
            intensity = 1 - limb_darkening + limb_darkening * mu
        else:
            intensity = 0.0
        
        return intensity
    
    def ejecta_density_profile(self, radius_norm: float) -> float:
        """
        Density profile of supernova ejecta
        
        Based on typical supernova ejecta models
        
        Parameters:
        -----------
        radius_norm : float
            Normalized radius (r/R_max)
            
        Returns:
        --------
        density : float
            Normalized density
        """
        if radius_norm <= 0 or radius_norm > 1.0:
            return 0.0
        
        # Power-law density profile typical for SN ejecta
        # ρ(r) ∝ r^(-n) for homologous expansion
        if self.params.sn_type == "Ia":
            # Type Ia: steeper profile
            power_index = 7.0
        else:
            # Core-collapse: shallower profile
            power_index = 5.0
        
        return radius_norm**(-power_index)
    
    def polarization_profile(self, radius_norm: float, azimuth: float) -> float:
        """
        Polarization profile for aspherical ejecta
        
        Based on the existing gamma() function that includes polarization
        
        Parameters:
        -----------
        radius_norm : float
            Normalized radius
        azimuth : float
            Azimuthal angle in radians
            
        Returns:
        --------
        polarization : float
            Polarization fraction
        """
        if radius_norm > 1.0:
            return 0.0
        
        # Based on the Pz function from existing code
        rmax = 2.25  # From existing code
        if radius_norm < rmax:
            y2 = rmax**2 - radius_norm**2
            cos2 = y2 / rmax**2
            pz = (1 - cos2) / (1 + cos2)
            
            # Intensity with polarization (from existing intensity function)
            intensity = 0.5 * (1 - pz) + pz * np.cos(azimuth)**2
        else:
            intensity = 0.0
        
        return intensity
    
    def intensity_2d(self, x: float, y: float, include_polarization: bool = True) -> float:
        """
        2D intensity profile in sky coordinates
        
        Parameters:
        -----------
        x, y : float
            Sky coordinates in units of photosphere radius
        include_polarization : bool
            Whether to include polarization effects
            
        Returns:
        --------
        intensity : float
            Intensity at position (x, y)
        """
        radius_norm = np.sqrt(x**2 + y**2)
        
        if radius_norm > 1.0:
            return 0.0
        
        if include_polarization:
            azimuth = np.arctan2(y, x)
            return self.polarization_profile(radius_norm, azimuth)
        else:
            return self.photosphere_intensity_profile(radius_norm)
    
    def visibility_amplitude(self, baseline_length: float, wavelength: float = 550e-9,
                           include_polarization: bool = True) -> float:
        """
        Calculate visibility amplitude for supernova
        
        Parameters:
        -----------
        baseline_length : float
            Baseline length in meters
        wavelength : float
            Observing wavelength in meters
        include_polarization : bool
            Whether to include polarization effects
            
        Returns:
        --------
        visibility : float
            Visibility amplitude |V|
        """
        # Convert baseline to angular units
        angular_baseline = baseline_length / wavelength  # radians^-1
        
        # Argument for visibility calculation
        # For uniform disk: |V| = |2J₁(πθB/λ) / (πθB/λ)|
        arg = np.pi * self.params.angular_radius * angular_baseline
        
        if arg == 0:
            return 1.0
        
        if include_polarization:
            # Use the gamma^2 calculation from existing code
            # This includes the polarization effects
            
            # Create 2D intensity map
            nbin = 101
            u_range = np.linspace(-2, 2, nbin)
            v_range = u_range
            
            intensity_map = np.zeros((nbin, nbin))
            for i, u_val in enumerate(u_range):
                for j, v_val in enumerate(v_range):
                    intensity_map[i, j] = self.intensity_2d(u_val, v_val, True)
            
            # Normalize
            intensity_map = intensity_map / intensity_map.sum()
            
            # Calculate visibility via FFT
            gamma = fft2(intensity_map)
            gamma2 = np.abs(gamma)**2
            
            # Extract visibility at the appropriate spatial frequency
            # This is a simplified extraction - full implementation would
            # interpolate to the exact baseline
            center = nbin // 2
            freq_index = min(int(arg * nbin / (4 * np.pi)), nbin // 2 - 1)
            
            if freq_index < center:
                visibility = np.sqrt(gamma2[center, center + freq_index])
            else:
                visibility = 0.0
        else:
            # Simple uniform disk with limb darkening
            visibility = abs(2 * special.j1(arg) / arg)
        
        return visibility
    
    def calculate_snr_for_observation(self, telescope_array, observing_time: float,
                                    wavelength: float = 550e-9) -> float:
        """
        Calculate SNR for supernova intensity interferometry observation
        
        Parameters:
        -----------
        telescope_array : TelescopeArray
            Telescope array configuration
        observing_time : float
            Observing time in seconds
        wavelength : float
            Observing wavelength in meters
            
        Returns:
        --------
        snr : float
            Signal-to-noise ratio
        """
        # Calculate apparent magnitude
        distance_modulus = 5 * np.log10(self.params.distance * 1e6 / 10)
        apparent_mag = self.params.absolute_magnitude + distance_modulus
        
        # Calculate photon rate per telescope
        photon_rate = telescope_array.photon_rate_per_telescope(apparent_mag, wavelength)
        
        # Calculate visibility amplitude
        baseline_length = telescope_array.max_baseline
        visibility = self.visibility_amplitude(baseline_length, wavelength)
        
        # Calculate SNR using intensity interferometry formula
        from intensity_interferometry import IntensityInterferometer, ObservationParameters
        
        obs_params = ObservationParameters(
            central_frequency=const.c.value / wavelength,
            bandwidth=const.c.value * 100e-9 / wavelength**2,  # 100 nm bandwidth
            observing_time=observing_time,
            timing_jitter_rms=telescope_array.telescopes[0].timing_jitter_rms,
            n_channels=1000  # Moderate spectroscopy
        )
        
        interferometer = IntensityInterferometer(obs_params)
        
        # Calculate dΓ/dν
        dgamma_dnu = photon_rate / obs_params.bandwidth
        
        snr = interferometer.photon_correlation_snr(dgamma_dnu, visibility)
        
        return snr
    
    def plot_intensity_profile(self, include_polarization: bool = True,
                             figsize: Tuple[float, float] = (12, 5)):
        """Plot 2D intensity profile of supernova"""
        # Create coordinate grid
        extent = 1.5  # Plot extent in units of photosphere radius
        n_points = 201
        x = np.linspace(-extent, extent, n_points)
        y = np.linspace(-extent, extent, n_points)
        X, Y = np.meshgrid(x, y)
        
        # Calculate intensity
        intensity = np.zeros_like(X)
        for i in range(n_points):
            for j in range(n_points):
                intensity[i, j] = self.intensity_2d(X[i, j], Y[i, j], include_polarization)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 2D intensity map
        im1 = ax1.imshow(intensity, extent=[-extent, extent, -extent, extent],
                        origin='lower', cmap='hot')
        ax1.set_xlabel('x (R_photosphere)')
        ax1.set_ylabel('y (R_photosphere)')
        title = f'SN {self.params.sn_type} Intensity'
        if include_polarization:
            title += ' (with polarization)'
        ax1.set_title(title)
        plt.colorbar(im1, ax=ax1, label='Normalized Intensity')
        
        # Radial profile
        radii = np.linspace(0, 1.5, 100)
        if include_polarization:
            # Average over azimuth
            azimuths = np.linspace(0, 2*np.pi, 20)
            radial_profile = np.zeros(len(radii))
            for i, r in enumerate(radii):
                intensities = [self.intensity_2d(r * np.cos(phi), r * np.sin(phi), True) 
                             for phi in azimuths]
                radial_profile[i] = np.mean(intensities)
        else:
            radial_profile = [self.photosphere_intensity_profile(r) for r in radii]
        
        ax2.plot(radii, radial_profile, 'b-', linewidth=2)
        ax2.axvline(1.0, color='red', linestyle='--', label='Photosphere edge')
        ax2.set_xlabel('Radius (R_photosphere)')
        ax2.set_ylabel('Normalized Intensity')
        ax2.set_title('Radial Intensity Profile')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    def plot_visibility_curve(self, baseline_range: Tuple[float, float] = (100, 10000),
                            wavelength: float = 550e-9, n_points: int = 50,
                            figsize: Tuple[float, float] = (10, 6)):
        """Plot visibility amplitude vs baseline length"""
        baselines = np.logspace(np.log10(baseline_range[0]), 
                               np.log10(baseline_range[1]), n_points)
        
        # Calculate visibilities with and without polarization
        vis_with_pol = []
        vis_without_pol = []
        
        for baseline in baselines:
            vis_with = self.visibility_amplitude(baseline, wavelength, True)
            vis_without = self.visibility_amplitude(baseline, wavelength, False)
            vis_with_pol.append(vis_with)
            vis_without_pol.append(vis_without)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.loglog(baselines / 1000, vis_with_pol, 'b-', linewidth=2, 
                 label=f'SN {self.params.sn_type} (with polarization)')
        ax.loglog(baselines / 1000, vis_without_pol, 'r--', linewidth=2,
                 label=f'SN {self.params.sn_type} (uniform disk)')
        
        ax.set_xlabel('Baseline Length (km)')
        ax.set_ylabel('Visibility Amplitude |V|')
        ax.set_title(f'Supernova Visibility vs Baseline\n'
                    f't = {self.params.explosion_time:.1f} days, '
                    f'θ = {self.params.angular_radius_microarcsec:.1f} μas')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add resolution scale
        resolution_baseline = wavelength / self.params.angular_radius
        ax.axvline(resolution_baseline / 1000, color='green', linestyle=':', 
                  label=f'Resolution scale ({resolution_baseline/1000:.1f} km)')
        ax.legend()
        
        plt.tight_layout()
        return fig, ax


def create_supernova_sample() -> List[SupernovaEjecta]:
    """Create a sample of different supernova types for analysis"""
    supernovae = []
    
    # Type Ia at different epochs
    for t in [5, 10, 20]:
        sn_ia = SupernovaEjecta(SupernovaParameters(
            sn_type="Ia",
            explosion_time=t,
            expansion_velocity=10000,  # km/s
            distance=20.0,  # Mpc
            absolute_magnitude=-19.46
        ))
        supernovae.append(sn_ia)
    
    # Core-collapse types
    for sn_type, abs_mag in [("II", -15.97), ("Ib", -18.26), ("Ic", -17.44)]:
        sn_cc = SupernovaEjecta(SupernovaParameters(
            sn_type=sn_type,
            explosion_time=15,
            expansion_velocity=8000,  # km/s (slightly slower than Ia)
            distance=20.0,  # Mpc
            absolute_magnitude=abs_mag
        ))
        supernovae.append(sn_cc)
    
    return supernovae


def example_supernova_interferometry():
    """Example supernova intensity interferometry calculations"""
    print("Supernova Intensity Interferometry Example")
    print("=" * 50)
    
    # Create a Type Ia supernova
    sn_ia = SupernovaEjecta(SupernovaParameters(
        sn_type="Ia",
        explosion_time=10,  # 10 days post-explosion
        expansion_velocity=10000,  # km/s
        distance=20.0,  # Mpc
        absolute_magnitude=-19.46
    ))
    
    print(f"Supernova Type: {sn_ia.params.sn_type}")
    print(f"Time since explosion: {sn_ia.params.explosion_time} days")
    print(f"Expansion velocity: {sn_ia.params.expansion_velocity} km/s")
    print(f"Distance: {sn_ia.params.distance} Mpc")
    print(f"Angular radius: {sn_ia.params.angular_radius_microarcsec:.1f} μas")
    
    # Calculate visibility at different baselines
    baselines = [1000, 5000, 10000]  # meters
    wavelength = 550e-9
    
    print(f"\nVisibility amplitudes at λ = {wavelength*1e9:.0f} nm:")
    for baseline in baselines:
        vis = sn_ia.visibility_amplitude(baseline, wavelength)
        print(f"  {baseline/1000:.0f} km baseline: |V| = {vis:.3f}")
    
    # Calculate SNR for observation
    from telescope_arrays import TelescopeArray
    array = TelescopeArray.cta_south_mst_like()
    
    observing_time = 3600  # 1 hour
    snr = sn_ia.calculate_snr_for_observation(array, observing_time, wavelength)
    print(f"\nSNR for 1-hour observation with CTA-like array: {snr:.1f}")


if __name__ == "__main__":
    example_supernova_interferometry()
    
    # Create and plot example supernova
    sn = SupernovaEjecta(SupernovaParameters(
        sn_type="Ia",
        explosion_time=10,
        expansion_velocity=10000,
        distance=20.0,
        absolute_magnitude=-19.46
    ))
    
    # Plot intensity profile
    fig1, axes1 = sn.plot_intensity_profile()
    plt.savefig('supernova_intensity.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot visibility curve
    fig2, ax2 = sn.plot_visibility_curve()
    plt.savefig('supernova_visibility.png', dpi=150, bbox_inches='tight')
    plt.show()