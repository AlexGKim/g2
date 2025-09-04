"""
Core Intensity Interferometry Calculations
Based on "Probing H0 and resolving AGN disks with ultrafast photon counters" (Dalal et al. 2024)

This module implements the fundamental intensity interferometry equations for AGN observations,
including visibility functions, signal-to-noise calculations, and timing jitter effects.
"""

import numpy as np
import scipy.special as special
import scipy.integrate as integrate
from scipy.fft import fft, ifft, fftfreq
from typing import Tuple, Optional, Callable, Union
import astropy.units as u
import astropy.constants as const
from dataclasses import dataclass


@dataclass
class ObservationParameters:
    """Parameters for intensity interferometry observations"""
    central_frequency: float  # Hz
    bandwidth: float  # Hz
    observing_time: float  # seconds
    timing_jitter_rms: float  # seconds (σ_t)
    n_channels: int = 1  # Number of spectral channels
    
    @property
    def central_wavelength(self) -> float:
        """Central wavelength in meters"""
        return const.c.value / self.central_frequency
    
    @property
    def timing_jitter_fwhm(self) -> float:
        """Timing jitter FWHM from RMS"""
        return self.timing_jitter_rms * 2.35
    
    @property
    def coherence_time(self) -> float:
        """Coherence time ~ 1/Δν"""
        return 1.0 / self.bandwidth


class IntensityInterferometer:
    """
    Core intensity interferometry calculations
    
    Implements the theoretical framework from Dalal et al. 2024 for measuring
    visibility amplitudes |V|² using photon count correlations.
    """
    
    def __init__(self, obs_params: ObservationParameters):
        self.obs_params = obs_params
    
    def visibility_function(self, intensity_profile: Callable[[float, float], float],
                          baseline_length: float, baseline_angle: float = 0.0,
                          source_distance: float = 1.0) -> complex:
        """
        Calculate the complex visibility function V(B) for a given intensity profile
        
        Based on Equation (1) from the paper:
        V(ν,B,Δt) = ∫ I_ν(ν,n̂) exp(i[k·B - ωΔt]) d²n̂
        
        Parameters:
        -----------
        intensity_profile : callable
            Function I(x, y) giving intensity at sky coordinates (x, y) in radians
        baseline_length : float
            Baseline length in meters
        baseline_angle : float
            Baseline position angle in radians
        source_distance : float
            Angular diameter distance to source in meters
            
        Returns:
        --------
        visibility : complex
            Complex visibility V(B)
        """
        # Convert baseline to u,v coordinates
        u = baseline_length * np.cos(baseline_angle) / self.obs_params.central_wavelength
        v = baseline_length * np.sin(baseline_angle) / self.obs_params.central_wavelength
        
        # Integration limits (should cover the source extent)
        # For AGN, typical sizes are ~100 R_s ~ 100 GM/c² ~ 1 AU for 10⁸ M_sun
        # At 20 Mpc, this corresponds to ~1 μas
        max_angle = 10e-6  # 10 microarcseconds in radians
        
        def integrand_real(y, x):
            """Real part of the integrand"""
            phase = 2 * np.pi * (u * x + v * y)
            return intensity_profile(x, y) * np.cos(phase)
        
        def integrand_imag(y, x):
            """Imaginary part of the integrand"""
            phase = 2 * np.pi * (u * x + v * y)
            return intensity_profile(x, y) * np.sin(phase)
        
        # Perform 2D integration
        real_part, _ = integrate.dblquad(
            integrand_real, -max_angle, max_angle, -max_angle, max_angle
        )
        imag_part, _ = integrate.dblquad(
            integrand_imag, -max_angle, max_angle, -max_angle, max_angle
        )
        
        return complex(real_part, imag_part)
    
    def normalized_visibility(self, intensity_profile: Callable[[float, float], float],
                            baseline_length: float, baseline_angle: float = 0.0,
                            source_distance: float = 1.0) -> complex:
        """
        Calculate normalized visibility V̄ (Equation 2 from paper)
        
        V̄ = V / ∫ I d²n̂
        """
        visibility = self.visibility_function(intensity_profile, baseline_length, 
                                            baseline_angle, source_distance)
        
        # Calculate total flux for normalization
        max_angle = 10e-6  # Same integration limits as visibility
        total_flux, _ = integrate.dblquad(
            intensity_profile, -max_angle, max_angle, -max_angle, max_angle
        )
        
        return visibility / total_flux if total_flux > 0 else 0.0
    
    def photon_correlation_snr(self, photon_rate: float, visibility_amplitude: float) -> float:
        """
        Calculate signal-to-noise ratio for intensity correlations
        
        Based on Equations (13-14) from the paper:
        SNR = |V|² / σ|V|²
        σ⁻¹|V|² = (dΓ/dν)^(1/2) * (T_obs/σ_t)^(1/2) * (128π)^(-1/4)
        
        Parameters:
        -----------
        photon_rate : float
            Photon detection rate dΓ/dν in Hz/Hz
        visibility_amplitude : float
            Visibility amplitude |V|
            
        Returns:
        --------
        snr : float
            Signal-to-noise ratio
        """
        # Calculate σ⁻¹|V|² from Equation (14)
        sigma_inv_v2 = (
            np.sqrt(photon_rate) * 
            np.sqrt(self.obs_params.observing_time / self.obs_params.timing_jitter_rms) *
            (128 * np.pi)**(-0.25)
        )
        
        # Include spectroscopic enhancement
        if self.obs_params.n_channels > 1:
            sigma_inv_v2 *= np.sqrt(self.obs_params.n_channels)
        
        # Calculate SNR
        snr = visibility_amplitude**2 * sigma_inv_v2
        
        return snr
    
    def timing_jitter_convolution(self, visibility_time_series: np.ndarray,
                                 time_grid: np.ndarray) -> np.ndarray:
        """
        Apply timing jitter convolution to visibility measurements
        
        Based on Equations (4-5) from the paper, convolves |V|² with Gaussian
        timing jitter distribution.
        
        Parameters:
        -----------
        visibility_time_series : np.ndarray
            |V|² as function of time delay
        time_grid : np.ndarray
            Time delay grid in seconds
            
        Returns:
        --------
        convolved : np.ndarray
            Timing-jitter convolved correlation function
        """
        # Gaussian timing jitter kernel
        sigma_t = self.obs_params.timing_jitter_rms
        jitter_kernel = np.exp(-time_grid**2 / (4 * sigma_t**2)) / np.sqrt(4 * np.pi * sigma_t**2)
        
        # Convolve with timing jitter (for two detectors, use √2 σ_t)
        combined_sigma = np.sqrt(2) * sigma_t
        combined_kernel = np.exp(-time_grid**2 / (4 * combined_sigma**2)) / np.sqrt(4 * np.pi * combined_sigma**2)
        
        # Perform convolution
        convolved = np.convolve(visibility_time_series, combined_kernel, mode='same')
        
        return convolved
    
    def spectroscopic_enhancement(self, base_snr: float) -> float:
        """
        Calculate SNR enhancement from spectroscopy
        
        From the paper: SNR increases as √n_c for n_c independent channels
        """
        return base_snr * np.sqrt(self.obs_params.n_channels)
    
    def optimal_bandwidth_condition(self) -> bool:
        """
        Check if bandwidth satisfies optimal condition σ_t Δω >> 1
        
        From the paper, this condition ensures maximum SNR enhancement.
        """
        delta_omega = 2 * np.pi * self.obs_params.bandwidth
        return self.obs_params.timing_jitter_rms * delta_omega > 1.0
    
    def coherence_length(self) -> float:
        """Calculate coherence length c/Δν in meters"""
        return const.c.value / self.obs_params.bandwidth
    
    def fringe_visibility_contrast(self, visibility_amplitude: float) -> float:
        """
        Calculate fringe contrast for amplitude interferometry comparison
        
        This would be the contrast measured by traditional amplitude interferometry
        """
        return visibility_amplitude  # |V| sets the fringe contrast


class VisibilityCalculator:
    """
    Specialized calculator for different source geometries
    """
    
    @staticmethod
    def uniform_disk(radius: float, baseline_length: float, wavelength: float) -> float:
        """
        Visibility amplitude for uniform circular disk
        
        |V| = |2J₁(πθB/λ) / (πθB/λ)|
        
        Parameters:
        -----------
        radius : float
            Angular radius in radians
        baseline_length : float
            Baseline length in meters
        wavelength : float
            Wavelength in meters
            
        Returns:
        --------
        visibility : float
            Visibility amplitude |V|
        """
        # Calculate argument of Bessel function
        arg = np.pi * radius * baseline_length / wavelength
        
        if arg == 0:
            return 1.0
        else:
            return abs(2 * special.j1(arg) / arg)
    
    @staticmethod
    def gaussian_source(sigma: float, baseline_length: float, wavelength: float) -> float:
        """
        Visibility amplitude for Gaussian source
        
        |V| = exp(-2π²σ²B²/λ²)
        
        Parameters:
        -----------
        sigma : float
            Gaussian width (1σ) in radians
        baseline_length : float
            Baseline length in meters
        wavelength : float
            Wavelength in meters
            
        Returns:
        --------
        visibility : float
            Visibility amplitude |V|
        """
        arg = 2 * np.pi**2 * sigma**2 * baseline_length**2 / wavelength**2
        return np.exp(-arg)
    
    @staticmethod
    def ring_source(radius: float, width: float, baseline_length: float, 
                   wavelength: float) -> float:
        """
        Visibility amplitude for thin ring source
        
        Useful for modeling photon rings around black holes
        
        Parameters:
        -----------
        radius : float
            Ring radius in radians
        width : float
            Ring width in radians
        baseline_length : float
            Baseline length in meters
        wavelength : float
            Wavelength in meters
            
        Returns:
        --------
        visibility : float
            Visibility amplitude |V|
        """
        # For thin ring, use Bessel function J₀
        arg = 2 * np.pi * radius * baseline_length / wavelength
        
        if width > 0:
            # For finite width, integrate over ring
            # Simplified approximation for narrow rings
            return abs(special.j0(arg)) * np.exp(-np.pi**2 * width**2 * baseline_length**2 / wavelength**2)
        else:
            return abs(special.j0(arg))


def calculate_agn_photon_rate(magnitude: float, telescope_area: float,
                            wavelength: float = 550e-9, bandwidth: float = 100e-9,
                            efficiency: float = 0.8) -> float:
    """
    Calculate photon detection rate for AGN observations
    
    Parameters:
    -----------
    magnitude : float
        Apparent magnitude
    telescope_area : float
        Telescope collecting area in m²
    wavelength : float
        Central wavelength in meters
    bandwidth : float
        Observing bandwidth in meters
    efficiency : float
        Overall detection efficiency
        
    Returns:
    --------
    rate : float
        Photon detection rate in Hz
    """
    # Convert magnitude to flux density (using g=0 → 3730 Jy from paper)
    flux_jy = 3730 * 10**(-magnitude / 2.5)  # Jy
    flux_si = flux_jy * 1e-26  # W m⁻² Hz⁻¹
    
    # Convert to photon flux
    frequency = const.c.value / wavelength
    photon_energy = const.h.value * frequency
    photon_flux = flux_si / photon_energy  # photons m⁻² s⁻¹ Hz⁻¹
    
    # Total photon rate
    freq_bandwidth = const.c.value * bandwidth / wavelength**2
    rate = photon_flux * freq_bandwidth * telescope_area * efficiency
    
    return rate


def example_agn_observation():
    """
    Example calculation for AGN intensity interferometry
    Based on the numerical example from the paper (Section II)
    """
    print("AGN Intensity Interferometry Example")
    print("=" * 50)
    
    # Observation parameters from paper
    obs_params = ObservationParameters(
        central_frequency=const.c.value / 550e-9,  # 550 nm
        bandwidth=const.c.value * 100e-9 / (550e-9)**2,  # 100 nm bandwidth
        observing_time=1e5,  # 10⁵ seconds ≈ 28 hours
        timing_jitter_rms=13e-12,  # 30 ps FWHM → 13 ps RMS
        n_channels=5000  # 5000 spectral channels
    )
    
    # Create interferometer
    interferometer = IntensityInterferometer(obs_params)
    
    # AGN parameters from paper
    agn_magnitude = 12  # g = 12 mag
    telescope_area = 88.0  # m² (CTA MST)
    
    # Calculate photon rate
    photon_rate = calculate_agn_photon_rate(agn_magnitude, telescope_area)
    print(f"Photon rate per telescope: {photon_rate:.2e} Hz")
    
    # Calculate photon rate per unit frequency
    dgamma_dnu = photon_rate / obs_params.bandwidth
    print(f"dΓ/dν: {dgamma_dnu:.2e} Hz/Hz")
    
    # For unresolved source (|V| = 1)
    visibility_amplitude = 1.0
    snr = interferometer.photon_correlation_snr(dgamma_dnu, visibility_amplitude)
    print(f"SNR for unresolved source: {snr:.0f}")
    
    # Check against paper's prediction
    paper_snr = 1800  # From paper for n_c = 5000
    print(f"Paper prediction: {paper_snr}")
    print(f"Ratio: {snr/paper_snr:.2f}")
    
    # Check optimal bandwidth condition
    optimal = interferometer.optimal_bandwidth_condition()
    print(f"Optimal bandwidth condition (σ_t Δω >> 1): {optimal}")
    
    # Calculate some example visibilities
    print(f"\nExample visibility calculations:")
    baseline_length = 1000  # 1 km baseline
    
    # Uniform disk with 1 μas radius
    disk_radius = 1e-6 / 206265  # 1 μas in radians
    vis_disk = VisibilityCalculator.uniform_disk(
        disk_radius, baseline_length, obs_params.central_wavelength
    )
    print(f"1 μas uniform disk at 1 km baseline: |V| = {vis_disk:.3f}")
    
    # Gaussian source with 0.5 μas width
    gaussian_sigma = 0.5e-6 / 206265  # 0.5 μas in radians
    vis_gaussian = VisibilityCalculator.gaussian_source(
        gaussian_sigma, baseline_length, obs_params.central_wavelength
    )
    print(f"0.5 μas Gaussian at 1 km baseline: |V| = {vis_gaussian:.3f}")


if __name__ == "__main__":
    example_agn_observation()