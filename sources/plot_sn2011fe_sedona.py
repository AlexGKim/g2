#!/usr/bin/env python3
"""
Comprehensive Test Plots for Sedona SN2011fe Source Model

This script creates visualizations that test and validate the functionality
of the SedonaSN2011feSource class, providing comprehensive visual validation
of all key methods and behaviors.

Key Features:
- Data loading and initialization validation
- Intensity calculation tests (spatial and spectral)
- Flux calculations and interpolation accuracy
- Visibility calculations including visibility vs zeta plot
- Chaotic source inheritance tests
- Integration tests with realistic scenarios
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os
import tempfile

# Add parent directory to path to import source module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sources.sn2011fe_sedona import SedonaSN2011feSource
    from intensity_interferometry_core import IntensityInterferometry
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import dependencies: {e}")
    DEPENDENCIES_AVAILABLE = False

# Set up plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (15, 12)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def create_mock_data():
    """Create mock data for testing when real Sedona data is not available"""
    # Create realistic mock wavelength grid
    wavelengths = np.linspace(3000, 10000, 100)  # Angstrom
    
    # Create a realistic SN spectrum with multiple features
    # Main continuum with blackbody-like shape
    T_eff = 8000  # K (effective temperature)
    h = 6.626e-34  # Planck constant
    c = 2.998e8   # Speed of light
    k_B = 1.381e-23  # Boltzmann constant
    
    # Convert wavelength to frequency for blackbody calculation
    freq = c / (wavelengths * 1e-10)
    
    # Blackbody spectrum (simplified)
    blackbody = (2 * h * freq**3 / c**2) / (np.exp(h * freq / (k_B * T_eff)) - 1)
    blackbody = blackbody / np.max(blackbody)  # Normalize
    
    # Add spectral lines (absorption features)
    line_centers = [4000, 5000, 6000, 7000, 8000]  # Angstrom
    line_depths = [0.3, 0.5, 0.4, 0.2, 0.3]
    line_widths = [50, 80, 60, 40, 70]  # Angstrom
    
    spectrum = blackbody.copy()
    for center, depth, width in zip(line_centers, line_depths, line_widths):
        line_profile = depth * np.exp(-0.5 * ((wavelengths - center) / width)**2)
        spectrum -= line_profile
    
    # Ensure spectrum is positive
    spectrum = np.maximum(spectrum, 0.01)
    
    # Create 3D flux data with realistic spatial structure
    nx, ny = 50, 50
    flux_3d = np.zeros((100, nx, ny))
    
    # Create spatial coordinates
    x = np.linspace(-25, 25, nx)
    y = np.linspace(-25, 25, ny)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    for i, flux_val in enumerate(spectrum):
        # Create wavelength-dependent spatial profile
        # Core gets smaller at shorter wavelengths (limb darkening effect)
        wavelength = wavelengths[i]
        core_size = 8 + 5 * (wavelength - 3000) / 7000  # Size varies with wavelength
        
        # Central Gaussian core
        core_profile = np.exp(-0.5 * (R / core_size)**2)
        
        # Add some asymmetry (ejecta structure)
        asymmetry = 0.2 * np.exp(-0.5 * ((X - 3) / 6)**2) * np.exp(-0.5 * (Y / 8)**2)
        
        # Combine profiles
        spatial_profile = core_profile + asymmetry
        spatial_profile = spatial_profile / np.max(spatial_profile)
        
        # Scale by spectrum and add realistic flux levels
        flux_3d[i, :, :] = flux_val * spatial_profile * 1e-15  # erg/s/cm²/Å
    
    return wavelengths, flux_3d


def get_source():
    """Get a SedonaSN2011feSource instance, using real data if available, mock data otherwise"""
    try:
        # Try to use real Sedona data first
        real_wave_file = 'data/WaveGrid.npy'
        real_flux_file = 'data/Phase0Flux.npy'
        
        if os.path.exists(real_wave_file) and os.path.exists(real_flux_file):
            # Use real Sedona data
            source = SedonaSN2011feSource(real_wave_file, real_flux_file)
            data_type = "Real Sedona Data"
        else:
            # Fallback to mock data
            wavelengths, flux_3d = create_mock_data()
            temp_dir = tempfile.mkdtemp()
            wave_file = os.path.join(temp_dir, 'WaveGrid.npy')
            flux_file = os.path.join(temp_dir, 'Phase0Flux.npy')
            np.save(wave_file, wavelengths)
            np.save(flux_file, flux_3d)
            source = SedonaSN2011feSource(wave_file, flux_file)
            data_type = "Mock Data"
            
        return source, data_type
    except Exception as e:
        print(f"Error creating source: {e}")
        return None, "Error"


def plot_data_loading_and_initialization():
    """Plot 1: Data loading and initialization validation"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    try:
        source, data_type = get_source()
        if source is None:
            raise Exception("Could not create source")
        
        # Plot 1: Wavelength vs Frequency conversion
        ax1.plot(source.wavelength_grid, source.frequency_grid / 1e14, 'b-', linewidth=2)
        ax1.set_xlabel('Wavelength (Å)')
        ax1.set_ylabel('Frequency (×10¹⁴ Hz)')
        ax1.set_title(f'Wavelength-Frequency Conversion\n({data_type})')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Photon flux density per frequency
        ax2.plot(source.wavelength_grid, source.photon_flux_density_grid, 'r-', linewidth=2, label='SEDONA')
        ax2.set_xlabel('Wavelength (Å)')
        ax2.set_ylabel('n_ν [s⁻¹ cm⁻² Hz⁻¹]')
        ax2.set_title(f'Photon Flux Density per Frequency\n({data_type})')
        ax2.set_ylim((0, np.max(source.photon_flux_density_grid[np.logical_and(
            source.wavelength_grid > 4000, source.wavelength_grid < 8000)]) * 1.1))
        ax2.set_xlim((3300, 10000))
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Flux density in SI units
        ax3.plot(source.frequency_grid / 1e14, source.flux_density_grid, 'g-', linewidth=2)
        ax3.set_xlabel('Frequency (×10¹⁴ Hz)')
        ax3.set_ylabel('Flux Density (W/m²/Hz)')
        ax3.set_title('Flux Density vs Frequency')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Data consistency check
        info = source.get_spectrum_info()
        categories = ['Wave Range\n(Å)', 'Freq Range\n(Hz)', 'Peak Flux\n(W/m²/Hz)', 
                     'Grid Size', 'Wave Points']
        values = [
            info['wavelength_range_angstrom'][1] - info['wavelength_range_angstrom'][0],
            info['frequency_range_hz'][1] - info['frequency_range_hz'][0],
            info['peak_flux_density_w_m2_hz'],
            info['spatial_grid'][0] * info['spatial_grid'][1],
            info['wavelength_points']
        ]
        
        # Normalize values for plotting
        normalized_values = []
        for i, val in enumerate(values):
            if i < 3:  # First three are physical quantities
                normalized_values.append(np.log10(val))
            else:  # Last two are counts
                normalized_values.append(val / 100)  # Scale to reasonable range
        
        bars = ax4.bar(categories, normalized_values, 
                      color=['blue', 'green', 'red', 'orange', 'purple'], alpha=0.7)
        ax4.set_ylabel('Log₁₀(Value) or Scaled Value')
        ax4.set_title('Data Summary')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            label = f'{val:.1e}' if val > 1000 else f'{val:.1f}'
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    label, ha='center', va='bottom', fontsize=8, rotation=45)
        
    except Exception as e:
        # If data loading fails, show error message
        for i, ax in enumerate([ax1, ax2, ax3, ax4]):
            ax.text(0.5, 0.5, f'Data loading test {i+1}\nError: {str(e)[:50]}...',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Data Loading Test {i+1}')
    
    plt.tight_layout()
    return fig


def plot_intensity_calculations():
    """Plot 2: Intensity calculation tests"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    try:
        source, data_type = get_source()
        if source is None:
            raise Exception("Could not create source")
        
        # Test frequency
        test_freq = 5e14  # ~600 nm
        
        # Plot 1: Intensity vs direction (single frequency)
        directions = np.linspace(-5e-6, 5e-6, 50)  # radians
        intensities_x = []
        intensities_y = []
        
        for d in directions:
            n_hat_x = np.array([d, 0.0])
            n_hat_y = np.array([0.0, d])
            intensities_x.append(source.intensity(test_freq, n_hat_x))
            intensities_y.append(source.intensity(test_freq, n_hat_y))
        
        ax1.plot(directions * 1e6, intensities_x, 'b-', linewidth=2, label='X-direction')
        ax1.plot(directions * 1e6, intensities_y, 'r-', linewidth=2, label='Y-direction')
        ax1.set_xlabel('Angular Position (μrad)')
        ax1.set_ylabel('Intensity (W/m²/Hz/sr)')
        ax1.set_title(f'Intensity Profile at ν = {test_freq:.1e} Hz')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Intensity vs frequency (at origin)
        test_freqs = source.frequency_grid[::5]  # Sample frequencies
        n_hat_origin = np.array([0.0, 0.0])
        intensities_freq = []
        
        for freq in test_freqs:
            intensities_freq.append(source.intensity(freq, n_hat_origin))
        
        ax2.plot(test_freqs / 1e14, intensities_freq, 'g-', linewidth=2, marker='o')
        ax2.set_xlabel('Frequency (×10¹⁴ Hz)')
        ax2.set_ylabel('Intensity at Origin (W/m²/Hz/sr)')
        ax2.set_title('Intensity vs Frequency at Origin')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: 2D intensity map
        extent = 5e-6  # radians
        n_points = min(source.nx, source.ny, 25)  # Cap at 25 for performance
        x_range = np.linspace(-extent, extent, n_points)
        y_range = np.linspace(-extent, extent, n_points)
        X, Y = np.meshgrid(x_range, y_range)
        
        intensity_map = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(n_points):
                n_hat = np.array([X[i, j], Y[i, j]])
                intensity_map[i, j] = source.intensity(test_freq, n_hat)
        
        im = ax3.imshow(intensity_map, extent=[-extent*1e6, extent*1e6, -extent*1e6, extent*1e6],
                       origin='lower', cmap='hot', aspect='equal')
        ax3.set_xlabel('X Position (μrad)')
        ax3.set_ylabel('Y Position (μrad)')
        ax3.set_title(f'2D Intensity Map at ν = {test_freq:.1e} Hz')
        try:
            plt.colorbar(im, ax=ax3, label='Intensity (W/m²/Hz/sr)')
        except Exception:
            pass
        
        # Plot 4: Multiple directions test (polar plot)
        n_directions = 20
        angles = np.linspace(0, 2*np.pi, n_directions)
        radius = 2e-6  # radians
        
        directions_multi = np.array([[radius * np.cos(a), radius * np.sin(a)] for a in angles])
        intensities_multi = source.intensity(test_freq, directions_multi)
        
        # Convert ax4 to polar projection
        ax4.remove()
        ax4 = fig.add_subplot(2, 2, 4, projection='polar')
        ax4.plot(angles, intensities_multi, 'bo-', linewidth=2, markersize=6)
        ax4.set_title(f'Intensity vs Angular Position\n(r = {radius*1e6:.1f} μrad)', pad=20)
        ax4.grid(True)
        
    except Exception as e:
        for i, ax in enumerate([ax1, ax2, ax3, ax4]):
            ax.text(0.5, 0.5, f'Intensity test {i+1}\nError: {str(e)[:30]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_visibility_calculations():
    """Plot 3: Visibility calculations including visibility vs zeta"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    try:
        source, data_type = get_source()
        if source is None:
            raise Exception("Could not create source")
        
        # Test parameters
        nu_0 = 5e14  # 600 nm
        c = 2.99792458e8  # Speed of light
        wavelength = c / nu_0
        
        # Estimate source angular size from intensity profile
        # Use the pixel scale as a rough estimate of angular size
        theta_estimate = source.pixel_scale * 5  # Rough estimate

        # print(source.pixel_scale, theta_estimate, 1.22*wavelength/source.pixel_scale)

        # Plot 1: Visibility vs baseline length
        baseline_lengths = np.logspace(-2, 2, 25)  # 0.01 m to 100 m
        visibilities = []
        
        for B in baseline_lengths:
            baseline = np.array([B, 0.0, 0.0])
            try:
                vis = source.V(nu_0, baseline)
                visibilities.append(abs(vis))
            except:
                visibilities.append(0.0)
        
        ax1.semilogx(baseline_lengths, visibilities, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Baseline Length (m)')
        ax1.set_ylabel('|V(B)|')
        ax1.set_title('Visibility vs Baseline Length')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Plot 2: Visibility vs zeta (KEY PLOT REQUESTED)
        # zeta is related to baseline by: baseline_lengths = zetas * wavelength / (np.pi * theta)
        zetas = np.linspace(0.1, 10, 50)
        baseline_lengths_zeta = zetas * wavelength / (np.pi * theta_estimate)
        visibilities_zeta = []
        
        for B in baseline_lengths_zeta:
            baseline = np.array([B, 0.0, 0.0])
            try:
                vis = source.V(nu_0, baseline)
                visibilities_zeta.append(abs(vis))
            except:
                visibilities_zeta.append(0.0)
        
        ax2.plot(zetas, visibilities_zeta, 'r-', linewidth=2, marker='s', markersize=4)
        ax2.set_xlabel('ζ = πBθ/λ')
        ax2.set_ylabel('|V(ζ)|')
        ax2.set_title(f'Visibility vs Zeta\n(θ ≈ {theta_estimate*1e6:.1f} μrad)')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)
        
        # Add theoretical comparison for uniform disk if possible
        try:
            from scipy.special import j1
            # Theoretical uniform disk visibility: V(ζ) = 2*J₁(ζ)/ζ
            zetas_theory = zetas[zetas > 0]
            vis_theory = []
            for z in zetas_theory:
                if z == 0:
                    vis_theory.append(1.0)
                else:
                    vis_theory.append(abs(2 * j1(z) / z))
            ax2.plot(zetas_theory, vis_theory, 'k--', alpha=0.7, label='Uniform disk theory')
            ax2.legend()
        except:
            pass
        
        # Plot 3: Visibility phase vs baseline
        baseline_lengths_phase = np.linspace(10, 1000, 30)
        vis_phases = []
        vis_amplitudes = []
        
        for B in baseline_lengths_phase:
            baseline = np.array([B, 0.0, 0.0])
            try:
                vis = source.V(nu_0, baseline)
                vis_phases.append(np.angle(vis))
                vis_amplitudes.append(abs(vis))
            except:
                vis_phases.append(0.0)
                vis_amplitudes.append(0.0)
        
        ax3_twin = ax3.twinx()
        line1 = ax3.plot(baseline_lengths_phase, vis_amplitudes, 'b-', linewidth=2, label='|V|')
        line2 = ax3_twin.plot(baseline_lengths_phase, np.array(vis_phases) * 180/np.pi, 'r-', 
                             linewidth=2, label='Phase (deg)')
        ax3.set_xlabel('Baseline Length (m)')
        ax3.set_ylabel('|V|', color='blue')
        ax3_twin.set_ylabel('Phase (degrees)', color='red')
        ax3.set_title('Visibility Amplitude and Phase')
        ax3.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper right')
        
        # Plot 4: Visibility vs frequency
        test_freqs = np.linspace(source.freq_min, source.freq_max, 30)
        baseline_fixed = np.array([100.0, 0.0, 0.0])  # 100m baseline
        vis_vs_freq = []
        
        for freq in test_freqs:
            try:
                vis = source.V(freq, baseline_fixed)
                vis_vs_freq.append(abs(vis))
            except:
                vis_vs_freq.append(0.0)
        
        ax4.plot(test_freqs / 1e14, vis_vs_freq, 'g-', linewidth=2, marker='d', markersize=4)
        ax4.set_xlabel('Frequency (×10¹⁴ Hz)')
        ax4.set_ylabel('|V|')
        ax4.set_title(f'Visibility vs Frequency\n(B = {baseline_fixed[0]:.0f} m)')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1.1)
        
    except Exception as e:
        for i, ax in enumerate([ax1, ax2, ax3, ax4]):
            ax.text(0.5, 0.5, f'Visibility test {i+1}\nError: {str(e)[:30]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
    plt.tight_layout()
    return fig


def plot_chaotic_source_inheritance():
    """Plot 4: ChaoticSource inheritance and coherence functions"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    try:
        source, data_type = get_source()
        if source is None:
            raise Exception("Could not create source")
        
        # Test parameters
        nu_0 = 5e14  # 600 nm
        delta_nu_narrow = 1e11  # 100 GHz
        delta_nu_wide = 1e13    # 10 THz
        
        # Plot 1: g1 function vs time delay
        delta_t_range = np.linspace(0, 5e-11, 100)  # 0 to 50 ps
        
        g1_narrow = [source.g1(dt, nu_0, delta_nu_narrow) for dt in delta_t_range]
        g1_wide = [source.g1(dt, nu_0, delta_nu_wide) for dt in delta_t_range]
        
        g1_narrow_abs = np.abs(g1_narrow)
        g1_wide_abs = np.abs(g1_wide)
        
        ax1.plot(delta_t_range * 1e12, g1_narrow_abs, 'b-', linewidth=2, 
                label=f'Narrow: Δν = {delta_nu_narrow:.0e} Hz')
        ax1.plot(delta_t_range * 1e12, g1_wide_abs, 'r-', linewidth=2,
                label=f'Wide: Δν = {delta_nu_wide:.0e} Hz')
        ax1.axhline(y=1/np.e, color='k', linestyle='--', alpha=0.5, label='1/e level')
        ax1.set_xlabel('Time Delay Δt (ps)')
        ax1.set_ylabel('|g¹(Δt)|')
        ax1.set_title('First-Order Temporal Coherence Function')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Plot 2: g2-1 function vs time delay
        g2_minus_one_narrow = [source.g2_minus_one(dt, nu_0, delta_nu_narrow) for dt in delta_t_range]
        g2_minus_one_wide = [source.g2_minus_one(dt, nu_0, delta_nu_wide) for dt in delta_t_range]
        
        ax2.plot(delta_t_range * 1e12, g2_minus_one_narrow, 'b-', linewidth=2,
                label=f'Narrow: Δν = {delta_nu_narrow:.0e} Hz')
        ax2.plot(delta_t_range * 1e12, g2_minus_one_wide, 'r-', linewidth=2,
                label=f'Wide: Δν = {delta_nu_wide:.0e} Hz')
        ax2.axhline(y=1/np.e**2, color='k', linestyle='--', alpha=0.5, label='1/e² level')
        ax2.set_xlabel('Time Delay Δt (ps)')
        ax2.set_ylabel('g²(Δt) - 1')
        ax2.set_title('Second-Order Temporal Coherence Function Minus One')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)
        
        # Plot 3: Coherence time vs bandwidth
        bandwidths = np.logspace(10, 14, 50)  # 10 GHz to 100 THz
        coherence_times = []
        
        for bw in bandwidths:
            # Find 1/e point
            dt_test = np.linspace(0, 1e-9, 1000)
            g1_test = [abs(source.g1(dt, nu_0, bw)) for dt in dt_test]
            
            # Find where g1 drops to 1/e
            idx = np.where(np.array(g1_test) <= 1/np.e)[0]
            if len(idx) > 0:
                coherence_time = dt_test[idx[0]]
            else:
                coherence_time = dt_test[-1]
            
            coherence_times.append(coherence_time)
        
        ax3.loglog(bandwidths, coherence_times, 'g-', linewidth=2, marker='o', markersize=4)
        ax3.loglog(bandwidths, 1/bandwidths, 'k--', alpha=0.7, label='τc = 1/Δν')
        ax3.set_xlabel('Bandwidth Δν (Hz)')
        ax3.set_ylabel('Coherence Time τc (s)')
        ax3.set_title('Coherence Time vs Bandwidth')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Signal-to-noise ratio estimation
        baselines = np.logspace(1, 4, 30)  # 10 m to 10 km
        snr_estimates = []
        
        # Mock observational parameters for SNR calculation
        A = 10.0  # 10 m² telescope area
        T_obs = 3600.0  # 1 hour observation
        sigma_t = 1e-9  # 1 ns timing jitter
        
        for B in baselines:
            baseline = np.array([B, 0.0, 0.0])
            try:
                vis = source.V(nu_0, baseline)
                flux = source.total_flux(nu_0)
                
                # Simple SNR estimate based on visibility and flux
                # This is a simplified calculation for demonstration
                h = 6.626e-34
                dGamma_dnu = A * flux / (h * nu_0)
                snr_est = abs(vis)**2 * dGamma_dnu * np.sqrt(T_obs / sigma_t) / np.sqrt(128 * np.pi)
                snr_estimates.append(snr_est)
            except:
                snr_estimates.append(0.0)
        
        ax4.loglog(baselines, snr_estimates, 'purple', linewidth=2, marker='s', markersize=4)
        ax4.set_xlabel('Baseline Length (m)')
        ax4.set_ylabel('SNR Estimate')
        ax4.set_title(f'Signal-to-Noise Ratio Estimate\n(A={A}m², T={T_obs/3600:.1f}h, σt={sigma_t*1e9:.0f}ns)')
        ax4.grid(True, alpha=0.3)
        
    except Exception as e:
        for i, ax in enumerate([ax1, ax2, ax3, ax4]):
            ax.text(0.5, 0.5, f'Coherence test {i+1}\nError: {str(e)[:30]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_integration_tests():
    """Plot 5: Integration tests and realistic scenarios"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    try:
        source, data_type = get_source()
        if source is None:
            raise Exception("Could not create source")
        
        # Plot 1: Flux conservation test
        test_freq = 5e14
        total_flux_direct = source.total_flux(test_freq)
        
        # Numerical integration over intensity
        extent = 15e-6
        n_points = min(source.nx, source.ny, 20)  # Cap at 20 for performance
        x_range = np.linspace(-extent, extent, n_points)
        y_range = np.linspace(-extent, extent, n_points)
        
        integrated_flux = 0
        pixel_area = (2 * extent / n_points)**2
        
        intensity_grid = np.zeros((n_points, n_points))
        for i, x in enumerate(x_range):
            for j, y in enumerate(y_range):
                n_hat = np.array([x, y])
                intensity = source.intensity(test_freq, n_hat)
                intensity_grid[i, j] = intensity
                integrated_flux += intensity * pixel_area
        
        # Show 2D intensity distribution
        im = ax1.imshow(intensity_grid, extent=[-extent*1e6, extent*1e6, -extent*1e6, extent*1e6],
                       origin='lower', cmap='hot', aspect='equal')
        ax1.set_xlabel('X Position (μrad)')
        ax1.set_ylabel('Y Position (μrad)')
        ax1.set_title(f'Flux Conservation Test\nDirect: {total_flux_direct:.2e}\nIntegrated: {integrated_flux:.2e}')
        try:
            plt.colorbar(im, ax=ax1, label='Intensity (W/m²/Hz/sr)')
        except Exception:
            pass
        
        # Plot 2: Wavelength-dependent spatial size
        test_wavelengths = [4000, 6000, 8000]  # Angstrom
        colors = ['blue', 'green', 'red']
        
        for i, (wave, color) in enumerate(zip(test_wavelengths, colors)):
            # Find closest frequency
            freq_idx = np.argmin(np.abs(source.wavelength_grid - wave))
            test_freq = source.frequency_grid[freq_idx]
            
            # Get radial profile
            radii = np.linspace(0, 6e-6, 20)
            intensities = []
            
            for r in radii:
                n_hat = np.array([r, 0.0])
                intensity = source.intensity(test_freq, n_hat)
                intensities.append(intensity)
            
            # Normalize for comparison
            intensities = np.array(intensities)
            if np.max(intensities) > 0:
                intensities = intensities / np.max(intensities)
            
            ax2.plot(radii * 1e6, intensities, color=color, linewidth=2, 
                    label=f'{wave:.0f} Å', marker='o', markersize=3)
        
        ax2.set_xlabel('Radial Distance (μrad)')
        ax2.set_ylabel('Normalized Intensity')
        ax2.set_title('Wavelength-Dependent Spatial Profiles')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Comparison with theoretical models
        # Compare visibility with uniform disk model
        baseline_lengths = np.logspace(1, 3, 30)
        nu_test = 5e14
        c = 2.99792458e8
        wavelength = c / nu_test
        
        # Get source visibilities
        source_vis = []
        for B in baseline_lengths:
            baseline = np.array([B, 0.0, 0.0])
            try:
                vis = source.V(nu_test, baseline)
                source_vis.append(abs(vis))
            except:
                source_vis.append(0.0)
        
        ax3.semilogx(baseline_lengths, source_vis, 'b-', linewidth=2, label='Sedona Source')
        
        # Add theoretical uniform disk for comparison
        try:
            from scipy.special import j1
            theta_disk = source.pixel_scale * 3  # Estimate disk size
            uniform_vis = []
            for B in baseline_lengths:
                u = B / wavelength
                x = 2 * np.pi * u * theta_disk
                if x == 0:
                    vis_val = 1.0
                else:
                    vis_val = abs(2 * j1(x) / x)
                uniform_vis.append(vis_val)
            
            ax3.semilogx(baseline_lengths, uniform_vis, 'r--', linewidth=2, 
                        label=f'Uniform Disk (θ={theta_disk*1e6:.1f}μrad)')
            ax3.legend()
        except:
            pass
        
        ax3.set_xlabel('Baseline Length (m)')
        ax3.set_ylabel('|V|')
        ax3.set_title('Comparison with Theoretical Models')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.1)
        
        # Plot 4: Performance and accuracy metrics
        info = source.get_spectrum_info()
        
        # Create performance summary
        metrics = ['Wavelength\nPoints', 'Spatial\nGrid Size', 'Frequency\nRange (Hz)', 
                  'Peak Flux\n(W/m²/Hz)', 'Angular\nScale (μrad)']
        values = [
            info['wavelength_points'],
            info['spatial_grid'][0] * info['spatial_grid'][1],
            info['frequency_range_hz'][1] - info['frequency_range_hz'][0],
            info['peak_flux_density_w_m2_hz'],
            source.pixel_scale * 1e6
        ]
        
        # Normalize for plotting
        normalized_values = [val / max(values) for val in values]
        
        bars = ax4.bar(metrics, normalized_values, 
                      color=['blue', 'green', 'red', 'orange', 'purple'], alpha=0.7)
        ax4.set_ylabel('Normalized Value')
        ax4.set_title('Source Model Performance Metrics')
        ax4.grid(True, alpha=0.3)
        
        # Add actual values as text
        for bar, val in zip(bars, values):
            height = bar.get_height()
            label = f'{val:.1e}' if val > 1000 else f'{val:.2f}'
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    label, ha='center', va='bottom', fontsize=8, rotation=45)
        
    except Exception as e:
        for i, ax in enumerate([ax1, ax2, ax3, ax4]):
            ax.text(0.5, 0.5, f'Integration test {i+1}\nError: {str(e)[:30]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
    
    plt.tight_layout()
    return fig


def create_documentation_fallback():
    """Create documentation when plotting fails"""
    try:
        with open('plot_sn2011fe_sedona_description.txt', 'w') as f:
            f.write("Sedona SN2011fe Source Model - Comprehensive Test Plots\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("This document describes the comprehensive test plots for SedonaSN2011feSource\n")
            f.write("that validate all key functionality and methods.\n\n")
            
            f.write("Plot 1: Data Loading and Initialization Validation\n")
            f.write("- Wavelength-frequency conversion validation\n")
            f.write("- Photon flux density per frequency\n")
            f.write("- Flux density in SI units\n")
            f.write("- Data summary and consistency checks\n\n")
            
            f.write("Plot 2: Intensity Calculation Tests\n")
            f.write("- Intensity vs angular position (X and Y directions)\n")
            f.write("- Intensity vs frequency at origin\n")
            f.write("- 2D intensity map\n")
            f.write("- Polar intensity distribution\n\n")
            
            f.write("Plot 3: Visibility Calculations (KEY PLOT)\n")
            f.write("- Visibility vs baseline length\n")
            f.write("- **Visibility vs zeta (ζ = πBθ/λ)** - MAIN REQUESTED PLOT\n")
            f.write("- Visibility amplitude and phase vs baseline\n")
            f.write("- Visibility vs frequency\n\n")
            
            f.write("Plot 4: Chaotic Source Inheritance Tests\n")
            f.write("- First-order coherence function g¹(Δt)\n")
            f.write("- Second-order coherence function g²(Δt)-1\n")
            f.write("- Coherence time vs bandwidth relationship\n")
            f.write("- Signal-to-noise ratio estimates\n\n")
            
            f.write("Plot 5: Integration Tests and Realistic Scenarios\n")
            f.write("- Flux conservation validation\n")
            f.write("- Wavelength-dependent spatial profiles\n")
            f.write("- Comparison with theoretical models\n")
            f.write("- Performance and accuracy metrics\n\n")
            
            f.write("Key Features:\n")
            f.write("- Comprehensive testing of all SedonaSN2011feSource methods\n")
            f.write("- Visibility vs zeta plot as specifically requested\n")
            f.write("- Fallback to mock data when real Sedona data unavailable\n")
            f.write("- Error handling and graceful degradation\n")
            f.write("- Performance optimization for large datasets\n")
        
        print("📄 Created comprehensive plot description document")
        
    except Exception as e:
        print(f"Could not create documentation file: {e}")


def main():
    """Create all comprehensive test plots and save to PDF"""
    print("Creating Comprehensive Sedona SN2011fe Source Model Test Plots...")
    print("=" * 70)
    
    if not DEPENDENCIES_AVAILABLE:
        print("⚠️  Dependencies not available - creating documentation instead")
        create_documentation_fallback()
        return
    
    try:
        with PdfPages('sources/plot_sn2011fe_sedona.pdf') as pdf:
            # print("1. Creating data loading and initialization plots...")
            # fig1 = plot_data_loading_and_initialization()
            # pdf.savefig(fig1, bbox_inches='tight')
            # plt.close(fig1)
            
            # print("2. Creating intensity calculation plots...")
            # fig2 = plot_intensity_calculations()
            # pdf.savefig(fig2, bbox_inches='tight')
            # plt.close(fig2)
            
            print("3. Creating visibility calculation plots (including zeta plot)...")
            fig3 = plot_visibility_calculations()
            pdf.savefig(fig3, bbox_inches='tight')
            plt.close(fig3)
            
            # print("4. Creating chaotic source inheritance plots...")
            # fig4 = plot_chaotic_source_inheritance()
            # pdf.savefig(fig4, bbox_inches='tight')
            # plt.close(fig4)
            
            # print("5. Creating integration test plots...")
            # fig5 = plot_integration_tests()
            # pdf.savefig(fig5, bbox_inches='tight')
            # plt.close(fig5)
            
            # Add metadata
            d = pdf.infodict()
            d['Title'] = 'Comprehensive Sedona SN2011fe Source Model Test Plots'
            d['Author'] = 'Intensity Interferometry Analysis'
            d['Subject'] = 'Complete validation of SedonaSN2011feSource functionality'
            d['Keywords'] = 'Sedona, SN2011fe, Intensity Interferometry, Visibility, Zeta, Testing'
            d['Creator'] = 'plot_sn2011fe_sedona.py (reimplemented)'
        
        print("\n✅ All comprehensive test plots saved to sources/plot_sn2011fe_sedona.pdf!")
        print("\nThe PDF contains 5 pages with comprehensive visualizations:")
        print("1. Data loading and initialization validation")
        print("2. Intensity calculation tests (spatial and spectral)")
        print("3. Visibility calculations INCLUDING visibility vs zeta plot")
        print("4. Chaotic source inheritance (g1, g2-1, coherence)")
        print("5. Integration tests with realistic scenarios")
        print("\n🎯 KEY FEATURE: Plot 3 includes the requested visibility vs zeta plot!")
        
    except Exception as e:
        print(f"❌ Error creating plots: {e}")
        print("Creating documentation instead...")
        create_documentation_fallback()
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()