#!/usr/bin/env python3
"""
Plotting script for Sedona SN2011fe Source Model

This script creates visualizations that reflect the contents and functionality
tested in test_sn2011fe_sedona.py, providing visual validation of the
SedonaSN2011feSource class behavior.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os
import tempfile

# Add sources directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from sources.sn2011fe_sedona import SedonaSN2011feSource
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import SedonaSN2011feSource: {e}")
    DEPENDENCIES_AVAILABLE = False

# Set up plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def create_mock_data():
    """Create mock data for testing and plotting"""
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


def plot_initialization_and_data_loading():
    """Plot 1: Data loading and initialization validation"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
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
        
        # Plot 1: Wavelength vs Frequency conversion
        ax1.plot(source.wavelength_grid, source.frequency_grid / 1e14, 'b-', linewidth=2)
        ax1.set_xlabel('Wavelength (Å)')
        ax1.set_ylabel('Frequency (×10¹⁴ Hz)')
        ax1.set_title(f'Wavelength-Frequency Conversion\n({data_type})')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Photon flux density per frequency - matching Figure 3 of II_Telescopes.pdf
        ax2.plot(source.wavelength_grid, source.photon_flux_density_grid, 'r-', linewidth=2, label='SEDONA')
        ax2.set_xlabel('Wavelength (Å)')
        ax2.set_ylabel('n_ν [s⁻¹ cm⁻² Hz⁻¹]')
        ax2.set_title(f'Photon Flux Density per Frequency\n({data_type}) - cf. Figure 3 II_Telescopes.pdf')
        # ax2.set_yscale('log')
        ax2.set_ylim((0,np.max(source.photon_flux_density_grid[np.logical_and(source.wavelength_grid > 4000, 
                                                                                 source.wavelength_grid < 8000)])*1.1))
        ax2.set_xlim((3300,10000))
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Flux density in SI units
        ax3.plot(source.frequency_grid / 1e14, source.flux_density_grid, 'g-', linewidth=2)
        ax3.set_xlabel('Frequency (×10¹⁴ Hz)')
        ax3.set_ylabel('Flux Density (W/m²/Hz)')
        ax3.set_title('Flux Density vs Frequency')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Data consistency check
        freq_range = [source.freq_min, source.freq_max]
        flux_range = [np.min(source.flux_density_grid[source.flux_density_grid > 0]), 
                     np.max(source.flux_density_grid)]
        
        ax4.bar(['Min Freq', 'Max Freq'], [f/1e14 for f in freq_range], 
                color='blue', alpha=0.7, label='Frequency (×10¹⁴ Hz)')
        ax4_twin = ax4.twinx()
        ax4_twin.bar(['Min Flux', 'Max Flux'], flux_range, 
                    color='red', alpha=0.7, label='Flux Density (W/m²/Hz)')
        ax4.set_title('Data Range Summary')
        ax4.set_ylabel('Frequency (×10¹⁴ Hz)', color='blue')
        ax4_twin.set_ylabel('Flux Density (W/m²/Hz)', color='red')
        ax4_twin.set_yscale('log')
        
    except Exception as e:
        # If real data loading fails, show error message
        ax1.text(0.5, 0.5, f'Data loading test\nError: {str(e)[:50]}...',
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Data Loading Test')
        
        for ax in [ax2, ax3, ax4]:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}...',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    finally:
        # Clean up temporary files if they were created
        if 'temp_dir' in locals():
            import shutil
            shutil.rmtree(temp_dir)
    
    plt.tight_layout()
    return fig


def plot_intensity_calculations():
    """Plot 2: Intensity calculation tests"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    try:
        # Try to use real Sedona data first
        real_wave_file = 'data/WaveGrid.npy'
        real_flux_file = 'data/Phase0Flux.npy'
        
        if os.path.exists(real_wave_file) and os.path.exists(real_flux_file):
            source = SedonaSN2011feSource(real_wave_file, real_flux_file)
        else:
            # Fallback to mock data
            wavelengths, flux_3d = create_mock_data()
            temp_dir = tempfile.mkdtemp()
            wave_file = os.path.join(temp_dir, 'WaveGrid.npy')
            flux_file = os.path.join(temp_dir, 'Phase0Flux.npy')
            np.save(wave_file, wavelengths)
            np.save(flux_file, flux_3d)
            source = SedonaSN2011feSource(wave_file, flux_file)
        
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
        
        # Plot 3: 2D intensity map (using native resolution)
        extent = 5e-6  # radians
        # Use native grid size from the source
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
        ax3.set_title(f'2D Intensity Map at ν = {test_freq:.1e} Hz\n(Grid: {n_points}×{n_points})')
        try:
            plt.colorbar(im, ax=ax3, label='Intensity (W/m²/Hz/sr)')
        except Exception:
            # Fallback if colorbar fails
            pass
        
        # Plot 4: Multiple directions test
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
    
    finally:
        # Clean up temporary files if they were created
        if 'temp_dir' in locals():
            import shutil
            shutil.rmtree(temp_dir)
    
    plt.tight_layout()
    return fig


def plot_flux_and_interpolation():
    """Plot 3: Flux calculations and interpolation tests"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    try:
        # Try to use real Sedona data first
        real_wave_file = 'data/WaveGrid.npy'
        real_flux_file = 'data/Phase0Flux.npy'
        
        if os.path.exists(real_wave_file) and os.path.exists(real_flux_file):
            source = SedonaSN2011feSource(real_wave_file, real_flux_file)
        else:
            # Fallback to mock data
            wavelengths, flux_3d = create_mock_data()
            temp_dir = tempfile.mkdtemp()
            wave_file = os.path.join(temp_dir, 'WaveGrid.npy')
            flux_file = os.path.join(temp_dir, 'Phase0Flux.npy')
            np.save(wave_file, wavelengths)
            np.save(flux_file, flux_3d)
            source = SedonaSN2011feSource(wave_file, flux_file)
        
        # Plot 1: Total flux vs frequency
        test_freqs = np.linspace(source.freq_min, source.freq_max, 100)
        total_fluxes = [source.total_flux(freq) for freq in test_freqs]
        
        ax1.plot(test_freqs / 1e14, total_fluxes, 'b-', linewidth=2)
        ax1.set_xlabel('Frequency (×10¹⁴ Hz)')
        ax1.set_ylabel('Total Flux (W/m²/Hz)')
        ax1.set_title('Total Flux vs Frequency')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Mark the actual data points
        ax1.scatter(source.frequency_grid / 1e14, source.flux_density_grid, 
                   c='red', s=20, alpha=0.6, label='Data points')
        ax1.legend()
        
        # Plot 2: Interpolation bounds test
        freq_extended = np.linspace(source.freq_min * 0.5, source.freq_max * 1.5, 200)
        flux_extended = [source.total_flux(freq) for freq in freq_extended]
        
        ax2.plot(freq_extended / 1e14, flux_extended, 'g-', linewidth=2)
        ax2.axvline(source.freq_min / 1e14, color='red', linestyle='--', 
                   label=f'Min freq: {source.freq_min:.2e} Hz')
        ax2.axvline(source.freq_max / 1e14, color='red', linestyle='--', 
                   label=f'Max freq: {source.freq_max:.2e} Hz')
        ax2.set_xlabel('Frequency (×10¹⁴ Hz)')
        ax2.set_ylabel('Total Flux (W/m²/Hz)')
        ax2.set_title('Flux Interpolation with Bounds')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Flux conservation test
        # Compare total flux to integrated intensity
        test_freq = source.frequency_grid[50]
        
        # Calculate flux by integrating intensity over small area (using native resolution)
        extent = 10e-6  # radians
        # Use native grid size from the source
        n_points = min(source.nx, source.ny, 15)  # Cap at 15 for performance
        x_range = np.linspace(-extent, extent, n_points)
        y_range = np.linspace(-extent, extent, n_points)
        
        total_intensity = 0
        pixel_area = (2 * extent / n_points)**2
        
        for x in x_range:
            for y in y_range:
                n_hat = np.array([x, y])
                intensity = source.intensity(test_freq, n_hat)
                total_intensity += intensity * pixel_area
        
        flux_direct = source.total_flux(test_freq)
        
        ax3.bar(['Direct Flux', 'Integrated Intensity'], 
               [flux_direct, total_intensity], 
               color=['blue', 'orange'], alpha=0.7)
        ax3.set_ylabel('Flux (W/m²/Hz)')
        ax3.set_title(f'Flux Conservation Test\nat ν = {test_freq:.1e} Hz')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Add text with ratio
        ratio = total_intensity / flux_direct if flux_direct > 0 else 0
        ax3.text(0.5, 0.8, f'Ratio: {ratio:.3f}', 
                transform=ax3.transAxes, ha='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 4: Interpolation accuracy test
        # Test interpolation at different frequencies
        test_indices = [10, 30, 50, 70, 90]
        exact_freqs = source.frequency_grid[test_indices]
        exact_fluxes = source.flux_density_grid[test_indices]
        
        # Test nearby frequencies
        interp_freqs = exact_freqs + (exact_freqs * 0.001)  # 0.1% offset
        interp_fluxes = [source.total_flux(freq) for freq in interp_freqs]
        
        ax4.scatter(exact_freqs / 1e14, exact_fluxes, 
                   c='blue', s=100, label='Exact values', marker='o')
        ax4.scatter(interp_freqs / 1e14, interp_fluxes, 
                   c='red', s=100, label='Interpolated', marker='x')
        
        # Draw lines connecting exact and interpolated values
        for i in range(len(exact_freqs)):
            ax4.plot([exact_freqs[i]/1e14, interp_freqs[i]/1e14], 
                    [exact_fluxes[i], interp_fluxes[i]], 
                    'gray', alpha=0.5, linestyle=':')
        
        ax4.set_xlabel('Frequency (×10¹⁴ Hz)')
        ax4.set_ylabel('Flux Density (W/m²/Hz)')
        ax4.set_title('Interpolation Accuracy Test')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
    except Exception as e:
        for i, ax in enumerate([ax1, ax2, ax3, ax4]):
            ax.text(0.5, 0.5, f'Flux test {i+1}\nError: {str(e)[:30]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
    
    finally:
        # Clean up temporary files if they were created
        if 'temp_dir' in locals():
            import shutil
            shutil.rmtree(temp_dir)
    
    plt.tight_layout()
    return fig


def plot_chaotic_source_inheritance():
    """Plot 4: ChaoticSource inheritance and coherence functions"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    try:
        # Try to use real Sedona data first
        real_wave_file = 'data/WaveGrid.npy'
        real_flux_file = 'data/Phase0Flux.npy'
        
        if os.path.exists(real_wave_file) and os.path.exists(real_flux_file):
            source = SedonaSN2011feSource(real_wave_file, real_flux_file)
        else:
            # Fallback to mock data
            wavelengths, flux_3d = create_mock_data()
            temp_dir = tempfile.mkdtemp()
            wave_file = os.path.join(temp_dir, 'WaveGrid.npy')
            flux_file = os.path.join(temp_dir, 'Phase0Flux.npy')
            np.save(wave_file, wavelengths)
            np.save(flux_file, flux_3d)
            source = SedonaSN2011feSource(wave_file, flux_file)
        
        # Test parameters
        nu_0 = 5e14  # 600 nm
        delta_nu_narrow = 1e11  # 100 GHz
        delta_nu_wide = 1e13    # 10 THz
        
        # Plot 1: g1 function vs time delay
        delta_t_range = np.linspace(0, 5e-11, 1000)  # 0 to 50 ps
        
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
        
        # Plot 4: Visibility calculation test
        baselines = np.logspace(1, 4, 50)  # 10 m to 10 km
        visibilities = []
        
        for baseline_length in baselines:
            baseline = np.array([baseline_length, 0.0, 0.0])
            try:
                vis = source.V(nu_0, baseline)
                visibilities.append(abs(vis))
            except:
                visibilities.append(0.0)
        
        ax4.semilogx(baselines, visibilities, 'purple', linewidth=2, marker='s', markersize=4)
        ax4.set_xlabel('Baseline Length (m)')
        ax4.set_ylabel('|V(B)|')
        ax4.set_title('Visibility vs Baseline Length')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1.1)
        
    except Exception as e:
        for i, ax in enumerate([ax1, ax2, ax3, ax4]):
            ax.text(0.5, 0.5, f'Coherence test {i+1}\nError: {str(e)[:30]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
    
    finally:
        # Clean up temporary files if they were created
        if 'temp_dir' in locals():
            import shutil
            shutil.rmtree(temp_dir)
    
    plt.tight_layout()
    return fig


def plot_integration_tests():
    """Plot 5: Integration tests and realistic scenarios"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    try:
        # Try to use real Sedona data first
        real_wave_file = 'data/WaveGrid.npy'
        real_flux_file = 'data/Phase0Flux.npy'
        
        if os.path.exists(real_wave_file) and os.path.exists(real_flux_file):
            source = SedonaSN2011feSource(real_wave_file, real_flux_file)
        else:
            # Fallback to realistic mock data with Gaussian spatial profile
            wavelengths = np.linspace(3000, 10000, 50)
            central_wave = 6000  # Angstrom
            sigma = 1000
            spectrum = np.exp(-0.5 * ((wavelengths - central_wave) / sigma)**2)
            
            nx, ny = 20, 20
            flux_3d = np.zeros((50, nx, ny))
            
            x = np.linspace(-10, 10, nx)
            y = np.linspace(-10, 10, ny)
            X, Y = np.meshgrid(x, y)
            
            for i, flux_val in enumerate(spectrum):
                spatial_profile = np.exp(-(X**2 + Y**2) / (2 * 3**2))
                flux_3d[i, :, :] = flux_val * spatial_profile * 1e-15
            
            temp_dir = tempfile.mkdtemp()
            wave_file = os.path.join(temp_dir, 'WaveGrid.npy')
            flux_file = os.path.join(temp_dir, 'Phase0Flux.npy')
            
            np.save(wave_file, wavelengths)
            np.save(flux_file, flux_3d)
            source = SedonaSN2011feSource(wave_file, flux_file)
        
        # Plot 1: Realistic intensity profile comparison
        central_freq = 5e14
        
        # Radial profile
        radii = np.linspace(0, 8e-6, 30)
        intensities_radial = []
        
        for r in radii:
            n_hat = np.array([r, 0.0])
            intensity = source.intensity(central_freq, n_hat)
            intensities_radial.append(intensity)
        
        ax1.plot(radii * 1e6, intensities_radial, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Radial Distance (μrad)')
        ax1.set_ylabel('Intensity (W/m²/Hz/sr)')
        ax1.set_title('Radial Intensity Profile')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Mark center vs edge comparison
        center_intensity = intensities_radial[0]
        edge_intensity = intensities_radial[-1]
        ax1.axhline(center_intensity, color='red', linestyle='--', alpha=0.7, label='Center')
        ax1.axhline(edge_intensity, color='green', linestyle='--', alpha=0.7, label='Edge')
        ax1.legend()
        
        # Plot 2: Flux conservation check
        test_freq = 5e14
        total_flux_direct = source.total_flux(test_freq)
        
        # Numerical integration over intensity (using native resolution)
        extent = 15e-6
        # Use native grid size from the source
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
        im = ax2.imshow(intensity_grid, extent=[-extent*1e6, extent*1e6, -extent*1e6, extent*1e6],
                       origin='lower', cmap='hot', aspect='equal')
        ax2.set_xlabel('X Position (μrad)')
        ax2.set_ylabel('Y Position (μrad)')
        ax2.set_title(f'2D Intensity Map (Grid: {n_points}×{n_points})\nDirect flux: {total_flux_direct:.2e}\nIntegrated: {integrated_flux:.2e}')
        try:
            plt.colorbar(im, ax=ax2, label='Intensity (W/m²/Hz/sr)')
        except Exception:
            # Fallback if colorbar fails
            pass
        
        # Plot 3: Wavelength-dependent spatial size
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
            
            ax3.plot(radii * 1e6, intensities, color=color, linewidth=2, 
                    label=f'{wave:.0f} Å', marker='o', markersize=3)
        
        ax3.set_xlabel('Radial Distance (μrad)')
        ax3.set_ylabel('Normalized Intensity')
        ax3.set_title('Wavelength-Dependent Spatial Profiles')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Spectrum information summary
        info = source.get_spectrum_info()
        
        # Create a summary plot
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
        ax4.set_title('Spectrum Information Summary')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if isinstance(val, tuple):
                label = f'{val[0]:.1e}-{val[1]:.1e}'
            else:
                label = f'{val:.1e}' if val > 1000 else f'{val:.1f}'
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    label, ha='center', va='bottom', fontsize=8, rotation=45)
        
    except Exception as e:
        for i, ax in enumerate([ax1, ax2, ax3, ax4]):
            ax.text(0.5, 0.5, f'Integration test {i+1}\nError: {str(e)[:30]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
    
    finally:
        # Clean up temporary files if they were created
        if 'temp_dir' in locals():
            import shutil
            shutil.rmtree(temp_dir)
    
    plt.tight_layout()
    return fig


def create_documentation_pdf():
    """Create a text-based documentation when plotting is not available"""
    try:
        # Create a simple text file that describes what would be plotted
        with open('plot_sn2011fe_sedona_description.txt', 'w') as f:
            f.write("Sedona SN2011fe Source Model - Plot Descriptions\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("This document describes the plots that would be generated by plot_sn2011fe_sedona.py\n")
            f.write("to validate the functionality tested in test_sn2011fe_sedona.py\n\n")
            
            f.write("Plot 1: Data Loading and Initialization Validation\n")
            f.write("- Wavelength-frequency conversion validation\n")
            f.write("- Total flux spectrum from 3D Sedona data\n")
            f.write("- Flux density in SI units\n")
            f.write("- Data range summary\n\n")
            
            f.write("Plot 2: Intensity Calculation Tests\n")
            f.write("- Intensity vs angular position (X and Y directions)\n")
            f.write("- Intensity vs frequency at origin\n")
            f.write("- 2D intensity map\n")
            f.write("- Polar intensity distribution\n\n")
            
            f.write("Plot 3: Flux Calculations and Interpolation Tests\n")
            f.write("- Total flux vs frequency with interpolation\n")
            f.write("- Interpolation bounds testing\n")
            f.write("- Flux conservation validation\n")
            f.write("- Interpolation accuracy assessment\n\n")
            
            f.write("Plot 4: Chaotic Source Inheritance Tests\n")
            f.write("- First-order coherence function g1(Δt)\n")
            f.write("- Second-order coherence function g2(Δt)-1\n")
            f.write("- Coherence time vs bandwidth relationship\n")
            f.write("- Visibility vs baseline length\n\n")
            
            f.write("Plot 5: Integration Tests and Realistic Scenarios\n")
            f.write("- Realistic radial intensity profile\n")
            f.write("- 2D intensity map with flux conservation check\n")
            f.write("- Wavelength-dependent spatial profiles\n")
            f.write("- Spectrum information summary\n\n")
        
        print("📄 Created plot_sn2011fe_sedona_description.txt with plot descriptions")
        
    except Exception as e:
        print(f"Could not create documentation file: {e}")


def main():
    """Create all plots and save to PDF"""
    print("Creating Sedona SN2011fe Source Model Plots...")
    print("=" * 50)
    
    if not DEPENDENCIES_AVAILABLE:
        print("⚠️  Some dependencies are not available:")
        print("   - SedonaSN2011feSource import failed (likely missing scipy)")
        print("   - Creating documentation instead of plots")
        create_documentation_pdf()
        return
    
    try:
        with PdfPages('plot_sn2011fe_sedona.pdf') as pdf:
            print("1. Creating initialization and data loading plots...")
            fig1 = plot_initialization_and_data_loading()
            pdf.savefig(fig1, bbox_inches='tight')
            plt.close(fig1)
            
            print("2. Creating intensity calculation plots...")
            fig2 = plot_intensity_calculations()
            pdf.savefig(fig2, bbox_inches='tight')
            plt.close(fig2)
            
            print("3. Creating flux and interpolation plots...")
            fig3 = plot_flux_and_interpolation()
            pdf.savefig(fig3, bbox_inches='tight')
            plt.close(fig3)
            
            print("4. Creating chaotic source inheritance plots...")
            fig4 = plot_chaotic_source_inheritance()
            pdf.savefig(fig4, bbox_inches='tight')
            plt.close(fig4)
            
            print("5. Creating integration test plots...")
            fig5 = plot_integration_tests()
            pdf.savefig(fig5, bbox_inches='tight')
            plt.close(fig5)
            
            # Add metadata
            d = pdf.infodict()
            d['Title'] = 'Sedona SN2011fe Source Model Test Plots'
            d['Author'] = 'Intensity Interferometry Analysis'
            d['Subject'] = 'Visual validation of test_sn2011fe_sedona.py functionality'
            d['Keywords'] = 'Sedona, SN2011fe, Intensity Interferometry, Source Model, Testing'
            d['Creator'] = 'plot_sn2011fe_sedona.py'
        
        print("\n✅ All plots saved to plot_sn2011fe_sedona.pdf!")
        print("\nThe PDF contains 5 pages with the following visualizations:")
        print("1. Data loading and initialization validation")
        print("2. Intensity calculation tests (spatial and spectral)")
        print("3. Flux calculations and interpolation accuracy")
        print("4. Chaotic source inheritance (g1, g2-1, visibility)")
        print("5. Integration tests with realistic scenarios")
        
    except Exception as e:
        print(f"❌ Error creating plots: {e}")
        print("Creating documentation instead...")
        create_documentation_pdf()
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()