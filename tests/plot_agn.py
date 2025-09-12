#!/usr/bin/env python3
"""
Comprehensive Plotting Script for AGN Source Models

This script creates visualizations that demonstrate the intensity and visibility
characteristics of the AGN source models: ShakuraSunyaevDisk, BroadLineRegion,
and RelativisticDisk.

Key Features:
- Intensity profiles and 2D maps
- Visibility vs baseline and zeta plots
- Comparison between different AGN models
- ChaoticSource inheritance validation
- Relativistic effects demonstration
"""

import sys
import os

# Add parent directory to path to import source module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    # Set backend after import
    plt.switch_backend('Agg')
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not available - will create documentation instead")
    MATPLOTLIB_AVAILABLE = False

try:
    import numpy as np
    import astropy.constants
    from g2.sources.agn import ShakuraSunyaevDisk, BroadLineRegion, RelativisticDisk, power_law_beta, lognormal_beta
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import AGN dependencies: {e}")
    DEPENDENCIES_AVAILABLE = False

# Set up plotting style
if MATPLOTLIB_AVAILABLE:
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (15, 12)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


def create_test_sources():
    """Create test AGN source instances for plotting"""
    sources = {}
    
    try:
        # Shakura-Sunyaev Disk
        sources['ss_disk'] = ShakuraSunyaevDisk(
            I_0=1e-15,  # W m^-2 Hz^-1 sr^-1
            R_0=100.0,   # GM/c^2 units
            R_in=9.0,   # GM/c^2 units
            n=3.0,
            inclination= 0, # np.pi/4,  # 45 degrees
            phi_B= 0, # np.pi/3.6,  # 
            distance= 20e6 * astropy.constants.pc.value ,  # m
            GM_over_c2=  astropy.constants.au.value # m
        )
        
        # Relativistic Disk (moderate spin)
        sources['rel_disk'] = RelativisticDisk(
            I_0=1e-15,
            R_0=100.0 ,
            R_in=9.0 ,
            n=3.0,
            inclination=np.pi/4,
            distance =20e6 * astropy.constants.pc.value,  # 1e20,
            GM_over_c2 = astropy.constants.au.value,  # 1e9
            spin_parameter=0.7  # High spin
        )
        
        # Broad Line Region
        beta_func = lambda R: power_law_beta(R, 1e15, -1.0, 1e-20)
        sources['blr'] = BroadLineRegion(
            beta_function=beta_func,
            R_in=1e14,  # m
            R_out=1e16,  # m
            GM=1e39,  # m^3/s^2
            inclination=np.pi/6,  # 30 degrees
            distance=1e25,  # m
            line_center_freq=4.57e14  # Hz (Hα line)
        )
        
        return sources
        
    except Exception as e:
        print(f"Error creating test sources: {e}")
        return {}
    
def plot_intensity_profiles():
    """Plot 1: Intensity profiles for different AGN models"""
    if not MATPLOTLIB_AVAILABLE:
        return None
        
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    try:
        sources = create_test_sources()
        if not sources:
            raise Exception("Could not create test sources")
        
        nu_test = 5e14  # 600 nm
        
        # Plot 1: Radial intensity profiles
        
        
        for name, source in sources.items():
            if name == 'blr':
                continue  # Skip BLR for radial profile (different physics)
            radii_angular = np.linspace(source.R_in * source.GM_over_c2 / source.distance, 1.1 * source.R_0 * source.GM_over_c2 / source.distance, 200)  # radians    
            intensities = []
            intensities2 = []
            for r in radii_angular:
                n_hat = np.array([r, 0.0])
                intensity = source.intensity(nu_test, n_hat)
                intensities.append(intensity)

                n_hat = np.array([0.0, r])
                intensity = source.intensity(nu_test, n_hat)
                intensities2.append(intensity)

            label = 'Shakura-Sunyaev' if name == 'ss_disk' else 'Relativistic'
            ax1.plot(radii_angular, intensities, linewidth=2, label=label+r" x=0")
            ax1.plot(radii_angular, intensities2, linewidth=2, label=label+r" y=0")

        ax1.set_xlabel('Radial Distance (μrad)')
        ax1.set_ylabel('Intensity (W/m²/Hz/sr)')
        ax1.set_title('Radial Intensity Profiles')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: 2D intensity map for SS disk
        ss_disk = sources['ss_disk']
        extent = (1.2 * ss_disk.R_0 * ss_disk.GM_over_c2 / ss_disk.distance) # radians
        n_points = 100
        x_range = np.linspace(-extent, extent, n_points)
        y_range = np.linspace(-extent, extent, n_points)
        X, Y = np.meshgrid(x_range, y_range)
        
        intensity_map = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(n_points):
                n_hat = np.array([X[i, j], Y[i, j]])
                intensity_map[i, j] = ss_disk.intensity(nu_test, n_hat)
        
        im = ax2.imshow(intensity_map, extent=[-extent*1e6, extent*1e6, -extent*1e6, extent*1e6],
                       origin='lower', cmap='hot', aspect='equal')
        ax2.set_xlabel('X Position (μrad)')
        ax2.set_ylabel('Y Position (μrad)')
        ax2.set_title('2D Intensity Map: Shakura-Sunyaev Disk')
        try:
            plt.colorbar(im, ax=ax2, label='Intensity (W/m²/Hz/sr)')
        except Exception:
            pass
        
        # Plot 3: Comparison of SS vs Relativistic disk
        rel_disk = sources['rel_disk']
        intensities_ss = []
        intensities_rel = []
        
        for r in radii_angular:
            n_hat = np.array([r, 0.0])
            intensities_ss.append(ss_disk.intensity(nu_test, n_hat))
            intensities_rel.append(rel_disk.intensity(nu_test, n_hat))
        
        ax3.plot(radii_angular * 1e6, intensities_ss, 'b-', linewidth=2, label='Standard SS')
        ax3.plot(radii_angular * 1e6, intensities_rel, 'r-', linewidth=2, label='Relativistic')
        ax3.set_xlabel('Radial Distance (μrad)')
        ax3.set_ylabel('Intensity (W/m²/Hz/sr)')
        ax3.set_title('Standard vs Relativistic Disk Comparison')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: BLR intensity at line center
        blr = sources['blr']
        blr_intensities = []
        
        for r in radii_angular:
            n_hat = np.array([r, 0.0])
            intensity = blr.intensity(blr.nu_c, n_hat)  # At line center
            blr_intensities.append(intensity)
        
        ax4.plot(radii_angular * 1e6, blr_intensities, 'g-', linewidth=2, label='BLR (Hα)')
        ax4.set_xlabel('Radial Distance (μrad)')
        ax4.set_ylabel('Intensity (W/m²/Hz/sr)')
        ax4.set_title('Broad Line Region Intensity Profile')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
    except Exception as e:
        print(e)
        for i, ax in enumerate([ax1, ax2, ax3, ax4]):
            ax.text(0.5, 0.5, f'Intensity plot {i+1}\nError: {str(e)[:30]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_visibility_characteristics():
    """Plot 2: Visibility characteristics including zeta plots"""
    if not MATPLOTLIB_AVAILABLE:
        return None
        
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    try:
        sources = create_test_sources()
        if not sources:
            raise Exception("Could not create test sources")
        

        c = 2.99792458e8
        nu_0 = c / 5500e-10  # 600 nm
        wavelength = c / nu_0
        
        # Plot 1: Visibility vs baseline length (showing oscillatory behavior like Fig 1 in PhysRev paper)
        # Calculate the characteristic baseline where minimum should occur
        ss_disk = sources['ss_disk']
        B_min_expected = wavelength * ss_disk.distance / (2 * np.pi * ss_disk.R_in * ss_disk.GM_over_c2)
        
        # Create log-scale baseline range from 0.5 to 1000 km like Fig 1 in the paper
        baseline_lengths = np.logspace(np.log10(500), np.log10(1000000), 500)  # 0.5 to 1000 km in meters
        
        for name, source in sources.items():
            if name == 'blr':
                continue  # Skip BLR for baseline plot
                
            visibilities = []
            for B in baseline_lengths:
                baseline = np.array([B, 0.0, 0.0])
                try:
                    vis = source.V(nu_0, baseline)
                    visibilities.append(abs(vis)**2)  # Plot |V|² like in Fig 1
                except Exception as e:
                    print(e)
                    visibilities.append(0.0)
            
            label = 'Shakura-Sunyaev' if name == 'ss_disk' else 'Relativistic'
            ax1.loglog(baseline_lengths/1000, visibilities, linewidth=2, label=label)  # Log-log plot in km
        
        # Mark the expected minimum location (convert to km)
        ax1.axvline(x=B_min_expected/1000, color='red', linestyle=':', alpha=0.7,
                   label=f'λD/(2πR_in) = {B_min_expected/1000:.1f} km')
        
        ax1.set_xlabel('Baseline Length (km)')
        ax1.set_ylabel('|V(B)|²')
        ax1.set_title('Squared Visibility vs Baseline Length (Log Scale)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0.5, 1000)
        ax1.set_ylim(1e-6, 1.1)
        
        # Plot 2: Visibility vs zeta for SS disk
        ss_disk = sources['ss_disk']
        
        # Estimate characteristic angular size from disk parameters
        theta_char = ss_disk.GM_over_c2 * ss_disk.R_0 / ss_disk.distance
        
        # Calculate zeta range: zeta = pi * B * theta / lambda
        zetas = np.logspace(-1, 2)
        baseline_lengths_zeta = zetas * wavelength / (np.pi * theta_char)
        visibilities_zeta = []
        
        for B in baseline_lengths_zeta:
            baseline = np.array([B, 0.0, 0.0])
            try:
                vis = ss_disk.V(nu_0, baseline)
                visibilities_zeta.append(abs(vis)**2)
            except:
                visibilities_zeta.append(0.0)

        ax2.loglog(zetas, visibilities_zeta, linewidth=2)  # Log-log plot in km        
        # ax2.plot(zetas, visibilities_zeta**2, 'b-', linewidth=2, label='SS Disk')
        ax2.set_xlabel('ζ = πBθ/λ')
        ax2.set_ylabel('|V(ζ)|²')
        ax2.set_title(f'Squared Visibility vs Zeta\n(θ ≈ {theta_char*1e6:.1f} μrad)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(1e-6, 1.1)
        
        # Plot 3: Visibility comparison between models
        baseline_test = np.array([1000.0, 0.0, 0.0])  # 1 km baseline
        frequencies = np.linspace(4e14, 6e14, 30)
        
        for name, source in sources.items():
            if name == 'blr':
                continue
                
            vis_vs_freq = []
            for freq in frequencies:
                try:
                    vis = source.V(freq, baseline_test)
                    vis_vs_freq.append(abs(vis)**2)
                except:
                    vis_vs_freq.append(0.0)
            
            label = 'Shakura-Sunyaev' if name == 'ss_disk' else 'Relativistic'
            ax3.plot(frequencies / 1e14, vis_vs_freq, linewidth=2, label=label)
        
        ax3.set_xlabel('Frequency (×10¹⁴ Hz)')
        ax3.set_ylabel('|V|²')
        ax3.set_title(f'Squared Visibility vs Frequency\n(B = {baseline_test[0]:.0f} m)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.1)
        
        # Plot 4: Phase of visibility
        baseline_lengths_phase = np.linspace(100, 2000, 30)
        vis_phases_ss = []
        vis_amplitudes_ss = []
        
        for B in baseline_lengths_phase:
            baseline = np.array([B, 0.0, 0.0])
            try:
                vis = ss_disk.V(nu_0, baseline)
                vis_phases_ss.append(np.angle(vis) * 180/np.pi)
                vis_amplitudes_ss.append(abs(vis)**2)
            except:
                vis_phases_ss.append(0.0)
                vis_amplitudes_ss.append(0.0)
        
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(baseline_lengths_phase, vis_amplitudes_ss, 'b-', linewidth=2, label='|V|')
        line2 = ax4_twin.plot(baseline_lengths_phase, vis_phases_ss, 'r-', linewidth=2, label='Phase (deg)')
        
        ax4.set_xlabel('Baseline Length (m)')
        ax4.set_ylabel('|V|²', color='blue')
        ax4_twin.set_ylabel('Phase (degrees)', color='red')
        ax4.set_title('Square Visibility Amplitude and Phase')
        ax4.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper right')
        
    except Exception as e:
        print(e)
        for i, ax in enumerate([ax1, ax2, ax3, ax4]):
            ax.text(0.5, 0.5, f'Visibility plot {i+1}\nError: {str(e)[:30]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_chaotic_source_properties():
    """Plot 3: ChaoticSource inheritance properties"""
    if not MATPLOTLIB_AVAILABLE:
        return None
        
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    try:
        sources = create_test_sources()
        if not sources:
            raise Exception("Could not create test sources")
        
        # Test parameters
        nu_0 = 5e14
        delta_nu_narrow = 1e11  # 100 GHz
        delta_nu_wide = 1e13    # 10 THz
        
        # Plot 1: g1 function vs time delay
        delta_t_range = np.linspace(0, 5e-11, 1000)  # 0 to 50 ps
        
        ss_disk = sources['ss_disk']
        g1_narrow = [ss_disk.g1(dt, nu_0, delta_nu_narrow) for dt in delta_t_range]
        g1_wide = [ss_disk.g1(dt, nu_0, delta_nu_wide) for dt in delta_t_range]
        
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
        g2_minus_one_narrow = [ss_disk.g2_minus_one(dt, nu_0, delta_nu_narrow) for dt in delta_t_range]
        g2_minus_one_wide = [ss_disk.g2_minus_one(dt, nu_0, delta_nu_wide) for dt in delta_t_range]
        
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
        bandwidths = np.logspace(10, 14, 50)
        coherence_times = []
        
        for bw in bandwidths:
            # Find 1/e point
            dt_test = np.linspace(0, 1e-9, 1000)
            g1_test = [abs(ss_disk.g1(dt, nu_0, bw)) for dt in dt_test]
            
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
        
        # Plot 4: Comparison of g2-1 for different AGN models
        delta_t_test = 1e-11  # 10 ps
        delta_nu_test = 1e12  # 1 THz
        
        model_names = []
        g2_values = []
        
        for name, source in sources.items():
            if name == 'blr':
                continue  # Skip BLR for this comparison
            try:
                g2_val = source.g2_minus_one(delta_t_test, nu_0, delta_nu_test)
                model_names.append('SS Disk' if name == 'ss_disk' else 'Rel. Disk')
                g2_values.append(g2_val)
            except:
                pass
        
        if model_names:
            bars = ax4.bar(model_names, g2_values, color=['blue', 'red'], alpha=0.7)
            ax4.set_ylabel('g²(Δt) - 1')
            ax4.set_title(f'g²-1 Comparison\n(Δt={delta_t_test*1e12:.0f}ps, Δν={delta_nu_test:.0e}Hz)')
            ax4.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, val in zip(bars, g2_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
    except Exception as e:
        for i, ax in enumerate([ax1, ax2, ax3, ax4]):
            ax.text(0.5, 0.5, f'Coherence plot {i+1}\nError: {str(e)[:30]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_relativistic_effects():
    """Plot 4: Relativistic effects demonstration"""
    if not MATPLOTLIB_AVAILABLE:
        return None
        
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    try:
        # Create disks with different spin parameters
        spins = [0.0, 0.5, 0.9]
        colors = ['blue', 'green', 'red']
        
        disks = {}
        for spin in spins:
            disks[spin] = RelativisticDisk(
                I_0=1e-15, R_0=10.0, R_in=3.0, n=3.0,
                inclination=np.pi/4, distance=1e20, GM_over_c2=1e9,
                spin_parameter=spin
            )
        
        nu_test = 5e14
        
        # Plot 1: ISCO radius vs spin
        spin_range = np.linspace(-0.99, 0.99, 50)
        isco_radii = []
        
        for spin in spin_range:
            test_disk = RelativisticDisk(
                I_0=1e-15, R_0=10.0, R_in=1.0, n=3.0,
                inclination=0, distance=1e20, GM_over_c2=1e9,
                spin_parameter=spin
            )
            isco_radii.append(test_disk.R_isco)
        
        ax1.plot(spin_range, isco_radii, 'b-', linewidth=2)
        ax1.set_xlabel('Spin Parameter a/M')
        ax1.set_ylabel('ISCO Radius (GM/c²)')
        ax1.set_title('ISCO Radius vs Black Hole Spin')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=6.0, color='k', linestyle='--', alpha=0.5, label='Schwarzschild (a=0)')
        ax1.legend()
        
        # Plot 2: Intensity profiles for different spins
        radii_angular = np.linspace(0, 3e-6, 30)
        
        for spin, color in zip(spins, colors):
            intensities = []
            for r in radii_angular:
                n_hat = np.array([r, 0.0])
                intensity = disks[spin].intensity(nu_test, n_hat)
                intensities.append(intensity)
            
            ax2.plot(radii_angular * 1e6, intensities, color=color, linewidth=2, 
                    label=f'a/M = {spin}')
        
        ax2.set_xlabel('Radial Distance (μrad)')
        ax2.set_ylabel('Intensity (W/m²/Hz/sr)')
        ax2.set_title('Intensity Profiles: Spin Effects')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Visibility vs baseline for different spins
        baseline_lengths = np.logspace(2, 4, 30)
        
        for spin, color in zip(spins, colors):
            visibilities = []
            for B in baseline_lengths:
                baseline = np.array([B, 0.0, 0.0])
                try:
                    vis = disks[spin].V(nu_test, baseline)
                    visibilities.append(abs(vis))
                except:
                    visibilities.append(0.0)
            
            ax3.semilogx(baseline_lengths, visibilities, color=color, linewidth=2,
                        label=f'a/M = {spin}')
        
        ax3.set_xlabel('Baseline Length (m)')
        ax3.set_ylabel('|V(B)|')
        ax3.set_title('Visibility: Relativistic Effects')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.1)
        
        # Plot 4: Doppler factor demonstration
        R_test = 6.0  # GM/c^2 units
        phi_range = np.linspace(0, 2*np.pi, 100)
        
        for spin, color in zip([0.0, 0.9], ['blue', 'red']):
            doppler_factors = []
            for phi in phi_range:
                doppler_factor = disks[spin]._doppler_factor(R_test, phi)
                doppler_factors.append(doppler_factor)
            
            ax4.plot(phi_range * 180/np.pi, doppler_factors, color=color, linewidth=2,
                    label=f'a/M = {spin}')
        
        ax4.set_xlabel('Azimuthal Angle φ (degrees)')
        ax4.set_ylabel('Doppler Factor D')
        ax4.set_title(f'Doppler Factor vs Azimuth\n(R = {R_test} GM/c²)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
    except Exception as e:
        for i, ax in enumerate([ax1, ax2, ax3, ax4]):
            ax.text(0.5, 0.5, f'Relativistic plot {i+1}\nError: {str(e)[:30]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
    
    plt.tight_layout()
    return fig


def create_documentation_fallback():
    """Create documentation when plotting fails"""
    try:
        with open('plot_agn_description.txt', 'w') as f:
            f.write("AGN Source Models - Comprehensive Test Plots\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("This document describes the comprehensive test plots for AGN source models\n")
            f.write("that demonstrate intensity and visibility characteristics.\n\n")
            
            f.write("Plot 1: Intensity Profiles\n")
            f.write("- Radial intensity profiles for different AGN models\n")
            f.write("- 2D intensity maps\n")
            f.write("- Comparison between standard and relativistic disks\n")
            f.write("- Broad Line Region intensity characteristics\n\n")
            
            f.write("Plot 2: Visibility Characteristics\n")
            f.write("- Visibility vs baseline length\n")
            f.write("- **Visibility vs zeta (ζ = πBθ/λ)** for AGN disks\n")
            f.write("- Visibility vs frequency comparison\n")
            f.write("- Visibility amplitude and phase\n\n")
            
            f.write("Plot 3: ChaoticSource Properties\n")
            f.write("- First-order coherence function g¹(Δt)\n")
            f.write("- Second-order coherence function g²(Δt)-1\n")
            f.write("- Coherence time vs bandwidth relationship\n")
            f.write("- Comparison of g²-1 between AGN models\n\n")
            
            f.write("Plot 4: Relativistic Effects\n")
            f.write("- ISCO radius vs black hole spin\n")
            f.write("- Intensity profiles for different spins\n")
            f.write("- Visibility changes due to relativistic effects\n")
            f.write("- Doppler factor demonstration\n\n")
            
            f.write("Key Features:\n")
            f.write("- Comprehensive testing of all AGN source models\n")
            f.write("- Demonstration of ChaoticSource inheritance\n")
            f.write("- Relativistic effects in accretion disks\n")
            f.write("- Visibility vs zeta plots for AGN sources\n")
            f.write("- Comparison between different AGN components\n")
        
        print("📄 Created comprehensive AGN plot description document")
        
    except Exception as e:
        print(f"Could not create documentation file: {e}")


def main():
    """Create all comprehensive AGN plots and save to PDF"""
    print("Creating Comprehensive AGN Source Model Plots...")
    print("=" * 50)
    
    if not DEPENDENCIES_AVAILABLE:
        print("⚠️  Dependencies not available - creating documentation instead")
        create_documentation_fallback()
        return
    
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  Matplotlib not available - creating documentation instead")
        create_documentation_fallback()
        return
    
    try:
        with PdfPages('plot_agn.pdf') as pdf:
            print("1. Creating intensity profile plots...")
            fig1 = plot_intensity_profiles()
            if fig1:
                pdf.savefig(fig1, bbox_inches='tight')
                plt.close(fig1)
            
            print("2. Creating visibility characteristic plots...")
            fig2 = plot_visibility_characteristics()
            if fig2:
                pdf.savefig(fig2, bbox_inches='tight')
                plt.close(fig2)
            
            print("3. Creating chaotic source property plots...")
            fig3 = plot_chaotic_source_properties()
            if fig3:
                pdf.savefig(fig3, bbox_inches='tight')
                plt.close(fig3)
            
            print("4. Creating relativistic effects plots...")
            fig4 = plot_relativistic_effects()
            if fig4:
                pdf.savefig(fig4, bbox_inches='tight')
                plt.close(fig4)
            
            # Add metadata
            d = pdf.infodict()
            d['Title'] = 'Comprehensive AGN Source Model Plots'
            d['Author'] = 'Intensity Interferometry Analysis'
            d['Subject'] = 'AGN intensity and visibility characteristics'
            d['Keywords'] = 'AGN, Accretion Disk, BLR, Relativistic, Visibility, Intensity'
            d['Creator'] = 'plot_agn.py'
        
        print("\n✅ All AGN plots saved to plot_agn.pdf!")
        print("\nThe PDF contains 4 pages with comprehensive visualizations:")
        print("1. Intensity profiles for different AGN models")
        print("2. Visibility characteristics including zeta plots")
        print("3. ChaoticSource inheritance properties")
        print("4. Relativistic effects in accretion disks")
        print("\n🎯 KEY FEATURES: Intensity and visibility plots for AGN models!")
        
    except Exception as e:
        print(f"❌ Error creating plots: {e}")
        print("Creating documentation instead...")
        create_documentation_fallback()
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
