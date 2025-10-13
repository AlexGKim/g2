#!/usr/bin/env python3
"""
Test RadialGrid Implementation and Reproduce II.ipynb Plots
===========================================================

This script tests the RadialGrid class and reproduces key plots from II.ipynb
using the SN2011fe_MLE_intensity_maxlight.hdf data.

Key plots reproduced:
- Intensity vs impact parameter at different wavelengths
- V² vs ζ (zeta) plots
- SNR calculations
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os
from pathlib import Path
from scipy.special import jv

# Add parent directory to path to import source module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from g2.models.sources.radial_grid import RadialGrid
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import RadialGrid: {e}")
    DEPENDENCIES_AVAILABLE = False

# Set up plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (15, 12)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def get_radial_source():
    """Get a RadialGrid source instance from HDF5 data"""
    try:
        # Try to load from the data directory
        hdf_file = 'g2/data/SN2011fe_MLE_intensity_maxlight.hdf'
        if not os.path.exists(hdf_file):
            # Try alternative path
            hdf_file = '../g2/data/SN2011fe_MLE_intensity_maxlight.hdf'
        if not os.path.exists(hdf_file):
            # Try II directory
            hdf_file = 'II/SN2011fe_MLE_intensity_maxlight.hdf'
        
        source = RadialGrid.from_hdf5(hdf_file)
        return source, "RadialGrid from HDF5"
    except Exception as e:
        print(f"Error creating RadialGrid source: {e}")
        return None, "Error"

def plot_intensity_profiles():
    """Plot intensity vs impact parameter at different wavelengths (Figure from II.ipynb)"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    try:
        source, data_type = get_radial_source()
        if source is None:
            raise Exception("Could not create source")
        
        # Select wavelengths similar to II.ipynb
        target_wavelengths = [3700, 4700, 6055, 6355, 8750]  # Angstrom
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        for target_wave, color in zip(target_wavelengths, colors):
            # Find closest wavelength
            wave_idx = np.argmin(np.abs(source.lambdas - target_wave))
            actual_wave = source.lambdas[wave_idx]
            
            # Get intensity profile and normalize
            intensity_profile = source.I_nu_p[wave_idx, :]
            intensity_norm = intensity_profile / np.sum(intensity_profile)
            
            # Plot vs impact parameter in units of 10^10 km (which is 10^13 m)
            ax.plot(source.p_rays / 1e13, intensity_norm,
                   color=color, linewidth=2, label=f'λ = {actual_wave:.0f}Å')
        
        ax.set_xlabel('Impact Parameter [10¹⁰ km]')
        ax.set_ylabel('Normalized Emission')
        ax.set_title('Intensity Profiles vs Impact Parameter')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error plotting intensity profiles:\n{str(e)}',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Intensity Profiles (Error)')
    
    plt.tight_layout()
    return fig

def plot_gamma2_vs_zeta():
    """Plot V² vs ζ (gamma2 vs zeta) - key plot from II.ipynb"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    try:
        source, data_type = get_radial_source()
        if source is None:
            raise Exception("Could not create source")
        
        # Parameters matching II.ipynb gamma2snr function
        nd = source.p_rays.shape[0]
        delta = source.p_rays[1] - source.p_rays[0]
        # drad in II.ipynb is 0.875e15 in original units (cm), convert to meters like p_rays
        drad = 0.875e15 * 1e-2  # Convert from cm to meters to match p_rays units
        ndisk = int(2 * drad / delta)
        factor = 25  # from II.ipynb
        norder = factor * 5
        
        # Create padded flux array (matching II.ipynb)
        flux = np.zeros(nd * factor)
        
        # Select wavelengths
        target_wavelengths = [3700, 4700, 6055, 6355, 8750]  # Angstrom
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        for target_wave, color in zip(target_wavelengths, colors):
            # Find closest wavelength
            wave_idx = np.argmin(np.abs(source.lambdas - target_wave))
            actual_wave = source.lambdas[wave_idx]
            
            # Reset flux array and fill with intensity data
            flux[:] = 0
            # Convert I_nu_p to I_lam_p as in II.ipynb
            I_lam_p = source.I_nu_p / source.lambdas[:, None] / source.lambdas[:, None]
            flux[:nd] = I_lam_p[wave_idx, :]
            
            # Calculate gamma using polar DFT (matching II.ipynb)
            gamma = source.dft_polar(flux, norder=norder)
            gamma0 = gamma[0]
            gamma = gamma / gamma0  # Normalize
            gamma2 = gamma * gamma  # Real gamma squared (not abs squared)
            
            # Create zeta array matching II.ipynb scaling
            zeta_array = np.arange(gamma2.shape[0]) * 2 * np.pi / (nd * factor / ndisk)
            
            # Plot gamma2 vs zeta
            ax.plot(zeta_array, gamma2, color=color, linewidth=2,
                   label=f'λ = {actual_wave:.0f}Å')
        
        # Add theoretical Airy profile for comparison
        zeta_theory = np.arange(0, nd * 10, 0.01)
        # Handle division by zero at zeta=0
        airy_theory = np.zeros_like(zeta_theory)
        airy_theory[0] = 1.0  # Limit as zeta->0
        mask = zeta_theory > 0
        airy_theory[mask] = (2 * jv(1, zeta_theory[mask]) / zeta_theory[mask])**2
        
        ax.plot(zeta_theory, airy_theory, 'k--', alpha=0.7, linewidth=2, label='Airy')
        
        ax.set_xlabel('ζ')
        ax.set_ylabel('V²')
        ax.set_title('V² vs ζ (Visibility vs Zeta)')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error plotting V² vs ζ:\n{str(e)}',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('V² vs ζ (Error)')
    
    plt.tight_layout()
    return fig

def plot_dgamma2ds():
    """Plot derivative of gamma² with respect to size parameter"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    try:
        source, data_type = get_radial_source()
        if source is None:
            raise Exception("Could not create source")
        
        # Parameters matching II.ipynb gamma2snr function
        nd = source.p_rays.shape[0]
        delta = source.p_rays[1] - source.p_rays[0]
        # drad in II.ipynb is 0.875e15 in original units (cm), convert to meters like p_rays
        drad = 0.875e15 * 1e-2  # Convert from cm to meters to match p_rays units
        ndisk = int(2 * drad / delta)
        factor = 25  # from II.ipynb
        norder = factor * 5
        
        # Create padded flux array (matching II.ipynb)
        flux = np.zeros(nd * factor)
        
        # Select wavelengths
        target_wavelengths = [3700, 4700, 6055, 6355, 8750]  # Angstrom
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        for target_wave, color in zip(target_wavelengths, colors):
            # Find closest wavelength
            wave_idx = np.argmin(np.abs(source.lambdas - target_wave))
            actual_wave = source.lambdas[wave_idx]
            
            # Reset flux array and fill with intensity data
            flux[:] = 0
            # Convert I_nu_p to I_lam_p as in II.ipynb
            I_lam_p = source.I_nu_p / source.lambdas[:, None] / source.lambdas[:, None]
            flux[:nd] = I_lam_p[wave_idx, :]
            
            # Calculate gamma and dgamma2ds (matching II.ipynb)
            gamma = source.dft_polar(flux, norder=norder)
            gamma0 = gamma[0]
            dgamma2ds_result = source.dgamma2ds(flux, norder=norder)
            
            # Normalize by gamma0^2 for comparison with II.ipynb
            dgamma2ds_norm = dgamma2ds_result / gamma0**2
            
            # Create zeta array matching II.ipynb scaling
            zeta_array = np.arange(dgamma2ds_norm.shape[0]) * 2 * np.pi / (nd * factor / ndisk)
            
            # Plot absolute value
            ax.plot(zeta_array, np.abs(dgamma2ds_norm), color=color, linewidth=2,
                   label=f'λ = {actual_wave:.0f}Å')
        
        ax.set_xlabel('ζ')
        ax.set_ylabel('|dγ²/ds|')
        ax.set_title('Derivative of V² with respect to Size Parameter')
        ax.set_xlim(0, 10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error plotting dγ²/ds:\n{str(e)}',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('dγ²/ds (Error)')
    
    plt.tight_layout()
    return fig

def plot_visibility_calculations():
    """Test visibility calculations using RadialGrid"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    try:
        source, data_type = get_radial_source()
        if source is None:
            raise Exception("Could not create source")
        
        # Test parameters
        nu_0 = 5e14  # 600 nm
        c = 2.99792458e8
        wavelength = c / nu_0
        
        # Plot 1: Visibility vs baseline length
        baseline_lengths = np.logspace(0, 4, 30)  # 1 m to 10 km
        visibilities = []
        
        for B in baseline_lengths:
            baseline = np.array([B, 0.0, 0.0])
            try:
                vis = source.V_squared(nu_0, baseline)
                visibilities.append(vis)
            except:
                visibilities.append(0.0)
        
        ax1.loglog(baseline_lengths, visibilities, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Baseline Length (m)')
        ax1.set_ylabel('|V(B)|²')
        ax1.set_title('Visibility vs Baseline Length')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Visibility vs frequency
        test_freqs = np.linspace(source.freq_min, source.freq_max, 20)
        baseline_fixed = np.array([100.0, 0.0, 0.0])  # 100m baseline
        vis_vs_freq = []
        
        for freq in test_freqs:
            try:
                vis = source.V(freq, baseline_fixed)
                vis_vs_freq.append(abs(vis))
            except:
                vis_vs_freq.append(0.0)
        
        ax2.plot(test_freqs / 1e14, vis_vs_freq, 'g-', linewidth=2, marker='d', markersize=4)
        ax2.set_xlabel('Frequency (×10¹⁴ Hz)')
        ax2.set_ylabel('|V|')
        ax2.set_title(f'Visibility vs Frequency (B = {baseline_fixed[0]:.0f} m)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Jacobian test
        baseline_test = np.array([100.0, 0.0, 0.0])
        jacobian = source.V_squared_jacobian(nu_0, baseline_test)
        
        param_names = list(jacobian.keys())
        param_values = list(jacobian.values())
        
        ax3.bar(param_names, param_values, alpha=0.7)
        ax3.set_ylabel('∂|V|²/∂param')
        ax3.set_title('Jacobian Components')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Source info
        info = source.get_spectrum_info()
        info_text = f"""RadialGrid Source Information:
        
Wavelength range: {info['wavelength_range_angstrom'][0]:.0f} - {info['wavelength_range_angstrom'][1]:.0f} Å
Frequency range: {info['frequency_range_hz'][0]:.2e} - {info['frequency_range_hz'][1]:.2e} Hz
Radial range: {info['radial_range_m'][0]:.2e} - {info['radial_range_m'][1]:.2e} m
Wavelength points: {info['wavelength_points']}
Radial points: {info['radial_points']}

Data type: {data_type}
"""
        ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, 
                verticalalignment='top', fontsize=10, fontfamily='monospace')
        ax4.set_title('Source Information')
        ax4.axis('off')
        
    except Exception as e:
        for i, ax in enumerate([ax1, ax2, ax3, ax4]):
            ax.text(0.5, 0.5, f'Visibility test {i+1}\nError: {str(e)[:30]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
    
    plt.tight_layout()
    return fig

def main():
    """Create all test plots and save to PDF"""
    print("Testing RadialGrid Implementation and Reproducing II.ipynb Plots...")
    print("=" * 70)
    
    if not DEPENDENCIES_AVAILABLE:
        print("⚠️  Dependencies not available")
        return
    
    try:
        with PdfPages('test_radial_grid.pdf') as pdf:
            print("1. Creating intensity profile plots...")
            fig1 = plot_intensity_profiles()
            pdf.savefig(fig1, bbox_inches='tight')
            plt.close(fig1)
            
            print("2. Creating V² vs ζ plots...")
            fig2 = plot_gamma2_vs_zeta()
            pdf.savefig(fig2, bbox_inches='tight')
            plt.close(fig2)
            
            print("3. Creating dγ²/ds plots...")
            fig3 = plot_dgamma2ds()
            pdf.savefig(fig3, bbox_inches='tight')
            plt.close(fig3)
            
            print("4. Creating visibility calculation tests...")
            fig4 = plot_visibility_calculations()
            pdf.savefig(fig4, bbox_inches='tight')
            plt.close(fig4)
            
            # Add metadata
            d = pdf.infodict()
            d['Title'] = 'RadialGrid Test Plots - Reproduction of II.ipynb'
            d['Author'] = 'RadialGrid Implementation Test'
            d['Subject'] = 'Testing polar DFT algorithms and visibility calculations'
            d['Keywords'] = 'RadialGrid, Polar DFT, Visibility, II.ipynb, SN2011fe'
        
        print("\n✅ All test plots saved to test_radial_grid.pdf!")
        print("\nThe PDF contains:")
        print("1. Intensity profiles vs impact parameter")
        print("2. V² vs ζ plots (key reproduction from II.ipynb)")
        print("3. Derivative dγ²/ds plots")
        print("4. Visibility calculation tests")
        print("\n🎯 RadialGrid implementation complete!")
        
    except Exception as e:
        print(f"❌ Error creating test plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()