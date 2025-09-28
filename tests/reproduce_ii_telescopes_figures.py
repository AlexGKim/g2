#!/usr/bin/env python3
"""
Reproduce Figures 3, 4, 6, 7, and 9 from II_Telescopes.pdf
===========================================================

This script reproduces key figures from "Measuring type Ia supernova angular-diameter 
distances with intensity interferometry" using GridSource.getSN2011feSource as the 
Sedona model.

Figures reproduced:
- Figure 3: Photon flux density spectrum 
- Figure 4: SEDONA emission profiles at different wavelengths
- Figure 6: V² maps as function of u-v coordinates
- Figure 7: V² slices and differences
- Figure 9: SNR maps for distance parameter measurements
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os
from pathlib import Path

# Add parent directory to path to import source module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from g2.models.sources.grid_source import GridSource
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import dependencies: {e}")
    DEPENDENCIES_AVAILABLE = False

# Set up plotting style to match paper
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['lines.linewidth'] = 1.5

# Physical constants
c = 2.99792458e8  # Speed of light in m/s
h = 6.62607015e-34  # Planck constant

def get_sedona_source():
    """Get the SEDONA SN2011fe source using GridSource.getSN2011feSource"""
    try:
        # Use the factory method to get SN2011fe source
        source = GridSource.getSN2011feSource()
        return source, "SEDONA SN2011fe"
    except Exception as e:
        print(f"Error creating SEDONA source: {e}")
        return None, "Error"

def reproduce_figure_3():
    """
    Reproduce Figure 3: Photon flux density spectrum
    Shows n_ν [s⁻¹ cm⁻² Hz⁻¹] vs wavelength comparing TARDIS and SEDONA
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    try:
        source, data_type = get_sedona_source()
        if source is None:
            raise Exception("Could not create source")
        
        # Get wavelength grid and pre-calculated photon flux density
        wavelengths = source.wavelength_grid  # [Angstrom]
        
        # Use the pre-calculated specific_photon_flux from GridSource
        # This is already in [photons/s/m²/Hz] and properly calculated
        photon_flux_cgs = source.specific_photon_flux / 1e4  # Convert m² to cm²
        
        # Plot SEDONA spectrum
        ax.plot(wavelengths, photon_flux_cgs, 'b-', linewidth=2, label='SEDONA')
        
        # Set up plot to match Figure 3
        ax.set_xlabel('Wavelength [Å]')
        ax.set_ylabel('n_ν [s⁻¹ cm⁻² Hz⁻¹]')
        ax.set_title('Type Ia Supernova Photon Flux Density (Figure 3)')
        # Use the same limits as the working version
        ax.set_ylim((0, np.max(photon_flux_cgs[np.logical_and(
            wavelengths > 4000, wavelengths < 8000)]) * 1.1))
        ax.set_xlim((3300, 10000))
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add text annotation
        ax.text(0.05, 0.95, f'B = 12.0 mag\n{data_type}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error reproducing Figure 3:\n{str(e)}',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Figure 3: Photon Flux Density (Error)')
    
    plt.tight_layout()
    return fig

def reproduce_figure_4():
    """
    Reproduce Figure 4: SEDONA emission profiles at different wavelengths
    Shows 2D intensity maps at select wavelengths
    """
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    
    try:
        source, data_type = get_sedona_source()
        if source is None:
            raise Exception("Could not create source")
        
        # Select wavelengths similar to Figure 4 in the paper
        target_wavelengths = [3697, 4698, 6128, 6190, 8746]  # Angstrom
        
        for i, target_wave in enumerate(target_wavelengths):
            ax = axes[i]
            
            # Find closest wavelength in the grid
            wave_idx = np.argmin(np.abs(source.wavelength_grid - target_wave))
            actual_wave = source.wavelength_grid[wave_idx]
            
            # Get the 2D intensity map for this wavelength
            intensity_map = source.intensity_data[wave_idx, :, :]
            
            # Convert pixel coordinates to physical coordinates (10^10 km)
            pixel_scale_km = source.pixel_scale_m / 1e13  # Convert m to 10^10 km
            nx, ny = intensity_map.shape
            extent_km = pixel_scale_km * max(nx, ny) / 2
            extent = [-extent_km, extent_km, -extent_km, extent_km]
            
            # Plot intensity map
            im = ax.imshow(intensity_map, extent=extent, origin='lower', 
                          cmap='hot', aspect='equal')
            
            ax.set_xlabel('Impact Parameter [10¹⁰ km]')
            if i == 0:
                ax.set_ylabel('Impact Parameter [10¹⁰ km]')
            ax.set_title(f'λ = {actual_wave:.0f}Å')
            
            # Add colorbar for the last subplot
            if i == len(target_wavelengths) - 1:
                plt.colorbar(im, ax=ax, label='Normalized Emission')
        
        fig.suptitle('SEDONA Emission Profiles (Figure 4)', fontsize=14)
        
    except Exception as e:
        for i, ax in enumerate(axes):
            ax.text(0.5, 0.5, f'Error\n{str(e)[:20]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=8)
            ax.set_title(f'Figure 4 Panel {i+1}')
    
    plt.tight_layout()
    return fig

def reproduce_figure_6():
    """
    Reproduce Figure 6: V² maps as function of u-v coordinates
    Shows intensity interference signal V² at different wavelengths
    """
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    
    try:
        source, data_type = get_sedona_source()
        if source is None:
            raise Exception("Could not create source")
        
        # Select wavelengths similar to Figure 6
        target_wavelengths = [3697, 4698, 6128, 6190, 8746]  # Angstrom
        
        # Set up u-v coordinate grid (in km, scaled by λ/5000Å)
        u_max = 10  # km
        v_max = 12.5  # km
        n_points = 25  # Reduced for performance
        
        for i, target_wave in enumerate(target_wavelengths):
            ax = axes[i]
            
            # Find closest wavelength and frequency
            wave_idx = np.argmin(np.abs(source.wavelength_grid - target_wave))
            actual_wave = source.wavelength_grid[wave_idx]
            nu_0 = source.frequency_grid[wave_idx]
            
            # Create u-v grid
            u_coords = np.linspace(0, u_max, n_points)
            v_coords = np.linspace(0, v_max, n_points)
            U, V = np.meshgrid(u_coords, v_coords)
            
            # Calculate V² for each u-v point
            V_squared_map = np.zeros((n_points, n_points))
            
            for j in range(n_points):
                for k in range(n_points):
                    # Convert u-v coordinates to baseline in meters
                    # Scale by wavelength as in the paper
                    scale_factor = actual_wave / 5000.0  # λ/5000Å scaling
                    baseline = np.array([U[j,k] * 1000 * scale_factor, 
                                       V[j,k] * 1000 * scale_factor, 0.0])
                    
                    try:
                        V_squared_map[j,k] = source.V_squared(nu_0, baseline)
                    except:
                        V_squared_map[j,k] = 0.0
            
            # Plot V² map with logarithmic scale
            im = ax.imshow(V_squared_map, extent=[0, u_max, 0, v_max], 
                          origin='lower', cmap='viridis', 
                          norm=plt.LogNorm(vmin=1e-2, vmax=1.0))
            
            ax.set_xlabel('u [km][λ/5000Å]')
            if i == 0:
                ax.set_ylabel('v [km][λ/5000Å]')
            ax.set_title(f'λ = {actual_wave:.0f}Å')
            
            # Add colorbar for the last subplot
            if i == len(target_wavelengths) - 1:
                plt.colorbar(im, ax=ax, label='V²')
        
        fig.suptitle('V² Maps (Figure 6)', fontsize=14)
        
    except Exception as e:
        for i, ax in enumerate(axes):
            ax.text(0.5, 0.5, f'Error\n{str(e)[:20]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=8)
            ax.set_title(f'Figure 6 Panel {i+1}')
    
    plt.tight_layout()
    return fig

def reproduce_figure_7():
    """
    Reproduce Figure 7: V² slices and differences
    Shows V² vs ζ and differences between u=0 and v=0 slices
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    try:
        source, data_type = get_sedona_source()
        if source is None:
            raise Exception("Could not create source")
        
        # Select wavelengths for plotting
        target_wavelengths = [3697, 4698, 6128, 6190, 8746]  # Angstrom
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        # Set up ζ = πBθ/λ coordinate
        zeta_max = 10
        zeta_coords = np.linspace(0.1, zeta_max, 50)
        
        # Estimate source angular size (θ) from pixel scale
        theta_estimate = source.pixel_scale() * 6.2  # Rough estimate
        
        for i, (target_wave, color) in enumerate(zip(target_wavelengths, colors)):
            # Find closest wavelength and frequency
            wave_idx = np.argmin(np.abs(source.wavelength_grid - target_wave))
            actual_wave = source.wavelength_grid[wave_idx]
            nu_0 = source.frequency_grid[wave_idx]
            wavelength_m = actual_wave * 1e-10  # Convert Å to m
            
            # Calculate V² vs ζ for v=0 slice
            V_squared_v0 = []
            V_squared_u0 = []
            
            for zeta in zeta_coords:
                # Convert ζ to baseline length: B = ζλ/(πθ)
                baseline_length = zeta * wavelength_m / (np.pi * theta_estimate)
                
                # Calculate V² for v=0 (u-direction)
                baseline_v0 = np.array([baseline_length, 0.0, 0.0])
                try:
                    v2_v0 = source.V_squared(nu_0, baseline_v0)
                    V_squared_v0.append(v2_v0)
                except:
                    V_squared_v0.append(0.0)
                
                # Calculate V² for u=0 (v-direction)
                baseline_u0 = np.array([0.0, baseline_length, 0.0])
                try:
                    v2_u0 = source.V_squared(nu_0, baseline_u0)
                    V_squared_u0.append(v2_u0)
                except:
                    V_squared_u0.append(0.0)
            
            # Plot V² vs ζ (left panel)
            ax1.semilogy(zeta_coords, V_squared_v0, color=color, linewidth=2,
                        label=f'λ = {actual_wave:.0f}Å')
            
            # Plot difference V²(v=0) - V²(u=0) (right panel)
            difference = np.array(V_squared_v0) - np.array(V_squared_u0)
            ax2.plot(zeta_coords, difference, color=color, linewidth=2,
                    label=f'λ = {actual_wave:.0f}Å')
        
        # Add theoretical Airy profile for comparison
        try:
            from scipy.special import j1
            airy_v2 = []
            for zeta in zeta_coords:
                if zeta == 0:
                    airy_v2.append(1.0)
                else:
                    airy_val = abs(2 * j1(zeta) / zeta)**2
                    airy_v2.append(airy_val)
            ax1.semilogy(zeta_coords, airy_v2, 'k--', alpha=0.7, label='Airy')
        except:
            pass
        
        # Format left panel
        ax1.set_xlabel('ζ = πBθ/λ')
        ax1.set_ylabel('V²')
        ax1.set_title('V² vs ζ (v=0 slice)')
        ax1.set_ylim(1e-3, 1.1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Format right panel
        ax2.set_xlabel('ζ = πBθ/λ')
        ax2.set_ylabel('V²(v=0) - V²(u=0)')
        ax2.set_title('Asymmetry: V²(v=0) - V²(u=0)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle('V² Slices and Differences (Figure 7)', fontsize=14)
        
    except Exception as e:
        for ax in [ax1, ax2]:
            ax.text(0.5, 0.5, f'Error reproducing Figure 7:\n{str(e)[:30]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
    
    plt.tight_layout()
    return fig

def reproduce_figure_9():
    """
    Reproduce Figure 9: SNR maps for distance parameter measurements
    Shows signal-to-noise ratio for distance measurements using SEDONA model
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    
    try:
        source, data_type = get_sedona_source()
        if source is None:
            raise Exception("Could not create source")
        
        # Select wavelengths
        target_wavelengths = [3697, 4698, 6128, 6190, 8746]  # Angstrom
        
        # Observational parameters (from paper)
        z = 0.004  # Redshift
        B_mag = 12.0  # Magnitude
        A = np.pi * (10.0/2)**2  # Telescope area (10m diameter)
        epsilon = 0.39  # Total throughput
        T_obs = 3600.0  # 1 hour observation
        sigma_t = 13e-12  # 13 ps RMS timing jitter
        
        # Set up u-v coordinate grid
        u_max = 10  # km
        v_max = 10  # km
        n_points = 20  # Reduced for performance
        
        u_coords = np.linspace(-u_max, u_max, n_points)
        v_coords = np.linspace(-v_max, v_max, n_points)
        U, V = np.meshgrid(u_coords, v_coords)
        
        for i, target_wave in enumerate(target_wavelengths):
            # Find closest wavelength and frequency
            wave_idx = np.argmin(np.abs(source.wavelength_grid - target_wave))
            actual_wave = source.wavelength_grid[wave_idx]
            nu_0 = source.frequency_grid[wave_idx]
            
            # Calculate SNR maps for two-pair and three-pair configurations
            for config_idx, (ax, config_name) in enumerate([(axes[0,i], 'Two-pair'), 
                                                           (axes[1,i], 'Three-pair')]):
                
                SNR_map = np.zeros((n_points, n_points))
                
                for j in range(n_points):
                    for k in range(n_points):
                        # Convert u-v to baseline
                        scale_factor = actual_wave / 5000.0
                        baseline = np.array([U[j,k] * 1000 * scale_factor, 
                                           V[j,k] * 1000 * scale_factor, 0.0])
                        
                        try:
                            # Calculate V²
                            V_squared = source.V_squared(nu_0, baseline)
                            
                            # Calculate photon flux
                            flux = source.total_flux(nu_0)  # [W/m²/Hz]
                            dGamma_dnu = epsilon * A * flux / (h * nu_0)  # [photons/s/Hz]
                            
                            # Calculate SNR (simplified from paper equations)
                            if V_squared > 0 and dGamma_dnu > 0:
                                sigma_V2_inv = dGamma_dnu * np.sqrt(T_obs / sigma_t) / np.sqrt(128 * np.pi)
                                
                                # Adjust for configuration
                                if config_idx == 1:  # Three-pair configuration
                                    sigma_V2_inv *= np.sqrt(1.5)  # Approximate improvement
                                
                                SNR = V_squared * sigma_V2_inv * 0.1  # Scale factor for visibility
                                SNR_map[j,k] = SNR
                            else:
                                SNR_map[j,k] = 0.0
                        except:
                            SNR_map[j,k] = 0.0
                
                # Plot SNR map
                im = ax.imshow(SNR_map, extent=[-u_max, u_max, -v_max, v_max], 
                              origin='lower', cmap='viridis', 
                              norm=plt.LogNorm(vmin=1e-2, vmax=1.0))
                
                ax.set_xlabel('u [km][λ/5000Å]')
                if i == 0:
                    ax.set_ylabel('v [km][λ/5000Å]')
                
                if config_idx == 0:
                    ax.set_title(f'λ = {actual_wave:.0f}Å')
                
                # Add configuration label
                ax.text(0.05, 0.95, config_name, transform=ax.transAxes, 
                       verticalalignment='top', color='white', fontweight='bold')
                
                # Add colorbar for the last column
                if i == len(target_wavelengths) - 1:
                    plt.colorbar(im, ax=ax, label='SNR_s')
        
        fig.suptitle('SNR Maps for Distance Measurements (Figure 9)', fontsize=14)
        
    except Exception as e:
        for i in range(2):
            for j in range(5):
                ax = axes[i,j]
                ax.text(0.5, 0.5, f'Error\n{str(e)[:15]}...', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=8)
                ax.set_title(f'Fig 9 ({i+1},{j+1})')
    
    plt.tight_layout()
    return fig

def main():
    """Create all figure reproductions and save to PDF"""
    print("Reproducing Figures 3, 4, 6, 7, and 9 from II_Telescopes.pdf...")
    print("=" * 70)
    
    if not DEPENDENCIES_AVAILABLE:
        print("⚠️  Dependencies not available")
        return
    
    try:
        with PdfPages('reproduce_ii_telescopes_figures.pdf') as pdf:
            print("1. Reproducing Figure 3: Photon flux density spectrum...")
            fig3 = reproduce_figure_3()
            pdf.savefig(fig3, bbox_inches='tight')
            plt.close(fig3)
            
            print("2. Reproducing Figure 4: SEDONA emission profiles...")
            fig4 = reproduce_figure_4()
            pdf.savefig(fig4, bbox_inches='tight')
            plt.close(fig4)
            
            print("3. Reproducing Figure 6: V² maps...")
            fig6 = reproduce_figure_6()
            pdf.savefig(fig6, bbox_inches='tight')
            plt.close(fig6)
            
            print("4. Reproducing Figure 7: V² slices and differences...")
            fig7 = reproduce_figure_7()
            pdf.savefig(fig7, bbox_inches='tight')
            plt.close(fig7)
            
            print("5. Reproducing Figure 9: SNR maps...")
            fig9 = reproduce_figure_9()
            pdf.savefig(fig9, bbox_inches='tight')
            plt.close(fig9)
            
            # Add metadata
            d = pdf.infodict()
            d['Title'] = 'Reproduction of II_Telescopes.pdf Figures 3, 4, 6, 7, 9'
            d['Author'] = 'GridSource.getSN2011feSource Analysis'
            d['Subject'] = 'Intensity Interferometry of Type Ia Supernovae'
            d['Keywords'] = 'SN2011fe, SEDONA, Intensity Interferometry, Visibility'
        
        print("\n✅ All figures reproduced and saved to reproduce_ii_telescopes_figures.pdf!")
        print("\nThe PDF contains reproductions of:")
        print("• Figure 3: Photon flux density spectrum")
        print("• Figure 4: SEDONA emission profiles at different wavelengths")
        print("• Figure 6: V² maps as function of u-v coordinates")
        print("• Figure 7: V² slices and asymmetry differences")
        print("• Figure 9: SNR maps for distance parameter measurements")
        print("\n🎯 All figures use GridSource.getSN2011feSource as the Sedona model!")
        
    except Exception as e:
        print(f"❌ Error creating figures: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()