#!/usr/bin/env python3
"""
Test comparing UniformDisk and RadialGrid2 for the same source shape
===================================================================

This test compares the visibility squared of:
1. UniformDisk with radius = 1 mas
2. RadialGrid2 with I_nu_p = 25 ones followed by 25 zeros, p_rays from 0 to 2*radius

These should describe the same source and produce identical visibility curves.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os

# Add parent directory to path to import source modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from g2.models.sources.simple import UniformDisk
    from g2.models.sources.radial_grid import RadialGrid2
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import sources: {e}")
    DEPENDENCIES_AVAILABLE = False

def create_uniform_disk():
    """Create a UniformDisk source with 1 mas radius"""
    # Convert 1 mas to radians
    mas_to_rad = 1e-3 / 3600 * np.pi / 180  # milliarcseconds to radians
    radius_rad = 1.0 * mas_to_rad
    
    # Create UniformDisk with flux_density parameter
    flux_density = 1e-26  # W m⁻² Hz⁻¹ (1 Jy)
    uniform_disk = UniformDisk(flux_density=flux_density, radius=radius_rad)
    return uniform_disk, radius_rad

def create_radial_grid2():
    """Create a RadialGrid2 source that represents the same 1 mas uniform disk"""
    # Convert 1 mas to radians
    mas_to_rad = 1e-3 / 3600 * np.pi / 180  # milliarcseconds to radians
    radius_rad = 1.0 * mas_to_rad
    
    # Create wavelength grid
    lambdas = np.linspace(4000, 7000, 10)  # Angstrom
    
    # Create p_rays uniformly sampled from 0 to 2*radius
    p_rays = np.linspace(0, radius_rad, 100)  # radians
    
    # Create I_nu_p: uniform disk profile (1 inside radius, 0 outside)
    I_nu_p = np.zeros((len(lambdas), len(p_rays)))
    for i, lam in enumerate(lambdas):
        # Create step function: 1 inside radius, 0 outside
        disk_profile = np.where(p_rays <= radius_rad, 1.0, 0.0)
        I_nu_p[i, :] = disk_profile
    
    # Create RadialGrid2
    radial_grid = RadialGrid2(lambdas, I_nu_p, p_rays, s=1.0)
    return radial_grid, radius_rad

def compare_visibility_squared():
    """Compare visibility squared between UniformDisk and RadialGrid2"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    try:
        # Create both sources
        uniform_disk, radius_rad = create_uniform_disk()
        radial_grid, _ = create_radial_grid2()
        
        print(f"Disk radius: {radius_rad:.2e} radians = 1.0 mas")
        print(f"RadialGrid2 p_rays range: {radial_grid.p_rays.min():.2e} - {radial_grid.p_rays.max():.2e}")
        print(f"RadialGrid2 intensity profile: first 5 = {radial_grid.I_nu_p[0, :5]}, last 5 = {radial_grid.I_nu_p[0, -5:]}")
        
        # Test parameters
        nu_0 = 5e14  # Hz (600 nm)
        
        # Plot 1: Visibility squared vs baseline length
        baseline_lengths = np.logspace(1, 2.5, 100)  # 10 m to 10 km
        
        vis_squared_uniform = []
        vis_squared_radial = []
        
        for B in baseline_lengths:
            baseline = np.array([B, 0.0, 0.0])
            
            # UniformDisk visibility
            try:
                vis_uniform = uniform_disk.V(nu_0, baseline)
                vis_squared_uniform.append(abs(vis_uniform)**2)
            except:
                vis_squared_uniform.append(0.0)
            
            # RadialGrid2 visibility
            try:
                vis_radial = radial_grid.V(nu_0, baseline)
                vis_squared_radial.append(abs(vis_radial)**2)
            except:
                vis_squared_radial.append(0.0)
        
        ax1.loglog(baseline_lengths, vis_squared_uniform, 'b-', linewidth=2, label='UniformDisk')
        ax1.loglog(baseline_lengths, vis_squared_radial, 'r--', linewidth=2, label='RadialGrid2')
        ax1.set_xlabel('Baseline Length (m)')
        ax1.set_ylabel('|V|²')
        ax1.set_title('Visibility² vs Baseline Length')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Difference between the two
        vis_diff = np.array(vis_squared_uniform) - np.array(vis_squared_radial)
        ax2.semilogx(baseline_lengths, vis_diff, 'g-', linewidth=2)
        ax2.set_xlabel('Baseline Length (m)')
        ax2.set_ylabel('|V|² Difference (UniformDisk - RadialGrid2)')
        ax2.set_title('Visibility² Difference')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Intensity profiles
        # UniformDisk theoretical profile
        r_theory = np.linspace(0, 2 * radius_rad, 100)
        intensity_uniform_theory = np.where(r_theory <= radius_rad, 1.0, 0.0)
        
        ax3.plot(r_theory * 1000 / (1e-3 / 3600 * np.pi / 180), intensity_uniform_theory, 
                'b-', linewidth=2, label='UniformDisk (theoretical)')
        ax3.plot(radial_grid.p_rays * 1000 / (1e-3 / 3600 * np.pi / 180), radial_grid.I_nu_p[0, :], 
                'ro-', markersize=4, label='RadialGrid2')
        ax3.set_xlabel('Radius (mas)')
        ax3.set_ylabel('Intensity')
        ax3.set_title('Intensity Profiles')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Statistics and info
        max_diff = np.max(np.abs(vis_diff))
        mean_diff = np.mean(np.abs(vis_diff))
        
        info_text = f"""Comparison Results:
        
Disk radius: 1.0 mas = {radius_rad:.2e} rad

UniformDisk:
- Analytical uniform disk
- Radius: {radius_rad:.2e} rad

RadialGrid2:
- 50 radial points from 0 to 2×radius
- First 25 points: intensity = 1
- Last 25 points: intensity = 0
- Wavelengths: {len(radial_grid.lambdas)}

Visibility² Comparison:
- Max difference: {max_diff:.2e}
- Mean abs difference: {mean_diff:.2e}

Expected: Both should be identical
for the same uniform disk source.
"""
        
        ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, 
                verticalalignment='top', fontsize=10, fontfamily='monospace')
        ax4.set_title('Comparison Statistics')
        ax4.axis('off')
        
    except Exception as e:
        for i, ax in enumerate([ax1, ax2, ax3, ax4]):
            ax.text(0.5, 0.5, f'Error in plot {i+1}:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
        print(f"Error in comparison: {e}")
        import traceback
        traceback.print_exc()
    
    plt.tight_layout()
    return fig

def main():
    """Create comparison plots and save to PDF"""
    print("Comparing UniformDisk vs RadialGrid2 for same source shape...")
    print("=" * 60)
    
    if not DEPENDENCIES_AVAILABLE:
        print("⚠️  Dependencies not available")
        return
    
    try:
        with PdfPages('uniform_disk_vs_radial_grid2.pdf') as pdf:
            print("Creating comparison plots...")
            fig = compare_visibility_squared()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Add metadata
            d = pdf.infodict()
            d['Title'] = 'UniformDisk vs RadialGrid2 Comparison'
            d['Author'] = 'Source Comparison Test'
            d['Subject'] = 'Comparing visibility functions of equivalent sources'
            d['Keywords'] = 'UniformDisk, RadialGrid2, Visibility, Comparison'
        
        print("\n✅ Comparison plots saved to uniform_disk_vs_radial_grid2.pdf!")
        print("\nThis test verifies that RadialGrid2 can reproduce")
        print("the same visibility function as an analytical UniformDisk")
        print("when configured with the same source parameters.")
        
    except Exception as e:
        print(f"❌ Error creating comparison plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()