"""
Plotting script for source module tests and demonstrations.

This script creates visualizations related to the test cases in test_source.py,
demonstrating key concepts in intensity interferometry including:
- Airy function behavior vs zeta parameter
- FFT vs analytical visibility comparisons
- Temporal coherence functions (g1 and g2-1)
- Source intensity profiles

All plots are saved to a single PDF file: plots_source.pdf
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
from source import PointSource, UniformDisk, ChaoticSource
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages

# Set up plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def plot_airy_function_vs_zeta():
    """Plot Airy function V = 2J₁(ζ)/ζ vs zeta parameter."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Zeta range
    zeta = np.linspace(0.01, 15, 1000)
    
    # Airy function: V = 2J₁(ζ)/ζ
    airy = 2 * j1(zeta) / zeta
    
    # Plot 1: Linear scale
    ax1.plot(zeta, airy, 'b-', linewidth=2, label='Airy function: 2J₁(ζ)/ζ')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Mark first few zeros
    zeros = [3.8317, 7.0156, 10.1735]  # First three zeros
    for i, zero in enumerate(zeros):
        ax1.axvline(x=zero, color='r', linestyle=':', alpha=0.7)
        ax1.text(zero, 0.1, f'Zero {i+1}\nζ={zero:.2f}', 
                ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('ζ = πρθ/λ')
    ax1.set_ylabel('|V|')
    ax1.set_title('Visibility vs Zeta Parameter (Linear Scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 15)
    ax1.set_ylim(-0.3, 1.1)
    
    # Plot 2: Log scale for better visibility of oscillations
    ax2.semilogy(zeta, np.abs(airy), 'b-', linewidth=2, label='|Airy function|')
    for zero in zeros:
        ax2.axvline(x=zero, color='r', linestyle=':', alpha=0.7)
    
    ax2.set_xlabel('ζ = πρθ/λ')
    ax2.set_ylabel('|V| (log scale)')
    ax2.set_title('Visibility vs Zeta Parameter (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 15)
    ax2.set_ylim(1e-3, 1.1)
    
    plt.tight_layout()
    return fig

def plot_fft_vs_analytical_comparison():
    """Compare FFT-based and analytical visibility calculations vs zeta."""
    # Create UniformDisk
    flux_density = 1e-26  # W/m²/Hz
    radius = 1e-8  # radians
    disk = UniformDisk(flux_density, radius)
    
    # Test parameters
    nu_0 = 5e14  # Hz
    c = 2.99792458e8
    wavelength = c / nu_0
    theta = 2 * radius  # Angular diameter
    
    # Range of zeta values
    zetas = np.linspace(0.1, 12, 100)
    
    # Calculate corresponding baseline lengths: ρ = ζλ/(πθ)
    baseline_lengths = zetas * wavelength / (np.pi * theta)
    
    # Calculate visibilities
    V_analytical = []
    V_fft = []
    
    for baseline_length in baseline_lengths:
        baseline = np.array([baseline_length, 0.0, 0.0])
        
        # Analytical (UniformDisk.V)
        V_anal = disk.V(nu_0, baseline)
        V_analytical.append(abs(V_anal))
        
        # FFT-based (AbstractSource.V)
        V_fft_val = super(UniformDisk, disk).V(nu_0, baseline)
        V_fft.append(abs(V_fft_val))
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Visibility vs zeta (main comparison)
    ax1.plot(zetas, V_analytical, 'b-', linewidth=2,
             label='Analytical (Airy function)')
    ax1.plot(zetas, V_fft, 'r--', linewidth=2, alpha=0.7,
             label='FFT-based (AbstractSource.V)')
    ax1.axvline(x=3.8317, color='k', linestyle=':', alpha=0.7,
                label='First zero (ζ=3.83)')
    ax1.set_xlabel('ζ = πρθ/λ')
    ax1.set_ylabel('|V|')
    ax1.set_title('Visibility vs Zeta Parameter: FFT vs Analytical Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 12)
    ax1.set_ylim(-0.3, 1.1)
    
    # Plot 2: Residuals (difference between methods)
    residuals = np.array(V_fft) - np.array(V_analytical)
    ax2.plot(zetas, residuals, 'g-', linewidth=2, label='FFT - Analytical')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.axvline(x=3.8317, color='k', linestyle=':', alpha=0.7,
                label='First zero (ζ=3.83)')
    ax2.set_xlabel('ζ = πρθ/λ')
    ax2.set_ylabel('Residual |V|')
    ax2.set_title('FFT vs Analytical Residuals')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 12)
    
    plt.tight_layout()
    return fig

def plot_temporal_coherence_functions():
    """Plot g1 and g2-1 temporal coherence functions."""
    # Create a point source for testing
    point = PointSource(lambda nu: 1e-26)
    
    # Test parameters
    nu_0 = 5e14  # Hz
    delta_nu_narrow = 1e11  # Narrow bandwidth
    delta_nu_wide = 1e13    # Wide bandwidth
    
    # Time delay range
    delta_t = np.linspace(0, 5e-11, 1000)  # 0 to 50 ps
    
    # Calculate coherence functions
    g1_narrow = [point.g1(dt, nu_0, delta_nu_narrow) for dt in delta_t]
    g1_wide = [point.g1(dt, nu_0, delta_nu_wide) for dt in delta_t]
    
    g2_minus_one_narrow = [point.g2_minus_one(dt, nu_0, delta_nu_narrow) for dt in delta_t]
    g2_minus_one_wide = [point.g2_minus_one(dt, nu_0, delta_nu_wide) for dt in delta_t]
    
    # Convert to arrays and take absolute values for g1
    g1_narrow = np.abs(g1_narrow)
    g1_wide = np.abs(g1_wide)
    g2_minus_one_narrow = np.array(g2_minus_one_narrow)
    g2_minus_one_wide = np.array(g2_minus_one_wide)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: g1 functions
    ax1.plot(delta_t * 1e12, g1_narrow, 'b-', linewidth=2, 
             label=f'|g¹(Δt)| narrow (Δν = {delta_nu_narrow:.0e} Hz)')
    ax1.plot(delta_t * 1e12, g1_wide, 'r-', linewidth=2,
             label=f'|g¹(Δt)| wide (Δν = {delta_nu_wide:.0e} Hz)')
    ax1.axhline(y=1/np.e, color='k', linestyle='--', alpha=0.5, 
                label='1/e level')
    ax1.set_xlabel('Time Delay Δt (ps)')
    ax1.set_ylabel('|g¹(Δt)|')
    ax1.set_title('First-Order Temporal Coherence Function')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # Plot 2: g2-1 functions
    ax2.plot(delta_t * 1e12, g2_minus_one_narrow, 'b-', linewidth=2,
             label=f'g²(Δt)-1 narrow (Δν = {delta_nu_narrow:.0e} Hz)')
    ax2.plot(delta_t * 1e12, g2_minus_one_wide, 'r-', linewidth=2,
             label=f'g²(Δt)-1 wide (Δν = {delta_nu_wide:.0e} Hz)')
    ax2.axhline(y=1/np.e**2, color='k', linestyle='--', alpha=0.5,
                label='1/e² level')
    ax2.set_xlabel('Time Delay Δt (ps)')
    ax2.set_ylabel('g²(Δt) - 1')
    ax2.set_title('Second-Order Temporal Coherence Function Minus One')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    plt.tight_layout()
    return fig

def plot_source_intensity_profiles():
    """Plot intensity profiles for different source types."""
    # Create sources
    point = PointSource(lambda nu: 1e-26)
    disk = UniformDisk(1e-26, 1e-8)
    
    # Test parameters
    nu_0 = 5e14
    
    # Angular coordinate grid
    theta_max = 3e-8  # 3 times the disk radius
    theta = np.linspace(-theta_max, theta_max, 1000)
    
    # Calculate intensity profiles (1D slice through center)
    I_point = []
    I_disk = []
    
    for t in theta:
        n_hat = np.array([t, 0.0])
        I_point.append(point.intensity(nu_0, n_hat))
        I_disk.append(disk.intensity(nu_0, n_hat))
    
    # Convert to arrays and normalize
    I_point = np.array(I_point)
    I_disk = np.array(I_disk)
    
    # Normalize for plotting
    I_point = I_point / np.max(I_point) if np.max(I_point) > 0 else I_point
    I_disk = I_disk / np.max(I_disk) if np.max(I_disk) > 0 else I_disk
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Intensity profiles
    ax1.plot(theta * 1e9, I_point, 'b-', linewidth=2, label='Point Source')
    ax1.plot(theta * 1e9, I_disk, 'r-', linewidth=2, label='Uniform Disk')
    ax1.axvline(x=disk.radius * 1e9, color='r', linestyle='--', alpha=0.5,
                label='Disk edge')
    ax1.axvline(x=-disk.radius * 1e9, color='r', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Angular Position θ (nrad)')
    ax1.set_ylabel('Normalized Intensity')
    ax1.set_title('Source Intensity Profiles')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: 2D intensity map for uniform disk
    theta_2d = np.linspace(-theta_max, theta_max, 200)
    X, Y = np.meshgrid(theta_2d, theta_2d)
    I_2d = np.zeros_like(X)
    
    for i in range(len(theta_2d)):
        for j in range(len(theta_2d)):
            n_hat = np.array([X[i, j], Y[i, j]])
            I_2d[i, j] = disk.intensity(nu_0, n_hat)
    
    im = ax2.imshow(I_2d, extent=[-theta_max*1e9, theta_max*1e9, 
                                  -theta_max*1e9, theta_max*1e9],
                    origin='lower', cmap='hot', aspect='equal')
    
    # Add circle to show disk boundary
    circle = plt.Circle((0, 0), disk.radius*1e9, fill=False, 
                       color='white', linewidth=2, linestyle='--')
    ax2.add_patch(circle)
    
    ax2.set_xlabel('θₓ (nrad)')
    ax2.set_ylabel('θᵧ (nrad)')
    ax2.set_title('Uniform Disk Intensity Map')
    plt.colorbar(im, ax=ax2, label='Intensity (W m⁻² Hz⁻¹ sr⁻¹)')
    
    plt.tight_layout()
    return fig

def plot_zeta_parameter_demonstration():
    """Demonstrate the zeta parameter concept showing universal behavior."""
    # Parameters
    nu_0 = 5e14  # Hz (600 nm)
    c = 2.99792458e8
    wavelength = c / nu_0
    
    # Different source sizes
    radii = [5e-9, 1e-8, 2e-8]  # radians
    labels = ['Small (5 nrad)', 'Medium (10 nrad)', 'Large (20 nrad)']
    colors = ['blue', 'red', 'green']
    
    # Zeta range
    zetas = np.linspace(0.1, 15, 1000)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Universal visibility curve vs zeta
    # All source sizes follow the same curve when plotted vs zeta
    airy_universal = 2 * j1(zetas) / zetas
    ax1.plot(zetas, airy_universal, 'k-', linewidth=3, alpha=0.8,
             label='Universal Airy function: 2J₁(ζ)/ζ')
    
    # Show individual source curves (they all overlap)
    for radius, label, color in zip(radii, labels, colors):
        ax1.plot(zetas, airy_universal, color=color, linewidth=1, alpha=0.6,
                linestyle='--', label=f'{label} (overlaps universal)')
    
    # Mark zeros
    zeros = [3.8317, 7.0156, 10.1735]
    for i, zero in enumerate(zeros):
        ax1.axvline(x=zero, color='r', linestyle=':', alpha=0.7)
        if i == 0:
            ax1.text(zero, 0.5, f'Zero {i+1}\nζ={zero:.2f}',
                    ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('ζ = πρθ/λ')
    ax1.set_ylabel('|V|')
    ax1.set_title('Universal Visibility Curve vs Zeta Parameter')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 15)
    ax1.set_ylim(-0.3, 1.1)
    
    # Plot 2: Required baseline lengths for each source size
    zeta_range = np.linspace(0.5, 10, 50)
    
    for radius, label, color in zip(radii, labels, colors):
        theta = 2 * radius  # Angular diameter
        # Calculate required baseline: ρ = ζλ/(πθ)
        required_baselines = zeta_range * wavelength / (np.pi * theta)
        
        ax2.plot(zeta_range, required_baselines, color=color, linewidth=2, label=label)
        
        # Mark first zero baseline
        first_zero_baseline = 3.8317 * wavelength / (np.pi * theta)
        ax2.scatter([3.8317], [first_zero_baseline], color=color, s=100,
                   marker='o', zorder=5)
        ax2.text(3.8317, first_zero_baseline, f'{first_zero_baseline:.0f}m',
                ha='left', va='bottom', color=color, fontsize=8)
    
    ax2.axvline(x=3.8317, color='black', linestyle='--', alpha=0.7,
                label='First zero (ζ=3.83)')
    
    ax2.set_xlabel('ζ = πρθ/λ')
    ax2.set_ylabel('Required Baseline Length ρ (m)')
    ax2.set_title('Baseline Requirements vs Zeta for Different Source Sizes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.5, 10)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    return fig

def plot_test_validation_summary():
    """Create a summary plot showing key test validation points vs zeta."""
    # Test data from our zeta parameter tests
    zeta_test_values = [0.5, 1.0, 2.0, 3.0, 3.8317, 5.0]
    visibility_values = [0.969074, 0.880101, 0.576725, 0.226039, 0.000001, -0.131032]
    
    # Theoretical Airy function
    zeta_theory = np.linspace(0.1, 8, 1000)
    airy_theory = 2 * j1(zeta_theory) / zeta_theory
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Test points on Airy function vs zeta
    ax1.plot(zeta_theory, airy_theory, 'b-', linewidth=2, alpha=0.7,
             label='Theoretical Airy function')
    ax1.scatter(zeta_test_values, visibility_values, color='red', s=100,
                zorder=5, label='Test validation points')
    
    # Annotate test points
    for i, (zeta, vis) in enumerate(zip(zeta_test_values, visibility_values)):
        if zeta == 3.8317:
            ax1.annotate(f'First zero\nζ={zeta:.3f}',
                        xy=(zeta, vis), xytext=(zeta+0.5, 0.3),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=9, ha='left')
        else:
            ax1.annotate(f'ζ={zeta:.1f}',
                        xy=(zeta, vis), xytext=(5, 5),
                        textcoords='offset points', fontsize=8)
    
    ax1.set_xlabel('ζ = πρθ/λ')
    ax1.set_ylabel('|V|')
    ax1.set_title('Test Validation Points on Airy Function vs Zeta')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 8)
    ax1.set_ylim(-0.1, 1.1)
    
    # Plot 2: Expanded zeta range showing more oscillations
    zeta_extended = np.linspace(0.1, 20, 2000)
    airy_extended = 2 * j1(zeta_extended) / zeta_extended
    
    ax2.plot(zeta_extended, airy_extended, 'b-', linewidth=1.5, alpha=0.8,
             label='Airy function: 2J₁(ζ)/ζ')
    ax2.scatter(zeta_test_values, visibility_values, color='red', s=80,
                zorder=5, label='Test points')
    
    # Mark zeros
    zeros = [3.8317, 7.0156, 10.1735, 13.3237, 16.4706]
    for i, zero in enumerate(zeros):
        ax2.axvline(x=zero, color='gray', linestyle=':', alpha=0.5)
        if i < 3:  # Label first three zeros
            ax2.text(zero, -0.25, f'Z{i+1}', ha='center', va='top', fontsize=7)
    
    ax2.set_xlabel('ζ = πρθ/λ')
    ax2.set_ylabel('|V|')
    ax2.set_title('Extended Zeta Range Showing Airy Function Oscillations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 20)
    ax2.set_ylim(-0.3, 1.1)
    
    plt.tight_layout()
    return fig

def main():
    """Run all plotting functions and save to PDF."""
    print("Creating plots for source module tests...")
    
    try:
        # Create PDF file
        with PdfPages('plots_source.pdf') as pdf:
            print("1. Plotting Airy function vs zeta parameter...")
            fig1 = plot_airy_function_vs_zeta()
            pdf.savefig(fig1, bbox_inches='tight')
            plt.close(fig1)
            
            print("2. Plotting FFT vs analytical comparison...")
            fig2 = plot_fft_vs_analytical_comparison()
            pdf.savefig(fig2, bbox_inches='tight')
            plt.close(fig2)
            
            print("3. Plotting temporal coherence functions...")
            fig3 = plot_temporal_coherence_functions()
            pdf.savefig(fig3, bbox_inches='tight')
            plt.close(fig3)
            
            print("4. Plotting source intensity profiles...")
            fig4 = plot_source_intensity_profiles()
            pdf.savefig(fig4, bbox_inches='tight')
            plt.close(fig4)
            
            print("5. Plotting zeta parameter demonstration...")
            fig5 = plot_zeta_parameter_demonstration()
            pdf.savefig(fig5, bbox_inches='tight')
            plt.close(fig5)
            
            print("6. Plotting test validation summary...")
            fig6 = plot_test_validation_summary()
            pdf.savefig(fig6, bbox_inches='tight')
            plt.close(fig6)
            
            # Add metadata to PDF
            d = pdf.infodict()
            d['Title'] = 'Source Module Test Plots'
            d['Author'] = 'Intensity Interferometry Analysis'
            d['Subject'] = 'Visualization of test cases from test_source.py'
            d['Keywords'] = 'Intensity Interferometry, Airy Function, Zeta Parameter, Visibility'
            d['Creator'] = 'plots_source.py'
        
        print("\nAll plots saved to plots_source.pdf!")
        print("The PDF contains 6 pages with the following plots:")
        print("1. Airy function vs zeta parameter (linear and log scales)")
        print("2. FFT vs analytical visibility comparison")
        print("3. Temporal coherence functions (g1 and g2-1)")
        print("4. Source intensity profiles (point source and uniform disk)")
        print("5. Zeta parameter demonstration for different source sizes")
        print("6. Test validation summary with key test points")
        
    except Exception as e:
        print(f"Error creating plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()