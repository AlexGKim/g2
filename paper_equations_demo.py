"""
Demonstration of Paper Equations Implementation

This module demonstrates the implementation of equations 1-14 from 
"Probing H0 and resolving AGN disks with ultrafast photon counters" (arXiv:2403.15903v1)

Examples include:
- Basic visibility calculations
- AGN disk modeling
- SNR forecasts
- Comparison with paper results
"""

import numpy as np
import matplotlib.pyplot as plt
from intensity_interferometry_core import (
    IntensityInterferometry, FactorizedVisibility, ObservationalParameters,
    PointSource, UniformDisk
)
from agn_source_models import (
    ShakuraSunyaevDisk, BroadLineRegion, RelativisticDisk,
    power_law_beta, lognormal_beta
)


def demo_basic_visibility():
    """Demonstrate basic visibility calculations - Equations (1-2)"""
    print("=== Basic Visibility Calculations ===")
    
    # Create a simple uniform disk source
    flux_density = 1e-12  # W m^-2 Hz^-1
    angular_radius = 1e-6  # radians (microarcsecond scale)
    source = UniformDisk(flux_density, angular_radius)
    
    # Initialize interferometry calculator
    interferometer = IntensityInterferometry(source)
    
    # Observational parameters
    nu_0 = 5e14  # 600 nm
    delta_nu = 1e12  # 1 THz bandwidth
    baseline = np.array([100.0, 0.0, 0.0])  # 100 m baseline
    
    # Calculate visibility (Equation 1)
    V = interferometer.visibility(nu_0, baseline)
    print(f"Complex visibility V = {V:.6f}")
    print(f"|V| = {abs(V):.6f}")
    
    # Calculate normalized fringe visibility (Equation 2)
    V_norm = interferometer.normalized_fringe_visibility(nu_0, delta_nu, baseline)
    print(f"Normalized visibility V̄ = {V_norm:.6f}")
    print(f"|V̄| = {abs(V_norm):.6f}")
    
    return V, V_norm


def demo_shakura_sunyaev_disk():
    """Demonstrate Shakura-Sunyaev disk model - Equations (21-22)"""
    print("\n=== Shakura-Sunyaev Disk Model ===")
    
    # Physical parameters for a typical AGN
    GM_over_c2 = 1.5e11  # 1 AU for 10^8 solar mass BH
    distance = 20e6 * 3.086e16  # 20 Mpc in meters
    I_0 = 1e-15  # Normalization intensity
    R_0 = 43.0  # In units of GM/c²
    R_in = 6.0   # ISCO for Schwarzschild BH
    inclination = np.pi/3  # 60 degrees
    
    # Create SS disk
    disk = ShakuraSunyaevDisk(I_0, R_0, R_in, n=3.0, 
                             inclination=inclination, 
                             distance=distance, 
                             GM_over_c2=GM_over_c2)
    
    # Calculate visibility as function of baseline
    nu_0 = 5e14  # Optical frequency
    baselines = np.logspace(1, 5, 50)  # 10 m to 100 km
    visibilities = []
    
    for B in baselines:
        baseline_vec = np.array([B, 0.0, 0.0])
        V = disk.visibility_analytical(nu_0, baseline_vec)
        visibilities.append(abs(V))
    
    visibilities = np.array(visibilities)
    
    # Plot visibility vs baseline (similar to Figure 1 in paper)
    plt.figure(figsize=(10, 6))
    plt.loglog(baselines/1000, visibilities**2, 'b-', linewidth=2)
    plt.xlabel('Baseline [km]')
    plt.ylabel('|V|²')
    plt.title('Shakura-Sunyaev Disk Visibility')
    plt.grid(True, alpha=0.3)
    
    # Mark characteristic scales
    lambda_0 = 3e8 / nu_0
    R_char_angular = R_0 * GM_over_c2 / distance
    B_char = lambda_0 / (2 * np.pi * R_char_angular) / 1000  # km
    plt.axvline(B_char, color='r', linestyle='--', 
                label=f'λD/(2πR₀) = {B_char:.1f} km')
    plt.legend()
    plt.tight_layout()
    plt.savefig('ss_disk_visibility.png', dpi=150)
    plt.show()
    
    print(f"Characteristic baseline: {B_char:.1f} km")
    print(f"Disk angular size: {R_char_angular*1e6:.2f} μas")
    
    return disk, baselines, visibilities


def demo_snr_calculations():
    """Demonstrate SNR calculations - Equations (6), (12-14)"""
    print("\n=== Signal-to-Noise Ratio Calculations ===")
    
    # Create a point source (unresolved)
    flux_density = 1e-12  # W m^-2 Hz^-1 (magnitude ~12)
    source = PointSource(flux_density)
    interferometer = IntensityInterferometry(source)
    
    # Observational parameters similar to paper examples
    params = ObservationalParameters(
        nu_0=5e14,           # 600 nm
        delta_nu=1e12,       # 1 THz bandwidth  
        baseline=np.array([10000.0, 0.0, 0.0]),  # 10 km baseline
        delta_t=0.0,         # Zero time lag
        sigma_t=13e-12,      # 13 ps timing jitter (30 ps FWHM)
        T_obs=24*3600,       # 24 hours observation
        A=88.0,              # 88 m² telescope area (CTA-like)
        n_t=14               # 14 telescopes
    )
    
    # Calculate SNR using different methods
    snr = interferometer.signal_to_noise_ratio(params)
    print(f"SNR for single channel: {snr:.1f}")
    
    # Calculate with spectroscopic enhancement
    n_channels = 5000
    snr_spectroscopic = snr * np.sqrt(n_channels)
    print(f"SNR with {n_channels} channels: {snr_spectroscopic:.0f}")
    
    # Calculate visibility error (Equation 14)
    sigma_v2 = interferometer.visibility_error(params)
    print(f"Visibility error σ_|V|²: {sigma_v2:.6f}")
    
    # Compare with paper estimate
    # Paper gives σ_{|V|²}^{-1} ≈ 0.04 for single channel
    F_nu = source.total_flux(params.nu_0)
    dGamma_dnu = params.A * F_nu / (6.626e-34 * params.nu_0)
    sigma_inv_paper = (dGamma_dnu * 
                      np.sqrt(params.T_obs / params.sigma_t) * 
                      (128 * np.pi)**(-0.25))
    
    print(f"Paper formula σ_|V|²^(-1): {sigma_inv_paper:.6f}")
    print(f"Our calculation σ_|V|²^(-1): {1/sigma_v2:.6f}")
    
    return snr, snr_spectroscopic


def demo_timing_jitter_effects():
    """Demonstrate timing jitter effects - Equations (9-12)"""
    print("\n=== Timing Jitter Effects ===")
    
    # Create test source
    source = PointSource(1e-12)
    interferometer = IntensityInterferometry(source)
    
    # Test different timing jitters
    sigma_t_values = np.logspace(-12, -9, 20)  # 1 ps to 1 ns
    delta_nu = 1e12  # 1 THz bandwidth
    
    jitter_factors = []
    for sigma_t in sigma_t_values:
        params = ObservationalParameters(
            nu_0=5e14, delta_nu=delta_nu, 
            baseline=np.array([1000.0, 0.0, 0.0]),
            delta_t=0.0, sigma_t=sigma_t, T_obs=3600.0, A=100.0, n_t=2
        )
        
        # Calculate timing jitter correlation
        C = interferometer.timing_jitter_correlation(params)
        jitter_factors.append(C)
    
    jitter_factors = np.array(jitter_factors)
    
    # Plot jitter factor vs σₜΔν
    plt.figure(figsize=(10, 6))
    sigma_t_delta_nu = sigma_t_values * delta_nu
    plt.semilogx(sigma_t_delta_nu, jitter_factors, 'b-', linewidth=2)
    plt.xlabel('σₜΔν')
    plt.ylabel('Correlation factor C')
    plt.title('Timing Jitter Effects on Intensity Correlations')
    plt.grid(True, alpha=0.3)
    
    # Mark transition at σₜΔν = 1
    plt.axvline(1.0, color='r', linestyle='--', label='σₜΔν = 1')
    plt.legend()
    plt.tight_layout()
    plt.savefig('timing_jitter_effects.png', dpi=150)
    plt.show()
    
    print(f"Jitter factor at σₜΔν = 0.1: {jitter_factors[5]:.6f}")
    print(f"Jitter factor at σₜΔν = 1.0: {jitter_factors[10]:.6f}")
    print(f"Jitter factor at σₜΔν = 10: {jitter_factors[15]:.6f}")


def demo_broad_line_region():
    """Demonstrate BLR model - Section IV equations"""
    print("\n=== Broad Line Region Model ===")
    
    # BLR parameters
    GM = 1.3e20 * 1e8  # 10^8 solar masses in SI units
    R_in = 1e16  # ~0.3 pc
    R_out = 1e18  # ~30 pc  
    distance = 170e6 * 3.086e16  # 170 Mpc
    inclination = np.pi/3
    nu_c = 4.57e14  # Hα line frequency
    
    # Create power-law response function β(R) ∝ R²
    def beta_func(R):
        return power_law_beta(R, R_in, 2.0, 1e-20)
    
    # Create BLR model
    blr = BroadLineRegion(beta_func, R_in, R_out, GM, 
                         inclination, distance, nu_c)
    
    # Calculate transfer function for different frequencies
    omega = 2 * np.pi / (30 * 24 * 3600)  # 30-day period
    velocities = np.linspace(-3000e3, 3000e3, 100)  # ±3000 km/s
    
    transfer_functions = []
    for v in velocities:
        psi = blr.transfer_function_fourier(omega, v)
        transfer_functions.append(abs(psi))
    
    transfer_functions = np.array(transfer_functions)
    
    # Plot transfer function vs velocity
    plt.figure(figsize=(10, 6))
    plt.plot(velocities/1000, transfer_functions, 'b-', linewidth=2)
    plt.xlabel('Velocity [km/s]')
    plt.ylabel('|ψ(ω,v)|')
    plt.title('BLR Transfer Function')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('blr_transfer_function.png', dpi=150)
    plt.show()
    
    print(f"BLR inner radius: {R_in/3.086e16:.2f} pc")
    print(f"BLR outer radius: {R_out/3.086e16:.2f} pc")
    print(f"Maximum velocity: {np.sqrt(GM/R_in)/1000:.0f} km/s")


def demo_relativistic_effects():
    """Demonstrate relativistic disk effects"""
    print("\n=== Relativistic Disk Effects ===")
    
    # Compare Newtonian vs relativistic disk
    GM_over_c2 = 1.5e11
    distance = 20e6 * 3.086e16
    I_0 = 1e-15
    R_0 = 43.0
    R_in = 6.0
    inclination = np.pi/3
    
    # Newtonian disk
    disk_newtonian = ShakuraSunyaevDisk(I_0, R_0, R_in, 
                                       inclination=inclination,
                                       distance=distance, 
                                       GM_over_c2=GM_over_c2)
    
    # Relativistic disk with spin a = 0.5
    disk_relativistic = RelativisticDisk(I_0, R_0, R_in,
                                        inclination=inclination,
                                        distance=distance,
                                        GM_over_c2=GM_over_c2,
                                        spin_parameter=0.5)
    
    print(f"Newtonian ISCO: {R_in:.1f} GM/c²")
    print(f"Relativistic ISCO: {disk_relativistic.R_isco:.1f} GM/c²")
    
    # Compare intensities at different positions
    nu_0 = 5e14
    positions = np.array([[0.1, 0], [0.2, 0], [0.5, 0]]) * GM_over_c2 / distance
    
    print("\nIntensity comparison:")
    print("Position [GM/c²/D]  Newtonian    Relativistic")
    for i, pos in enumerate(positions):
        I_newt = disk_newtonian.intensity(nu_0, pos)
        I_rel = disk_relativistic.intensity(nu_0, pos)
        pos_scale = pos[0] * distance / GM_over_c2
        print(f"{pos_scale:.1f}                {I_newt:.2e}    {I_rel:.2e}")


def run_all_demos():
    """Run all demonstration functions"""
    print("Running Paper Equations Demonstration")
    print("=" * 50)
    
    # Run demonstrations
    demo_basic_visibility()
    demo_shakura_sunyaev_disk()
    demo_snr_calculations()
    demo_timing_jitter_effects()
    demo_broad_line_region()
    demo_relativistic_effects()
    
    print("\n" + "=" * 50)
    print("All demonstrations completed!")
    print("Generated plots:")
    print("- ss_disk_visibility.png")
    print("- timing_jitter_effects.png") 
    print("- blr_transfer_function.png")


if __name__ == "__main__":
    run_all_demos()