#!/usr/bin/env python3
"""
Test basic inverse_noise implementation against equation 14
"""

import numpy as np
from intensity_interferometry_core import PointSource

def test_basic_equation_14():
    """
    Test that our inverse_noise method correctly implements equation 14
    σ⁻¹_{|V|²} = (dΓ/dν) * √(T_obs/σ_t) * (128π)^(-1/4)
    """
    
    print("Testing basic equation 14 implementation")
    print("=" * 40)
    
    # Simple test parameters
    nu_0 = 5e14  # Hz
    A = 100.0  # m²
    T_obs = 3600.0  # 1 hour
    sigma_t = 10e-12  # 10 ps
    F_nu = 1e-12  # W m⁻² Hz⁻¹
    
    # Create source
    source = PointSource(F_nu)
    
    # Calculate using our method
    inverse_noise = source.inverse_noise(nu_0, A, T_obs, sigma_t)
    
    # Calculate manually using equation 14
    h = 6.62607015e-34
    dGamma_dnu = A * F_nu / (h * nu_0)
    manual_inverse_noise = (dGamma_dnu * 
                           np.sqrt(T_obs / sigma_t) * 
                           (128 * np.pi)**(-0.25))
    
    print(f"Parameters:")
    print(f"  F_ν: {F_nu:.1e} W m⁻² Hz⁻¹")
    print(f"  A: {A} m²")
    print(f"  T_obs: {T_obs/3600:.1f} hours")
    print(f"  σ_t: {sigma_t*1e12:.0f} ps")
    print(f"  dΓ/dν: {dGamma_dnu:.2e}")
    print()
    
    print(f"Results:")
    print(f"  Our method: {inverse_noise:.6f}")
    print(f"  Manual calc: {manual_inverse_noise:.6f}")
    print(f"  Difference: {abs(inverse_noise - manual_inverse_noise):.2e}")
    
    if abs(inverse_noise - manual_inverse_noise) < 1e-10:
        print("  ✓ PASS: Perfect agreement")
    else:
        print("  ✗ FAIL: Methods disagree")
    
    # Now test with paper parameters but single telescope
    print("\n" + "="*50)
    print("Testing with paper parameters (single telescope)")
    
    # Paper parameters
    lambda_Ha = 6560e-10  # m
    c = 2.99792458e8
    nu_0_paper = c / lambda_Ha
    A_paper = 88.0  # m²
    T_obs_paper = 24 * 3600  # 24 hours
    sigma_t_paper = 30e-12 / 2.35  # 30 ps FWHM -> RMS
    dGamma_dnu_paper = 4e-7  # From paper
    F_nu_paper = dGamma_dnu_paper * h * nu_0_paper / A_paper
    
    source_paper = PointSource(F_nu_paper)
    inverse_noise_paper = source_paper.inverse_noise(nu_0_paper, A_paper, T_obs_paper, sigma_t_paper)
    
    print(f"Paper parameters (single telescope):")
    print(f"  dΓ/dν: {dGamma_dnu_paper:.1e}")
    print(f"  σ⁻¹_|V|²: {inverse_noise_paper:.2f}")
    print(f"  Paper mentions 73 for full array")
    print(f"  Ratio (73/our_result): {73/inverse_noise_paper:.1f}")
    
    # This ratio might give us a clue about the scaling
    scaling_factor = 73 / inverse_noise_paper
    print(f"  Implied scaling factor: {scaling_factor:.1f}")
    
    # Check if this matches n_t choose 2
    n_t = 14
    n_pairs = n_t * (n_t - 1) // 2
    sqrt_n_pairs = np.sqrt(n_pairs)
    
    print(f"  √(n_pairs) = √{n_pairs} = {sqrt_n_pairs:.1f}")
    print(f"  n_t = {n_t}")
    
    if abs(scaling_factor - sqrt_n_pairs) < 1:
        print("  ✓ Scaling matches √(n_pairs)")
    else:
        print("  ? Scaling doesn't match √(n_pairs)")
        print(f"    Maybe different normalization in paper?")

if __name__ == "__main__":
    test_basic_equation_14()