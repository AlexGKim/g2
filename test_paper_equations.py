"""
Test Suite for Paper Equations Implementation

Comprehensive tests for equations 1-14 from arXiv:2403.15903v1
"""

import numpy as np
import pytest
from intensity_interferometry_core import (
    IntensityInterferometry, FactorizedVisibility, ObservationalParameters,
    PointSource, UniformDisk
)
from agn_source_models import (
    ShakuraSunyaevDisk, BroadLineRegion, RelativisticDisk,
    power_law_beta, lognormal_beta
)


class TestBasicVisibility:
    """Test basic visibility calculations - Equations (1-2)"""
    
    def test_point_source_visibility(self):
        """Test that point source has constant visibility"""
        source = PointSource(1e-12)
        interferometer = IntensityInterferometry(source)
        
        nu_0 = 5e14
        baselines = [
            np.array([100.0, 0.0, 0.0]),
            np.array([1000.0, 0.0, 0.0]),
            np.array([0.0, 500.0, 0.0])
        ]
        
        visibilities = []
        for baseline in baselines:
            V = interferometer.visibility(nu_0, baseline)
            visibilities.append(abs(V))
        
        # Point source should have |V| ≈ 1 for all baselines
        for vis in visibilities:
            assert abs(vis - 1.0) < 0.1, f"Point source visibility {vis} should be ≈ 1"
    
    def test_uniform_disk_visibility(self):
        """Test uniform disk visibility follows expected pattern"""
        flux_density = 1e-12
        radius = 1e-6  # 1 microarcsecond
        source = UniformDisk(flux_density, radius)
        interferometer = IntensityInterferometry(source)
        
        nu_0 = 5e14
        lambda_0 = 3e8 / nu_0
        
        # Test at baseline where first null should occur
        # For uniform disk: first null at B = 1.22 λ / (2 * radius)
        B_null = 1.22 * lambda_0 / (2 * radius)
        baseline = np.array([B_null, 0.0, 0.0])
        
        V = interferometer.visibility(nu_0, baseline)
        
        # Should be close to zero at first null
        assert abs(V) < 0.1, f"Visibility at first null should be small, got {abs(V)}"
    
    def test_normalized_visibility(self):
        """Test normalized fringe visibility calculation"""
        source = PointSource(1e-12)
        interferometer = IntensityInterferometry(source)
        
        nu_0 = 5e14
        delta_nu = 1e12
        baseline = np.array([100.0, 0.0, 0.0])
        
        V_norm = interferometer.normalized_fringe_visibility(nu_0, delta_nu, baseline)
        
        # For point source, normalized visibility should be ≈ 1
        assert abs(abs(V_norm) - 1.0) < 0.1, f"Normalized visibility should be ≈ 1, got {abs(V_norm)}"


class TestPhotonStatistics:
    """Test photon count statistics - Equations (3-6)"""
    
    def test_photon_count_covariance(self):
        """Test photon count covariance calculation"""
        source = PointSource(1e-12)
        interferometer = IntensityInterferometry(source)
        
        params = ObservationalParameters(
            nu_0=5e14, delta_nu=1e12,
            baseline=np.array([100.0, 0.0, 0.0]),
            delta_t=0.0, sigma_t=10e-12, T_obs=3600.0, A=100.0, n_t=2
        )
        
        covariance = interferometer.photon_count_covariance(params)
        
        # Covariance should be positive
        assert covariance > 0, f"Photon count covariance should be positive, got {covariance}"
        
        # Should scale with telescope area squared
        params_large = params
        params_large.A = 400.0  # 4x area
        covariance_large = interferometer.photon_count_covariance(params_large)
        
        # Should scale as A²
        ratio = covariance_large / covariance
        assert abs(ratio - 16.0) < 2.0, f"Covariance should scale as A², got ratio {ratio}"
    
    def test_timing_jitter_effects(self):
        """Test timing jitter correlation function"""
        source = PointSource(1e-12)
        interferometer = IntensityInterferometry(source)
        
        # Test with small timing jitter
        params_good = ObservationalParameters(
            nu_0=5e14, delta_nu=1e12,
            baseline=np.array([100.0, 0.0, 0.0]),
            delta_t=0.0, sigma_t=1e-12, T_obs=3600.0, A=100.0, n_t=2
        )
        
        # Test with large timing jitter
        params_bad = ObservationalParameters(
            nu_0=5e14, delta_nu=1e12,
            baseline=np.array([100.0, 0.0, 0.0]),
            delta_t=0.0, sigma_t=100e-12, T_obs=3600.0, A=100.0, n_t=2
        )
        
        C_good = interferometer.timing_jitter_correlation(params_good)
        C_bad = interferometer.timing_jitter_correlation(params_bad)
        
        # Better timing should give higher correlation
        assert C_good > C_bad, f"Better timing should give higher correlation: {C_good} vs {C_bad}"


class TestSNRCalculations:
    """Test SNR calculations - Equations (12-14)"""
    
    def test_snr_scaling(self):
        """Test SNR scaling with observational parameters"""
        source = PointSource(1e-12)
        interferometer = IntensityInterferometry(source)
        
        base_params = ObservationalParameters(
            nu_0=5e14, delta_nu=1e12,
            baseline=np.array([100.0, 0.0, 0.0]),
            delta_t=0.0, sigma_t=10e-12, T_obs=3600.0, A=100.0, n_t=2
        )
        
        snr_base = interferometer.signal_to_noise_ratio(base_params)
        
        # Test scaling with observation time
        params_long = base_params
        params_long.T_obs = 4 * 3600.0  # 4x longer
        snr_long = interferometer.signal_to_noise_ratio(params_long)
        
        # SNR should scale as √T_obs
        ratio = snr_long / snr_base
        expected_ratio = np.sqrt(4.0)
        assert abs(ratio - expected_ratio) < 0.5, f"SNR should scale as √T, got ratio {ratio}"
        
        # Test scaling with telescope area
        params_big = base_params
        params_big.A = 400.0  # 4x area
        snr_big = interferometer.signal_to_noise_ratio(params_big)
        
        # SNR should scale as A (through photon rate)
        ratio = snr_big / snr_base
        assert abs(ratio - 4.0) < 1.0, f"SNR should scale as A, got ratio {ratio}"
    
    def test_visibility_error(self):
        """Test visibility error calculation"""
        source = PointSource(1e-12)
        interferometer = IntensityInterferometry(source)
        
        params = ObservationalParameters(
            nu_0=5e14, delta_nu=1e12,
            baseline=np.array([100.0, 0.0, 0.0]),
            delta_t=0.0, sigma_t=10e-12, T_obs=3600.0, A=100.0, n_t=2
        )
        
        sigma_v2 = interferometer.visibility_error(params)
        
        # Error should be positive and finite
        assert sigma_v2 > 0, f"Visibility error should be positive, got {sigma_v2}"
        assert np.isfinite(sigma_v2), f"Visibility error should be finite, got {sigma_v2}"


class TestShakuraSunyaevDisk:
    """Test Shakura-Sunyaev disk model - Equations (21-22)"""
    
    def test_disk_intensity_profile(self):
        """Test disk intensity profile"""
        GM_over_c2 = 1.5e11
        distance = 20e6 * 3.086e16
        I_0 = 1e-15
        R_0 = 43.0
        R_in = 6.0
        
        disk = ShakuraSunyaevDisk(I_0, R_0, R_in, 
                                 distance=distance, GM_over_c2=GM_over_c2)
        
        # Test that intensity is zero inside R_in
        nu_0 = 5e14
        pos_inside = np.array([3.0 * GM_over_c2 / distance, 0.0])
        I_inside = disk.intensity(nu_0, pos_inside)
        assert I_inside == 0.0, f"Intensity inside R_in should be zero, got {I_inside}"
        
        # Test that intensity is positive outside R_in
        pos_outside = np.array([10.0 * GM_over_c2 / distance, 0.0])
        I_outside = disk.intensity(nu_0, pos_outside)
        assert I_outside > 0, f"Intensity outside R_in should be positive, got {I_outside}"
        
        # Test that intensity decreases with radius
        pos_far = np.array([100.0 * GM_over_c2 / distance, 0.0])
        I_far = disk.intensity(nu_0, pos_far)
        assert I_far < I_outside, f"Intensity should decrease with radius: {I_far} vs {I_outside}"
    
    def test_disk_visibility(self):
        """Test disk visibility calculation"""
        GM_over_c2 = 1.5e11
        distance = 20e6 * 3.086e16
        I_0 = 1e-15
        R_0 = 43.0
        R_in = 6.0
        
        disk = ShakuraSunyaevDisk(I_0, R_0, R_in,
                                 distance=distance, GM_over_c2=GM_over_c2)
        
        nu_0 = 5e14
        
        # Test visibility at short baseline (should be ≈ 1)
        baseline_short = np.array([10.0, 0.0, 0.0])
        V_short = disk.visibility_analytical(nu_0, baseline_short)
        assert abs(V_short) > 0.8, f"Short baseline visibility should be high, got {abs(V_short)}"
        
        # Test visibility at long baseline (should be smaller)
        baseline_long = np.array([100000.0, 0.0, 0.0])
        V_long = disk.visibility_analytical(nu_0, baseline_long)
        assert abs(V_long) < abs(V_short), f"Long baseline should have lower visibility"


class TestBroadLineRegion:
    """Test BLR model - Section IV"""
    
    def test_blr_velocity_structure(self):
        """Test BLR velocity structure"""
        GM = 1.3e20 * 1e8
        R_in = 1e16
        R_out = 1e18
        distance = 170e6 * 3.086e16
        inclination = np.pi/3
        nu_c = 4.57e14
        
        def beta_func(R):
            return power_law_beta(R, R_in, 2.0, 1e-20)
        
        blr = BroadLineRegion(beta_func, R_in, R_out, GM,
                             inclination, distance, nu_c)
        
        # Test Keplerian velocity calculation
        R_test = 5e16  # meters
        phi_test = np.pi/2  # 90 degrees
        
        v_los = blr._keplerian_velocity(R_test, phi_test)
        v_expected = np.sqrt(GM / R_test) * np.sin(inclination)
        
        assert abs(v_los - v_expected) < 100e3, f"Velocity calculation error: {v_los} vs {v_expected}"
    
    def test_transfer_function(self):
        """Test transfer function calculation"""
        GM = 1.3e20 * 1e8
        R_in = 1e16
        R_out = 1e18
        distance = 170e6 * 3.086e16
        inclination = np.pi/3
        nu_c = 4.57e14
        
        def beta_func(R):
            return power_law_beta(R, R_in, 2.0, 1e-20)
        
        blr = BroadLineRegion(beta_func, R_in, R_out, GM,
                             inclination, distance, nu_c)
        
        omega = 2 * np.pi / (30 * 24 * 3600)  # 30-day period
        velocity = 1000e3  # 1000 km/s
        
        psi = blr.transfer_function_fourier(omega, velocity)
        
        # Transfer function should be finite and complex
        assert np.isfinite(psi), f"Transfer function should be finite, got {psi}"
        assert isinstance(psi, complex), f"Transfer function should be complex, got {type(psi)}"


class TestRelativisticEffects:
    """Test relativistic disk effects"""
    
    def test_isco_calculation(self):
        """Test ISCO radius calculation for different spins"""
        GM_over_c2 = 1.5e11
        distance = 20e6 * 3.086e16
        I_0 = 1e-15
        R_0 = 43.0
        R_in = 6.0
        
        # Schwarzschild case (a = 0)
        disk_schwarzschild = RelativisticDisk(I_0, R_0, R_in,
                                             distance=distance, GM_over_c2=GM_over_c2,
                                             spin_parameter=0.0)
        
        # Maximal prograde spin (a = 1)
        disk_maximal = RelativisticDisk(I_0, R_0, R_in,
                                       distance=distance, GM_over_c2=GM_over_c2,
                                       spin_parameter=1.0)
        
        # ISCO should be smaller for higher spin
        assert disk_maximal.R_isco < disk_schwarzschild.R_isco, \
            f"Higher spin should have smaller ISCO: {disk_maximal.R_isco} vs {disk_schwarzschild.R_isco}"
        
        # Schwarzschild ISCO should be 6 GM/c²
        assert abs(disk_schwarzschild.R_isco - 6.0) < 0.1, \
            f"Schwarzschild ISCO should be 6 GM/c², got {disk_schwarzschild.R_isco}"


class TestFactorizedVisibility:
    """Test factorized visibility - Equations (7-8)"""
    
    def test_temporal_factor(self):
        """Test temporal factor calculation"""
        source = PointSource(1e-12)
        factorized = FactorizedVisibility(source)
        
        delta_nu = 1e12
        
        # At zero time lag, should be 1
        factor_zero = factorized.temporal_factor(delta_nu, 0.0)
        assert abs(factor_zero - 1.0) < 0.01, f"Temporal factor at Δt=0 should be 1, got {factor_zero}"
        
        # At large time lag, should be small
        delta_t_large = 1.0 / delta_nu  # Coherence time
        factor_large = factorized.temporal_factor(delta_nu, delta_t_large)
        assert abs(factor_large) < 0.1, f"Temporal factor at large Δt should be small, got {factor_large}"
    
    def test_spatial_visibility(self):
        """Test spatial visibility calculation"""
        source = PointSource(1e-12)
        factorized = FactorizedVisibility(source)
        
        nu_0 = 5e14
        baseline_perp = np.array([100.0, 0.0])
        
        V_spatial = factorized.spatial_visibility(nu_0, baseline_perp)
        
        # For point source, spatial visibility should be ≈ 1
        assert abs(abs(V_spatial) - 1.0) < 0.1, f"Point source spatial visibility should be ≈ 1, got {abs(V_spatial)}"


def run_all_tests():
    """Run all tests"""
    print("Running comprehensive test suite...")
    
    # Create test instances
    test_classes = [
        TestBasicVisibility(),
        TestPhotonStatistics(),
        TestSNRCalculations(),
        TestShakuraSunyaevDisk(),
        TestBroadLineRegion(),
        TestRelativisticEffects(),
        TestFactorizedVisibility()
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n--- {class_name} ---")
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_class, method_name)
                method()
                print(f"✓ {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"✗ {method_name}: {str(e)}")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed_tests}/{total_tests} passed")
    if passed_tests == total_tests:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed")
    
    return passed_tests == total_tests


class TestPaperFigureReproduction:
    """Test reproduction of figures from the paper"""
    
    def test_figure_1_reproduction(self):
        """Reproduce Figure 1: Shakura-Sunyaev disk visibility"""
        import matplotlib.pyplot as plt
        
        print("\n--- Reproducing Figure 1 from Paper ---")
        
        # Parameters from paper for Figure 1
        GM_over_c2 = 1.0 * 1.496e11  # 1 AU (paper uses GM/c² = 1 AU)
        distance = 20e6 * 3.086e16   # 20 Mpc
        wavelength = 550e-9          # 5500 Å
        nu_0 = 3e8 / wavelength      # Frequency
        
        # Create two disk models with different R_in values
        # Paper shows R_in for maximal spin prograde and retrograde
        I_0 = 1e-15
        R_0 = 100. # 43.0  # From paper
        
        # Prograde maximal spin ISCO ≈ 1 GM/c²
        disk_prograde = ShakuraSunyaevDisk(I_0, R_0, R_in=1.0, n=3.0,
                                          distance=distance, GM_over_c2=GM_over_c2)
        
        # Retrograde maximal spin ISCO ≈ 9 GM/c²
        disk_retrograde = ShakuraSunyaevDisk(I_0, R_0, R_in=9.0, n=3.0,
                                            distance=distance, GM_over_c2=GM_over_c2)
        
        # Calculate baseline range (similar to paper)
        baselines = np.logspace(-1, 3, 100)  # 1 m to 1000 km
        
        visibilities_prograde = []
        visibilities_retrograde = []
        
        for B in baselines:
            baseline_vec = np.array([B * 1000, 0.0, 0.0])  # Convert km to m
            
            V_pro = disk_prograde.visibility_analytical(nu_0, baseline_vec)
            V_ret = disk_retrograde.visibility_analytical(nu_0, baseline_vec)
            
            visibilities_prograde.append(abs(V_pro)**2)
            visibilities_retrograde.append(abs(V_ret)**2)
        
        # Create the plot (similar to Figure 1)
        plt.figure(figsize=(10, 8))
        plt.loglog(baselines, visibilities_prograde, 'b-', linewidth=2,
                  label='R_in = 1 GM/c² (prograde)')
        plt.loglog(baselines, visibilities_retrograde, 'r-', linewidth=2,
                  label='R_in = 9 GM/c² (retrograde)')
        
        # Add vertical lines for characteristic baselines
        lambda_D = wavelength * distance
        
        # Baseline where we start to resolve R_in = 1
        B_resolve_1 = lambda_D / (2 * np.pi * 1.0 * GM_over_c2) / 1000  # km
        plt.axvline(B_resolve_1, color='b', linestyle='--', alpha=0.7,
                   label=f'λD/(2πR_in=1) = {B_resolve_1:.1f} km')
        
        # Baseline where we start to resolve R_in = 9
        B_resolve_9 = lambda_D / (2 * np.pi * 9.0 * GM_over_c2) / 1000  # km
        plt.axvline(B_resolve_9, color='r', linestyle='--', alpha=0.7,
                   label=f'λD/(2πR_in=9) = {B_resolve_9:.1f} km')
        
        plt.xlabel('Baseline [km]')
        plt.ylabel('|V|²')
        plt.title('Squared visibility |V|² for Shakura-Sunyaev disk\n' +
                 f'GM/c² = 1 AU, D = 20 Mpc, λ = 5500 Å')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(1, 1000)
        plt.ylim(1e-8, 1)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig('figure_1_reproduction.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Figure 1 reproduced and saved as 'figure_1_reproduction.png'")
        print(f"  Prograde resolution baseline: {B_resolve_1:.1f} km")
        print(f"  Retrograde resolution baseline: {B_resolve_9:.1f} km")
        
        # Verify that the visibilities behave as expected
        assert visibilities_prograde[0] > 0.9, "Short baseline visibility should be high"
        assert visibilities_prograde[-1] < 0.1, "Long baseline visibility should be low"
        assert visibilities_retrograde[0] > 0.9, "Short baseline visibility should be high"
        assert visibilities_retrograde[-1] < 0.1, "Long baseline visibility should be low"
        
        # The retrograde case should have higher visibility at intermediate baselines
        # due to larger inner hole
        mid_idx = len(baselines) // 2
        assert visibilities_retrograde[mid_idx] > visibilities_prograde[mid_idx], \
            "Retrograde should have higher visibility at intermediate baselines"
        
        return baselines, visibilities_prograde, visibilities_retrograde


def run_all_tests():
    """Run all tests"""
    print("Running comprehensive test suite...")
    
    # Create test instances
    test_classes = [
        TestBasicVisibility(),
        TestPhotonStatistics(),
        TestSNRCalculations(),
        TestShakuraSunyaevDisk(),
        TestBroadLineRegion(),
        TestRelativisticEffects(),
        TestFactorizedVisibility(),
        TestPaperFigureReproduction()
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n--- {class_name} ---")
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_class, method_name)
                method()
                print(f"✓ {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"✗ {method_name}: {str(e)}")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed_tests}/{total_tests} passed")
    if passed_tests == total_tests:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)