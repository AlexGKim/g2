#!/usr/bin/env python3
"""
Unit Tests for AGN Source Models

This test suite validates the functionality of the AGN source classes
including ShakuraSunyaevDisk, BroadLineRegion, and RelativisticDisk.
Tests cover initialization, intensity calculations, flux calculations,
visibility calculations, and ChaoticSource inheritance.
"""

import unittest
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# Add sources directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from g2.models.agn import ShakuraSunyaevDisk, BroadLineRegion, RelativisticDisk, power_law_beta, lognormal_beta


class TestShakuraSunyaevDisk(unittest.TestCase):
    """Test suite for ShakuraSunyaevDisk class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.I_0 = 1e-15  # W m^-2 Hz^-1 sr^-1
        self.R_0 = 10.0   # GM/c^2 units
        self.R_in = 3.0   # GM/c^2 units
        self.n = 3.0
        self.inclination = np.pi / 4  # 45 degrees
        self.distance = 1e20  # m
        self.GM_over_c2 = 1e9  # m
        
        self.disk = ShakuraSunyaevDisk(
            I_0=self.I_0,
            R_0=self.R_0,
            R_in=self.R_in,
            n=self.n,
            inclination=self.inclination,
            distance=self.distance,
            GM_over_c2=self.GM_over_c2
        )
    
    def test_initialization(self):
        """Test proper initialization of ShakuraSunyaevDisk"""
        self.assertEqual(self.disk.I_0, self.I_0)
        self.assertEqual(self.disk.R_0, self.R_0)
        self.assertEqual(self.disk.R_in, self.R_in)
        self.assertEqual(self.disk.n, self.n)
        self.assertEqual(self.disk.inclination, self.inclination)
        self.assertEqual(self.disk.distance, self.distance)
        self.assertEqual(self.disk.GM_over_c2, self.GM_over_c2)
        
        # Check precomputed values
        self.assertAlmostEqual(self.disk.cos_i, np.cos(self.inclination))
        self.assertAlmostEqual(self.disk.sin_i, np.sin(self.inclination))
    
    def test_f_function(self):
        """Test f(R) function calculation"""
        # Test at R_in (should return inf)
        f_at_rin = self.disk._f_function(self.R_in)
        self.assertEqual(f_at_rin, np.inf)
        
        # Test at R < R_in (should return inf)
        f_below_rin = self.disk._f_function(self.R_in - 1.0)
        self.assertEqual(f_below_rin, np.inf)
        
        # Test at R > R_in (should return finite value)
        f_above_rin = self.disk._f_function(self.R_0)
        self.assertTrue(np.isfinite(f_above_rin))
        self.assertGreater(f_above_rin, 0)
    
    def test_disk_intensity(self):
        """Test disk intensity calculation"""
        # Test at R_in (should return 0)
        I_at_rin = self.disk._disk_intensity(self.R_in)
        self.assertEqual(I_at_rin, 0.0)
        
        # Test at R < R_in (should return 0)
        I_below_rin = self.disk._disk_intensity(self.R_in - 1.0)
        self.assertEqual(I_below_rin, 0.0)
        
        # Test at R > R_in (should return positive value)
        I_above_rin = self.disk._disk_intensity(self.R_0)
        self.assertGreater(I_above_rin, 0)
        self.assertLess(I_above_rin, self.I_0)  # Should be less than normalization
    
    def test_intensity_calculation(self):
        """Test intensity calculation at sky positions"""
        # Test at origin
        nu_test = 5e14  # Hz
        n_hat_origin = np.array([0.0, 0.0])
        intensity_origin = self.disk.intensity(nu_test, n_hat_origin)
        self.assertGreaterEqual(intensity_origin, 0)
        
        # Test at offset position
        n_hat_offset = np.array([1e-6, 1e-6])  # Small angular offset
        intensity_offset = self.disk.intensity(nu_test, n_hat_offset)
        self.assertGreaterEqual(intensity_offset, 0)
        
        # Test with array frequency input
        nu_array = np.array([4e14, 5e14, 6e14])
        intensities = self.disk.intensity(nu_array, n_hat_origin)
        self.assertEqual(len(intensities), len(nu_array))
        self.assertTrue(np.all(intensities >= 0))
    
    def test_total_flux_calculation(self):
        """Test total flux calculation"""
        nu_test = 5e14
        total_flux = self.disk.total_flux(nu_test)
        
        # Should return positive value
        self.assertGreater(total_flux, 0)
        self.assertIsInstance(total_flux, float)
        
        # Should be finite
        self.assertTrue(np.isfinite(total_flux))
    
    def test_visibility_calculation(self):
        """Test visibility calculation"""
        nu_0 = 5e14
        baseline = np.array([100.0, 0.0, 0.0])  # 100m E-W baseline
        
        visibility = self.disk.V(nu_0, baseline)
        
        # Should return complex number
        self.assertIsInstance(visibility, complex)
        
        # Magnitude should be between 0 and 1
        vis_mag = abs(visibility)
        self.assertGreaterEqual(vis_mag, 0)
        self.assertLessEqual(vis_mag, 1)
        
        # Test zero baseline (should return 1)
        baseline_zero = np.array([0.0, 0.0, 0.0])
        vis_zero = self.disk.V(nu_0, baseline_zero)
        self.assertAlmostEqual(abs(vis_zero), 1.0, places=5)
    
    def test_chaotic_source_inheritance(self):
        """Test that disk properly inherits from ChaoticSource"""
        # Test g1 function
        delta_t = 1e-9  # 1 ns
        nu_0 = 5e14
        delta_nu = 1e12  # 1 THz
        
        g1_value = self.disk.g1(delta_t, nu_0, delta_nu)
        self.assertIsInstance(g1_value, complex)
        
        # Test g2_minus_one function
        g2_minus_one = self.disk.g2_minus_one(delta_t, nu_0, delta_nu)
        self.assertIsInstance(g2_minus_one, float)
        self.assertGreaterEqual(g2_minus_one, 0)
        self.assertLessEqual(g2_minus_one, 1)
        
        # Test relation g2_minus_one = |g1|^2
        expected_g2_minus_one = abs(g1_value)**2
        self.assertAlmostEqual(g2_minus_one, expected_g2_minus_one, places=10)


class TestBroadLineRegion(unittest.TestCase):
    """Test suite for BroadLineRegion class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Simple power-law beta function
        self.beta_func = lambda R: power_law_beta(R, 1e15, -1.0, 1e-20)
        self.R_in = 1e14  # m
        self.R_out = 1e16  # m
        self.GM = 1e39  # m^3/s^2
        self.inclination = np.pi / 6  # 30 degrees
        self.distance = 1e25  # m
        self.line_center_freq = 4.57e14  # Hz (Hα line)
        
        self.blr = BroadLineRegion(
            beta_function=self.beta_func,
            R_in=self.R_in,
            R_out=self.R_out,
            GM=self.GM,
            inclination=self.inclination,
            distance=self.distance,
            line_center_freq=self.line_center_freq
        )
    
    def test_initialization(self):
        """Test proper initialization of BroadLineRegion"""
        self.assertEqual(self.blr.beta_function, self.beta_func)
        self.assertEqual(self.blr.R_in, self.R_in)
        self.assertEqual(self.blr.R_out, self.R_out)
        self.assertEqual(self.blr.GM, self.GM)
        self.assertEqual(self.blr.inclination, self.inclination)
        self.assertEqual(self.blr.distance, self.distance)
        self.assertEqual(self.blr.nu_c, self.line_center_freq)
        
        # Check precomputed values
        self.assertAlmostEqual(self.blr.cos_i, np.cos(self.inclination))
        self.assertAlmostEqual(self.blr.sin_i, np.sin(self.inclination))
    
    def test_keplerian_velocity(self):
        """Test Keplerian velocity calculation"""
        R_test = 1e15  # m
        phi_test = np.pi / 2  # 90 degrees
        
        v_los = self.blr._keplerian_velocity(R_test, phi_test)
        
        # Should be finite and reasonable
        self.assertTrue(np.isfinite(v_los))
        self.assertLess(abs(v_los), 3e8)  # Less than speed of light
        
        # Test at phi = 0 (should give zero line-of-sight velocity)
        v_los_zero = self.blr._keplerian_velocity(R_test, 0.0)
        self.assertAlmostEqual(v_los_zero, 0.0, places=10)
    
    def test_doppler_shift(self):
        """Test Doppler shift calculation"""
        v_los_test = 1e6  # m/s
        nu_shifted = self.blr._doppler_shift(v_los_test)
        
        # Should be close to but different from center frequency
        self.assertNotEqual(nu_shifted, self.line_center_freq)
        self.assertGreater(nu_shifted, 0)
        
        # Test zero velocity (should return center frequency)
        nu_zero = self.blr._doppler_shift(0.0)
        self.assertAlmostEqual(nu_zero, self.line_center_freq)
    
    def test_intensity_calculation(self):
        """Test BLR intensity calculation"""
        # Test at center frequency
        n_hat_test = np.array([1e-6, 1e-6])
        intensity = self.blr.intensity(self.line_center_freq, n_hat_test)
        
        # Should return non-negative value
        self.assertGreaterEqual(intensity, 0)
        
        # Test outside BLR bounds
        n_hat_far = np.array([1e-3, 1e-3])  # Very large angular offset
        intensity_far = self.blr.intensity(self.line_center_freq, n_hat_far)
        self.assertEqual(intensity_far, 0.0)
    
    def test_total_flux_calculation(self):
        """Test total flux calculation"""
        total_flux = self.blr.total_flux(self.line_center_freq)
        
        # Should return positive value
        self.assertGreaterEqual(total_flux, 0)
        self.assertIsInstance(total_flux, float)
        self.assertTrue(np.isfinite(total_flux))
    
    def test_chaotic_source_inheritance(self):
        """Test that BLR properly inherits from ChaoticSource"""
        # Test g1 and g2_minus_one functions
        delta_t = 1e-9
        nu_0 = self.line_center_freq
        delta_nu = 1e11
        
        g1_value = self.blr.g1(delta_t, nu_0, delta_nu)
        g2_minus_one = self.blr.g2_minus_one(delta_t, nu_0, delta_nu)
        
        self.assertIsInstance(g1_value, complex)
        self.assertIsInstance(g2_minus_one, float)
        self.assertAlmostEqual(g2_minus_one, abs(g1_value)**2, places=10)


class TestRelativisticDisk(unittest.TestCase):
    """Test suite for RelativisticDisk class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.spin_parameter = 0.5  # Moderate spin
        
        self.rel_disk = RelativisticDisk(
            I_0=1e-15,
            R_0=10.0,
            R_in=3.0,
            n=3.0,
            inclination=np.pi/4,
            distance=1e20,
            GM_over_c2=1e9,
            spin_parameter=self.spin_parameter
        )
    
    def test_initialization(self):
        """Test proper initialization of RelativisticDisk"""
        self.assertEqual(self.rel_disk.spin_parameter, self.spin_parameter)
        
        # Check that ISCO radius was calculated
        self.assertGreater(self.rel_disk.R_isco, 0)
        self.assertLess(self.rel_disk.R_isco, 10)  # Should be reasonable
        
        # Check that R_in was updated to ISCO if necessary
        self.assertGreaterEqual(self.rel_disk.R_in, self.rel_disk.R_isco)
    
    def test_isco_radius_calculation(self):
        """Test ISCO radius calculation for different spins"""
        # Test zero spin (should be 6 GM/c^2)
        disk_zero_spin = RelativisticDisk(
            I_0=1e-15, R_0=10.0, R_in=3.0, n=3.0,
            inclination=0, distance=1e20, GM_over_c2=1e9,
            spin_parameter=0.0
        )
        self.assertAlmostEqual(disk_zero_spin.R_isco, 6.0, places=1)
        
        # Test maximum prograde spin (should be 1 GM/c^2)
        disk_max_spin = RelativisticDisk(
            I_0=1e-15, R_0=10.0, R_in=3.0, n=3.0,
            inclination=0, distance=1e20, GM_over_c2=1e9,
            spin_parameter=0.998
        )
        self.assertLess(disk_max_spin.R_isco, 2.0)
    
    def test_doppler_factor(self):
        """Test relativistic Doppler factor calculation"""
        R_test = 10.0  # GM/c^2 units
        phi_test = np.pi / 2
        
        doppler_factor = self.rel_disk._doppler_factor(R_test, phi_test)
        
        # Should be positive and finite
        self.assertGreater(doppler_factor, 0)
        self.assertTrue(np.isfinite(doppler_factor))
        
        # Should be different from 1 due to relativistic effects
        self.assertNotAlmostEqual(doppler_factor, 1.0, places=3)
    
    def test_intensity_with_relativistic_effects(self):
        """Test intensity calculation with relativistic corrections"""
        nu_test = 5e14
        n_hat_test = np.array([1e-6, 1e-6])
        
        # Get relativistic intensity
        intensity_rel = self.rel_disk.intensity(nu_test, n_hat_test)
        
        # Get base SS disk intensity for comparison
        base_disk = ShakuraSunyaevDisk(
            I_0=self.rel_disk.I_0,
            R_0=self.rel_disk.R_0,
            R_in=self.rel_disk.R_in,
            n=self.rel_disk.n,
            inclination=self.rel_disk.inclination,
            distance=self.rel_disk.distance,
            GM_over_c2=self.rel_disk.GM_over_c2
        )
        intensity_base = base_disk.intensity(nu_test, n_hat_test)
        
        # Relativistic intensity should be different from base
        if intensity_base > 0:
            self.assertNotAlmostEqual(intensity_rel, intensity_base, places=10)
        
        # Should still be non-negative
        self.assertGreaterEqual(intensity_rel, 0)


class TestUtilityFunctions(unittest.TestCase):
    """Test suite for utility functions"""
    
    def test_power_law_beta(self):
        """Test power law beta function"""
        R = 1e15
        R_0 = 1e14
        n = -1.0
        norm = 1e-20
        
        beta_val = power_law_beta(R, R_0, n, norm)
        
        # Should return expected value
        expected = norm * (R / R_0)**n
        self.assertAlmostEqual(beta_val, expected)
        
        # Should be positive for positive inputs
        self.assertGreater(beta_val, 0)
    
    def test_lognormal_beta(self):
        """Test lognormal beta function"""
        R = 1e15
        R_0 = 1e15  # Same as R for simplicity
        sigma = 0.5
        norm = 1e-20
        
        beta_val = lognormal_beta(R, R_0, sigma, norm)
        
        # Should return positive value
        self.assertGreater(beta_val, 0)
        self.assertTrue(np.isfinite(beta_val))
        
        # Test with R = 0 (should return 0)
        beta_zero = lognormal_beta(0.0, R_0, sigma, norm)
        self.assertEqual(beta_zero, 0.0)


class TestAGNIntegration(unittest.TestCase):
    """Integration tests for AGN models"""
    
    def test_disk_visibility_vs_baseline(self):
        """Test that visibility decreases with increasing baseline"""
        disk = ShakuraSunyaevDisk(
            I_0=1e-15, R_0=10.0, R_in=3.0, n=3.0,
            inclination=0, distance=1e20, GM_over_c2=1e9
        )
        
        nu_0 = 5e14
        baselines = [10.0, 100.0, 1000.0]  # Increasing baseline lengths
        visibilities = []
        
        for B in baselines:
            baseline = np.array([B, 0.0, 0.0])
            vis = disk.V(nu_0, baseline)
            visibilities.append(abs(vis))
        
        # Visibility should generally decrease with baseline
        # (though there may be oscillations for complex sources)
        self.assertGreaterEqual(visibilities[0], visibilities[-1])
    
    def test_flux_conservation(self):
        """Test flux conservation in intensity integration"""
        disk = ShakuraSunyaevDisk(
            I_0=1e-15, R_0=10.0, R_in=3.0, n=3.0,
            inclination=0, distance=1e20, GM_over_c2=1e9
        )
        
        nu_test = 5e14
        total_flux = disk.total_flux(nu_test)
        
        # Should be positive and finite
        self.assertGreater(total_flux, 0)
        self.assertTrue(np.isfinite(total_flux))
        
        # Should be consistent across multiple calls
        total_flux_2 = disk.total_flux(nu_test)
        self.assertAlmostEqual(total_flux, total_flux_2)


def run_tests():
    """Run all tests and return results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestShakuraSunyaevDisk))
    suite.addTests(loader.loadTestsFromTestCase(TestBroadLineRegion))
    suite.addTests(loader.loadTestsFromTestCase(TestRelativisticDisk))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestAGNIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("Running AGN Source Models Test Suite")
    print("=" * 50)
    
    result = run_tests()
    
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")
    
    print(f"\nRan {result.testsRun} tests in total")