"""
Unit tests for the source module.

This test suite validates the functionality of all source classes including:
- AbstractSource base class methods
- ChaoticSource temporal coherence functions
- PointSource analytical solutions
- UniformDisk analytical vs FFT comparison
"""

import unittest
import numpy as np
from source import AbstractSource, ChaoticSource, PointSource, UniformDisk


class TestAbstractSource(unittest.TestCase):
    """Test cases for AbstractSource base class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a UniformDisk instance for testing AbstractSource methods
        self.flux_density = 1e-26  # W/m²/Hz
        self.radius = 1e-8  # radians (~2 mas)
        self.disk = UniformDisk(self.flux_density, self.radius)
        
        # Test parameters
        self.nu_0 = 5e14  # 600 nm
        self.baseline_short = np.array([10.0, 0.0, 0.0])  # 10m E-W
        self.baseline_long = np.array([100.0, 0.0, 0.0])  # 100m E-W
        
    def test_V_method_exists(self):
        """Test that V method exists and is callable."""
        self.assertTrue(hasattr(self.disk, 'V'))
        self.assertTrue(callable(getattr(self.disk, 'V')))
        
    def test_V_returns_complex(self):
        """Test that V method returns complex number."""
        result = self.disk.V(self.nu_0, self.baseline_short)
        self.assertIsInstance(result, complex)
        
    def test_V_zero_baseline(self):
        """Test V method with zero baseline returns 1.0."""
        baseline_zero = np.array([0.0, 0.0, 0.0])
        result = self.disk.V(self.nu_0, baseline_zero)
        self.assertAlmostEqual(abs(result), 1.0, places=6)
        
    def test_V_vs_analytic_airy(self):
        """Test AbstractSource.V against UniformDisk.V (analytical Airy)."""
        # Get FFT-based result from AbstractSource.V (inherited)
        V_fft = super(UniformDisk, self.disk).V(self.nu_0, self.baseline_short)
        
        # Get analytical result from UniformDisk.V (overridden)
        V_analytic = self.disk.V(self.nu_0, self.baseline_short)
        
        # They should agree within numerical precision (FFT has discretization errors)
        self.assertAlmostEqual(abs(V_fft), abs(V_analytic), places=1,
                              msg=f"FFT: {abs(V_fft):.6f}, Analytic: {abs(V_analytic):.6f}")
        
    def test_V_vs_analytic_multiple_baselines(self):
        """Test AbstractSource.V vs analytical for multiple baselines."""
        baselines = [
            np.array([10.0, 0.0, 0.0]),
            np.array([50.0, 0.0, 0.0]),
            np.array([100.0, 0.0, 0.0]),
            np.array([0.0, 75.0, 0.0]),  # N-S baseline
            np.array([50.0, 50.0, 0.0])  # Diagonal baseline
        ]
        
        for baseline in baselines:
            with self.subTest(baseline=baseline):
                V_fft = super(UniformDisk, self.disk).V(self.nu_0, baseline)
                V_analytic = self.disk.V(self.nu_0, baseline)
                
                # Allow for some numerical error in FFT method (discretization)
                self.assertAlmostEqual(abs(V_fft), abs(V_analytic), places=1,
                                      msg=f"Baseline {baseline}: FFT={abs(V_fft):.6f}, Analytic={abs(V_analytic):.6f}")


class TestChaoticSource(unittest.TestCase):
    """Test cases for ChaoticSource class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a PointSource instance (inherits from ChaoticSource)
        self.point = PointSource(lambda nu: 1e-26)
        
        # Test parameters
        self.nu_0 = 5e14  # Hz
        self.delta_nu = 1e12  # Hz
        self.delta_t_zero = 0.0
        self.delta_t_small = 1e-12  # ps
        self.delta_t_large = 1e-9   # ns
        
    def test_g1_method_exists(self):
        """Test that g1 method exists and is callable."""
        self.assertTrue(hasattr(self.point, 'g1'))
        self.assertTrue(callable(getattr(self.point, 'g1')))
        
    def test_g2_minus_one_method_exists(self):
        """Test that g2_minus_one method exists and is callable."""
        self.assertTrue(hasattr(self.point, 'g2_minus_one'))
        self.assertTrue(callable(getattr(self.point, 'g2_minus_one')))
        
    def test_g1_returns_complex(self):
        """Test that g1 returns complex number."""
        result = self.point.g1(self.delta_t_zero, self.nu_0, self.delta_nu)
        self.assertIsInstance(result, complex)
        
    def test_g2_minus_one_returns_float(self):
        """Test that g2_minus_one returns float."""
        result = self.point.g2_minus_one(self.delta_t_zero, self.nu_0, self.delta_nu)
        self.assertIsInstance(result, (float, np.floating))
        
    def test_g1_at_zero_delay(self):
        """Test g1 at zero time delay equals 1."""
        result = self.point.g1(self.delta_t_zero, self.nu_0, self.delta_nu)
        self.assertAlmostEqual(abs(result), 1.0, places=10)
        
    def test_g2_minus_one_at_zero_delay(self):
        """Test g2_minus_one at zero time delay equals 0 (since |g1(0)|² = 1, so |g1(0)|² - 1 = 0)."""
        result = self.point.g2_minus_one(self.delta_t_zero, self.nu_0, self.delta_nu)
        self.assertAlmostEqual(result, 0.0, places=10)
        
    def test_g2_minus_one_equals_g1_squared_minus_one(self):
        """Test that g2_minus_one = |g1|² - 1."""
        for delta_t in [self.delta_t_zero, self.delta_t_small, self.delta_t_large]:
            with self.subTest(delta_t=delta_t):
                g1 = self.point.g1(delta_t, self.nu_0, self.delta_nu)
                g2_minus_one = self.point.g2_minus_one(delta_t, self.nu_0, self.delta_nu)
                
                expected = abs(g1)**2 - 1.0
                self.assertAlmostEqual(g2_minus_one, expected, places=10,
                                      msg=f"At Δt={delta_t}: g²-1={g2_minus_one}, |g¹|²-1={expected}")
                
    def test_g1_sinc_behavior(self):
        """Test that g1 exhibits sinc behavior."""
        # At large time delays, g1 should approach zero
        delta_t_large = 1.0 / self.delta_nu  # Coherence time
        g1_large = self.point.g1(delta_t_large, self.nu_0, self.delta_nu)
        
        # Should be much smaller than at zero delay
        self.assertLess(abs(g1_large), 0.1)
        
    def test_coherence_time_scaling(self):
        """Test that coherence time scales with 1/Δν."""
        delta_nu_narrow = 1e11  # Narrow bandwidth
        delta_nu_wide = 1e13    # Wide bandwidth
        delta_t_test = 5e-12    # Test time
        
        g1_narrow = self.point.g1(delta_t_test, self.nu_0, delta_nu_narrow)
        g1_wide = self.point.g1(delta_t_test, self.nu_0, delta_nu_wide)
        
        # Narrow bandwidth should have longer coherence time (higher correlation)
        # At this time scale, both will be < 1, but narrow should be larger
        self.assertGreater(abs(g1_narrow), abs(g1_wide))
        
        # Both should be positive and less than 1
        self.assertGreater(abs(g1_narrow), 0.0)
        self.assertGreater(abs(g1_wide), 0.0)
        self.assertLess(abs(g1_narrow), 1.0)
        self.assertLess(abs(g1_wide), 1.0)


class TestPointSource(unittest.TestCase):
    """Test cases for PointSource class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.flux_constant = 1e-26  # W/m²/Hz
        self.point = PointSource(lambda nu: self.flux_constant)
        
        # Test parameters
        self.nu_0 = 5e14
        self.baseline = np.array([100.0, 0.0, 0.0])
        
    def test_initialization(self):
        """Test PointSource initialization."""
        self.assertIsInstance(self.point, PointSource)
        self.assertIsInstance(self.point, ChaoticSource)
        self.assertIsInstance(self.point, AbstractSource)
        
    def test_total_flux(self):
        """Test total flux calculation."""
        flux = self.point.total_flux(self.nu_0)
        self.assertEqual(flux, self.flux_constant)
        
    def test_V_always_unity(self):
        """Test that V is always 1.0 for point source."""
        baselines = [
            np.array([0.0, 0.0, 0.0]),
            np.array([10.0, 0.0, 0.0]),
            np.array([1000.0, 0.0, 0.0]),
            np.array([0.0, 500.0, 0.0]),
            np.array([100.0, 100.0, 0.0])
        ]
        
        for baseline in baselines:
            with self.subTest(baseline=baseline):
                result = self.point.V(self.nu_0, baseline)
                self.assertAlmostEqual(abs(result), 1.0, places=10)
                self.assertAlmostEqual(result.imag, 0.0, places=10)
                
    def test_frequency_dependent_flux(self):
        """Test point source with frequency-dependent flux."""
        # Power-law spectrum: F_ν ∝ ν^(-0.7)
        alpha = -0.7
        nu_ref = 5e14
        flux_ref = 1e-26
        
        power_law_point = PointSource(lambda nu: flux_ref * (nu/nu_ref)**alpha)
        
        # Test at different frequencies
        frequencies = [1e14, 5e14, 1e15]
        for nu in frequencies:
            with self.subTest(nu=nu):
                expected_flux = flux_ref * (nu/nu_ref)**alpha
                actual_flux = power_law_point.total_flux(nu)
                self.assertAlmostEqual(actual_flux, expected_flux, places=10)


class TestUniformDisk(unittest.TestCase):
    """Test cases for UniformDisk class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.flux_density = 1e-26  # W/m²/Hz
        self.radius = 1e-8  # radians
        self.disk = UniformDisk(self.flux_density, self.radius)
        
        # Test parameters
        self.nu_0 = 5e14
        self.c = 2.99792458e8
        self.wavelength = self.c / self.nu_0
        
    def test_initialization(self):
        """Test UniformDisk initialization."""
        self.assertIsInstance(self.disk, UniformDisk)
        self.assertIsInstance(self.disk, ChaoticSource)
        self.assertIsInstance(self.disk, AbstractSource)
        
        self.assertEqual(self.disk.flux_density, self.flux_density)
        self.assertEqual(self.disk.radius, self.radius)
        
        # Check surface brightness calculation
        expected_brightness = self.flux_density / (np.pi * self.radius**2)
        self.assertAlmostEqual(self.disk.surface_brightness, expected_brightness)
        
    def test_total_flux(self):
        """Test total flux is conserved."""
        flux = self.disk.total_flux(self.nu_0)
        self.assertAlmostEqual(flux, self.flux_density, places=10)
        
    def test_intensity_inside_disk(self):
        """Test intensity inside the disk."""
        # Point at center
        n_hat_center = np.array([0.0, 0.0])
        intensity_center = self.disk.intensity(self.nu_0, n_hat_center)
        self.assertAlmostEqual(intensity_center, self.disk.surface_brightness)
        
        # Point at edge
        n_hat_edge = np.array([self.radius, 0.0])
        intensity_edge = self.disk.intensity(self.nu_0, n_hat_edge)
        self.assertAlmostEqual(intensity_edge, self.disk.surface_brightness)
        
    def test_intensity_outside_disk(self):
        """Test intensity outside the disk is zero."""
        n_hat_outside = np.array([2 * self.radius, 0.0])
        intensity_outside = self.disk.intensity(self.nu_0, n_hat_outside)
        self.assertEqual(intensity_outside, 0.0)
        
    def test_V_zero_baseline(self):
        """Test V at zero baseline equals 1."""
        baseline_zero = np.array([0.0, 0.0, 0.0])
        result = self.disk.V(self.nu_0, baseline_zero)
        self.assertAlmostEqual(abs(result), 1.0, places=10)
        
    def test_V_airy_function(self):
        """Test V implements correct Airy function."""
        from scipy.special import j1
        
        baseline = np.array([100.0, 0.0, 0.0])
        baseline_length = np.linalg.norm(baseline[:2])
        
        # Calculate expected Airy function value
        u = baseline_length / self.wavelength
        x = 2 * np.pi * u * self.radius
        
        if x == 0:
            expected = 1.0
        else:
            expected = 2 * j1(x) / x
            
        result = self.disk.V(self.nu_0, baseline)
        self.assertAlmostEqual(abs(result), abs(expected), places=10)
        
    def test_V_first_zero(self):
        """Test V first zero occurs at correct baseline."""
        # First zero of Airy function: 2J₁(x)/x = 0 when x = 3.8317...
        # This gives: 2πuθ = 3.8317, so u = 3.8317/(2πθ)
        # Therefore: B = λu = λ × 3.8317/(2πθ) = 1.22λ/(2θ)
        
        first_zero_u = 3.8317 / (2 * np.pi * self.radius)
        baseline_first_zero = first_zero_u * self.wavelength
        
        baseline = np.array([baseline_first_zero, 0.0, 0.0])
        result = self.disk.V(self.nu_0, baseline)
        
        # Should be very close to zero
        self.assertLess(abs(result), 0.01)
        
    def test_V_symmetry(self):
        """Test V is symmetric for different baseline orientations."""
        baseline_length = 50.0
        
        baselines = [
            np.array([baseline_length, 0.0, 0.0]),  # E-W
            np.array([0.0, baseline_length, 0.0]),  # N-S
            np.array([baseline_length/np.sqrt(2), baseline_length/np.sqrt(2), 0.0])  # Diagonal
        ]
        
        results = []
        for baseline in baselines:
            result = self.disk.V(self.nu_0, baseline)
            results.append(abs(result))
            
        # All should be equal (symmetric disk)
        for i in range(1, len(results)):
            self.assertAlmostEqual(results[i], results[0], places=8)


class TestIntegration(unittest.TestCase):
    """Integration tests across multiple classes."""
    
    def test_point_vs_disk_limit(self):
        """Test that very small disk approaches point source behavior."""
        # Very small disk
        tiny_radius = 1e-12  # Much smaller than typical baselines
        tiny_disk = UniformDisk(1e-26, tiny_radius)
        
        # Point source
        point = PointSource(lambda nu: 1e-26)
        
        # Test parameters
        nu_0 = 5e14
        baseline = np.array([100.0, 0.0, 0.0])
        
        # Both should give visibility ≈ 1
        disk_V = tiny_disk.V(nu_0, baseline)
        point_V = point.V(nu_0, baseline)
        
        self.assertAlmostEqual(abs(disk_V), abs(point_V), places=3)
        
    def test_chaotic_source_inheritance(self):
        """Test that all chaotic sources have consistent g1/g2_minus_one behavior."""
        sources = [
            PointSource(lambda nu: 1e-26),
            UniformDisk(1e-26, 1e-8)
        ]
        
        # Test temporal coherence (only for ChaoticSource methods)
        delta_t = 1e-12
        nu_0 = 5e14
        delta_nu = 1e12
        
        for source in sources:
            with self.subTest(source=type(source).__name__):
                # Test g2_minus_one method (inherited from ChaoticSource)
                g2_minus_one = source.g2_minus_one(delta_t, nu_0, delta_nu)
                
                # All ChaoticSource instances should have temporal g1 method
                g1_temporal = source.g1(delta_t, nu_0, delta_nu)
                expected = abs(g1_temporal)**2 - 1.0
                self.assertAlmostEqual(g2_minus_one, expected, places=10)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)