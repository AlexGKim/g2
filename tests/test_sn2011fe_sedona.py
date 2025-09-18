#!/usr/bin/env python3
"""
Unit tests for the Sedona SN2011fe Source Model

This test suite validates the functionality of the SedonaSN2011feSource class
including data loading, intensity calculations, flux calculations, and
compatibility with the AbstractSource interface.
"""

import unittest
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock
import tempfile

# Add sources directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'sources'))
from g2.models.sources.grid_source import GridSource   


class TestSedonaSN2011feSource(unittest.TestCase):
    """Test suite for SedonaSN2011feSource class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that will be used across multiple tests"""
        # Create mock data for testing
        cls.mock_wavelength_grid = np.linspace(3000, 10000, 100)  # Angstrom
        cls.mock_flux_data_3d = np.random.rand(100, 50, 50) * 1e-15  # erg/s/cm²/Å
        
        # Create temporary files for testing
        cls.temp_dir = tempfile.mkdtemp()
        cls.wave_file = os.path.join(cls.temp_dir, 'WaveGrid.npy')
        cls.flux_file = os.path.join(cls.temp_dir, 'Phase0Flux.npy')
        
        np.save(cls.wave_file, cls.mock_wavelength_grid)
        np.save(cls.flux_file, cls.mock_flux_data_3d)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """Set up for each test"""
        self.source = SedonaSN2011feSource(
            wave_grid_file=self.wave_file,
            flux_file=self.flux_file
        )
    
    def test_initialization(self):
        """Test proper initialization of the source"""
        # Check that data was loaded correctly
        self.assertEqual(len(self.source.wavelength_grid), 100)
        self.assertEqual(self.source.flux_data_3d.shape, (100, 50, 50))
        
        # Check frequency grid conversion
        self.assertEqual(len(self.source.frequency_grid), 100)
        self.assertTrue(np.all(self.source.frequency_grid > 0))
        
        # Check that frequency grid is in descending order (since wavelength is ascending)
        self.assertTrue(np.all(np.diff(self.source.frequency_grid) < 0))
        
        # Check flux interpolator exists
        self.assertIsNotNone(self.source.flux_interpolator)
        
        # Check frequency range
        self.assertGreater(self.source.freq_max, self.source.freq_min)
    
    
    def test_total_flux_calculation(self):
        """Test total flux calculation"""
        # Test at a frequency within the range
        test_freq = self.source.frequency_grid[50]  # Middle frequency
        flux = self.source.total_flux(test_freq)
        
        # Should return a positive value
        self.assertGreater(flux, 0)
        self.assertIsInstance(flux, float)
        
        # Test at frequency outside range (should return 0 due to fill_value)
        flux_outside = self.source.total_flux(1e20)  # Very high frequency
        self.assertEqual(flux_outside, 0.0)
    
    def test_intensity_single_direction(self):
        """Test intensity calculation for a single direction"""
        test_freq = self.source.frequency_grid[50]
        n_hat = np.array([0.0, 0.0])  # At origin
        
        intensity = self.source.intensity(test_freq, n_hat)
        
        # Should return a non-negative value
        self.assertGreaterEqual(intensity, 0)
        self.assertIsInstance(intensity, (float, np.floating))
        
        # Test at offset position
        n_hat_offset = np.array([1e-6, 1e-6])
        intensity_offset = self.source.intensity(test_freq, n_hat_offset)
        self.assertGreaterEqual(intensity_offset, 0)
    
    def test_intensity_multiple_directions(self):
        """Test intensity calculation for multiple directions"""
        test_freq = self.source.frequency_grid[50]
        n_hat_array = np.array([
            [0.0, 0.0],
            [1e-6, 0.0],
            [0.0, 1e-6],
            [1e-6, 1e-6]
        ])
        
        intensities = self.source.intensity(test_freq, n_hat_array)
        
        # Should return array of same length
        self.assertEqual(len(intensities), 4)
        self.assertTrue(np.all(intensities >= 0))
    
    def test_intensity_array_frequencies(self):
        """Test intensity calculation with array of frequencies"""
        test_freqs = self.source.frequency_grid[40:60:5]  # Sample of frequencies
        n_hat = np.array([0.0, 0.0])
        
        intensities = self.source.intensity(test_freqs, n_hat)
        
        # Should return array matching frequency array
        self.assertEqual(len(intensities), len(test_freqs))
        self.assertTrue(np.all(intensities >= 0))
    
    def test_intensity_out_of_bounds(self):
        """Test intensity calculation for directions outside the grid"""
        test_freq = self.source.frequency_grid[50]
        
        # Very large offset that should be outside the grid
        n_hat_far = np.array([1e-3, 1e-3])  # Much larger than pixel scale
        intensity_far = self.source.intensity(test_freq, n_hat_far)
        
        # Should return 0 for positions outside the grid
        self.assertEqual(intensity_far, 0.0)
    
    def test_interpolate_intensity_helper(self):
        """Test the _interpolate_intensity helper method"""
        # Create a simple test intensity map that matches the source's grid size
        test_map = np.ones((self.source.nx, self.source.ny)) * 100.0
        pixel_scale = 1e-6
        
        # Test at center
        n_hat_center = np.array([0.0, 0.0])
        intensity = self.source._interpolate_intensity(test_map, n_hat_center, pixel_scale)
        self.assertAlmostEqual(intensity, 100.0, places=5)
        
        # Test multiple directions
        n_hat_multi = np.array([[0.0, 0.0], [1e-6, 0.0]])
        intensities = self.source._interpolate_intensity(test_map, n_hat_multi, pixel_scale)
        self.assertEqual(len(intensities), 2)
    
    def test_get_spectrum_info(self):
        """Test spectrum information retrieval"""
        info = self.source.get_spectrum_info()
        
        # Check that all expected keys are present
        expected_keys = [
            'wavelength_range_angstrom',
            'frequency_range_hz',
            'peak_flux_density_w_m2_hz',
            'total_luminosity_estimate',
            'spatial_grid',
            'wavelength_points'
        ]
        
        for key in expected_keys:
            self.assertIn(key, info)
        
        # Check data types and ranges
        self.assertIsInstance(info['wavelength_range_angstrom'], tuple)
        self.assertIsInstance(info['frequency_range_hz'], tuple)
        self.assertGreater(info['peak_flux_density_w_m2_hz'], 0)
        self.assertEqual(info['spatial_grid'], (50, 50))
        self.assertEqual(info['wavelength_points'], 100)
    
    def test_chaotic_source_inheritance(self):
        """Test that the source properly inherits from ChaoticSource"""
        # Test g1 function (inherited from ChaoticSource)
        delta_t = 1e-9  # 1 ns
        nu_0 = 5e14    # 600 nm
        delta_nu = 1e12  # 1 THz
        
        g1_value = self.source.g1(delta_t, nu_0, delta_nu)
        self.assertIsInstance(g1_value, complex)
        
        # Test g2_minus_one function (inherited from ChaoticSource)
        g2_minus_one = self.source.g2_minus_one(delta_t, nu_0, delta_nu)
        self.assertIsInstance(g2_minus_one, float)
        self.assertGreaterEqual(g2_minus_one, 0)
        self.assertLessEqual(g2_minus_one, 1)
    
    def test_visibility_calculation(self):
        """Test visibility calculation using inherited V method"""
        nu_0 = 5e14  # 600 nm
        baseline = np.array([100.0, 0.0, 0.0])  # 100m E-W baseline
        
        # This should work since it inherits from ChaoticSource -> AbstractSource
        visibility = self.source.V(nu_0, baseline)
        self.assertIsInstance(visibility, complex)
    
    def test_units_and_conversions(self):
        """Test that unit conversions are correct"""
        # Test frequency-wavelength conversion
        c = 2.99792458e8  # m/s
        for i in range(0, len(self.source.wavelength_grid), 10):
            wavelength_m = self.source.wavelength_grid[i] * 1e-10
            expected_freq = c / wavelength_m
            actual_freq = self.source.frequency_grid[i]
            self.assertAlmostEqual(actual_freq, expected_freq, places=5)
    
    def test_flux_interpolation_bounds(self):
        """Test flux interpolation at boundaries"""
        # Test at minimum frequency
        flux_min = self.source.total_flux(self.source.freq_min)
        self.assertGreaterEqual(flux_min, 0)
        
        # Test at maximum frequency
        flux_max = self.source.total_flux(self.source.freq_max)
        self.assertGreaterEqual(flux_max, 0)
        
        # Test slightly outside bounds
        flux_below = self.source.total_flux(self.source.freq_min * 0.9)
        flux_above = self.source.total_flux(self.source.freq_max * 1.1)
        self.assertEqual(flux_below, 0.0)
        self.assertEqual(flux_above, 0.0)
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_plot_spectrum(self, mock_figure, mock_show):
        """Test spectrum plotting functionality"""
        # Test that plotting doesn't raise errors
        try:
            self.source.plot_spectrum('angstrom')
            self.source.plot_spectrum('nm')
            self.source.plot_spectrum('micron')
        except ImportError:
            # matplotlib not available, which is fine
            pass
        
        # Test invalid units
        with self.assertRaises(ValueError):
            self.source.plot_spectrum('invalid_unit')
    
    def test_data_consistency(self):
        """Test internal data consistency"""
        # Check that flux data dimensions are consistent
        self.assertEqual(self.source.flux_data_3d.shape[0], len(self.source.wavelength_grid))
        self.assertEqual(self.source.flux_data_3d.shape[0], len(self.source.frequency_grid))
        
        # Check that total flux spectrum has correct length
        self.assertEqual(len(self.source.total_flux_spectrum), len(self.source.wavelength_grid))
        
        # Check that flux density grid has correct length
        self.assertEqual(len(self.source.flux_density_grid), len(self.source.wavelength_grid))


class TestSedonaSourceIntegration(unittest.TestCase):
    """Integration tests for Sedona source with realistic scenarios"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        # Create more realistic mock data
        wavelengths = np.linspace(3000, 10000, 50)  # Smaller for faster tests
        
        # Create a simple Gaussian-like spectrum
        central_wave = 6000  # Angstrom
        sigma = 1000
        spectrum = np.exp(-0.5 * ((wavelengths - central_wave) / sigma)**2)
        
        # Create 3D flux data with spatial structure
        nx, ny = 20, 20
        flux_3d = np.zeros((50, nx, ny))
        
        for i, flux_val in enumerate(spectrum):
            # Create a simple 2D Gaussian spatial profile
            x = np.linspace(-10, 10, nx)
            y = np.linspace(-10, 10, ny)
            X, Y = np.meshgrid(x, y)
            spatial_profile = np.exp(-(X**2 + Y**2) / (2 * 3**2))
            flux_3d[i, :, :] = flux_val * spatial_profile * 1e-15
        
        # Save to temporary files
        self.temp_dir = tempfile.mkdtemp()
        self.wave_file = os.path.join(self.temp_dir, 'WaveGrid.npy')
        self.flux_file = os.path.join(self.temp_dir, 'Phase0Flux.npy')
        
        np.save(self.wave_file, wavelengths)
        np.save(self.flux_file, flux_3d)
        
        self.source = SedonaSN2011feSource(
            wave_grid_file=self.wave_file,
            flux_file=self.flux_file
        )
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_realistic_intensity_profile(self):
        """Test intensity calculation with realistic spatial profile"""
        # Test at central frequency
        central_freq = 5e14  # ~600 nm
        
        # Test intensity at center vs edge
        intensity_center = self.source.intensity(central_freq, np.array([0.0, 0.0]))
        intensity_edge = self.source.intensity(central_freq, np.array([5e-6, 5e-6]))
        
        # Center should have higher intensity than edge for Gaussian profile
        self.assertGreater(intensity_center, intensity_edge)
    
    def test_flux_conservation(self):
        """Test that total flux is conserved in calculations"""
        test_freq = 5e14
        total_flux = self.source.total_flux(test_freq)
        
        # Should be positive and reasonable
        self.assertGreater(total_flux, 0)
        self.assertLess(total_flux, 1e-20)  # Reasonable upper bound for test data


def run_tests():
    """Run all tests and return results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSedonaSN2011feSource))
    suite.addTests(loader.loadTestsFromTestCase(TestSedonaSourceIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("Running Sedona SN2011fe Source Test Suite")
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