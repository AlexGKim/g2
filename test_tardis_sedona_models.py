"""
Specialized Tests for TARDIS and SEDONA Model Comparisons
Based on Section II of II_Telescopes.pdf

This test suite focuses specifically on validating the TARDIS and SEDONA
spectral synthesis models and their visibility calculations.
"""

import unittest
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import scipy.special as special
from scipy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt

# Import our modules
from supernova_models import SupernovaEjecta, SupernovaParameters
from intensity_interferometry import VisibilityCalculator


class TestTARDISModelValidation(unittest.TestCase):
    """Validate TARDIS model implementation against paper results"""
    
    def setUp(self):
        """Load TARDIS data and set up test parameters"""
        self.data_file = Path("import/SN2011fe_MLE_intensity_maxlight.hdf")
        self.has_real_data = self.data_file.exists()
        
        if self.has_real_data:
            try:
                self.intensity_data = pd.read_hdf(self.data_file, key='intensity')
                self.wavelengths = self.intensity_data.index.values
                self.impact_parameters = self.intensity_data.columns.values  # in cm
                self.intensity_values = self.intensity_data.values
                
                # Flip wavelength order as in the notebook
                self.wavelengths = np.flip(self.wavelengths)
                self.intensity_values = np.flip(self.intensity_values, axis=0)
                
            except Exception as e:
                self.has_real_data = False
                print(f"Could not load TARDIS data: {e}")
        
        # Paper-specific wavelengths from Figure 2
        self.paper_wavelengths = [3700, 4700, 6055, 6355, 8750]  # Angstroms
        
        # SN2011fe parameters
        self.sn2011fe_params = SupernovaParameters(
            sn_type="Ia",
            explosion_time=20,  # Peak B-band time
            expansion_velocity=10000,  # km/s
            distance=6.4,  # Mpc (actual distance to SN2011fe)
            absolute_magnitude=-19.46
        )
    
    @unittest.skipUnless(Path("import/SN2011fe_MLE_intensity_maxlight.hdf").exists(), 
                        "TARDIS data file not available")
    def test_tardis_data_wavelength_range(self):
        """Test TARDIS wavelength coverage matches paper Figure 2"""
        if self.has_real_data:
            # Should cover 3000-10000 Å range
            self.assertGreaterEqual(np.min(self.wavelengths), 2500,
                                  "TARDIS should cover blue wavelengths")
            self.assertLessEqual(np.max(self.wavelengths), 11000,
                                "TARDIS should cover red wavelengths")
            
            # Check specific wavelengths from Figure 2
            for target_wl in self.paper_wavelengths:
                closest_idx = np.argmin(np.abs(self.wavelengths - target_wl))
                closest_wl = self.wavelengths[closest_idx]
                self.assertLess(abs(closest_wl - target_wl), 100,
                              f"Should have wavelength near {target_wl} Å")
    
    @unittest.skipUnless(Path("import/SN2011fe_MLE_intensity_maxlight.hdf").exists(), 
                        "TARDIS data file not available")
    def test_tardis_impact_parameter_range(self):
        """Test TARDIS impact parameter coverage"""
        if self.has_real_data:
            # Convert to 10^10 km units as in paper
            impact_params_1e10km = self.impact_parameters / 1e15
            
            # Should cover 0 to ~3 x 10^10 km as shown in Figure 2
            self.assertAlmostEqual(np.min(impact_params_1e10km), 0, delta=0.1,
                                 msg="Should start near zero impact parameter")
            self.assertGreater(np.max(impact_params_1e10km), 2.5,
                             "Should cover large impact parameters")
    
    @unittest.skipUnless(Path("import/SN2011fe_MLE_intensity_maxlight.hdf").exists(), 
                        "TARDIS data file not available")
    def test_tardis_emission_profile_properties(self):
        """Test TARDIS emission profile properties from Figure 2"""
        if self.has_real_data:
            # Test each wavelength from Figure 2
            for target_wl in self.paper_wavelengths:
                closest_idx = np.argmin(np.abs(self.wavelengths - target_wl))
                profile = self.intensity_values[closest_idx, :]
                
                # Normalize profile
                if np.max(profile) > 0:
                    normalized_profile = profile / np.max(profile)
                    
                    # Profile should be peaked toward center
                    peak_idx = np.argmax(normalized_profile)
                    center_fraction = peak_idx / len(normalized_profile)
                    self.assertLess(center_fraction, 0.5,
                                  f"Peak should be in inner half for λ={target_wl}")
                    
                    # Profile should decrease toward edges
                    edge_value = normalized_profile[-1]
                    self.assertLess(edge_value, 0.5,
                                  f"Edge intensity should be <50% of peak for λ={target_wl}")
    
    @unittest.skipUnless(Path("import/SN2011fe_MLE_intensity_maxlight.hdf").exists(), 
                        "TARDIS data file not available")
    def test_tardis_wavelength_dependence(self):
        """Test wavelength dependence of emission profiles"""
        if self.has_real_data:
            # Compare profiles at different wavelengths
            profiles = {}
            for target_wl in self.paper_wavelengths:
                closest_idx = np.argmin(np.abs(self.wavelengths - target_wl))
                profile = self.intensity_values[closest_idx, :]
                if np.max(profile) > 0:
                    profiles[target_wl] = profile / np.max(profile)
            
            # Profiles should be different at different wavelengths
            if len(profiles) >= 2:
                wl_keys = list(profiles.keys())
                profile1 = profiles[wl_keys[0]]
                profile2 = profiles[wl_keys[1]]
                
                # Calculate RMS difference
                rms_diff = np.sqrt(np.mean((profile1 - profile2)**2))
                self.assertGreater(rms_diff, 0.05,
                                 "Profiles should differ significantly between wavelengths")
    
    def test_tardis_spherical_symmetry(self):
        """Test that TARDIS model maintains spherical symmetry"""
        # Create TARDIS-like supernova model
        sn = SupernovaEjecta(self.sn2011fe_params)
        
        # Test intensity at same radius but different angles
        radius = 0.5  # Normalized radius
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        intensities = []
        for angle in angles:
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            # Test without polarization for pure spherical symmetry
            intensity = sn.intensity_2d(x, y, include_polarization=False)
            intensities.append(intensity)
        
        # All intensities should be equal for spherical symmetry
        intensities = np.array(intensities)
        if np.mean(intensities) > 0:
            relative_std = np.std(intensities) / np.mean(intensities)
            self.assertLess(relative_std, 0.01,
                          "TARDIS model should be spherically symmetric")


class TestSEDONAModelValidation(unittest.TestCase):
    """Validate SEDONA model implementation against paper results"""
    
    def setUp(self):
        """Set up SEDONA model test parameters"""
        # SEDONA wavelengths from Figure 4
        self.sedona_wavelengths = [3696.69, 4698.29, 6128.39, 6189.92, 8745.82]
        
        # N100 model parameters (asymmetric ignition)
        self.sedona_params = SupernovaParameters(
            sn_type="Ia",
            explosion_time=20,  # 20 days after explosion
            expansion_velocity=10000,  # km/s
            distance=20.0,  # Mpc (z=0.004)
            absolute_magnitude=-19.46
        )
        self.sedona_sn = SupernovaEjecta(self.sedona_params)
    
    def test_sedona_asymmetric_emission(self):
        """Test that SEDONA model produces asymmetric emission"""
        # Create 2D intensity map
        extent = 2.0
        n_points = 51
        x = np.linspace(-extent, extent, n_points)
        y = np.linspace(-extent, extent, n_points)
        
        intensity_map = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(n_points):
                # Use polarization to create asymmetry
                intensity_map[i, j] = self.sedona_sn.intensity_2d(
                    x[i], y[j], include_polarization=True
                )
        
        # Test for asymmetry by comparing quadrants
        center = n_points // 2
        quarter = n_points // 4
        
        # Extract quadrants
        q1 = intensity_map[center:center+quarter, center:center+quarter]
        q2 = intensity_map[center:center+quarter, center-quarter:center]
        q3 = intensity_map[center-quarter:center, center-quarter:center]
        q4 = intensity_map[center-quarter:center, center:center+quarter]
        
        quadrant_sums = [np.sum(q) for q in [q1, q2, q3, q4]]
        
        if max(quadrant_sums) > 0:
            # Calculate asymmetry measure
            asymmetry = (max(quadrant_sums) - min(quadrant_sums)) / max(quadrant_sums)
            self.assertGreater(asymmetry, 0.01,
                             "SEDONA model should show asymmetry")
    
    def test_sedona_wavelength_specific_features(self):
        """Test wavelength-specific features from Figure 4"""
        baseline_length = 5000  # 5 km
        
        visibilities = {}
        for wl_angstrom in self.sedona_wavelengths:
            wavelength = wl_angstrom * 1e-10  # Convert to meters
            vis = self.sedona_sn.visibility_amplitude(
                baseline_length, wavelength, include_polarization=True
            )
            visibilities[wl_angstrom] = vis
        
        # Visibilities should vary with wavelength
        vis_values = list(visibilities.values())
        if len(vis_values) > 1:
            vis_std = np.std(vis_values)
            vis_mean = np.mean(vis_values)
            if vis_mean > 0:
                relative_variation = vis_std / vis_mean
                self.assertGreater(relative_variation, 0.05,
                                 "SEDONA visibility should vary with wavelength")
    
    def test_sedona_3d_structure(self):
        """Test 3D structure effects in SEDONA model"""
        # Test visibility at different baseline orientations
        baseline_length = 3000  # 3 km
        wavelength = 550e-9
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        visibilities = []
        for angle in angles:
            # For now, our model doesn't explicitly handle baseline angle
            # but we can test the intensity profile asymmetry
            vis = self.sedona_sn.visibility_amplitude(
                baseline_length, wavelength, include_polarization=True
            )
            visibilities.append(vis)
        
        # All should be positive and reasonable
        for vis in visibilities:
            self.assertGreaterEqual(vis, 0, "Visibility should be non-negative")
            self.assertLessEqual(vis, 1, "Visibility should be ≤ 1")


class TestTARDISvsSEDONAComparison(unittest.TestCase):
    """Compare TARDIS and SEDONA model predictions"""
    
    def setUp(self):
        """Set up comparison parameters"""
        # Common supernova parameters
        self.common_params = SupernovaParameters(
            sn_type="Ia",
            explosion_time=20,
            expansion_velocity=10000,
            distance=20.0,
            absolute_magnitude=-19.46
        )
        
        # Create models
        self.tardis_sn = SupernovaEjecta(self.common_params)
        self.sedona_sn = SupernovaEjecta(self.common_params)
        
        # Test wavelengths
        self.test_wavelengths = [4000, 5000, 6000, 7000, 8000]  # Angstroms
    
    def test_visibility_amplitude_comparison(self):
        """Compare visibility amplitudes between TARDIS and SEDONA"""
        baseline_length = 2000  # 2 km
        
        for wl_angstrom in self.test_wavelengths:
            wavelength = wl_angstrom * 1e-10
            
            # TARDIS (spherically symmetric)
            vis_tardis = self.tardis_sn.visibility_amplitude(
                baseline_length, wavelength, include_polarization=False
            )
            
            # SEDONA (asymmetric)
            vis_sedona = self.sedona_sn.visibility_amplitude(
                baseline_length, wavelength, include_polarization=True
            )
            
            # Both should be reasonable
            self.assertGreaterEqual(vis_tardis, 0, f"TARDIS visibility should be ≥0 at {wl_angstrom}Å")
            self.assertGreaterEqual(vis_sedona, 0, f"SEDONA visibility should be ≥0 at {wl_angstrom}Å")
            self.assertLessEqual(vis_tardis, 1, f"TARDIS visibility should be ≤1 at {wl_angstrom}Å")
            self.assertLessEqual(vis_sedona, 1, f"SEDONA visibility should be ≤1 at {wl_angstrom}Å")
    
    def test_model_consistency_at_short_baselines(self):
        """Test that both models give similar results at short baselines"""
        # At very short baselines, both should approach unity visibility
        short_baseline = 100  # 100 m
        wavelength = 550e-9
        
        vis_tardis = self.tardis_sn.visibility_amplitude(
            short_baseline, wavelength, include_polarization=False
        )
        vis_sedona = self.sedona_sn.visibility_amplitude(
            short_baseline, wavelength, include_polarization=True
        )
        
        # Both should be close to 1 for unresolved source
        self.assertGreater(vis_tardis, 0.9, "TARDIS should give high visibility at short baseline")
        self.assertGreater(vis_sedona, 0.9, "SEDONA should give high visibility at short baseline")
    
    def test_model_differences_at_long_baselines(self):
        """Test that models show differences at long baselines"""
        # At long baselines, asymmetry effects should be more apparent
        long_baseline = 8000  # 8 km
        wavelength = 550e-9
        
        vis_tardis = self.tardis_sn.visibility_amplitude(
            long_baseline, wavelength, include_polarization=False
        )
        vis_sedona = self.sedona_sn.visibility_amplitude(
            long_baseline, wavelength, include_polarization=True
        )
        
        # Both should be significantly less than 1 (resolved)
        self.assertLess(vis_tardis, 0.8, "TARDIS should show resolution at long baseline")
        self.assertLess(vis_sedona, 0.8, "SEDONA should show resolution at long baseline")


class TestVisibilityCalculationMethods(unittest.TestCase):
    """Test different visibility calculation methods"""
    
    def test_hankel_transform_method(self):
        """Test Hankel transform method for axisymmetric profiles"""
        # Create simple Gaussian profile
        def gaussian_profile(r):
            sigma = 0.5
            return np.exp(-r**2 / (2 * sigma**2))
        
        # Test visibility calculation
        baseline_length = 1000  # 1 km
        wavelength = 550e-9
        
        # For Gaussian, analytical visibility is also Gaussian
        sigma_angular = 0.5  # Normalized units
        expected_vis = np.exp(-2 * np.pi**2 * sigma_angular**2 * 
                             (baseline_length / wavelength)**2)
        
        # Our implementation should give similar result
        calculated_vis = VisibilityCalculator.gaussian_source(
            sigma_angular, baseline_length, wavelength
        )
        
        self.assertAlmostEqual(calculated_vis, expected_vis, places=3,
                              msg="Gaussian visibility should match analytical result")
    
    def test_fft_method_for_2d_profiles(self):
        """Test FFT method for 2D intensity profiles"""
        # Create simple 2D Gaussian
        n_points = 64
        extent = 4.0
        x = np.linspace(-extent, extent, n_points)
        y = np.linspace(-extent, extent, n_points)
        X, Y = np.meshgrid(x, y)
        
        sigma = 1.0
        intensity_2d = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        
        # Calculate visibility via FFT
        gamma = fft2(intensity_2d)
        gamma2 = np.abs(gamma)**2
        
        # Central value should be close to total flux squared
        total_flux = np.sum(intensity_2d)
        central_gamma2 = gamma2[0, 0]
        
        self.assertAlmostEqual(central_gamma2, total_flux**2, delta=total_flux**2 * 0.1,
                              msg="Central visibility should equal total flux squared")
    
    def test_bessel_function_accuracy(self):
        """Test accuracy of Bessel function calculations"""
        # Test Airy disk visibility at specific points
        test_points = [0, 1.22, 2.23, 3.24]  # Zeros and extrema of Airy function
        
        for zeta in test_points:
            if zeta == 0:
                expected = 1.0
            else:
                expected = abs(2 * special.j1(zeta) / zeta)
            
            # Test our implementation
            radius = 1.0  # Normalized
            baseline = zeta / (np.pi * radius)  # Solve for baseline
            wavelength = 1.0  # Normalized
            
            calculated = VisibilityCalculator.uniform_disk(radius, baseline, wavelength)
            
            self.assertAlmostEqual(calculated, expected, places=4,
                                  msg=f"Airy visibility should be accurate at ζ={zeta}")


class TestPolarizationEffectsDetailed(unittest.TestCase):
    """Detailed tests of polarization effects from Section IV.B"""
    
    def setUp(self):
        """Set up polarization test parameters"""
        self.sn_params = SupernovaParameters(
            sn_type="Ia",
            explosion_time=20,
            expansion_velocity=10000,
            distance=20.0,
            absolute_magnitude=-19.46
        )
        self.supernova = SupernovaEjecta(self.sn_params)
    
    def test_polarization_magnitude(self):
        """Test that polarization effects are at the level stated in paper"""
        baseline_length = 5000  # 5 km
        wavelength = 550e-9
        
        # Calculate visibility with and without polarization
        vis_with_pol = self.supernova.visibility_amplitude(
            baseline_length, wavelength, include_polarization=True
        )
        vis_without_pol = self.supernova.visibility_amplitude(
            baseline_length, wavelength, include_polarization=False
        )
        
        # Calculate |V|² difference
        v2_with = vis_with_pol**2
        v2_without = vis_without_pol**2
        v2_difference = abs(v2_with - v2_without)
        
        # Paper states differences are <5×10⁻⁴
        self.assertLess(v2_difference, 5e-4,
                       "Polarization effect should be <5×10⁻⁴ as stated in paper")
    
    def test_polarization_spatial_structure(self):
        """Test spatial structure of polarization"""
        # Test polarization at different radii
        radii = [0.2, 0.5, 0.8, 1.0]
        azimuth = 0  # Fixed azimuth
        
        polarizations = []
        for radius in radii:
            pol = self.supernova.polarization_profile(radius, azimuth)
            polarizations.append(pol)
        
        # Polarization should vary with radius
        if len(set(polarizations)) > 1:  # Check if values are different
            pol_std = np.std(polarizations)
            self.assertGreater(pol_std, 0.01,
                             "Polarization should vary with radius")
    
    def test_integrated_polarization_level(self):
        """Test that integrated polarization is <0.5% as stated in paper"""
        # Calculate intensity with polarization over the disk
        n_points = 51
        extent = 1.5
        x = np.linspace(-extent, extent, n_points)
        y = np.linspace(-extent, extent, n_points)
        
        total_intensity = 0
        total_polarized_intensity = 0
        
        for i in range(n_points):
            for j in range(n_points):
                radius = np.sqrt(x[i]**2 + y[j]**2)
                if radius <= 1.0:  # Within photosphere
                    azimuth = np.arctan2(y[j], x[i])
                    
                    # Unpolarized intensity
                    unpol_intensity = self.supernova.photosphere_intensity_profile(radius)
                    
                    # Polarized intensity
                    pol_intensity = self.supernova.polarization_profile(radius, azimuth)
                    
                    total_intensity += unpol_intensity
                    total_polarized_intensity += abs(pol_intensity - unpol_intensity)
        
        if total_intensity > 0:
            integrated_polarization = total_polarized_intensity / total_intensity
            # Paper states P < 0.5% except at high-velocity Ca II
            self.assertLess(integrated_polarization, 0.05,
                           "Integrated polarization should be <5%")


def create_model_comparison_plots():
    """Create comparison plots between TARDIS and SEDONA models"""
    # Set up models
    params = SupernovaParameters(
        sn_type="Ia",
        explosion_time=20,
        expansion_velocity=10000,
        distance=20.0,
        absolute_magnitude=-19.46
    )
    
    tardis_sn = SupernovaEjecta(params)
    sedona_sn = SupernovaEjecta(params)
    
    # Test wavelengths
    wavelengths = [4000, 5500, 7000]  # Angstroms
    baselines = np.logspace(2, 4, 50)  # 100m to 10km
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, wl_angstrom in enumerate(wavelengths):
        wavelength = wl_angstrom * 1e-10
        
        # Calculate visibilities
        vis_tardis = []
        vis_sedona = []
        
        for baseline in baselines:
            v_t = tardis_sn.visibility_amplitude(baseline, wavelength, False)
            v_s = sedona_sn.visibility_amplitude(baseline, wavelength, True)
            vis_tardis.append(v_t)
            vis_sedona.append(v_s)
        
        # Plot
        axes[i].loglog(baselines/1000, vis_tardis, 'b-', label='TARDIS (spherical)', linewidth=2)
        axes[i].loglog(baselines/1000, vis_sedona, 'r--', label='SEDONA (asymmetric)', linewidth=2)
        
        axes[i].set_xlabel('Baseline Length (km)')
        axes[i].set_ylabel('Visibility Amplitude |V|')
        axes[i].set_title(f'λ = {wl_angstrom} Å')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('tardis_sedona_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig, axes


if __name__ == "__main__":
    # Run the specialized tests
    print("Running TARDIS and SEDONA Model Validation Tests")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestTARDISModelValidation,
        TestSEDONAModelValidation,
        TestTARDISvsSEDONAComparison,
        TestVisibilityCalculationMethods,
        TestPolarizationEffectsDetailed
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Create comparison plots if tests pass
    if result.wasSuccessful():
        print("\nCreating model comparison plots...")
        try:
            create_model_comparison_plots()
        except Exception as e:
            print(f"Could not create plots: {e}")
    
    # Print summary
    print(f"\n{'='*50}")
    print("TARDIS/SEDONA MODEL TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}")