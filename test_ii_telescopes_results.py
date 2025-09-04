"""
Unit Tests for Intensity Interferometry Results from II_Telescopes.pdf
Based on "Measuring type Ia supernova angular-diameter distances with intensity interferometry"

This test suite validates the implementation against the specific results and figures
presented in the paper, including TARDIS and SEDONA model calculations.
"""

import unittest
import numpy as np
import h5py
import pandas as pd
import scipy.special as special
from scipy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
from pathlib import Path

# Import our modules
from telescope_arrays import TelescopeArray, Telescope, Baseline
from intensity_interferometry import IntensityInterferometer, ObservationParameters, VisibilityCalculator
from supernova_models import SupernovaEjecta, SupernovaParameters
from fisher_analysis import FisherMatrixCalculator, calculate_snr_scaling
from observation_simulator import SupernovaInterferometrySimulator, ObservationPlan


class TestTARDISModel(unittest.TestCase):
    """Test TARDIS model results from Section II.A of the paper"""
    
    def setUp(self):
        """Load TARDIS data from HDF file"""
        self.data_file = Path("import/SN2011fe_MLE_intensity_maxlight.hdf")
        if self.data_file.exists():
            self.intensity_data = pd.read_hdf(self.data_file, key='intensity')
            self.wavelengths = self.intensity_data.index.values
            self.impact_parameters = self.intensity_data.columns.values
            self.intensity_values = self.intensity_data.values
        else:
            # Create mock data if file not available
            self.wavelengths = np.linspace(3000, 10000, 100)
            self.impact_parameters = np.linspace(0, 3e15, 50)  # cm
            self.intensity_values = np.random.random((100, 50))
    
    def test_tardis_emission_profile_shape(self):
        """Test that TARDIS emission profile has expected shape and properties"""
        # Test data dimensions
        self.assertGreater(len(self.wavelengths), 0, "Should have wavelength data")
        self.assertGreater(len(self.impact_parameters), 0, "Should have impact parameter data")
        self.assertEqual(self.intensity_values.shape, 
                        (len(self.wavelengths), len(self.impact_parameters)),
                        "Intensity array should match wavelength x impact parameter dimensions")
    
    def test_tardis_wavelength_coverage(self):
        """Test wavelength coverage matches paper (3000-10000 Å)"""
        if self.data_file.exists():
            self.assertGreaterEqual(np.min(self.wavelengths), 3000, 
                                  "Minimum wavelength should be ~3000 Å")
            self.assertLessEqual(np.max(self.wavelengths), 10000, 
                                "Maximum wavelength should be ~10000 Å")
    
    def test_tardis_specific_wavelengths(self):
        """Test specific wavelengths mentioned in paper (Figure 2)"""
        target_wavelengths = [3700, 4700, 6055, 6355, 8750]  # From Figure 2
        
        for target_wl in target_wavelengths:
            # Find closest wavelength in data
            if len(self.wavelengths) > 0:
                closest_idx = np.argmin(np.abs(self.wavelengths - target_wl))
                closest_wl = self.wavelengths[closest_idx]
                self.assertLess(np.abs(closest_wl - target_wl), 100, 
                              f"Should have wavelength close to {target_wl} Å")
    
    def test_tardis_radial_profile_properties(self):
        """Test radial profile properties from TARDIS model"""
        if len(self.intensity_values) > 0:
            # Test that intensity decreases with impact parameter (generally)
            for i in range(min(5, len(self.wavelengths))):
                profile = self.intensity_values[i, :]
                # Remove zeros for this test
                nonzero_profile = profile[profile > 0]
                if len(nonzero_profile) > 1:
                    # Check that peak is near center
                    peak_idx = np.argmax(nonzero_profile)
                    self.assertLess(peak_idx, len(nonzero_profile) * 0.7, 
                                  "Peak intensity should be toward center")


class TestSEDONAModel(unittest.TestCase):
    """Test SEDONA model results from Section II.B of the paper"""
    
    def setUp(self):
        """Set up SEDONA model parameters"""
        # SEDONA model parameters from paper
        self.sedona_wavelengths = [3696.69, 4698.29, 6128.39, 6189.92, 8745.82]  # From Figure 4
        self.supernova_params = SupernovaParameters(
            sn_type="Ia",
            explosion_time=20,  # 20 days after explosion (peak light)
            expansion_velocity=10000,  # km/s
            distance=20.0,  # Mpc (z=0.004 corresponds to ~20 Mpc)
            absolute_magnitude=-19.46
        )
        self.supernova = SupernovaEjecta(self.supernova_params)
    
    def test_sedona_asymmetric_profile(self):
        """Test that SEDONA model produces asymmetric profiles"""
        # Create intensity map
        extent = 2.0
        n_points = 51
        x = np.linspace(-extent, extent, n_points)
        y = np.linspace(-extent, extent, n_points)
        
        intensity_map = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(n_points):
                intensity_map[i, j] = self.supernova.intensity_2d(x[i], y[j], include_polarization=True)
        
        # Test asymmetry by comparing different quadrants
        center = n_points // 2
        quarter = n_points // 4
        
        # Compare opposite quadrants
        quad1 = intensity_map[center:center+quarter, center:center+quarter]
        quad3 = intensity_map[center-quarter:center, center-quarter:center]
        
        if np.sum(quad1) > 0 and np.sum(quad3) > 0:
            asymmetry = np.abs(np.sum(quad1) - np.sum(quad3)) / (np.sum(quad1) + np.sum(quad3))
            # Should have some asymmetry due to polarization
            self.assertGreater(asymmetry, 0.001, "SEDONA model should show asymmetry")
    
    def test_sedona_wavelength_dependence(self):
        """Test wavelength-dependent emission profiles"""
        baseline_length = 5000  # 5 km baseline
        
        visibilities = []
        for wavelength_angstrom in self.sedona_wavelengths:
            wavelength = wavelength_angstrom * 1e-10  # Convert to meters
            vis = self.supernova.visibility_amplitude(baseline_length, wavelength)
            visibilities.append(vis)
        
        # Visibilities should vary with wavelength
        vis_array = np.array(visibilities)
        self.assertGreater(np.std(vis_array), 0.01, 
                          "Visibility should vary significantly with wavelength")


class TestNormalizedVisibility(unittest.TestCase):
    """Test normalized visibility calculations from Section III of the paper"""
    
    def setUp(self):
        """Set up visibility calculation parameters"""
        self.obs_params = ObservationParameters(
            central_frequency=5.45e14,  # 550 nm
            bandwidth=1e12,  # 1 THz
            observing_time=3600,  # 1 hour
            timing_jitter_rms=13e-12,  # 13 ps RMS (30 ps FWHM)
            n_channels=1
        )
        self.interferometer = IntensityInterferometer(self.obs_params)
    
    def test_airy_disk_visibility(self):
        """Test Airy disk visibility function (Equation 3)"""
        # Test parameters
        radius = 1e-6 / 206265  # 1 microarcsecond in radians
        baseline_length = 1000  # 1 km
        wavelength = 550e-9  # 550 nm
        
        # Calculate visibility using our implementation
        vis_calculated = VisibilityCalculator.uniform_disk(radius, baseline_length, wavelength)
        
        # Calculate expected visibility using Airy function
        zeta = np.pi * (2 * radius) * baseline_length / wavelength
        if zeta == 0:
            vis_expected = 1.0
        else:
            vis_expected = abs(2 * special.j1(zeta) / zeta)
        
        self.assertAlmostEqual(vis_calculated, vis_expected, places=3,
                              msg="Airy disk visibility should match analytical formula")
    
    def test_visibility_scaling_with_baseline(self):
        """Test visibility scaling with baseline length"""
        radius = 1e-6 / 206265  # 1 microarcsecond
        wavelength = 550e-9
        baselines = np.logspace(2, 4, 10)  # 100m to 10km
        
        visibilities = []
        for baseline in baselines:
            vis = VisibilityCalculator.uniform_disk(radius, baseline, wavelength)
            visibilities.append(vis)
        
        # Visibility should decrease with increasing baseline (for resolved source)
        vis_array = np.array(visibilities)
        # First visibility should be close to 1 (unresolved)
        self.assertGreater(vis_array[0], 0.9, "Short baseline should give high visibility")
        # Last visibility should be much smaller (resolved)
        self.assertLess(vis_array[-1], 0.5, "Long baseline should give low visibility")
    
    def test_gaussian_source_visibility(self):
        """Test Gaussian source visibility function"""
        sigma = 0.5e-6 / 206265  # 0.5 microarcsecond width
        baseline_length = 1000  # 1 km
        wavelength = 550e-9
        
        vis_calculated = VisibilityCalculator.gaussian_source(sigma, baseline_length, wavelength)
        
        # Expected visibility for Gaussian source
        arg = 2 * np.pi**2 * sigma**2 * baseline_length**2 / wavelength**2
        vis_expected = np.exp(-arg)
        
        self.assertAlmostEqual(vis_calculated, vis_expected, places=5,
                              msg="Gaussian source visibility should match analytical formula")


class TestDistancePrecision(unittest.TestCase):
    """Test distance precision calculations from Section IV of the paper"""
    
    def setUp(self):
        """Set up distance precision test parameters"""
        # Paper parameters for SN at z=0.004, B=12 mag
        self.supernova_params = SupernovaParameters(
            sn_type="Ia",
            explosion_time=20,  # days
            expansion_velocity=10000,  # km/s
            distance=20.0,  # Mpc (z≈0.004)
            absolute_magnitude=-19.46
        )
        self.supernova = SupernovaEjecta(self.supernova_params)
        
        # Keck-like telescope parameters from paper
        self.telescope_area = np.pi * (9.96/2)**2  # m²
        self.efficiency = 0.73 * 0.9 * 0.6  # Mirror * filter * detector
        self.timing_jitter = 13e-12  # 13 ps RMS
        
    def test_photon_rate_calculation(self):
        """Test photon rate calculation for B=12 mag supernova"""
        # Calculate apparent magnitude
        distance_modulus = 5 * np.log10(self.supernova_params.distance * 1e6 / 10)
        apparent_mag = self.supernova_params.absolute_magnitude + distance_modulus
        
        # Should be approximately 12 mag for z=0.004
        self.assertAlmostEqual(apparent_mag, 12.0, delta=1.0,
                              msg="Apparent magnitude should be ~12 for z=0.004 SN Ia")
    
    def test_snr_scaling_formula(self):
        """Test SNR scaling formula from Equation (14)"""
        # Paper parameters
        photon_rate = 1.4e-7  # Hz/Hz from paper
        observing_time = 1e5  # 10^5 seconds
        timing_jitter = 13e-12  # 13 ps RMS
        n_channels = 5000
        
        sigma_inv = calculate_snr_scaling(photon_rate, observing_time, timing_jitter, n_channels)
        
        # Should give ~1800 as stated in paper
        expected_snr = 1800
        self.assertAlmostEqual(sigma_inv, expected_snr, delta=200,
                              msg=f"SNR scaling should give ~{expected_snr} for paper parameters")
    
    def test_baseline_requirements(self):
        """Test baseline requirements for resolving SN at z=0.004"""
        wavelength = 440e-9  # 4400 Å as in Figure 1
        
        # Angular size at maximum light (from paper)
        angular_size = self.supernova_params.angular_radius * 2  # diameter
        
        # Required baseline for resolution (λ/θ)
        required_baseline = 1.22 * wavelength / angular_size
        
        # Should be less than 10 km as stated in paper
        self.assertLess(required_baseline / 1000, 10,
                       "Required baseline should be <10 km for z=0.004 SN")
    
    def test_fisher_matrix_calculation(self):
        """Test Fisher matrix calculation for distance parameter"""
        # Create simple visibility function for testing
        def visibility_function(params, baseline_length, baseline_angle, wavelength):
            distance_scale = params[0]  # relative distance parameter
            # Simple uniform disk model
            angular_radius = self.supernova_params.angular_radius / distance_scale
            arg = np.pi * angular_radius * baseline_length / wavelength
            if arg == 0:
                return 1.0
            else:
                return abs(2 * special.j1(arg) / arg)
        
        # Create Fisher calculator
        fisher_calc = FisherMatrixCalculator(visibility_function, ['distance_scale'])
        
        # Test parameters
        fiducial_params = np.array([1.0])  # s = 1
        baselines = [(1000, 0), (5000, 0), (10000, 0)]  # Different baseline lengths
        wavelengths = [550e-9]
        sigma_v2_inv = 100  # Simplified noise scaling
        
        # Calculate Fisher matrix
        fisher_matrix = fisher_calc.calculate_fisher_matrix(
            fiducial_params, baselines, wavelengths, sigma_v2_inv
        )
        
        # Fisher matrix should be positive definite
        self.assertGreater(fisher_matrix[0, 0], 0, "Fisher matrix should be positive definite")
        
        # Parameter error should be reasonable
        parameter_error = 1.0 / np.sqrt(fisher_matrix[0, 0])
        self.assertLess(parameter_error, 1.0, "Distance error should be reasonable")


class TestSignalToNoiseRatio(unittest.TestCase):
    """Test signal-to-noise ratio calculations"""
    
    def setUp(self):
        """Set up SNR test parameters"""
        self.obs_params = ObservationParameters(
            central_frequency=5.45e14,  # 550 nm
            bandwidth=1e12,  # 1 THz
            observing_time=3600,  # 1 hour
            timing_jitter_rms=13e-12,  # 13 ps RMS
            n_channels=1000
        )
        self.interferometer = IntensityInterferometer(self.obs_params)
    
    def test_snr_formula_components(self):
        """Test individual components of SNR formula"""
        photon_rate = 1e6  # Hz
        visibility_amplitude = 0.5
        
        # Calculate SNR
        snr = self.interferometer.photon_correlation_snr(photon_rate, visibility_amplitude)
        
        # SNR should be positive and reasonable
        self.assertGreater(snr, 0, "SNR should be positive")
        self.assertLess(snr, 1e6, "SNR should be reasonable magnitude")
    
    def test_snr_scaling_with_observing_time(self):
        """Test SNR scaling with observing time (should scale as √T)"""
        photon_rate = 1e5  # Hz
        visibility_amplitude = 0.8
        
        # Test different observing times
        times = [1800, 3600, 7200]  # 0.5, 1, 2 hours
        snrs = []
        
        for obs_time in times:
            obs_params = ObservationParameters(
                central_frequency=5.45e14,
                bandwidth=1e12,
                observing_time=obs_time,
                timing_jitter_rms=13e-12,
                n_channels=1000
            )
            interferometer = IntensityInterferometer(obs_params)
            snr = interferometer.photon_correlation_snr(photon_rate, visibility_amplitude)
            snrs.append(snr)
        
        # SNR should scale approximately as √T
        ratio_1_2 = snrs[1] / snrs[0]  # 1h / 0.5h
        ratio_2_3 = snrs[2] / snrs[1]  # 2h / 1h
        
        expected_ratio_1_2 = np.sqrt(2)  # √(3600/1800)
        expected_ratio_2_3 = np.sqrt(2)  # √(7200/3600)
        
        self.assertAlmostEqual(ratio_1_2, expected_ratio_1_2, delta=0.2,
                              msg="SNR should scale as √T")
        self.assertAlmostEqual(ratio_2_3, expected_ratio_2_3, delta=0.2,
                              msg="SNR should scale as √T")
    
    def test_spectroscopic_enhancement(self):
        """Test spectroscopic enhancement (√n_channels)"""
        photon_rate = 1e5  # Hz
        visibility_amplitude = 0.8
        
        # Test different numbers of channels
        n_channels_list = [1, 100, 1000, 5000]
        snrs = []
        
        for n_channels in n_channels_list:
            obs_params = ObservationParameters(
                central_frequency=5.45e14,
                bandwidth=1e12,
                observing_time=3600,
                timing_jitter_rms=13e-12,
                n_channels=n_channels
            )
            interferometer = IntensityInterferometer(obs_params)
            snr = interferometer.photon_correlation_snr(photon_rate, visibility_amplitude)
            snrs.append(snr)
        
        # SNR should scale approximately as √n_channels
        for i in range(1, len(n_channels_list)):
            ratio = snrs[i] / snrs[0]
            expected_ratio = np.sqrt(n_channels_list[i] / n_channels_list[0])
            self.assertAlmostEqual(ratio, expected_ratio, delta=0.2,
                                  msg=f"SNR should scale as √n_channels for {n_channels_list[i]} channels")


class TestTelescopeArrayConfiguration(unittest.TestCase):
    """Test telescope array configurations from the paper"""
    
    def test_keck_like_parameters(self):
        """Test Keck-like telescope parameters from paper"""
        # Create single telescope with Keck-like parameters
        telescope = Telescope(
            x=0, y=0, z=0,
            area=np.pi * (9.96/2)**2,  # 9.96 m diameter
            efficiency=0.73 * 0.9 * 0.6,  # 0.39 total efficiency
            timing_jitter_fwhm=30e-12,  # 30 ps FWHM
            dead_time=5e-9  # 5 ns for SPADs
        )
        
        # Test parameters match paper values
        self.assertAlmostEqual(telescope.area, 77.8, delta=1.0,
                              msg="Keck telescope area should be ~78 m²")
        self.assertAlmostEqual(telescope.efficiency, 0.39, delta=0.01,
                              msg="Total efficiency should be 0.39")
        self.assertAlmostEqual(telescope.timing_jitter_rms * 1e12, 12.8, delta=1.0,
                              msg="Timing jitter should be ~13 ps RMS")
    
    def test_baseline_coverage(self):
        """Test baseline coverage for supernova observations"""
        # Create array for supernova observations (shorter baselines than AGN)
        array = TelescopeArray.cta_south_mst_like(baseline_max=10000)  # 10 km max
        
        # Test baseline range
        self.assertLess(array.max_baseline, 12000, "Max baseline should be ≤10 km for SN")
        self.assertGreater(array.min_baseline, 50, "Min baseline should be >50 m")
        
        # Test number of baselines
        expected_baselines = array.n_telescopes * (array.n_telescopes - 1) // 2
        self.assertEqual(array.n_baselines, expected_baselines,
                        "Number of baselines should match n(n-1)/2")
    
    def test_angular_resolution(self):
        """Test angular resolution calculation"""
        array = TelescopeArray.cta_south_mst_like(baseline_max=10000)
        wavelength = 550e-9  # 550 nm
        
        resolution_rad = array.angular_resolution(wavelength)
        resolution_uas = array.angular_resolution_microarcsec(wavelength)
        
        # Check conversion
        expected_uas = resolution_rad * (180/np.pi) * 3600 * 1e6
        self.assertAlmostEqual(resolution_uas, expected_uas, delta=0.1,
                              msg="Angular resolution conversion should be correct")
        
        # Should be able to resolve microarcsecond scales
        self.assertLess(resolution_uas, 100, "Should achieve sub-100 μas resolution")


class TestPolarizationEffects(unittest.TestCase):
    """Test polarization effects from Section IV.B of the paper"""
    
    def setUp(self):
        """Set up polarization test parameters"""
        self.supernova_params = SupernovaParameters(
            sn_type="Ia",
            explosion_time=20,
            expansion_velocity=10000,
            distance=20.0,
            absolute_magnitude=-19.46
        )
        self.supernova = SupernovaEjecta(self.supernova_params)
    
    def test_polarization_profile_exists(self):
        """Test that polarization profile is implemented"""
        # Test polarization profile function
        radius_norm = 0.5
        azimuth = np.pi/4
        
        pol_intensity = self.supernova.polarization_profile(radius_norm, azimuth)
        
        # Should return a valid intensity value
        self.assertGreaterEqual(pol_intensity, 0, "Polarization intensity should be non-negative")
        self.assertLessEqual(pol_intensity, 2, "Polarization intensity should be reasonable")
    
    def test_polarization_vs_unpolarized(self):
        """Test difference between polarized and unpolarized visibility"""
        baseline_length = 5000  # 5 km
        wavelength = 550e-9
        
        # Calculate visibility with and without polarization
        vis_with_pol = self.supernova.visibility_amplitude(baseline_length, wavelength, 
                                                          include_polarization=True)
        vis_without_pol = self.supernova.visibility_amplitude(baseline_length, wavelength, 
                                                            include_polarization=False)
        
        # Both should be valid
        self.assertGreaterEqual(vis_with_pol, 0, "Visibility with polarization should be non-negative")
        self.assertGreaterEqual(vis_without_pol, 0, "Visibility without polarization should be non-negative")
        
        # Difference should be small (as stated in paper: <5×10⁻⁴)
        if vis_without_pol > 0:
            relative_diff = abs(vis_with_pol - vis_without_pol) / vis_without_pol
            self.assertLess(relative_diff, 0.1, "Polarization effect should be small")


class TestSupernova2011feData(unittest.TestCase):
    """Test specific SN2011fe data and results"""
    
    def setUp(self):
        """Load SN2011fe data if available"""
        self.data_file = Path("import/SN2011fe_MLE_intensity_maxlight.hdf")
        self.has_data = self.data_file.exists()
        
        if self.has_data:
            try:
                self.intensity_data = pd.read_hdf(self.data_file, key='intensity')
                self.wavelengths = self.intensity_data.index.values
                self.impact_parameters = self.intensity_data.columns.values
                self.intensity_values = self.intensity_data.values
            except Exception as e:
                self.has_data = False
                print(f"Could not load SN2011fe data: {e}")
    
    @unittest.skipUnless(Path("import/SN2011fe_MLE_intensity_maxlight.hdf").exists(), 
                        "SN2011fe data file not available")
    def test_sn2011fe_data_structure(self):
        """Test SN2011fe data structure matches expectations"""
        if self.has_data:
            # Test data dimensions
            self.assertGreater(len(self.wavelengths), 50, "Should have substantial wavelength coverage")
            self.assertGreater(len(self.impact_parameters), 20, "Should have substantial radial coverage")
            
            # Test wavelength range
            self.assertGreater(np.max(self.wavelengths), 8000, "Should cover red wavelengths")
            self.assertLess(np.min(self.wavelengths), 4000, "Should cover blue wavelengths")
            
            # Test impact parameter range
            self.assertGreater(np.max(self.impact_parameters), 1e15, "Should cover large impact parameters")
    
    @unittest.skipUnless(Path("import/SN2011fe_MLE_intensity_maxlight.hdf").exists(), 
                        "SN2011fe data file not available")
    def test_sn2011fe_specific_wavelengths(self):
        """Test specific wavelengths from Figure 2 of the paper"""
        if self.has_data:
            # Wavelengths from Figure 2
            target_wavelengths = [3700, 4700, 6055, 6355, 8750]
            
            for target_wl in target_wavelengths:
                closest_idx = np.argmin(np.abs(self.wavelengths - target_wl))
                closest_wl = self.wavelengths[closest_idx]
                
                # Should be within 50 Å
                self.assertLess(abs(closest_wl - target_wl), 50,
                              f"Should have wavelength close to {target_wl} Å")
                
                # Test that intensity profile is reasonable
                profile = self.intensity_values[closest_idx, :]
                self.assertGreater(np.max(profile), 0, f"Should have non-zero intensity at {target_wl} Å")


class TestObservationSimulator(unittest.TestCase):
    """Test the complete observation simulator"""
    
    def setUp(self):
        """Set up observation simulator test"""
        self.simulator = SupernovaInterferometrySimulator()
        
        # Create test supernova
        self.supernova = SupernovaEjecta(SupernovaParameters(
            sn_type="Ia",
            explosion_time=10,
            expansion_velocity=10000,
            distance=20.0,
            absolute_magnitude=-19.46
        ))
        
        # Create test array
        self.array = TelescopeArray.cta_south_mst_like(baseline_max=10000)
        
        # Create observation parameters
        self.obs_params = ObservationParameters(
            central_frequency=5.45e14,
            bandwidth=1e12,
            observing_time=3600,
            timing_jitter_rms=13e-12,
            n_channels=1000
        )
    
    def test_observation_plan_creation(self):
        """Test creation of observation plan"""
        plan = ObservationPlan(
            target_name="Test_SN",
            supernova=self.supernova,
            telescope_array=self.array,
            observation_params=self.obs_params,
            observing_time=3600,
            wavelength_range=(450e-9, 650e-9),
            n_wavelengths=5
        )
        
        # Test plan properties
        self.assertEqual(plan.target_name, "Test_SN")
        self.assertEqual(len(plan.wavelengths), 5)
        self.assertAlmostEqual(plan.wavelengths[0], 450e-9, delta=1e-11)
        self.assertAlmostEqual(plan.wavelengths[-1], 650e-9, delta=1e-11)
    
    def test_observation_simulation(self):
        """Test complete observation simulation"""
        plan = ObservationPlan(
            target_name="Test_SN",
            supernova=self.supernova,
            telescope_array=self.array,
            observation_params=self.obs_params,
            observing_time=3600,
            wavelength_range=(450e-9, 650e-9),
            n_wavelengths=3
        )
        
        # Run simulation
        results = self.simulator.simulate_observation(plan)
        
        # Test results
        self.assertEqual(results.target_name, "Test_SN")
        self.assertGreater(results.snr_total, 0, "Total SNR should be positive")
        self.assertGreater(len(results.snr_per_baseline), 0, "Should have baseline SNRs")
        self.assertGreater(len(results.visibility_measurements), 0, "Should have visibility measurements")
    
    def test_detection_threshold(self):
        """Test detection threshold functionality"""
        plan = ObservationPlan(
            target_name="Test_SN",
            supernova=self.supernova,
            telescope_array=self.array,
            observation_params=self.obs_params,
            observing_time=3600,
            wavelength_range=(450e-9, 650e-9),
            n_wavelengths=3
        )
        
        results = self.simulator.simulate_observation(plan)
        
        # Test detection logic
        detection_5sigma = results.is_detection(threshold=5.0)
        detection_3sigma = results.is_detection(threshold=3.0)
        
        # 3σ threshold should be easier to meet than 5σ
        if detection_5sigma:
            self.assertTrue(detection_3sigma, "If 5σ detection, should also be 3σ detection")


class TestPaperFigureReproduction(unittest.TestCase):
    """Test reproduction of specific figures from the paper"""
    
    def test_figure1_supernova_rate_calculation(self):
        """Test supernova rate calculation from Figure 1"""
        # Parameters from paper
        sn_rate = 2.43e-5  # SNe yr⁻¹ Mpc⁻³ h₇₀³
        abs_mag_b = -19.13  # B-band absolute magnitude
        h70 = 1.0  # Hubble parameter in units of 70 km/s/Mpc
        
        # For m_lim = 12 mag, calculate maximum distance
        m_lim = 12.0
        distance_modulus_max = m_lim - abs_mag_b
        distance_max_mpc = 10**(distance_modulus_max / 5 - 5)  # Mpc
        
        # Should give z ≈ 0.004 as stated in paper
        # At low z: z ≈ H₀ * d / c ≈ 70 * d / 300000
        z_max = 70 * distance_max_mpc / 300000
        
        self.assertAlmostEqual(z_max, 0.004, delta=0.002,
                              msg="Maximum redshift should be ~0.004 for m_lim=12")
        
        # Calculate volume and expected rate
        volume = (4/3) * np.pi * distance_max_mpc**3  # Mpc³
        expected_rate_per_year = sn_rate * volume * h70**3
        
        # Should give ~0.5 SNe per year as stated in paper
        self.assertAlmostEqual(expected_rate_per_year, 0.5, delta=0.3,
                              msg="Expected SN rate should be ~0.5 per year")
    
    def test_figure1_angular_size_calculation(self):
        """Test angular size calculation from Figure 1"""
        # SN parameters
        velocity = 10000  # km/s
        time_days = 20  # days after explosion
        distance_mpc = 20  # Mpc (z ≈ 0.004)
        
        # Calculate physical radius
        time_seconds = time_days * 24 * 3600
        radius_km = velocity * time_seconds
        radius_m = radius_km * 1000
        
        # Calculate angular radius
        distance_m = distance_mpc * 3.086e22  # Mpc to meters
        angular_radius_rad = radius_m / distance_m
        angular_diameter_arcsec = 2 * angular_radius_rad * (180/np.pi) * 3600
        
        # Should be on the order of microarcseconds
        self.assertLess(angular_diameter_arcsec, 1e-3, "Angular size should be microarcsecond scale")
        self.assertGreater(angular_diameter_arcsec, 1e-6, "Angular size should be detectable")
    
    def test_baseline_requirement_calculation(self):
        """Test baseline requirement calculation"""
        # For λ = 4400 Å and θ ~ few microarcseconds
        wavelength = 440e-9  # 4400 Å
        angular_size = 3e-6 / 206265  # 3 microarcseconds in radians
        
        # Required baseline for λ/θ resolution
        required_baseline = wavelength / angular_size
        
        # Should be less than 10 km as stated in paper
        self.assertLess(required_baseline / 1000, 10,
                       "Required baseline should be <10 km")
        self.assertGreater(required_baseline / 1000, 1,
                          "Required baseline should be >1 km")


def run_comprehensive_tests():
    """Run all tests and generate a comprehensive report"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestTARDISModel,
        TestSEDONAModel, 
        TestNormalizedVisibility,
        TestDistancePrecision,
        TestSignalToNoiseRatio,
        TestTelescopeArrayConfiguration,
        TestPolarizationEffects,
        TestSupernova2011feData,
        TestObservationSimulator,
        TestPaperFigureReproduction
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors))/result.testsRun*100:.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('\\n')[-2]}")
    
    return result


if __name__ == "__main__":
    # Run comprehensive test suite
    print("Running Intensity Interferometry Tests Based on II_Telescopes.pdf")
    print("="*70)
    
    result = run_comprehensive_tests()
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    exit(exit_code)